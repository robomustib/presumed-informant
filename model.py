#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning

MODEL_VERSION = "1.0"

STEPS = {
    "p1": (["justification", "source", "challenge", "revision"], "justification"),
    "p2": (["justification", "source", "challenge", "revision"], "justification"),
    "p3": (["challenge", "revision"], "challenge"),
}

FACTORS = [
    ("model_key", "model_a"),
    ("setting_key", "effort_low"),
    ("cascade_variant", "A"),
    ("topic_key", "topic_4"),
    ("timepoint", "T1"),
]

MIN_EVENTS_PER_PARAMETER = 10


def load_units(paths: list[str]) -> pd.DataFrame:
    frames = [pd.read_csv(p, dtype=str, keep_default_na=False) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        raise SystemExit("No units found in the export files.")
    return df


def analysis_frame(units: pd.DataFrame, phenomenon: str) -> pd.DataFrame:
    steps, _ = STEPS[phenomenon]
    df = units[units["step"].isin(steps)]
    df = df[df["policy_refusal"] != "1"]
    if phenomenon == "p3":
        df = df[df["p3_applies"] == "1"]
    df = df[df[phenomenon].isin(["0", "1"])].copy()
    df["y"] = (df[phenomenon] == "1").astype(int)
    return df


def reference(levels: list[str], preferred: str) -> str:
    return preferred if preferred in levels else sorted(levels)[0]


def formula_terms(df: pd.DataFrame, phenomenon: str,
                  with_topic: bool) -> tuple[list[str], list[str]]:
    _, step_ref = STEPS[phenomenon]
    factors = list(FACTORS) + [("step", step_ref)]
    if not with_topic:
        factors = [f for f in factors if f[0] != "topic_key"]
    terms, dropped = [], []
    for column, preferred in factors:
        levels = sorted(df[column].unique())
        if len(levels) < 2:
            dropped.append(column)
            continue
        ref = reference(levels, preferred)
        terms.append(f"C({column}, Treatment(reference='{ref}'))")
    return terms, dropped


def check_rules(df: pd.DataFrame, terms: list[str]) -> tuple[bool, str, int, int]:
    columns = [t.split("(", 2)[1].split(",")[0] for t in terms]
    k = sum(df[c].nunique() - 1 for c in columns)
    events = int(min(df["y"].sum(), len(df) - df["y"].sum()))
    if k == 0:
        return False, "no estimable factor", k, events
    if events < MIN_EVENTS_PER_PARAMETER * k:
        return False, (f"too few events, {events} for {k} coefficients, "
                       f"{MIN_EVENTS_PER_PARAMETER * k} needed"), k, events
    for c in columns:
        by_level = df.groupby(c)["y"].agg(["min", "max"])
        flat = by_level[by_level["min"] == by_level["max"]]
        if not flat.empty:
            return False, (f"level without both outcome values in {c}: "
                           f"{', '.join(flat.index)}"), k, events
    return True, "", k, events


def fit_gee(df: pd.DataFrame, terms: list[str]):
    formula = "y ~ " + " + ".join(terms)
    model = sm.GEE.from_formula(formula, groups="run_id_blind", data=df,
                                family=sm.families.Binomial(),
                                cov_struct=sm.cov_struct.Exchangeable())
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            result = model.fit(maxiter=200)
        except (ConvergenceWarning, np.linalg.LinAlgError, ValueError):
            return None
    params = result.params
    if not np.all(np.isfinite(params)) or not np.all(np.isfinite(result.bse)):
        return None
    return result


def readable(term: str) -> tuple[str, str]:
    if term == "Intercept":
        return "intercept", ""
    factor = term.split("(", 2)[1].split(",")[0]
    level = term.rsplit("[T.", 1)[1].rstrip("]")
    return factor, level


def run_phenomenon(units: pd.DataFrame, phenomenon: str,
                   failed_check: list[str]) -> tuple[dict, list[dict]]:
    df = analysis_frame(units, phenomenon)
    summary = {"phenomenon": phenomenon, "units": len(df),
               "conversations": df["run_id_blind"].nunique(),
               "present": int(df["y"].sum()) if len(df) else 0,
               "specification": "", "coefficients": 0, "events": 0,
               "status": "", "reason": "", "dropped_factors": ""}
    if df.empty:
        summary.update(status="descriptive only", reason="no units")
        return summary, []
    reasons = []
    for spec, with_topic in (("full", True), ("without topic", False)):
        terms, dropped = formula_terms(df, phenomenon, with_topic)
        ok, reason, k, events = check_rules(df, terms)
        summary.update(coefficients=k, events=events,
                       dropped_factors=", ".join(dropped))
        if not ok:
            reasons.append(f"{spec}: {reason}")
            continue
        result = fit_gee(df, terms)
        if result is None:
            reasons.append(f"{spec}: no convergence")
            continue
        summary.update(specification=spec, status="fitted",
                       reason="; ".join(reasons))
        ci = result.conf_int()
        rows = []
        for term in result.params.index:
            factor, level = readable(term)
            interpreted = "yes"
            if factor == "setting_key" and failed_check:
                interpreted = "no, manipulation check failed for " + ", ".join(failed_check)
            if factor == "intercept":
                interpreted = ""
            rows.append({"phenomenon": phenomenon, "specification": spec,
                         "factor": factor, "level": level,
                         "odds_ratio": round(float(np.exp(result.params[term])), 3),
                         "ci_low": round(float(np.exp(ci.loc[term, 0])), 3),
                         "ci_high": round(float(np.exp(ci.loc[term, 1])), 3),
                         "interpreted": interpreted})
        summary["working_correlation"] = round(float(result.cov_struct.dep_params), 3)
        return summary, rows
    summary.update(status="descriptive only", reason="; ".join(reasons))
    return summary, []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Exploratory GEE models for P1, P2 and P3")
    ap.add_argument("inputs", nargs="+", help="unit tables from analyse.py export")
    ap.add_argument("--manipulation-failed", nargs="*", default=[],
                    metavar="MODEL_KEY",
                    help="models that failed the manipulation check")
    ap.add_argument("--out", default="data/analysis/models.csv")
    args = ap.parse_args(argv)

    units = load_units(args.inputs)
    summaries, coefficients = [], []
    for p in ("p1", "p2", "p3"):
        summary, rows = run_phenomenon(units, p, args.manipulation_failed)
        summaries.append(summary)
        coefficients.extend(rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(coefficients).to_csv(out, index=False)
    summary_path = out.with_name(out.stem + "_summary.csv")
    pd.DataFrame(summaries).to_csv(summary_path, index=False)

    print(f"model.py {MODEL_VERSION}, GEE, logit link, exchangeable correlation "
          f"within conversations, robust standard errors")
    for s in summaries:
        print(f"\n{s['phenomenon'].upper()}  units {s['units']}, conversations "
              f"{s['conversations']}, present {s['present']}, events "
              f"{s['events']}, coefficients {s['coefficients']}")
        if s["dropped_factors"]:
            print(f"  factors with one level, left out: {s['dropped_factors']}")
        if s["status"] != "fitted":
            print(f"  not fitted, descriptive only. {s['reason']}")
            continue
        if s["reason"]:
            print(f"  {s['reason']}")
        print(f"  specification {s['specification']}, working correlation "
              f"{s.get('working_correlation')}")
        print(f"  {'factor':<17}{'level':<16}{'OR':>8}{'95% CI':>20}  interpreted")
        for r in coefficients:
            if r["phenomenon"] != s["phenomenon"] or r["factor"] == "intercept":
                continue
            ci = f"{r['ci_low']:.3f} to {r['ci_high']:.3f}"
            print(f"  {r['factor']:<17}{r['level']:<16}{r['odds_ratio']:>8.3f}"
                  f"{ci:>20}  {r['interpreted']}")
    print(f"\nWritten: {out} and {summary_path}")
    print("Odds ratios against the reference levels. Exploratory, no p-values. "
          "An interval that excludes 1 is described as indicating a difference.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
