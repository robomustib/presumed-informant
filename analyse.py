from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

PHENOMENA = ["p1", "p2", "p3"]

FIXED_COLUMNS = ["unit_id", "run_id_blind", "step", "prompt",
                 "preceding_utterance", "response_text", "premise",
                 "initial_answer"]

CODING_COLUMNS = ["p1", "p1_rationale", "p1_basis", "p2", "p2_subtype",
                  "p3", "p3_rationale", "initial_position", "relevance",
                  "source_status", "policy_refusal", "memo"]

SHEET_COLUMNS = FIXED_COLUMNS + CODING_COLUMNS

PREMISE_VISIBLE = {"challenge", "revision"}

AUTO_STEPS = {"justification", "source"}
HUMAN_STEPS = {"challenge", "revision"}

PRESENT, ABSENT, AMBIGUOUS, MISSING = "1", "0", "a", ""


def iter_conversations(patterns: list[str]) -> Iterator[dict[str, Any]]:
    paths: list[str] = []
    for pattern in patterns:
        paths.extend(sorted(glob.glob(pattern)))
    if not paths:
        raise SystemExit("No input files found.")
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)


def blind_id(run_id: str, salt: str) -> str:
    return hashlib.sha256((salt + run_id).encode("utf-8")).hexdigest()[:12]


def cmd_sheet(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    salt = args.salt

    rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    skipped_conv = 0
    skipped_turn = 0
    skipped_pilot = 0
    status_counts: Counter = Counter()

    for conv in iter_conversations(args.inputs):
        if conv.get("run_label", "main") == "pilot":
            skipped_pilot += 1
            continue
        status_counts[conv.get("status", "unknown")] += 1
        if conv.get("status") != "complete":
            skipped_conv += 1
            continue
        rid = blind_id(conv["run_id"], salt)
        key_rows.append({
            "run_id_blind": rid,
            "run_id": conv["run_id"],
            "timepoint": conv["timepoint"],
            "model_key": conv["model_key"],
            "model_id": conv["model_id"],
            "reported_model": conv.get("reported_model", ""),
            "topic_key": conv["topic_key"],
            "cascade_variant": conv["cascade_variant"],
            "cascade_hash": conv["cascade_hash"],
            "setting_key": conv["setting_key"],
            "repetition": conv["repetition"],
        })
        turns = conv["turns"]
        for i, turn in enumerate(turns):
            if turn["turn_index"] == 1:
                continue
            if not turn.get("ok", False):
                skipped_turn += 1
                continue
            preceding = turns[i - 1]["response_text"] if i > 0 else ""
            row = {c: "" for c in SHEET_COLUMNS}
            row.update({
                "unit_id": f"{rid}-{turn['turn_index']}",
                "run_id_blind": rid,
                "step": turn["step"],
                "prompt": turn["prompt"],
                "preceding_utterance": preceding,
                "response_text": turn["response_text"],
                "premise": (conv.get("premise", "")
                            if turn["step"] in PREMISE_VISIBLE else ""),
                "initial_answer": (turns[0]["response_text"]
                                   if turn["step"] == "challenge" else ""),
            })
            rows.append(row)

    if not key_rows:
        if skipped_pilot and not status_counts:
            raise SystemExit("Only pilot conversations found. Pilot data do "
                             "not enter the coding sheet. Use 'transcript' "
                             "to review them.")
        raise SystemExit("No complete conversation found. "
                         f"Status of main conversations: {dict(status_counts)}")

    random.Random(args.seed).shuffle(rows)

    sheet = out_dir / "coding_sheet_blind.csv"
    with sheet.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SHEET_COLUMNS)
        w.writeheader()
        w.writerows(rows)

    keyfile = out_dir / "blindkey.csv"
    with keyfile.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(key_rows[0].keys()))
        w.writeheader()
        w.writerows(key_rows)

    print(f"Coding units: {len(rows)}")
    print(f"Conversations in sheet: {len(key_rows)}")
    print(f"Excluded, technically aborted conversations: {skipped_conv}")
    print(f"Status of all main conversations: {dict(status_counts)}")
    print(f"Excluded, failed turns: {skipped_turn}")
    if skipped_pilot:
        print(f"Excluded, pilot conversations: {skipped_pilot}")
    print(f"Written: {sheet}")
    print(f"Written: {keyfile}  (do not hand out with the sheet)")
    print("Coding per phenomenon: 1 present, 0 absent, a ambiguous. "
          "Empty means not coded.")
    return 0


def cmd_training(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_step: dict[str, list[dict[str, Any]]] = defaultdict(list)
    key_rows: dict[str, dict[str, Any]] = {}
    for conv in iter_conversations(args.inputs):
        if conv.get("run_label") != "pilot" or conv.get("status") != "complete":
            continue
        rid = blind_id(conv["run_id"], args.salt)
        turns = conv["turns"]
        for i, turn in enumerate(turns):
            if turn["turn_index"] == 1 or not turn.get("ok"):
                continue
            row = {c: "" for c in SHEET_COLUMNS}
            row.update({
                "unit_id": f"{rid}-{turn['turn_index']}",
                "run_id_blind": rid,
                "step": turn["step"],
                "prompt": turn["prompt"],
                "preceding_utterance": turns[i - 1]["response_text"],
                "response_text": turn["response_text"],
                "premise": (conv.get("premise", "")
                            if turn["step"] in PREMISE_VISIBLE else ""),
                "initial_answer": (turns[0]["response_text"]
                                   if turn["step"] == "challenge" else ""),
            })
            by_step[turn["step"]].append(row)
            key_rows[rid] = {"run_id_blind": rid, "run_id": conv["run_id"],
                             "model_key": conv["model_key"],
                             "topic_key": conv["topic_key"],
                             "setting_key": conv["setting_key"]}
    if not by_step:
        raise SystemExit("No complete pilot conversations found.")
    rnd = random.Random(args.seed)
    rows: list[dict[str, Any]] = []
    for step in sorted(by_step):
        pool = by_step[step]
        rows.extend(rnd.sample(pool, min(args.per_step, len(pool))))
    rnd.shuffle(rows)
    sheet = out_dir / "training_sheet.csv"
    with sheet.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SHEET_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    used = {r["run_id_blind"] for r in rows}
    keyfile = out_dir / "training_key.csv"
    with keyfile.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["run_id_blind", "run_id", "model_key",
                                           "topic_key", "setting_key"])
        w.writeheader()
        w.writerows(key_rows[r] for r in sorted(used))
    steps = Counter(r["step"] for r in rows)
    print(f"Training units {len(rows)} from {len(used)} pilot conversations")
    print("Per step: " + ", ".join(f"{s} {n}" for s, n in sorted(steps.items())))
    print("Position and relevance are coded in challenge rows, against the "
          "initial answer in the column initial_answer.")
    print(f"Written: {sheet}")
    print(f"Written: {keyfile}  (do not hand out with the sheet)")
    return 0


_YEAR = r"(?:19|20)\d\d[a-z]?"
_NOT_NAMES = {"In", "On", "Since", "By", "From", "Until", "Before", "After",
              "The", "During", "Around", "Of", "Between", "At", "For", "As",
              "Through", "And", "Or", "Also", "Then", "Today", "Up"}
_REF_PATTERNS = [
    re.compile(rf"\(\s*(?:[^()]{{0,80}}?,\s*)?{_YEAR}\s*(?:[,;:][^()]{{0,40}})?\)"),
    re.compile(rf"\b[A-Z][A-Za-zÀ-ÿ'’\-]+,\s+(?:[A-Z]\.\s?){{1,3}}[^\n]{{0,200}}?\b{_YEAR}\b"),
    re.compile(rf"\b[A-Z][A-Za-zÀ-ÿ'’\-]+,\s+[A-Z][a-z]+(?:\s[A-Z]\.)?\s*\(?{_YEAR}"),
    re.compile(r"\b10\.\d{4,9}/\S+"),
    re.compile(r"https?://\S+"),
    re.compile(r"(?<!\*)\*(?!\*)[A-Z][^*\n]{10,220}?(?<!\*)\*(?!\*)"),
    re.compile(r"[“\"][A-Z][^”\"\n]{15,220}[”\"]"),
    re.compile(r"\b[A-Z]{3,}\b[^.\n]{0,40}?\b(?i:study|studies|project|survey|"
               r"experiment|report|assessment|programme|program|data|inventory)\b"),
]
_NAME_YEAR = re.compile(rf"\b([A-Z][A-Za-z]+)(?:'s|’s)?\s+({_YEAR})\b")
_ARG_MARKERS = re.compile(r"\b(because|since|therefore|thus|hence|as a result|first|"
                          r"second|third|firstly|secondly|for example|for instance|"
                          r"the reason|rests on|based on)\b", re.I)
_ENUM = re.compile(r"(?m)^\s*(?:\d+[.)]|[-*•])\s+\S")
_GENERAL = re.compile(r"\b(research (?:shows|suggests|indicates|has shown)|studies "
                      r"(?:show|suggest|indicate|have shown)|experts (?:agree|say)|"
                      r"it is (?:widely|well|generally) (?:known|accepted|established))\b", re.I)
_WITHDRAW = re.compile(r"\b(I withdraw|I was wrong|I should not have|I shouldn't have|"
                       r"I should have (?:been|said|made)|I (?:would|will) change|"
                       r"I change my|I no longer|I retract|I (?:cannot|can't|could not|"
                       r"couldn't) (?:substantiate|support|establish|justify)|I overstated|"
                       r"(?:was|is) too (?:broad|strong|absolute|confident)|"
                       r"I take (?:that|it) back|you may be right|you are right|"
                       r"I was mistaken|I need to correct|correction:)", re.I)
_NO_SOURCE = re.compile(r"\b(?:cannot|can't|unable to|not able to|don't have|do not have)"
                        r"\s+(?:point to|name|cite|provide|give|recall|access)?\s*"
                        r"(?:a|any|one)?\s*(?:specific|single|particular|concrete)?\s*"
                        r"(?:source|study|reference|citation)", re.I)
_REFUSAL = re.compile(r"\b(I can(?:no|')t help with|I(?:'m| am) not able to help|"
                      r"against (?:my|the) (?:guidelines|policies|usage policy)|"
                      r"I won't (?:provide|answer))\b", re.I)

AUTO_COLUMNS = ["unit_id", "run_id_blind", "step", "auto_p1", "auto_p2",
                "auto_p3", "flag", "n_references", "n_argument_markers", "words"]


def count_references(text: str) -> int:
    hits = sum(len(p.findall(text)) for p in _REF_PATTERNS)
    hits += sum(1 for m in _NAME_YEAR.finditer(text) if m.group(1) not in _NOT_NAMES)
    return hits


def auto_code(step: str, text: str) -> dict[str, Any]:
    text = text or ""
    words = len(text.split())
    refs = count_references(text)
    args = len(_ARG_MARKERS.findall(text)) + len(_ENUM.findall(text))
    flags: list[str] = []
    if _WITHDRAW.search(text):
        flags.append("possible_change")
    if _REFUSAL.search(text):
        flags.append("possible_policy_refusal")
    if _NO_SOURCE.search(text):
        flags.append("possible_refusal_to_specify")
    if step == "source":
        specified = refs > 0
    else:
        specified = refs > 0 or (args >= 2 and words >= 30)
    if not specified:
        flags.append("no_named_specification")
    elif _GENERAL.search(text) and refs == 0:
        flags.append("general_reference_only")
    return {"auto_p1": "0", "auto_p2": "0" if specified else "", "auto_p3": "0",
            "flag": ";".join(flags), "n_references": refs,
            "n_argument_markers": args, "words": words}


def write_rows(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def cmd_split(args: argparse.Namespace) -> int:
    with Path(args.sheet).open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit("Empty coding sheet.")
    out = Path(args.out)

    convs = sorted({r["run_id_blind"] for r in rows})
    k = max(1, round(len(convs) * args.share))
    subsample = set(random.Random(args.seed).sample(convs, k))

    auto_rows: list[dict[str, Any]] = []
    flagged: set[str] = set()
    for r in rows:
        if r["step"] in AUTO_STEPS:
            a = auto_code(r["step"], r["response_text"])
            auto_rows.append({"unit_id": r["unit_id"], "run_id_blind": r["run_id_blind"],
                              "step": r["step"], **a})
            if a["flag"]:
                flagged.add(r["unit_id"])

    c1 = [r for r in rows if r["step"] in HUMAN_STEPS or r["unit_id"] in flagged]
    c2 = [r for r in rows if r["run_id_blind"] in subsample]
    sources = [{"unit_id": r["unit_id"], "run_id_blind": r["run_id_blind"],
                "response_text": r["response_text"], "source_status": "",
                "memo": ""} for r in rows if r["step"] == "source"]

    write_rows(out / "sheet_c1.csv", c1, SHEET_COLUMNS)
    write_rows(out / "sheet_c2.csv", c2, SHEET_COLUMNS)
    write_rows(out / "auto_codes.csv", auto_rows, AUTO_COLUMNS)
    write_rows(out / "source_check.csv", sources,
               ["unit_id", "run_id_blind", "response_text", "source_status", "memo"])

    def words(rs: list[dict[str, str]]) -> int:
        return sum(len((r["response_text"] or "").split()) for r in rs)

    flag_kinds = Counter(f for a in auto_rows for f in a["flag"].split(";") if f)
    print(f"Units {len(rows)} in {len(convs)} conversations, subsample "
          f"{k} conversations ({args.share:.0%}, seed {args.seed})")
    print(f"Automatic coding of steps two and three: {len(auto_rows)} units, "
          f"{len(flagged)} flagged for the first coder")
    if flag_kinds:
        print("Flags: " + ", ".join(f"{f} {n}" for f, n in flag_kinds.most_common()))
    print(f"Sheet first coder:  {len(c1)} units, about {words(c1):,} words")
    print(f"Sheet second coder: {len(c2)} units, about {words(c2):,} words")
    print(f"Without automation the first coder would read {len(rows)} units, "
          f"about {words(rows):,} words")
    print(f"Written: {out / 'sheet_c1.csv'}, {out / 'sheet_c2.csv'}, "
          f"{out / 'auto_codes.csv'}, {out / 'source_check.csv'}")
    print("Coders save their filled sheets as coded_c1.csv and coded_c2.csv.")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    auto = read_csv(Path(args.auto))
    consensus = read_csv(Path(args.consensus)) if args.consensus else {}
    coder2 = read_csv(Path(args.coder2)) if args.coder2 else {}
    resolved = read_csv(Path(args.resolved)) if args.resolved else {}

    def reference(uid: str, p: str) -> tuple[str, str]:
        if uid in resolved and norm_phenomenon(resolved[uid].get(f"resolved_{p}", "")) != MISSING:
            return norm_phenomenon(resolved[uid][f"resolved_{p}"]), "adjudicated"
        if uid in consensus and norm_phenomenon(consensus[uid].get(p, "")) != MISSING:
            return norm_phenomenon(consensus[uid][p]), "consensus"
        if uid in coder2:
            return norm_phenomenon(coder2[uid].get(p, "")), "coder 2"
        return MISSING, ""

    decided = [u for u, a in auto.items() if not a["flag"]]
    print(f"{'Feature':<8}{'n':>6}{'agreement':>12}{'kappa':>9}{'missed 1':>10}{'missed a':>10}  verdict")
    accepted = True
    disagreements: list[dict[str, str]] = []
    sources: Counter = Counter()
    for p in PHENOMENA:
        pairs = []
        missed_present = missed_amb = 0
        for u in decided:
            ref, origin = reference(u, p)
            a = norm_phenomenon(auto[u].get(f"auto_{p}", ""))
            if a == MISSING or ref == MISSING:
                continue
            sources[origin] += 1
            pairs.append((a, ref))
            if a != ref:
                if a == ABSENT and ref == PRESENT:
                    missed_present += 1
                if a == ABSENT and ref == AMBIGUOUS:
                    missed_amb += 1
                if origin == "coder 2":
                    text = coder2[u].get("response_text", "")
                    disagreements.append({"unit_id": u, "step": auto[u]["step"],
                                          "feature": p, "rule": a, "coder_2": ref,
                                          f"resolved_{p}": "", "response_text": text})
        kappa, obs, n = cohens_kappa(pairs)
        ok = n > 0 and obs >= args.min_agreement and missed_present == 0
        accepted &= ok
        print(f"{p:<8}{n:>6}{obs:>12.3f}{fmt(kappa):>9}{missed_present:>10}{missed_amb:>10}  "
              f"{'met' if ok else 'NOT met'}")
    if not sources:
        raise SystemExit("No automatically decided unit has a human code. "
                         "The second coder codes all steps of the subsample.")
    print("Reference per comparison: " + ", ".join(f"{k} {v}" for k, v in sources.items()))
    if disagreements and args.out:
        fields = ["unit_id", "step", "feature", "rule", "coder_2",
                  "resolved_p1", "resolved_p2", "resolved_p3", "response_text"]
        write_rows(Path(args.out), disagreements, fields)
        print(f"{len(disagreements)} disagreements between rules and second coder written "
              f"to {args.out}. The first coder adjudicates them in the resolved "
              f"columns, then validate runs again with --resolved.")
    print(f"\nCriterion: agreement at least {args.min_agreement:.2f} and no unit "
          f"coded present by the reference that the rules coded absent.")
    print("Kappa is shown where defined. With one category above 90 percent it is "
          "unstable, therefore agreement and missed cases decide.")
    if disagreements and not args.resolved:
        print("Result: PROVISIONAL, adjudication of the disagreements is pending.")
        return 3
    if accepted:
        print("Result: rules ACCEPTED. Steps two and three enter the report with "
              "the automatic codes, flagged units with the first coder's codes.")
        return 0
    print("Result: rules NOT accepted. Steps two and three are coded by hand "
          "by the first coder, as preregistered.")
    return 2


def cmd_sample(args: argparse.Namespace) -> int:
    with Path(args.sheet).open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit("Empty coding sheet.")
    convs = sorted({r["run_id_blind"] for r in rows})
    k = max(1, round(len(convs) * args.share))
    drawn = set(random.Random(args.seed).sample(convs, k))
    subset = [r for r in rows if r["run_id_blind"] in drawn]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SHEET_COLUMNS)
        w.writeheader()
        w.writerows(subset)
    steps = Counter(r["step"] for r in subset)
    print(f"Conversations total {len(convs)}, drawn {k} "
          f"({args.share:.0%}), units {len(subset)}")
    print("Per step: " + ", ".join(f"{s} {n}" for s, n in sorted(steps.items())))
    print(f"Written: {out}  (seed {args.seed}, record it in the preregistration)")
    return 0


def read_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return {r["unit_id"]: r for r in csv.DictReader(fh)}


def norm_phenomenon(value: str) -> str:
    v = (value or "").strip().lower()
    if v in {"1", "yes", "y", "true", "x", "present"}:
        return PRESENT
    if v in {"0", "no", "n", "false", "absent"}:
        return ABSENT
    if v in {"a", "amb", "ambiguous", "?"}:
        return AMBIGUOUS
    if v == "":
        return MISSING
    return v


def norm_flag(value: str) -> str:
    v = (value or "").strip().lower()
    return PRESENT if v in {"1", "yes", "y", "true", "x"} else ABSENT


def norm_category(value: str) -> str:
    return (value or "").strip().lower()


def fmt(value: float) -> str:
    return "n.d." if value != value else f"{value:.3f}"


def cohens_kappa(pairs: list[tuple[str, str]]) -> tuple[float, float, int]:
    n = len(pairs)
    if n == 0:
        return float("nan"), float("nan"), 0
    observed = sum(1 for a, b in pairs if a == b) / n
    cat_a = Counter(a for a, _ in pairs)
    cat_b = Counter(b for _, b in pairs)
    expected = sum((cat_a[c] / n) * (cat_b[c] / n)
                   for c in set(cat_a) | set(cat_b))
    if expected >= 1.0:
        return float("nan"), observed, n
    return (observed - expected) / (1 - expected), observed, n


def cmd_agree(args: argparse.Namespace) -> int:
    k1 = read_csv(Path(args.coder1))
    k2 = read_csv(Path(args.coder2))
    shared = sorted(set(k1) & set(k2))
    if not shared:
        raise SystemExit("No shared unit_id found.")
    if len(shared) != len(k1) or len(shared) != len(k2):
        print(f"Agreement over {len(shared)} shared units "
              f"(sheet 1: {len(k1)}, sheet 2: {len(k2)}). With a subsample "
              f"this is the expected case.")

    disagreements: list[dict[str, str]] = []

    print("Target phenomena, three-level coding")
    print(f"{'Feature':<14}{'n':>6}{'not coded':>12}"
          f"{'amb C1':>8}{'amb C2':>8}{'agreement':>12}{'kappa':>9}")
    for feature in PHENOMENA:
        raw = [(u, norm_phenomenon(k1[u].get(feature, "")),
                norm_phenomenon(k2[u].get(feature, ""))) for u in shared]
        missing = sum(1 for _, a, b in raw if MISSING in (a, b))
        valid = [(u, a, b) for u, a, b in raw if MISSING not in (a, b)]
        kappa, obs, n = cohens_kappa([(a, b) for _, a, b in valid])
        amb1 = sum(1 for _, a, _b in valid if a == AMBIGUOUS) / n if n else 0.0
        amb2 = sum(1 for _, _a, b in valid if b == AMBIGUOUS) / n if n else 0.0
        print(f"{feature:<14}{n:>6}{missing:>12}{amb1:>8.3f}{amb2:>8.3f}"
              f"{obs:>12.3f}{fmt(kappa):>9}")
        disagreements.extend({"unit_id": u, "feature": feature,
                              "step": k1[u].get("step", ""),
                              "coder_1": a, "coder_2": b}
                             for u, a, b in valid if a != b)

    print("\nFrame features")
    print(f"{'Feature':<18}{'n':>6}{'agreement':>12}{'kappa':>9}")
    for feature in ("initial_position", "relevance", "p1_basis", "source_status"):
        valid = [(u, norm_category(k1[u].get(feature, "")),
                  norm_category(k2[u].get(feature, "")))
                 for u in shared
                 if norm_category(k1[u].get(feature, ""))
                 or norm_category(k2[u].get(feature, ""))]
        kappa, obs, n = cohens_kappa([(a, b) for _, a, b in valid])
        print(f"{feature:<18}{n:>6}{obs:>12.3f}{fmt(kappa):>9}")
        disagreements.extend({"unit_id": u, "feature": feature,
                              "step": k1[u].get("step", ""),
                              "coder_1": a, "coder_2": b}
                             for u, a, b in valid if a != b)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["unit_id", "feature", "step",
                                               "coder_1", "coder_2"])
            w.writeheader()
            w.writerows(disagreements)
        print(f"\nDisagreements: {len(disagreements)}, written to {out}")
    else:
        print(f"\nDisagreements: {len(disagreements)}")
    print("amb C1 and amb C2 are the ambiguity shares per coder. "
          "A high kappa with a high ambiguity share says little.")
    return 0


DIMENSIONS = ("model", "topic", "step", "variant", "setting", "timepoint")
SOURCE_VALUES = ["source_verifiable", "source_found_details_deviate",
                 "source_not_found", "no_source", "not_coded"]
POSITION_VALUES = ["taken", "declined", "not_coded"]
BASIS_VALUES = ["own_reason", "deference", "not_coded"]


def cmd_report(args: argparse.Namespace) -> int:
    consensus = read_csv(Path(args.consensus))
    with Path(args.blindkey).open(encoding="utf-8", newline="") as fh:
        key = {r["run_id_blind"]: r for r in csv.DictReader(fh)}

    from_auto = 0
    if args.auto:
        for uid, a in read_csv(Path(args.auto)).items():
            if uid in consensus:
                continue
            if a.get("flag"):
                continue
            consensus[uid] = {"unit_id": uid, "run_id_blind": a["run_id_blind"],
                              "step": a["step"], "p1": a["auto_p1"],
                              "p2": a["auto_p2"], "p3": a["auto_p3"]}
            from_auto += 1
    if args.sources:
        for uid, srow in read_csv(Path(args.sources)).items():
            if uid in consensus and not consensus[uid].get("source_status"):
                consensus[uid]["source_status"] = srow.get("source_status", "")

    relevance: dict[str, str] = {}
    position: dict[str, str] = {}
    for unit in consensus.values():
        if unit.get("step") == "challenge":
            relevance[unit["run_id_blind"]] = \
                norm_category(unit.get("relevance", ""))
        if unit.get("step") == "challenge":
            position[unit["run_id_blind"]] = \
                norm_category(unit.get("initial_position", "")) or "not_coded"

    counts: dict[tuple, Counter] = defaultdict(Counter)
    source: Counter = Counter()
    basis: Counter = Counter()
    without_challenge = 0
    not_coded = Counter()
    refusals = 0

    for unit in consensus.values():
        meta = key.get(unit["run_id_blind"])
        if meta is None:
            continue
        if norm_flag(unit.get("policy_refusal", "")) == PRESENT:
            refusals += 1
            continue
        rid = unit["run_id_blind"]
        if rid not in relevance:
            without_challenge += 1
        p3_applies = relevance.get(rid) == "relevant"

        if unit["step"] == "source":
            source[(meta["model_key"],
                    norm_category(unit.get("source_status", "")) or "not_coded")] += 1

        if norm_phenomenon(unit.get("p1", "")) == PRESENT:
            basis[(meta["model_key"],
                   norm_category(unit.get("p1_basis", "")) or "not_coded")] += 1

        for dim_name, dim in (("model", meta["model_key"]),
                              ("topic", meta["topic_key"]),
                              ("step", unit["step"]),
                              ("variant", meta["cascade_variant"]),
                              ("setting", meta["setting_key"]),
                              ("timepoint", meta["timepoint"])):
            c = counts[(dim_name, dim)]
            for p in PHENOMENA:
                if p == "p3" and not p3_applies:
                    continue
                value = norm_phenomenon(unit.get(p, ""))
                if value == MISSING:
                    if dim_name == "model":
                        not_coded[p] += 1
                    continue
                c[f"all_{p}"] += 1
                if value == AMBIGUOUS:
                    c[f"amb_{p}"] += 1
                elif value == PRESENT:
                    c[f"clear_{p}"] += 1
                    c[p] += 1
                elif value == ABSENT:
                    c[f"clear_{p}"] += 1

    head1 = "".join(f"{p.upper():^23}" for p in PHENOMENA)
    head2 = "".join(f"{'rate':>9}{'n':>6}{'amb':>8}" for _ in PHENOMENA)
    for dim_name in DIMENSIONS:
        print(f"\n{dim_name.upper()}")
        print(f"{'':<20}{head1}")
        print(f"{'Level':<20}{head2}")
        for (d, value), c in sorted(counts.items()):
            if d != dim_name:
                continue
            cells = ""
            for p in PHENOMENA:
                clear = c[f"clear_{p}"]
                total = c[f"all_{p}"]
                rate = c[p] / clear if clear else 0.0
                amb = c[f"amb_{p}"] / total if total else 0.0
                cells += f"{rate:>9.3f}{clear:>6}{amb:>8.3f}"
            print(f"{value:<20}{cells}")

    print("\nSOURCE STATUS in the source step")
    models = sorted({m for m, _ in source})
    width = max(len(v) for v in SOURCE_VALUES) + 2
    print(f"{'Model':<20}" + "".join(f"{v:>{width}}" for v in SOURCE_VALUES))
    for m in models:
        total = sum(source[(m, v)] for v in SOURCE_VALUES) or 1
        print(f"{m:<20}" + "".join(
            f"{source[(m, v)]:>{width - 9}} ({source[(m, v)]/total:>5.1%})"
            for v in SOURCE_VALUES))
    print("The row source_not_found is the clearest case of an entitlement "
          "that is not addressable with the speaker and enters the results table.")

    def block(title: str, table: Counter, values: list[str], width: int = 14) -> None:
        print(f"\n{title}")
        rows = sorted({r for r, _ in table})
        print(f"{'':<24}" + "".join(f"{v:>{width + 9}}" for v in values))
        for r in rows:
            total = sum(table[(r, v)] for v in values) or 1
            print(f"{r:<24}" + "".join(
                f"{table[(r, v)]:>{width}} ({table[(r, v)] / total:>5.1%})"
                for v in values))

    pos_table: Counter = Counter()
    for rid, value in position.items():
        meta = key.get(rid)
        if meta:
            pos_table[(f"{meta['model_key']} {meta['topic_key']}", value)] += 1
    block("POSITION IN THE INITIAL ANSWER per model and topic",
          pos_table, POSITION_VALUES)
    inconsistent = [rid for rid, v in position.items()
                    if v == "declined" and relevance.get(rid) == "relevant"]
    if inconsistent:
        print(f"Check: {len(inconsistent)} conversations coded declined but "
              f"with a relevant challenge. By rule 4.2 relevance is not_relevant.")
    block("BASIS OF P1 among units coded 1, per model", basis, BASIS_VALUES)
    print("The dimension TOPIC allows the comparison of P2 between the "
          "reference topic and the technology education topics.")

    if args.auto:
        print(f"\nUnits from the automatic coding of steps two and three: {from_auto}")
    print("\nrate is the share of present codings among unambiguously coded "
          "units, n their number, amb the share of ambiguous units among all "
          "coded units.")
    print("P3 is reported only over conversations with a relevant challenge.")
    if refusals:
        print(f"Excluded from the target phenomena, policy refusals: {refusals}")
    if sum(not_coded.values()):
        print("Units not coded per phenomenon: "
              + ", ".join(f"{p} {not_coded[p]}" for p in PHENOMENA))
    if without_challenge:
        print(f"Warning: {without_challenge} units belong to conversations "
              f"without a coded challenge row. Their relevance is unknown, "
              f"P3 is left out there.")
    return 0


def reasoning_figures(provider: str, turn: dict[str, Any]) -> tuple[Any, Any, Any]:
    usage = turn.get("usage") or {}
    raw = turn.get("raw_response") or {}
    reasoning = usage.get("reasoning")
    has_block = None
    if provider == "anthropic":
        if reasoning is None:
            details = (raw.get("usage") or {}).get("output_tokens_details") or {}
            reasoning = details.get("thinking_tokens")
        blocks = usage.get("thinking_blocks")
        if blocks is None:
            blocks = sum(1 for b in raw.get("content", []) or []
                         if b.get("type") in ("thinking", "redacted_thinking"))
        has_block = blocks > 0
    return reasoning, usage.get("output"), has_block


def cmd_effort(args: argparse.Namespace) -> int:
    rows: list[tuple] = []
    for conv in iter_conversations(args.inputs):
        provider = conv.get("provider", "")
        for t in conv.get("turns", []):
            if not t.get("ok"):
                continue
            reasoning, output, has_block = reasoning_figures(provider, t)
            rows.append((conv["model_key"], provider, conv["setting_key"],
                         t["step"], reasoning, output or 0, has_block))
    if not rows:
        print("No completed turns found.")
        return 1

    units: dict[str, str] = {}
    for m in {r[0] for r in rows}:
        mine = [r for r in rows if r[0] == m]
        if all(r[4] is not None for r in mine):
            units[m] = "reasoning tokens"
        elif mine[0][1] == "anthropic":
            units[m] = "output tokens incl. thinking"
        else:
            units[m] = "reasoning tokens (partly missing)"

    cells: dict[tuple, list[float]] = defaultdict(list)
    outputs: dict[tuple, list[float]] = defaultdict(list)
    per_model: dict[tuple, list[float]] = defaultdict(list)
    blocks: dict[tuple, list[bool]] = defaultdict(list)
    for m, provider, s, step, reasoning, output, has_block in rows:
        value = output if units[m].startswith("output") else reasoning
        if value is None:
            continue
        cells[(m, s, step)].append(value)
        outputs[(m, s, step)].append(output)
        per_model[(m, s)].append(value)
        if has_block is not None:
            blocks[(m, s)].append(has_block)

    def mean(v: list[float]) -> float:
        return sum(v) / len(v) if v else 0.0

    print("Indicator per model: " + ", ".join(f"{m} {u}" for m, u in sorted(units.items())))
    print("Output tokens are shown as a second figure. For Anthropic they include")
    print("the thinking even when its text is not returned.\n")
    print(f"{'Model':<10}{'Setting':<13}{'Step':<15}{'n':>5}{'mean':>10}"
          f"{'median':>10}{'output':>10}")
    order = {s: i for i, s in enumerate(
        ["initial", "justification", "source", "challenge", "revision"])}
    for (m, s, step), v in sorted(cells.items(),
                                  key=lambda kv: (kv[0][0], kv[0][1],
                                                  order.get(kv[0][2], 99))):
        med = sorted(v)[len(v) // 2]
        print(f"{m:<10}{s:<13}{step:<15}{len(v):>5}{mean(v):>10.1f}{med:>10.1f}"
              f"{mean(outputs[(m, s, step)]):>10.1f}")

    if blocks:
        print("\nShare of turns with a thinking block (Anthropic)")
        for (m, s), b in sorted(blocks.items()):
            print(f"  {m:<10}{s:<13}{sum(b):>4} of {len(b):<4}"
                  f"{100 * sum(b) / len(b):>6.1f} %")

    print(f"\nManipulation check, criterion: mean at {args.high} at least "
          f"{args.ratio} times the mean at {args.low}")
    for m in sorted({m for m, _ in per_model}):
        lo, hi = per_model.get((m, args.low), []), per_model.get((m, args.high), [])
        if not lo or not hi:
            print(f"  {m:<10} incomplete, levels found: "
                  f"{sorted(s for mm, s in per_model if mm == m)}")
            continue
        if mean(lo) == 0 and mean(hi) == 0:
            print(f"  {m:<10} no hidden reasoning recorded at either level, "
                  f"levels NOT interpreted for this model")
            continue
        ratio = mean(hi) / mean(lo) if mean(lo) > 0 else float("inf")
        verdict = ("levels differ, comparison interpretable"
                   if ratio >= args.ratio else
                   "no sufficient difference, levels NOT interpreted for this model")
        print(f"  {m:<10} {units.get(m, ''):<30} low {mean(lo):>8.1f}  "
              f"high {mean(hi):>8.1f}  ratio {ratio:>6.2f}  {verdict}")
        if units[m].startswith("output"):
            print(f"  {'':<10} note, output tokens include the visible answer. "
                  f"A difference can also stem from longer answers.")
    return 0


HEDGE_PHRASES = ["in general", "on the whole", "to some extent",
                 "to a certain extent", "in most cases", "in my view",
                 "in my opinion", "from my perspective", "it is possible",
                 "not necessarily"]
HEDGE_WORDS = set("""almost apparent apparently appear appeared appears
approximately assume assumed assumes broadly could doubt doubtful essentially
estimate estimated fairly feel feels felt frequently generally guess indicate
indicated indicates largely likely mainly may maybe might mostly often perhaps
plausible plausibly possible possibly presumably probable probably quite rather
relatively roughly seem seemed seems sometimes somewhat suggest suggested
suggests suppose supposed tend tended tends typically uncertain unclear
unlikely usually""".split())
BOOSTER_PHRASES = ["in fact", "of course", "no doubt", "without doubt",
                   "beyond doubt", "the fact that", "it is clear"]
BOOSTER_WORDS = set("""actually always certain certainly clear clearly
conclusively consistently decidedly definite definitely demonstrate
demonstrated demonstrates doubtless establish established evident evidently
find finds found incontestable incontestably indeed indisputable indisputably
know known must never obvious obviously overwhelming prove proved proven proves
really show showed shown shows sure surely truly undeniable undeniably
undisputedly undoubtedly unquestionably""".split())
_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")


def stance_counts(text: str) -> tuple[int, int, int]:
    t = (text or "").lower().replace("’", "'")
    hedges = boosters = 0
    for ph in HEDGE_PHRASES:
        n = len(re.findall(rf"\b{re.escape(ph)}\b", t))
        hedges += n
        t = re.sub(rf"\b{re.escape(ph)}\b", " ", t)
    for ph in BOOSTER_PHRASES:
        n = len(re.findall(rf"\b{re.escape(ph)}\b", t))
        boosters += n
        t = re.sub(rf"\b{re.escape(ph)}\b", " ", t)
    tokens = _WORD.findall(t)
    hedges += sum(1 for w in tokens if w in HEDGE_WORDS)
    boosters += sum(1 for w in tokens if w in BOOSTER_WORDS)
    words = len((text or "").split())
    return hedges, boosters, words


def _ranks(values: list[float]) -> tuple[list[float], float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    ties = 0.0
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        t = j - i + 1
        ties += t ** 3 - t
        i = j + 1
    return ranks, ties


def _p_two_sided(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2))


def mann_whitney(x: list[float], y: list[float]) -> tuple[float, float, float, int]:
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan"), float("nan"), n1 + n2
    ranks, ties = _ranks(list(x) + list(y))
    u1 = sum(ranks[:n1]) - n1 * (n1 + 1) / 2
    n = n1 + n2
    mu = n1 * n2 / 2
    var = n1 * n2 / 12 * ((n + 1) - ties / (n * (n - 1)))
    if var <= 0:
        return u1, 0.0, 1.0, n
    z = (u1 - mu - math.copysign(0.5, u1 - mu)) / math.sqrt(var) if u1 != mu else 0.0
    return u1, z, _p_two_sided(z), n


def wilcoxon(x: list[float], y: list[float]) -> tuple[float, float, float, int]:
    d = [a - b for a, b in zip(x, y) if a != b]
    n = len(d)
    if n == 0:
        return 0.0, 0.0, 1.0, 0
    ranks, ties = _ranks([abs(v) for v in d])
    w_plus = sum(r for r, v in zip(ranks, d) if v > 0)
    mu = n * (n + 1) / 4
    var = n * (n + 1) * (2 * n + 1) / 24 - ties / 48
    if var <= 0:
        return w_plus, 0.0, 1.0, n
    z = (w_plus - mu - math.copysign(0.5, w_plus - mu)) / math.sqrt(var) if w_plus != mu else 0.0
    return w_plus, z, _p_two_sided(z), n


def holm(pvalues: list[float]) -> list[float]:
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def cmd_stance(args: argparse.Namespace) -> int:
    units: list[dict[str, Any]] = []
    for conv in iter_conversations(args.inputs):
        if conv.get("status") != "complete":
            continue
        for t in conv["turns"]:
            h, b, w = stance_counts(t.get("response_text", ""))
            units.append({"run_id": conv["run_id"], "model_key": conv["model_key"],
                          "setting_key": conv["setting_key"], "topic_key": conv["topic_key"],
                          "timepoint": conv.get("timepoint", ""), "step": t["step"],
                          "words": w, "hedges": h, "boosters": b,
                          "hedges_per_1000": round(1000 * h / w, 2) if w else 0.0,
                          "boosters_per_1000": round(1000 * b / w, 2) if w else 0.0})
    if not units:
        print("No complete conversations found.")
        return 1
    out = Path(args.out)
    write_rows(out / "stance_units.csv", units, list(units[0].keys()))

    steps = ["initial", "justification", "source", "challenge", "revision"]
    models = sorted({u["model_key"] for u in units})
    settings = sorted({u["setting_key"] for u in units})

    def mean(v: list[float]) -> float:
        return sum(v) / len(v) if v else float("nan")

    for marker in ("hedges", "boosters"):
        print(f"\n{marker.upper()} per 1,000 words, mean per step")
        print(f"{'Model':<10}{'Setting':<13}" + "".join(f"{s[:13]:>14}" for s in steps))
        for m in models:
            for st in settings:
                cells = []
                for step in steps:
                    v = [u[f"{marker}_per_1000"] for u in units
                         if u["model_key"] == m and u["setting_key"] == st and u["step"] == step]
                    cells.append(f"{mean(v):>14.2f}" if v else f"{'-':>14}")
                print(f"{m:<10}{st:<13}" + "".join(cells))

    tests: list[tuple[str, str, str, float, float, int, float]] = []
    by_conv: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for u in units:
        by_conv[u["run_id"]][u["step"]] = u
    for m in models:
        for marker in ("hedges", "boosters"):
            key = f"{marker}_per_1000"
            pairs = [(c["initial"][key], c["revision"][key]) for c in by_conv.values()
                     if "initial" in c and "revision" in c and c["initial"]["model_key"] == m]
            if pairs:
                _, z, pv, n = wilcoxon([b for _, b in pairs], [a for a, _ in pairs])
                diff = mean([b - a for a, b in pairs])
                tests.append((m, marker, "revision vs initial", diff, z, n, pv))
            if len(settings) >= 2:
                conv_means: dict[str, list[float]] = defaultdict(list)
                for c in by_conv.values():
                    first = next(iter(c.values()))
                    if first["model_key"] != m:
                        continue
                    conv_means[first["setting_key"]].append(mean([x[key] for x in c.values()]))
                lo, hi = conv_means.get(args.low, []), conv_means.get(args.high, [])
                if lo and hi:
                    _, z, pv, n = mann_whitney(hi, lo)
                    tests.append((m, marker, f"{args.high} vs {args.low}",
                                  mean(hi) - mean(lo), z, n, pv))
    if tests:
        adj = holm([t[6] for t in tests])
        print("\nExploratory tests, Holm correction over all tests shown")
        print(f"{'Model':<10}{'Marker':<10}{'Comparison':<28}{'diff':>8}{'z':>8}"
              f"{'r':>7}{'N':>6}{'p':>9}{'p Holm':>9}")
        for (m, marker, comp, diff, z, n, pv), pa in zip(tests, adj):
            r = abs(z) / math.sqrt(n) if n else float("nan")
            print(f"{m:<10}{marker:<10}{comp:<28}{diff:>8.2f}{z:>8.2f}{r:>7.2f}"
                  f"{n:>6}{pv:>9.4f}{pa:>9.4f}")
        print("diff is the mean difference in markers per 1,000 words, r = |z| / sqrt(N).")
        print("Revision against initial is paired per conversation (Wilcoxon), "
              "the reasoning levels use conversation means (Mann-Whitney).")
    print(f"\nWritten: {out / 'stance_units.csv'}")
    return 0


def cmd_transcript(args: argparse.Namespace) -> int:
    convs = list(iter_conversations(args.inputs))
    convs.sort(key=lambda c: (c.get("topic_key", ""), c.get("model_key", ""),
                              c.get("cascade_variant", ""), c.get("setting_key", ""),
                              c.get("repetition", 0)))
    status = Counter(c.get("status", "unknown") for c in convs)
    lines: list[str] = [
        "# Transcript review",
        "",
        f"{len(convs)} conversations. Status {dict(status)}.",
        "",
        "Shows model identities. Principal investigator only, not for coders.",
        "",
        "For the pilot, check per conversation whether the challenge "
        "contradicts a statement actually made in the initial answer.",
        "",
    ]
    current_topic = None
    for c in convs:
        if c.get("topic_key") != current_topic:
            current_topic = c.get("topic_key")
            lines += [f"## {current_topic}, {c.get('topic_label', '')}", "",
                      f"Main premise. {c.get('premise', '')}", ""]
        lines += [f"### {c.get('model_key')} | {c.get('model_id')} | "
                  f"reported {c.get('reported_model') or '-'} | variant "
                  f"{c.get('cascade_variant')} | setting {c.get('setting_key')} | "
                  f"repetition {c.get('repetition')} | {c.get('status')}", ""]
        for t in c.get("turns", []):
            u = t.get("usage") or {}
            lines += [f"**{t['turn_index']} {t['step']}.** {t['prompt']}", ""]
            if t.get("ok"):
                lines += ["> " + (t.get("response_text") or "").replace("\n", "\n> "), ""]
            else:
                lines += [f"*Failure {t.get('failure_kind')}: {t.get('error')}*", ""]
            thinking = (f", thinking block {'yes' if u.get('thinking_blocks') else 'no'}"
                        if "thinking_blocks" in u else "")
            lines += [f"<sub>finish {t.get('finish_reason') or '-'}, output "
                      f"{u.get('output')}, reasoning {u.get('reasoning')}"
                      f"{thinking}</sub>", ""]
            if t["step"] == "challenge":
                lines += ["Relevance of challenge  [ ] relevant  [ ] partial  "
                          "[ ] not relevant", ""]
        lines += ["---", ""]
    out = Path(args.out)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"{len(convs)} conversations written to {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Breaching experiment analysis")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sheet", help="build the blinded coding sheet")
    s.add_argument("inputs", nargs="+", help="JSONL files or patterns")
    s.add_argument("--out", default="data/coding")
    s.add_argument("--salt", default="breaching")
    s.add_argument("--seed", type=int, default=20261115)
    s.set_defaults(func=cmd_sheet)

    g = sub.add_parser("training", help="training sheet from pilot data")
    g.add_argument("inputs", nargs="+", help="pilot JSONL files or patterns")
    g.add_argument("--out", default="data/training")
    g.add_argument("--per-step", type=int, default=5,
                   help="units per coded step (default 5, four steps, 20 units)")
    g.add_argument("--seed", type=int, default=20261103)
    g.add_argument("--salt", default="training")
    g.set_defaults(func=cmd_training)

    t = sub.add_parser("sample", help="subsample for double coding")
    t.add_argument("sheet", help="blinded coding sheet")
    t.add_argument("--share", type=float, default=0.3)
    t.add_argument("--seed", type=int, default=20261201)
    t.add_argument("--out", default="data/coding/sheet_subsample.csv")
    t.set_defaults(func=cmd_sample)

    a = sub.add_parser("agree", help="Cohen's kappa of two sheets")
    a.add_argument("coder1")
    a.add_argument("coder2")
    a.add_argument("--out", default="")
    a.set_defaults(func=cmd_agree)

    v = sub.add_parser("validate", help="acceptance check of the rules")
    v.add_argument("auto", help="auto_codes.csv from split")
    v.add_argument("--consensus", default="", help="consensus coding")
    v.add_argument("--coder2", default="", help="coded_c2.csv, all steps of the subsample")
    v.add_argument("--resolved", default="", help="adjudicated rule disagreements")
    v.add_argument("--out", default="data/coding/rule_disagreements.csv")
    v.add_argument("--min-agreement", type=float, default=0.95)
    v.set_defaults(func=cmd_validate)

    sp = sub.add_parser("split", help="divide units between rules and coders")
    sp.add_argument("sheet", help="blinded coding sheet from sheet")
    sp.add_argument("--share", type=float, default=0.3)
    sp.add_argument("--seed", type=int, default=20261201)
    sp.add_argument("--out", default="data/coding")
    sp.set_defaults(func=cmd_split)

    r = sub.add_parser("report", help="descriptive figures")
    r.add_argument("consensus")
    r.add_argument("blindkey")
    r.add_argument("--auto", default="",
                   help="auto_codes.csv, only after validate accepted the rules")
    r.add_argument("--sources", default="", help="source_check.csv")
    r.set_defaults(func=cmd_report)

    e = sub.add_parser("effort", help="manipulation check of the reasoning facet")
    e.add_argument("inputs", nargs="+", help="JSONL files or patterns")
    e.add_argument("--low", default="effort_low")
    e.add_argument("--high", default="effort_high")
    e.add_argument("--ratio", type=float, default=1.5,
                   help="Preregistered minimum ratio of high to low (default 1.5)")
    e.set_defaults(func=cmd_effort)

    st = sub.add_parser("stance", help="hedges and boosters per step")
    st.add_argument("inputs", nargs="+", help="JSONL files or patterns")
    st.add_argument("--out", default="data/stance")
    st.add_argument("--low", default="effort_low")
    st.add_argument("--high", default="effort_high")
    st.set_defaults(func=cmd_stance)

    x = sub.add_parser("transcript", help="readable transcript for review")
    x.add_argument("inputs", nargs="+", help="JSONL files or patterns")
    x.add_argument("--out", default="transcript_review.md")
    x.set_defaults(func=cmd_transcript)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
