#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

import harness
from harness import (KEY_VARIABLES, ModelSpec, ProviderClient, SettingSpec,
                     load_config, load_dotenv, check_dotenv_permissions,
                     PLACEHOLDER, reasoning_indicator)

LOG = logging.getLogger("modelcheck")

DEFAULT_FIND = ["fable", "carbon", "terra"]

PROBE_TOKENS = {
    "openai": {"max_completion_tokens": 4000},
    "anthropic": {"max_tokens": 4000},
    "google": {"maxOutputTokens": 4000},
}

PROBES: dict[str, dict[str, dict[str, Any]]] = {
    "temperature_0": {
        "openai": {"temperature": 0.0},
        "anthropic": {"temperature": 0.0},
        "google": {"temperature": 0.0},
    },
    "effort_low": {
        "openai": {"reasoning_effort": "low"},
        "anthropic": {"thinking": {"type": "adaptive"},
                      "output_config": {"effort": "low"}},
        "google": {"thinkingConfig": {"thinkingLevel": "low"}},
    },
    "effort_high": {
        "openai": {"reasoning_effort": "high"},
        "anthropic": {"thinking": {"type": "adaptive"},
                      "output_config": {"effort": "high"}},
        "google": {"thinkingConfig": {"thinkingLevel": "high"}},
    },
}

PROMPT_SINGLE = [{"role": "user", "content": "Reply with the single word OK."}]

PROMPT_REASONING = [{"role": "user", "content": (
    "How many whole numbers from 1 to 500 contain the digit 7 at least once? "
    "Answer with the number only.")}]
REASONING_ANSWER = "95"

PROMPT_MULTI = [
    {"role": "user", "content": "Name one primary colour. Answer with one word."},
    {"role": "assistant", "content": "Red."},
    {"role": "user", "content": "Name a different one. Answer with one word."},
]


KEY_PREFIXES = {
    "ANTHROPIC_API_KEY": [
        ("sk-ant-api", "Console API key, the correct type"),
        ("sk-ant-oat", "OAuth token from a Claude subscription, NOT valid for the API"),
        ("sk-ant-admin", "Admin key, NOT valid for model requests"),
    ],
    "OPENAI_API_KEY": [
        ("sk-proj-", "project API key"),
        ("sk-svcacct-", "service account key"),
        ("sk-admin-", "Admin key, NOT valid for model requests"),
        ("sk-", "API key"),
    ],
    "GOOGLE_API_KEY": [("AIza", "API key")],
}


def describe_key(var: str) -> str:
    value = os.environ.get(var, "")
    problems = []
    if value != value.strip():
        problems.append("leading or trailing whitespace")
    if any(c.isspace() for c in value.strip()):
        problems.append("whitespace inside the key")
    if any(c in value for c in "\"'"):
        problems.append("quote characters")
    if "#" in value:
        problems.append("contains '#', possibly a comment")
    if not value.isascii():
        problems.append("non-ASCII characters")
    kind = "unknown format"
    for prefix, label in KEY_PREFIXES.get(var, []):
        if value.startswith(prefix):
            kind = label
            break
    text = f"key check {var}: length {len(value)}, {kind}"
    return text + (f", problems: {', '.join(problems)}" if problems
                   else ", no formatting problems found")


async def list_openai(client: httpx.AsyncClient) -> list[dict[str, Any]]:
    r = await client.get("https://api.openai.com/v1/models",
                         headers={"Authorization":
                                  f"Bearer {os.environ['OPENAI_API_KEY']}"})
    r.raise_for_status()
    return [{"id": m["id"], "created": m.get("created")}
            for m in r.json().get("data", [])]


async def list_anthropic(client: httpx.AsyncClient) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    after: str | None = None
    while True:
        params = {"limit": 1000}
        if after:
            params["after_id"] = after
        r = await client.get("https://api.anthropic.com/v1/models",
                             params=params,
                             headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"],
                                      "anthropic-version": "2023-06-01"})
        r.raise_for_status()
        data = r.json()
        out += [{"id": m["id"], "created": m.get("created_at"),
                 "display_name": m.get("display_name")}
                for m in data.get("data", [])]
        if not data.get("has_more"):
            return out
        after = data.get("last_id")


async def list_google(client: httpx.AsyncClient) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    token: str | None = None
    while True:
        params: dict[str, Any] = {"pageSize": 1000}
        if token:
            params["pageToken"] = token
        r = await client.get("https://generativelanguage.googleapis.com/v1beta/models",
                             params=params,
                             headers={"x-goog-api-key": os.environ["GOOGLE_API_KEY"]})
        r.raise_for_status()
        data = r.json()
        for m in data.get("models", []):
            if "generateContent" in m.get("supportedGenerationMethods", []):
                out.append({"id": m["name"].removeprefix("models/"),
                            "created": None,
                            "display_name": m.get("displayName")})
        token = data.get("nextPageToken")
        if not token:
            return out


LISTERS = {"openai": list_openai, "anthropic": list_anthropic,
           "google": list_google}

RELEVANT_PREFIX = {"openai": ("gpt-", "o"), "anthropic": ("claude",),
                   "google": ("gemini",)}


def sort_newest(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(m: dict[str, Any]) -> tuple:
        created = m.get("created")
        return (str(created) if created is not None else "", m["id"])
    return sorted(models, key=key, reverse=True)


async def probe(spec: ModelSpec, name: str, history: list[dict[str, str]],
                extra: dict[str, Any], timeout: float) -> dict[str, Any]:
    client = ProviderClient(spec, timeout=timeout, max_retries=2)
    try:
        res = await client.chat(history, extra)
    finally:
        await client.aclose()
    return {
        "probe": name,
        "params_sent": {**spec.params, **extra},
        "ok": res["ok"],
        "failure_kind": res.get("failure_kind"),
        "http_status": res["http_status"],
        "reported_model": res["reported_model"],
        "fingerprint": res.get("fingerprint", ""),
        "finish_reason": res["finish_reason"],
        "usage": res.get("usage", {}),
        "usage_raw": ((res.get("raw") or {}).get("usage")
                      or (res.get("raw") or {}).get("usageMetadata") or {}),
        "elapsed_s": res["elapsed_s"],
        "text": res["text"] or "",
        "error": (res["error"] or "")[:400] if res["error"] else None,
    }


def print_probe(r: dict[str, Any]) -> None:
    u = r.get("usage") or {}
    status = "OK" if r["ok"] else f"FAIL[{r['failure_kind']}]"
    detail = r["text"].replace("\n", " ")[:40] if r["ok"] else (r["error"] or "")[:110]
    print(f"    {r['probe']:<20} {status:<16} "
          f"{(r['reported_model'] or '-'):<34} "
          f"finish={r['finish_reason'] or '-':<12} "
          f"out={u.get('output')} reasoning={u.get('reasoning')}"
          + (f" thinking_blocks={u.get('thinking_blocks')}"
             if "thinking_blocks" in u else "")
          + f"  {detail}")


async def check_model(provider: str, model_id: str, base_params: dict[str, Any],
                      settings: list[SettingSpec], repeat: int,
                      timeout: float, effort_repeats: int = 3) -> dict[str, Any]:
    spec = ModelSpec(key=f"{provider}:{model_id}", provider=provider,
                     model_id=model_id, version_label="", params=base_params)
    results: list[dict[str, Any]] = []
    print(f"\n  {provider}:{model_id}")

    r = await probe(spec, "base", PROMPT_SINGLE, {}, timeout)
    results.append(r); print_probe(r)
    if not r["ok"] and r["failure_kind"] in ("http", "transport", "format"):
        if r["http_status"] in (401, 403):
            print("    -> " + describe_key(KEY_VARIABLES[provider]))
        print("    -> not reachable with this key, further probes skipped")
        return {"provider": provider, "model_id": model_id, "reachable": False,
                "probes": results}

    r = await probe(spec, "multiturn", PROMPT_MULTI, {}, timeout)
    results.append(r); print_probe(r)

    r = await probe(spec, "temperature_0", PROMPT_SINGLE,
                    dict(PROBES["temperature_0"].get(provider, {})), timeout)
    results.append(r); print_probe(r)
    levels = {"effort_default": {},
              "effort_low": dict(PROBES["effort_low"].get(provider, {})),
              "effort_high": dict(PROBES["effort_high"].get(provider, {}))}
    for level, extra in levels.items():
        if provider == "anthropic" and "budget_tokens" in extra.get("thinking", {}):
            budget = extra["thinking"]["budget_tokens"]
            extra["max_tokens"] = max(base_params.get("max_tokens", 0), budget + 2000)
        for i in range(effort_repeats):
            r = await probe(spec, f"{level}#{i + 1}", PROMPT_REASONING, extra, timeout)
            results.append(r); print_probe(r)
            if not r["ok"] and r["failure_kind"] == "http":
                break

    if provider == "openai":
        legacy = replace(spec, params={k: v for k, v in base_params.items()
                                       if k != "max_completion_tokens"}
                         | {"max_tokens": base_params.get("max_completion_tokens", 4000)})
        r = await probe(legacy, "legacy_max_tokens", PROMPT_SINGLE, {}, timeout)
        results.append(r); print_probe(r)

    for s in settings:
        r = await probe(spec, f"config:{s.key}", PROMPT_SINGLE,
                        s.for_provider(provider), timeout)
        results.append(r); print_probe(r)

    versions: list[str] = []
    fingerprints: list[str] = []
    for i in range(repeat):
        r = await probe(spec, f"repeat_{i + 1}", PROMPT_SINGLE, {}, timeout)
        results.append(r)
        if r["reported_model"]:
            versions.append(r["reported_model"])
        if r["fingerprint"]:
            fingerprints.append(r["fingerprint"])
    base = results[0]
    if base["reported_model"]:
        versions.insert(0, base["reported_model"])
    if base.get("fingerprint"):
        fingerprints.insert(0, base["fingerprint"])
    print(f"    reported versions over {len(versions)} calls: "
          f"{dict(Counter(versions)) or '-'}")
    print(f"    backend fingerprints: {dict(Counter(fingerprints)) or 'not provided'}")

    contrast: dict[str, Any] = {}
    unit = ""
    for level in ("effort_default", "effort_low", "effort_high"):
        runs = [x for x in results if x["probe"].startswith(level + "#") and x["ok"]]
        if not runs:
            continue
        values, outputs = [], []
        for x in runs:
            u = x.get("usage") or {}
            v, unit = reasoning_indicator(provider, u)
            values.append(v if v is not None else 0)
            outputs.append(u.get("output") or 0)
        contrast[level] = {
            "values": values,
            "mean": round(sum(values) / len(values), 1),
            "output_mean": round(sum(outputs) / len(outputs), 1),
            "correct": f"{sum(REASONING_ANSWER in x['text'] for x in runs)}/{len(runs)}",
        }
    if contrast:
        print(f"    {unit} on the reasoning task, mean of up to {effort_repeats}:")
        for level, c in contrast.items():
            print(f"      {level:<15} mean {c['mean']:>8}  values {c['values']}  "
                  f"output mean {c['output_mean']:>7}  correct {c['correct']}")
        if provider == "anthropic":
            for level in contrast:
                runs = [x for x in results
                        if x["probe"].startswith(level + "#") and x["ok"]]
                n = sum(1 for x in runs
                        if ((x.get("usage") or {}).get("thinking_blocks") or 0) > 0)
                contrast[level]["with_thinking"] = f"{n}/{len(runs)}"
            print("    requests with a thinking block: "
                  + ", ".join(f"{k.removeprefix('effort_')} {v['with_thinking']}"
                              for k, v in contrast.items()))
            if not any(int(v["with_thinking"].split("/")[0]) for v in contrast.values()):
                print("    note: no thinking block at any level. The model did not "
                      "think on this task, the cascade in the pilot decides.")

    return {"provider": provider, "model_id": model_id, "reachable": True,
            "reported_versions": dict(Counter(versions)),
            "fingerprints": dict(Counter(fingerprints)),
            "reasoning_contrast": contrast,
            "probes": results}


def summarize(model_result: dict[str, Any]) -> dict[str, Any]:
    by_name = {p["probe"]: p for p in model_result["probes"]}

    def state(name: str) -> str:
        group = [p for k, p in by_name.items() if k == name or k.startswith(name + "#")]
        if not group:
            return "not tested"
        if any(p["ok"] for p in group):
            return "accepted"
        return f"failed ({group[0]['failure_kind']})"

    return {
        "model": f"{model_result['provider']}:{model_result['model_id']}",
        "reachable": model_result["reachable"],
        "versions": model_result.get("reported_versions", {}),
        "fingerprints": model_result.get("fingerprints", {}),
        "reasoning_contrast": model_result.get("reasoning_contrast", {}),
        "multiturn": state("multiturn"),
        "temperature_0": state("temperature_0"),
        "effort_low": state("effort_low"),
        "effort_high": state("effort_high"),
        "legacy_max_tokens": state("legacy_max_tokens"),
        "config_settings": {k.removeprefix("config:"): state(k)
                            for k in by_name if k.startswith("config:")},
    }


async def run(args: argparse.Namespace) -> int:
    available = [p for p, var in KEY_VARIABLES.items() if os.environ.get(var)]
    missing = [var for p, var in KEY_VARIABLES.items() if not os.environ.get(var)]
    print(f"Providers with key: {', '.join(available) or 'none'}")
    if missing:
        print(f"No key for: {', '.join(missing)}")
    if not available:
        print("No API keys found. Fill in .env (see .env.example).", file=sys.stderr)
        return 2

    report: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "harness_version": harness.HARNESS_VERSION,
        "find_terms": args.find,
        "listing": {},
        "models": [],
        "summary": [],
    }

    print("\n=== Step 1. Models available to this account ===")
    matches: dict[str, list[str]] = {}
    async with httpx.AsyncClient(timeout=args.timeout) as client:
        for provider in available:
            try:
                models = await LISTERS[provider](client)
            except httpx.HTTPStatusError as exc:
                msg = f"HTTP {exc.response.status_code}: {exc.response.text[:200]}"
                print(f"\n  {provider}: listing failed, {msg}")
                if exc.response.status_code in (401, 403):
                    print("    " + describe_key(KEY_VARIABLES[provider]))
                report["listing"][provider] = {"error": msg}
                continue
            except httpx.HTTPError as exc:
                print(f"\n  {provider}: listing failed, {exc}")
                report["listing"][provider] = {"error": str(exc)}
                continue

            ids = [m["id"] for m in models]
            report["listing"][provider] = {"count": len(ids), "ids": sorted(ids)}
            found = [i for i in ids if any(t.lower() in i.lower() for t in args.find)]
            matches[provider] = sorted(found)
            print(f"\n  {provider}: {len(ids)} models")
            if found:
                print("    matching " + ", ".join(args.find) + ":")
                for i in sorted(found):
                    print(f"      {i}")
            else:
                print(f"    none matching {', '.join(args.find)}. Newest language models:")
                relevant = [m for m in sort_newest(models)
                            if m["id"].startswith(RELEVANT_PREFIX[provider])]
                for m in relevant[:12]:
                    print(f"      {m['id']}")

    if args.list_only:
        write_report(report, args.out)
        return 0

    targets: list[tuple[str, str, dict[str, Any]]] = []
    settings: list[SettingSpec] = []
    if args.models:
        for item in args.models:
            if ":" not in item:
                print(f"Skipping '{item}', expected provider:model_id", file=sys.stderr)
                continue
            provider, model_id = item.split(":", 1)
            targets.append((provider, model_id, dict(PROBE_TOKENS.get(provider, {}))))
    elif args.config:
        cfg = load_config(args.config)
        settings = cfg.settings
        for m in cfg.models:
            if PLACEHOLDER.search(m.model_id):
                print(f"  Skipping {m.key}, model_id is still a placeholder: {m.model_id}")
                continue
            targets.append((m.provider, m.model_id, dict(m.params)))
    else:
        for provider, ids in matches.items():
            for model_id in ids[:args.max_per_provider]:
                targets.append((provider, model_id, dict(PROBE_TOKENS[provider])))
            if len(ids) > args.max_per_provider:
                print(f"  {provider}: {len(ids)} matches, probing the first "
                      f"{args.max_per_provider}. Use --models for others.")

    targets = [t for t in targets if t[0] in available]
    if not targets:
        print("\nNothing to probe. None of the requested models is available "
              "to this account, or no key is set for its provider.")
        write_report(report, args.out)
        return 1

    print("\n=== Step 2. Probes (one short request each) ===")
    for provider, model_id, params in targets:
        result = await check_model(provider, model_id, params, settings,
                                   args.repeat, args.timeout, args.effort_repeats)
        report["models"].append(result)
        report["summary"].append(summarize(result))

    print("\n=== Step 3. Summary ===")
    for s in report["summary"]:
        print(f"\n  {s['model']}")
        if not s["reachable"]:
            print("    not reachable with this key")
            continue
        print(f"    reported versions : {s['versions']}")
        print(f"    fingerprints      : {s['fingerprints'] or 'not provided'}")
        version_ids = list(s["versions"])
        if version_ids and all(v == s["model"].split(":", 1)[1] for v in version_ids):
            print("    note              : the provider reports only the alias, "
                  "no dated version. A silent update would not be visible "
                  "in the reported version.")
        print(f"    multi-turn        : {s['multiturn']}")
        print(f"    temperature 0     : {s['temperature_0']}")
        print(f"    reasoning low     : {s['effort_low']}")
        print(f"    reasoning high    : {s['effort_high']}")
        if s["reasoning_contrast"]:
            means = {k.removeprefix("effort_"): v["mean"]
                     for k, v in s["reasoning_contrast"].items()}
            print(f"    reasoning (mean)  : {means}")
        if s["legacy_max_tokens"] != "not tested":
            print(f"    legacy max_tokens : {s['legacy_max_tokens']}")
        for k, v in s["config_settings"].items():
            print(f"    config setting {k:<6}: {v}")

    usable = [s for s in report["summary"]
              if s["reachable"] and s["multiturn"] == "accepted"]
    if usable:
        print("\n  Usable in harness.py, model_id lines for the config:")
        for s in usable:
            provider, model_id = s["model"].split(":", 1)
            print(f"    {provider:<10} model_id: \"{model_id}\"")

    write_report(report, args.out)
    return 0


def write_report(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    path = out_dir / f"model_check_{stamp}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport written: {path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Check models, versions and parameters")
    ap.add_argument("--find", nargs="+", default=DEFAULT_FIND,
                    help="Search terms for model IDs (default: fable carbon terra)")
    ap.add_argument("--models", nargs="+",
                    help="Explicit models as provider:model_id, "
                         "e.g. anthropic:claude-fable-5-1")
    ap.add_argument("--config", type=Path,
                    help="Probe the models and settings of a harness config")
    ap.add_argument("--list-only", action="store_true",
                    help="Only list available models, send no requests")
    ap.add_argument("--repeat", type=int, default=3,
                    help="Repeated base calls per model to check version stability")
    ap.add_argument("--effort-repeats", type=int, default=3,
                    help="Repetitions per reasoning level on the reasoning task")
    ap.add_argument("--max-per-provider", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--env", type=Path, default=None,
                    help="Path to .env, default is .env next to this script")
    ap.add_argument("--out", type=Path, default=Path("model_checks"),
                    help="Folder for the JSON report")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    env_path = args.env or (Path(__file__).resolve().parent / ".env")
    loaded = load_dotenv(env_path)
    if loaded:
        check_dotenv_permissions(env_path)
        print(f"Loaded {len(loaded)} key(s) from {env_path}")

    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
