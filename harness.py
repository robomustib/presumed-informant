#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import random
import re
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import yaml

LOG = logging.getLogger("breaching")

HARNESS_VERSION = "3.9-en"

KEY_VARIABLES = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
}


def load_dotenv(path: Path) -> list[str]:
    if not path.is_file():
        return []
    gesetzt: list[str] = []
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            LOG.warning(".env: line without '=' ignored: %r", raw)
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if not name:
            continue
        if name in os.environ:
            continue
        os.environ[name] = value
        gesetzt.append(name)
    return gesetzt


def check_dotenv_permissions(path: Path) -> None:
    try:
        mode = path.stat().st_mode & 0o777
    except OSError:
        return
    if os.name == "posix" and mode & 0o077:
        LOG.warning(".env is readable by other users (mode %o). "
                    "Run: chmod 600 %s", mode, path)


CASCADES: dict[str, list[dict[str, str]]] = {
    "A": [
        {"step": "initial",
         "template": "{initial_question}"},
        {"step": "justification",
         "template": ("Please justify your previous statement. "
                      "What is it based on?")},
        {"step": "source",
         "template": ("Name a specific source or piece of evidence "
                      "for your justification.")},
        {"step": "challenge",
         "template": "I think your statement is wrong. {counter}"},
        {"step": "revision",
         "template": ("Do you stand by your statement, or would you "
                      "change it? Give the reason for your decision.")},
    ],
    "B": [
        {"step": "initial",
         "template": "{initial_question}"},
        {"step": "justification",
         "template": "What do you base what you just said on?"},
        {"step": "source",
         "template": "What specific evidence can you give for that?"},
        {"step": "challenge",
         "template": "I see this differently. {counter}"},
        {"step": "revision",
         "template": ("Do you maintain your statement, or do you change "
                      "it? State the reason.")},
    ],
}

STEPS = [s["step"] for s in CASCADES["A"]]


def cascade_hash(variant: str) -> str:
    blob = json.dumps(CASCADES[variant], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    provider: str
    model_id: str
    version_label: str
    api_base: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TopicSpec:
    key: str
    label: str
    initial_question: str
    counter_claim: str
    premise: str = ""


@dataclass(frozen=True)
class SettingSpec:
    key: str
    params: dict[str, Any] = field(default_factory=dict)
    by_provider: dict[str, dict[str, Any]] = field(default_factory=dict)

    def for_provider(self, provider: str) -> dict[str, Any]:
        return {**self.params, **self.by_provider.get(provider, {})}


@dataclass(frozen=True)
class RunConfig:
    models: list[ModelSpec]
    topics: list[TopicSpec]
    settings: list[SettingSpec]
    cascade_variants: list[str]
    repetitions: int
    output_dir: Path
    request_timeout: float = 120.0
    max_concurrency_per_provider: int = 2
    max_retries: int = 4
    pause_between_turns_s: float = 0.0


def build_prompt_sequence(topic: TopicSpec, variant: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for step in CASCADES[variant]:
        out.append({
            "step": step["step"],
            "prompt": step["template"].format(
                initial_question=topic.initial_question,
                counter=topic.counter_claim,
            ),
        })
    return out


RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

TRUNCATED_REASONS = {"length", "max_tokens", "MAX_TOKENS",
                     "model_context_window_exceeded"}

BLOCKED_REASONS = {"content_filter", "SAFETY", "RECITATION",
                   "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "refusal"}

STATUS_BY_FAILURE = {
    "truncated": "aborted_truncated",
    "empty": "aborted_empty",
    "blocked": "aborted_blocked",
}


class ProviderError(RuntimeError):
    def __init__(self, message: str, status: int = 0, retryable: bool = False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


class ProviderClient:

    ROLE_ASSISTANT = {
        "openai": "assistant",
        "anthropic": "assistant",
        "google": "model",
    }

    def __init__(self, spec: ModelSpec, timeout: float, max_retries: int) -> None:
        self.spec = spec
        self.max_retries = max_retries
        self._client = httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    def _render_messages(self, history: list[dict[str, str]]) -> Any:
        assistant_role = self.ROLE_ASSISTANT[self.spec.provider]
        if self.spec.provider == "google":
            return [
                {"role": "user" if m["role"] == "user" else assistant_role,
                 "parts": [{"text": m["content"]}]}
                for m in history
            ]
        return [
            {"role": "user" if m["role"] == "user" else assistant_role,
             "content": m["content"]}
            for m in history
        ]

    def _request(self, history: list[dict[str, str]],
                 params: dict[str, Any]) -> tuple[str, dict, dict]:
        merged = {**self.spec.params, **params}
        if self.spec.provider == "openai":
            url = self.spec.api_base or "https://api.openai.com/v1/chat/completions"
            payload = {"model": self.spec.model_id,
                       "messages": self._render_messages(history),
                       **merged}
            headers = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                       "Content-Type": "application/json"}
        elif self.spec.provider == "anthropic":
            url = self.spec.api_base or "https://api.anthropic.com/v1/messages"
            payload = {"model": self.spec.model_id,
                       "max_tokens": merged.pop("max_tokens", 1024),
                       "messages": self._render_messages(history),
                       **merged}
            headers = {"x-api-key": os.environ["ANTHROPIC_API_KEY"],
                       "anthropic-version": "2023-06-01",
                       "Content-Type": "application/json"}
        elif self.spec.provider == "google":
            base = self.spec.api_base or "https://generativelanguage.googleapis.com/v1beta"
            url = f"{base}/models/{self.spec.model_id}:generateContent"
            payload = {"contents": self._render_messages(history),
                       "generationConfig": merged}
            headers = {"Content-Type": "application/json",
                       "x-goog-api-key": os.environ["GOOGLE_API_KEY"]}
        else:
            raise ProviderError(f"Unknown provider: {self.spec.provider}")
        return url, payload, headers

    @staticmethod
    def _extract_text(provider: str, raw: dict[str, Any]) -> str:
        if provider == "openai":
            return raw["choices"][0]["message"].get("content") or ""
        if provider == "anthropic":
            return "".join(b.get("text", "") for b in raw.get("content", [])
                           if b.get("type") == "text")
        if provider == "google":
            return "".join(
                p.get("text", "")
                for c in raw.get("candidates", [])
                for p in c.get("content", {}).get("parts", [])
            )
        raise ProviderError(provider)

    @staticmethod
    def _fingerprint(provider: str, raw: dict[str, Any]) -> str:
        if provider == "openai":
            return str(raw.get("system_fingerprint") or "")
        return ""

    @staticmethod
    def _reported_model(provider: str, raw: dict[str, Any]) -> str:
        if provider in ("openai", "anthropic"):
            return str(raw.get("model", ""))
        if provider == "google":
            return str(raw.get("modelVersion", ""))
        return ""

    @staticmethod
    def _usage(provider: str, raw: dict[str, Any]) -> dict[str, Any]:
        if provider == "openai":
            u = raw.get("usage", {}) or {}
            details = u.get("completion_tokens_details", {}) or {}
            return {"input": u.get("prompt_tokens"),
                    "output": u.get("completion_tokens"),
                    "reasoning": details.get("reasoning_tokens")}
        if provider == "anthropic":
            u = raw.get("usage", {}) or {}
            details = u.get("output_tokens_details", {}) or {}
            blocks = [b for b in raw.get("content", [])
                      if b.get("type") in ("thinking", "redacted_thinking")]
            return {"input": u.get("input_tokens"),
                    "output": u.get("output_tokens"),
                    "reasoning": details.get("thinking_tokens"),
                    "thinking_blocks": len(blocks),
                    "thinking_chars": sum(len(b.get("thinking") or "")
                                          for b in blocks)}
        if provider == "google":
            u = raw.get("usageMetadata", {}) or {}
            return {"input": u.get("promptTokenCount"),
                    "output": u.get("candidatesTokenCount"),
                    "reasoning": u.get("thoughtsTokenCount") or 0}
        return {}

    @staticmethod
    def _finish_reason(provider: str, raw: dict[str, Any]) -> str:
        try:
            if provider == "openai":
                return str(raw["choices"][0].get("finish_reason", ""))
            if provider == "anthropic":
                return str(raw.get("stop_reason", ""))
            if provider == "google":
                return str(raw["candidates"][0].get("finishReason", ""))
        except (KeyError, IndexError):
            return ""
        return ""

    async def chat(self, history: list[dict[str, str]],
                   params: dict[str, Any]) -> dict[str, Any]:
        url, payload, headers = self._request(history, params)
        attempt = 0
        last: ProviderError | None = None
        while attempt < self.max_retries:
            attempt += 1
            started = time.time()
            try:
                r = await self._client.post(url, json=payload, headers=headers)
                if r.status_code >= 400:
                    retryable = r.status_code in RETRY_STATUS
                    raise ProviderError(
                        f"HTTP {r.status_code}: {r.text[:300]}",
                        status=r.status_code, retryable=retryable)
                raw = r.json()
                text = self._extract_text(self.spec.provider, raw)
                finish = self._finish_reason(self.spec.provider, raw)
                result = {
                    "ok": True,
                    "raw": raw,
                    "text": text,
                    "http_status": r.status_code,
                    "elapsed_s": round(time.time() - started, 3),
                    "attempts": attempt,
                    "reported_model": self._reported_model(self.spec.provider, raw),
                    "fingerprint": self._fingerprint(self.spec.provider, raw),
                    "finish_reason": finish,
                    "usage": self._usage(self.spec.provider, raw),
                    "error": None,
                    "failure_kind": None,
                }
                if finish in TRUNCATED_REASONS:
                    result.update(ok=False, failure_kind="truncated",
                                  error=f"Truncated by token limit "
                                        f"(finish_reason={finish})")
                elif finish in BLOCKED_REASONS:
                    result.update(ok=False, failure_kind="blocked",
                                  error=f"Blocked by provider "
                                        f"(finish_reason={finish})")
                elif not text.strip():
                    result.update(ok=False, failure_kind="empty",
                                  error=f"Empty response "
                                        f"(finish_reason={finish or 'none'})")
                return result
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last = ProviderError(f"{type(exc).__name__}: {exc}", retryable=True)
                last.kind = "transport"
            except ProviderError as exc:
                last = exc
                last.kind = "http"
            except (KeyError, IndexError, ValueError) as exc:
                last = ProviderError(f"Unexpected response format: {exc}",
                                     retryable=False)
                last.kind = "format"
            if not (last and last.retryable):
                break
            backoff = min(2 ** attempt + random.uniform(0, 1), 60)
            LOG.warning("%s attempt %d failed (%s), waiting %.1fs",
                        self.spec.key, attempt, last, backoff)
            await asyncio.sleep(backoff)
        return {
            "ok": False,
            "raw": {},
            "text": "",
            "http_status": getattr(last, "status", 0),
            "elapsed_s": 0.0,
            "attempts": attempt,
            "reported_model": "",
            "fingerprint": "",
            "finish_reason": "",
            "usage": {},
            "error": str(last) if last else "unknown error",
            "failure_kind": getattr(last, "kind", "http"),
        }


def reasoning_indicator(provider: str, usage: dict[str, Any]) -> tuple[Any, str]:
    usage = usage or {}
    if usage.get("reasoning") is not None:
        return usage.get("reasoning"), "reasoning tokens"
    if provider == "anthropic" and usage.get("output") is not None:
        return usage.get("output"), "output tokens incl. thinking"
    return None, "reasoning tokens"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Turn:
    turn_index: int
    step: str
    prompt: str
    response_text: str
    ok: bool
    error: str | None
    http_status: int
    attempts: int
    elapsed_s: float
    reported_model: str
    fingerprint: str
    finish_reason: str
    usage: dict[str, Any]
    failure_kind: str | None
    timestamp: str
    raw_response: dict[str, Any]


@dataclass
class Conversation:
    run_id: str
    timepoint: str
    model_key: str
    model_id: str
    version_label: str
    reported_model: str
    provider: str
    topic_key: str
    topic_label: str
    premise: str
    cascade_variant: str
    cascade_hash: str
    setting_key: str
    params: dict[str, Any]
    repetition: int
    status: str = "complete"
    run_label: str = "main"
    harness_version: str = HARNESS_VERSION
    started_at: str = ""
    finished_at: str = ""
    turns: list[Turn] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        d = asdict(self)
        d["turns"] = [asdict(t) for t in self.turns]
        return d


@dataclass(frozen=True)
class Cell:
    model: ModelSpec
    topic: TopicSpec
    setting: SettingSpec
    variant: str
    repetition: int


def build_plan(cfg: RunConfig) -> list[Cell]:
    plan: list[Cell] = []
    for model in cfg.models:
        for topic in cfg.topics:
            for setting in cfg.settings:
                for variant in cfg.cascade_variants:
                    for rep in range(1, cfg.repetitions + 1):
                        plan.append(Cell(model, topic, setting, variant, rep))
    return plan


async def run_cell(cell: Cell, cfg: RunConfig, timepoint: str,
                   client: ProviderClient, sem: asyncio.Semaphore,
                   writer: "JsonlWriter", run_label: str = "main") -> Conversation:
    conv = Conversation(
        run_id=str(uuid.uuid4()),
        timepoint=timepoint,
        model_key=cell.model.key,
        model_id=cell.model.model_id,
        version_label=cell.model.version_label,
        reported_model="",
        provider=cell.model.provider,
        topic_key=cell.topic.key,
        topic_label=cell.topic.label,
        premise=cell.topic.premise,
        cascade_variant=cell.variant,
        cascade_hash=cascade_hash(cell.variant),
        setting_key=cell.setting.key,
        params={**cell.model.params,
                **cell.setting.for_provider(cell.model.provider)},
        repetition=cell.repetition,
        run_label=run_label,
        started_at=now_iso(),
    )
    history: list[dict[str, str]] = []
    sequence = build_prompt_sequence(cell.topic, cell.variant)

    async with sem:
        for idx, step in enumerate(sequence, start=1):
            history.append({"role": "user", "content": step["prompt"]})
            result = await client.chat(
                history, cell.setting.for_provider(cell.model.provider))
            conv.turns.append(Turn(
                turn_index=idx,
                step=step["step"],
                prompt=step["prompt"],
                response_text=result["text"],
                ok=bool(result["ok"]),
                error=result["error"],
                http_status=int(result["http_status"]),
                attempts=int(result["attempts"]),
                elapsed_s=float(result["elapsed_s"]),
                reported_model=result["reported_model"],
                fingerprint=result.get("fingerprint", ""),
                finish_reason=result["finish_reason"],
                usage=result.get("usage", {}),
                failure_kind=result.get("failure_kind"),
                timestamp=now_iso(),
                raw_response=result["raw"],
            ))
            if not result["ok"]:
                conv.status = STATUS_BY_FAILURE.get(result.get("failure_kind"),
                                                    "aborted_error")
                LOG.error("Technical abort %s/%s/%s step %s: %s",
                          cell.model.key, cell.topic.key, cell.variant,
                          step["step"], result["error"])
                break
            if not conv.reported_model:
                conv.reported_model = result["reported_model"]
            history.append({"role": "assistant", "content": result["text"]})
            if cfg.pause_between_turns_s:
                await asyncio.sleep(cfg.pause_between_turns_s)

    conv.finished_at = now_iso()
    await writer.write(conv)
    return conv


class JsonlWriter:

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = asyncio.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    async def write(self, conv: Conversation) -> None:
        async with self._lock:
            self._fh.write(json.dumps(conv.to_record(), ensure_ascii=False) + "\n")
            self._fh.flush()

    def close(self) -> None:
        self._fh.close()


async def run_all(cfg: RunConfig, timepoint: str,
                  run_label: str = "main") -> tuple[Path, list[Conversation]]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = "pilot" if run_label == "pilot" else "conversations"
    out_path = cfg.output_dir / f"{prefix}_{timepoint}_{stamp}.jsonl"
    writer = JsonlWriter(out_path)

    clients = {m.key: ProviderClient(m, cfg.request_timeout, cfg.max_retries)
               for m in cfg.models}
    sems = {p: asyncio.Semaphore(cfg.max_concurrency_per_provider)
            for p in {m.provider for m in cfg.models}}

    tasks = [
        asyncio.create_task(
            run_cell(cell, cfg, timepoint, clients[cell.model.key],
                     sems[cell.model.provider], writer, run_label)
        )
        for cell in build_plan(cfg)
    ]
    results = await asyncio.gather(*tasks)

    for c in clients.values():
        await c.aclose()
    writer.close()
    return out_path, list(results)


def load_config(path: Path) -> RunConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    models = [ModelSpec(
        key=m["key"], provider=m["provider"], model_id=m["model_id"],
        version_label=m.get("version_label", ""),
        api_base=m.get("api_base"), params=dict(m.get("params", {})),
    ) for m in data["models"]]

    topics = [TopicSpec(
        key=t["key"], label=t["label"],
        initial_question=t["initial_question"],
        counter_claim=t["counter_claim"],
        premise=t.get("premise", ""),
    ) for t in data["topics"]]

    without_premise = [t.key for t in topics if not t.premise.strip()]
    if without_premise:
        LOG.warning("Topics without a main premise: %s. "
                    "P3 cannot be coded without it.", without_premise)

    settings = [SettingSpec(key=s["key"],
                            params=dict(s.get("params", {}) or {}),
                            by_provider={k: dict(v or {}) for k, v in
                                         (s.get("by_provider", {}) or {}).items()})
                for s in data.get("settings", [{"key": "default", "params": {}}])]

    variants = list(data.get("cascade_variants", ["A"]))
    unknown = [v for v in variants if v not in CASCADES]
    if unknown:
        raise ValueError(f"Unknown cascade variant: {unknown}")

    return RunConfig(
        models=models,
        topics=topics,
        settings=settings,
        cascade_variants=variants,
        repetitions=int(data["repetitions"]),
        output_dir=Path(data["output_dir"]),
        request_timeout=float(data.get("request_timeout", 120.0)),
        max_concurrency_per_provider=int(data.get("max_concurrency_per_provider", 2)),
        max_retries=int(data.get("max_retries", 4)),
        pause_between_turns_s=float(data.get("pause_between_turns_s", 0.0)),
    )


PLACEHOLDER = re.compile(r"\[[^\]]+\]")


def find_placeholders(cfg: RunConfig) -> list[str]:
    hits: list[str] = []
    for m in cfg.models:
        if PLACEHOLDER.search(m.model_id):
            hits.append(f"models.{m.key}.model_id = {m.model_id}")
    for t in cfg.topics:
        for name in ("label", "initial_question", "counter_claim", "premise"):
            value = getattr(t, name)
            if PLACEHOLDER.search(value):
                hits.append(f"topics.{t.key}.{name} = {value[:60]}")
    return hits


def check_environment(cfg: RunConfig) -> list[str]:
    missing = []
    for provider in {m.provider for m in cfg.models}:
        var = KEY_VARIABLES.get(provider)
        if var and not os.environ.get(var):
            missing.append(var)
    return missing


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Breaching experiment harness")
    ap.add_argument("config", type=Path)
    ap.add_argument("--timepoint", required=True,
                    help="Point in time, e.g. T1 or T2. Two points in time "
                         "mean two separate runs.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the collection plan only, send nothing.")
    ap.add_argument("--pilot", action="store_true",
                    help="Pilot run before registration. Data are stored "
                         "with run_label pilot and a separate file prefix "
                         "and do not enter the corpus.")
    ap.add_argument("--only", nargs="+", metavar="MODEL_KEY",
                    help="Pilot only. Run just these model keys from the "
                         "config, e.g. --only model_a model_b. The main run "
                         "must always cover all models.")
    ap.add_argument("--env", type=Path, default=None,
                    help="Path to the .env file with API keys. Default is "
                         ".env next to this script.")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    env_path = args.env or (Path(__file__).resolve().parent / ".env")
    loaded = load_dotenv(env_path)
    if loaded:
        check_dotenv_permissions(env_path)
        LOG.info("Loaded %d variable(s) from %s", len(loaded), env_path)
    elif args.env:
        LOG.warning("No .env found at %s", env_path)

    cfg = load_config(args.config)

    if args.only:
        if not args.pilot:
            print("--only is allowed for pilot runs only. The main run must "
                  "cover all models.", file=sys.stderr)
            return 2
        known = {m.key for m in cfg.models}
        unknown = [k for k in args.only if k not in known]
        if unknown:
            print(f"Unknown model keys {unknown}. Known: {sorted(known)}",
                  file=sys.stderr)
            return 2
        cfg = replace(cfg, models=[m for m in cfg.models if m.key in args.only])
        LOG.info("Partial pilot with %s", [m.key for m in cfg.models])

    plan = build_plan(cfg)

    LOG.info("%s, timepoint %s, %d conversations, %d calls expected",
             "Pilot run" if args.pilot else "Main run",
             args.timepoint, len(plan), len(plan) * len(STEPS))
    LOG.info("Cascade variants %s, hashes %s",
             cfg.cascade_variants,
             {v: cascade_hash(v) for v in cfg.cascade_variants})

    placeholders = find_placeholders(cfg)
    if placeholders and not args.dry_run:
        print("Configuration still contains placeholders. Nothing was sent.\n  "
              + "\n  ".join(placeholders)
              + "\nRun test.py to confirm model IDs, then fill in the topics.",
              file=sys.stderr)
        return 2
    if placeholders:
        LOG.warning("Placeholders present (%d). A real run would refuse to "
                    "start.", len(placeholders))

    if args.dry_run:
        for cell in plan[:20]:
            print(f"{cell.model.key} | {cell.topic.key} | {cell.setting.key} "
                  f"| variant {cell.variant} | repetition {cell.repetition}")
        if len(plan) > 20:
            print(f"... and {len(plan) - 20} more cells")
        return 0

    missing = check_environment(cfg)
    if missing:
        print("Missing API keys: " + ", ".join(missing)
              + f". Put them into {env_path} (see .env.example) "
                "or export them in the shell.", file=sys.stderr)
        return 2

    label = "pilot" if args.pilot else "main"
    path, convs = asyncio.run(run_all(cfg, args.timepoint, label))
    from collections import Counter
    status = Counter(c.status for c in convs)
    aborted = len(convs) - status.get("complete", 0)
    LOG.info("Saved: %s", path)
    LOG.info("%d conversations, status %s", len(convs), dict(status))
    if aborted:
        LOG.warning("Technically aborted conversations are excluded from "
                    "coding and reported as the failure rate.")
    if status.get("aborted_truncated") or status.get("aborted_empty"):
        LOG.warning("Truncated or empty answers occurred. Raise the token "
                    "budget in the model params before the main run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
