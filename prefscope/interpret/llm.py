"""Backend-agnostic LLM text completion.

`openai` backend talks to any OpenAI-compatible endpoint (DeepSeek-V3.2 on
OpenRouter by default, or a local vLLM server via api_base). `claude-cli` and
`codex-cli` shell out to the local `claude` / `codex` CLIs (no API key — uses
each tool's own auth). WIMHF prompts expect free-text output, so json_mode
defaults to False.
"""
from __future__ import annotations

import json
import math
import os
import random
import subprocess
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


def _backoff(attempt: int) -> None:
    # exponential backoff + jitter — rides out sustained 429s from lower-throughput
    # providers (e.g. GLM on OpenRouter) instead of exhausting instant retries.
    time.sleep(min(30.0, 2.0 ** attempt) + random.uniform(0, 1.0))


def _nonretryable(e: Exception) -> bool:
    """True for errors that will NEVER recover within a run — a bad/expired key (401),
    exhausted credit (402), or forbidden (403). Retrying these just burns requests
    (the dead-key retry storm), so we fail fast and abort the whole run instead."""
    code = getattr(e, "status_code", None) or getattr(e, "code", None)
    try:
        if int(code) in (401, 402, 403):
            return True
    except (TypeError, ValueError):
        pass
    msg = str(e).lower()
    return any(s in msg for s in (
        "insufficient", "payment required", "insufficient_quota", "credit",
        "unauthorized", "invalid api key", "no auth credentials", "402", "401", "403"))

DEFAULT_API_BASE = "https://openrouter.ai/api/v1"
# OpenRouter slug for DeepSeek-V3.2 — confirm/override with --model if needed.
DEFAULT_MODEL = "deepseek/deepseek-v3.2"


def _finish_reason(resp) -> str | None:
    try:
        return resp.choices[0].finish_reason
    except Exception:
        return None


def _response_text(resp) -> str:
    msg = resp.choices[0].message
    content = getattr(msg, "content", None)
    if isinstance(content, str) and content.strip():
        return content
    reasoning = getattr(msg, "reasoning_content", None)
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(getattr(item, "text", "")))
        return "\n".join(p for p in parts if p)
    return ""


def _field(obj, name: str, default=None):
    """Read an SDK response field, including provider-specific Pydantic extras."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    value = getattr(obj, name, default)
    if value is not default:
        return value
    extra = getattr(obj, "model_extra", None)
    if isinstance(extra, dict):
        return extra.get(name, default)
    return default


def _number(value, cast, default=0):
    try:
        out = cast(value)
        return out if math.isfinite(float(out)) else default
    except (TypeError, ValueError, OverflowError):
        return default


_USAGE_COUNTERS = (
    "attempted_requests", "responses", "failed_requests", "accepted_responses",
    "empty_responses", "truncated_responses", "usage_responses", "costed_responses",
    "prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens",
    "cached_tokens", "cache_write_tokens", "cost_credits",
)


def _empty_usage() -> dict:
    return {key: (0.0 if key == "cost_credits" else 0) for key in _USAGE_COUNTERS}


def _compact_number(value: int) -> str:
    value = int(value)
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)


class UsageTracker:
    """Thread-safe request/token/cost accounting for LLM calls.

    OpenRouter includes native token counts and the charged cost in ``response.usage``.
    We aggregate those exact provider-reported values. An optional JSONL event log is
    appended after every response/error so a long batch run leaves an audit trail even
    when interrupted. Prompts and completions are deliberately never written to it.
    """

    def __init__(self, event_log: str | Path | None = None, *, resume: bool = False) -> None:
        self.run_id = uuid.uuid4().hex
        self.event_log = Path(event_log) if event_log is not None else None
        self._lock = threading.Lock()
        self._total = _empty_usage()
        self._stages: dict[str, dict] = {}
        self._models: dict[str, dict] = {}
        self.resumed_events = 0
        if resume and self.event_log is not None and self.event_log.exists():
            self._resume_event_log()

    def _resume_event_log(self) -> None:
        """Rebuild cumulative counters from the append-only ledger after interruption."""
        for line in self.event_log.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except (TypeError, json.JSONDecodeError):
                continue  # tolerate one truncated final line after abrupt termination
            kind = event.get("event")
            values = _empty_usage()
            if kind == "error":
                values.update({"attempted_requests": 1, "failed_requests": 1})
            elif kind == "response":
                accepted = bool(event.get("accepted"))
                cost = event.get("cost_credits")
                prompt = _number(event.get("prompt_tokens"), int)
                completion = _number(event.get("completion_tokens"), int)
                total = _number(event.get("total_tokens"), int, prompt + completion)
                reasoning = _number(event.get("reasoning_tokens"), int)
                cached = _number(event.get("cached_tokens"), int)
                cache_write = _number(event.get("cache_write_tokens"), int)
                # Schema-v1 ledgers predate an explicit usage_available field. Provider
                # responses with token counts or a cost had usage; local CLI backends did not.
                usage_available = bool(event.get("usage_available",
                    any((prompt, completion, total, reasoning, cached, cache_write))
                    or cost is not None))
                values.update({
                    "attempted_requests": 1, "responses": 1,
                    "accepted_responses": int(accepted),
                    "empty_responses": int(not accepted),
                    "truncated_responses": int(
                        not accepted and event.get("finish_reason") == "length"),
                    "usage_responses": int(usage_available),
                    "costed_responses": int(cost is not None),
                    "prompt_tokens": prompt, "completion_tokens": completion,
                    "total_tokens": total, "reasoning_tokens": reasoning,
                    "cached_tokens": cached, "cache_write_tokens": cache_write,
                    "cost_credits": _number(cost, float) if cost is not None else 0.0,
                })
            else:
                continue
            stage = str(event.get("stage") or "llm")
            model = str(event.get("model") or event.get("requested_model") or "unknown")
            self._add(self._total, values)
            self._add(self._stages.setdefault(stage, _empty_usage()), values)
            self._add(self._models.setdefault(model, _empty_usage()), values)
            self.resumed_events += 1

    @staticmethod
    def _public_bucket(bucket: dict) -> dict:
        out = dict(bucket)
        out["usage_available"] = bool(out["usage_responses"])
        out["cost_available"] = bool(out["costed_responses"])
        return out

    @staticmethod
    def _add(bucket: dict, values: dict) -> None:
        for key in _USAGE_COUNTERS:
            bucket[key] += values.get(key, 0)

    def _record(self, event: dict, values: dict, *, stage: str, model: str) -> None:
        event = {"schema_version": 1, "run_id": self.run_id,
                 "timestamp": datetime.now(timezone.utc).isoformat(), **event}
        with self._lock:
            self._add(self._total, values)
            stage_bucket = self._stages.setdefault(stage, _empty_usage())
            model_bucket = self._models.setdefault(model, _empty_usage())
            self._add(stage_bucket, values)
            self._add(model_bucket, values)
            if self.event_log is not None:
                self.event_log.parent.mkdir(parents=True, exist_ok=True)
                with self.event_log.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")

    def record_response(self, response, *, requested_model: str, backend: str,
                        stage: str, attempt: int, accepted: bool) -> None:
        usage = _field(response, "usage")
        prompt = _number(_field(usage, "prompt_tokens"), int)
        completion = _number(_field(usage, "completion_tokens"), int)
        total = _number(_field(usage, "total_tokens"), int, prompt + completion)
        prompt_details = _field(usage, "prompt_tokens_details")
        completion_details = _field(usage, "completion_tokens_details")
        reasoning = _number(_field(completion_details, "reasoning_tokens"), int)
        cached = _number(_field(prompt_details, "cached_tokens"), int)
        cache_write = _number(_field(prompt_details, "cache_write_tokens"), int)
        raw_cost = _field(usage, "cost")
        cost = (_number(raw_cost, float) if raw_cost is not None else None)
        finish_reason = _finish_reason(response)
        actual_model = str(_field(response, "model", requested_model) or requested_model)
        values = _empty_usage()
        values.update({
            "attempted_requests": 1, "responses": 1,
            "accepted_responses": int(accepted),
            "empty_responses": int(not accepted),
            "truncated_responses": int(not accepted and finish_reason == "length"),
            "usage_responses": int(usage is not None),
            "costed_responses": int(cost is not None),
            "prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": total, "reasoning_tokens": reasoning,
            "cached_tokens": cached, "cache_write_tokens": cache_write,
            "cost_credits": cost or 0.0,
        })
        self._record({
            "event": "response", "backend": backend, "stage": stage,
            "attempt": int(attempt), "requested_model": requested_model,
            "model": actual_model, "generation_id": _field(response, "id"),
            "finish_reason": finish_reason, "accepted": bool(accepted),
            "usage_available": bool(usage is not None),
            "prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": total, "reasoning_tokens": reasoning,
            "cached_tokens": cached, "cache_write_tokens": cache_write,
            "cost_credits": cost,
        }, values, stage=stage, model=actual_model)

    def record_error(self, error: Exception, *, requested_model: str, backend: str,
                     stage: str, attempt: int) -> None:
        values = _empty_usage()
        values.update({"attempted_requests": 1, "failed_requests": 1})
        status = getattr(error, "status_code", None) or getattr(error, "code", None)
        self._record({
            "event": "error", "backend": backend, "stage": stage,
            "attempt": int(attempt), "requested_model": requested_model,
            "model": requested_model, "error_type": type(error).__name__,
            "status_code": status,
        }, values, stage=stage, model=requested_model)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "schema_version": 1,
                "run_id": self.run_id,
                "resumed_events": self.resumed_events,
                "cost_unit": "OpenRouter credits",
                "total": self._public_bucket(self._total),
                "stages": {k: self._public_bucket(v) for k, v in self._stages.items()},
                "models": {k: self._public_bucket(v) for k, v in self._models.items()},
                "event_log": str(self.event_log) if self.event_log is not None else None,
            }

    def write_summary(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(self.snapshot(), indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
        tmp.replace(path)
        return path

    def progress(self) -> str:
        total = self.snapshot()["total"]
        requests = total["attempted_requests"]
        if total["usage_available"]:
            tokens = (f"input {_compact_number(total['prompt_tokens'])} | "
                      f"output {_compact_number(total['completion_tokens'])}")
        else:
            tokens = "tokens n/a"
        cost = (f"cost {total['cost_credits']:.4f} credits"
                if total["cost_available"] else "cost n/a")
        return f"requests {requests} | {tokens} | {cost}"


class LLMClient:
    def __init__(self, *, backend: str = "openai", model: str = DEFAULT_MODEL,
                 api_base: str | None = DEFAULT_API_BASE,
                 api_key_env: str = "OPENROUTER_API_KEY",
                 temperature: float = 0.2, max_tokens: int = 512,
                 json_mode: bool = False, timeout: int = 180,
                 retries: int = 3, reasoning_effort: str | None = None,
                 usage_tracker: UsageTracker | None = None,
                 usage_stage: str = "llm",
                 _client=None) -> None:
        if backend not in ("openai", "claude-cli", "codex-cli"):
            raise ValueError(
                f"backend must be 'openai', 'claude-cli', or 'codex-cli', got {backend!r}")
        self.backend = backend
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.json_mode = json_mode
        self.timeout = timeout
        self.retries = max(1, retries)
        # reasoning models (gpt-5-mini, o-series) spend tokens THINKING before the answer,
        # counting against max_tokens — so heavy reasoning on a simple naming task both
        # wastes tokens and can truncate the output. 'minimal'/'low' curbs it. None = leave
        # the provider default. Sent as OpenRouter's `reasoning.effort` (ignored by models
        # that don't reason).
        self.reasoning_effort = reasoning_effort
        self.usage_tracker = usage_tracker or UsageTracker()
        self.usage_stage = usage_stage
        self._client = _client
        if backend == "openai" and self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=os.environ.get(api_key_env, "EMPTY"),
                                  base_url=api_base)

    def usage_snapshot(self) -> dict:
        return self.usage_tracker.snapshot()

    def usage_progress(self) -> str:
        return self.usage_tracker.progress()

    def write_usage(self, path: str | Path) -> Path:
        return self.usage_tracker.write_summary(path)

    def raw(self, messages, *, max_tokens: int | None = None,
            json_mode: bool | None = None, response_schema: dict | None = None) -> str:
        mt = self.max_tokens if max_tokens is None else max_tokens
        jm = self.json_mode if json_mode is None else json_mode
        if self.backend == "claude-cli":
            prompt = "\n\n".join(m["content"] for m in messages)
            last_exc: Exception | None = None
            for attempt in range(self.retries):
                try:
                    proc = subprocess.run(
                        ["claude", "-p", "--model", self.model, "--tools", "",
                         "--strict-mcp-config", "--no-session-persistence"],
                        input=prompt, capture_output=True, text=True,
                        timeout=self.timeout)
                    if proc.returncode != 0:
                        raise RuntimeError(
                            f"claude CLI failed (rc={proc.returncode}): "
                            f"{(proc.stderr or proc.stdout).strip()[:300]}")
                    out = proc.stdout.strip()
                    if not out:
                        # empty output is a transient CLI failure, not a valid answer
                        raise RuntimeError("claude CLI returned empty output")
                    self.usage_tracker.record_response(
                        None, requested_model=self.model, backend=self.backend,
                        stage=self.usage_stage, attempt=attempt + 1, accepted=True)
                    return out
                except Exception as e:
                    self.usage_tracker.record_error(
                        e, requested_model=self.model, backend=self.backend,
                        stage=self.usage_stage, attempt=attempt + 1)
                    last_exc = e
                    if attempt < self.retries - 1:
                        time.sleep(1.5 * (attempt + 1))
            raise last_exc
        if self.backend == "codex-cli":
            prompt = "\n\n".join(m["content"] for m in messages)
            # -o writes ONLY the final assistant message (no event/log chatter);
            # read-only sandbox + ephemeral = no writes, no session files, no
            # approval prompts. Prompt is piped on stdin ('-').
            cmd = ["codex", "exec", "-s", "read-only", "--skip-git-repo-check",
                   "--ephemeral", "--color", "never"]
            if self.model and self.model != DEFAULT_MODEL:
                cmd += ["-m", self.model]      # else use codex's configured default
            last_exc: Exception | None = None
            for attempt in range(self.retries):
                out_path = None
                try:
                    fd, out_path = tempfile.mkstemp(suffix=".txt")
                    os.close(fd)
                    proc = subprocess.run(
                        cmd + ["-o", out_path, "-"], input=prompt,
                        capture_output=True, text=True, timeout=self.timeout)
                    if proc.returncode != 0:
                        raise RuntimeError(
                            f"codex CLI failed (rc={proc.returncode}): "
                            f"{(proc.stderr or proc.stdout).strip()[:300]}")
                    with open(out_path) as f:
                        out = f.read().strip()
                    if not out:
                        raise RuntimeError("codex CLI returned empty output")
                    self.usage_tracker.record_response(
                        None, requested_model=self.model, backend=self.backend,
                        stage=self.usage_stage, attempt=attempt + 1, accepted=True)
                    return out
                except Exception as e:
                    self.usage_tracker.record_error(
                        e, requested_model=self.model, backend=self.backend,
                        stage=self.usage_stage, attempt=attempt + 1)
                    last_exc = e
                    if attempt < self.retries - 1:
                        time.sleep(1.5 * (attempt + 1))
                finally:
                    if out_path and os.path.exists(out_path):
                        os.unlink(out_path)
            raise last_exc
        # progressive response_format fallback: strict json_schema (PINS the key) ->
        # plain json_object -> none. First non-empty wins; works on providers that don't
        # support json_schema. Empty content counts as a failure (transient).
        formats: list = []
        if response_schema is not None:
            formats.append({"type": "json_schema", "json_schema": {
                "name": "concept", "strict": True, "schema": response_schema}})
        if jm or response_schema is not None:
            formats.append({"type": "json_object"})
        formats.append(None)

        # Total-request budget = self.retries (a HARD cap, e.g. 3 — NOT retries*formats).
        # Each attempt advances through the format list (json_schema -> json_object -> none)
        # to negotiate provider compatibility, then repeats the last (widely-supported) one.
        # A non-retryable error (dead key / no credit) aborts immediately — no storm.
        last_exc: Exception | None = None
        for attempt in range(self.retries):
            rf = formats[min(attempt, len(formats) - 1)]
            kwargs = {"model": self.model, "messages": messages,
                      "temperature": self.temperature, "max_tokens": mt}
            if rf is not None:
                kwargs["response_format"] = rf
            if self.reasoning_effort:
                kwargs["extra_body"] = {"reasoning": {"effort": self.reasoning_effort}}
            try:
                resp = self._client.chat.completions.create(**kwargs)
            except Exception as e:
                self.usage_tracker.record_error(
                    e, requested_model=self.model, backend=self.backend,
                    stage=self.usage_stage, attempt=attempt + 1)
                if _nonretryable(e):
                    raise                          # fast-fail: don't retry a dead key/credit
                last_exc = e
                if attempt < self.retries - 1:
                    _backoff(attempt)
                continue

            out = _response_text(resp)
            accepted = bool(out.strip())
            # Account before validating content: empty/truncated responses can still be
            # billed, and the retry must not make that cost disappear from the ledger.
            self.usage_tracker.record_response(
                resp, requested_model=self.model, backend=self.backend,
                stage=self.usage_stage, attempt=attempt + 1, accepted=accepted)
            if accepted:
                return out
            # Distinguish TRUNCATION (reasoning ate the budget) from a real empty,
            # so it's visible in logs/debug dumps instead of silently abstaining.
            if _finish_reason(resp) == "length":
                last_exc = RuntimeError(
                    f"truncated (finish_reason=length) at max_tokens={mt} — the "
                    "model likely spent the budget reasoning; raise --max-tokens or "
                    "set --reasoning-effort none")
            else:
                last_exc = RuntimeError("empty response")
            if attempt < self.retries - 1:
                _backoff(attempt)
        raise last_exc
