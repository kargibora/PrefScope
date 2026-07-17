"""Backend-agnostic LLM text completion.

`openai` backend talks to any OpenAI-compatible endpoint (DeepSeek-V3.2 on
OpenRouter by default, or a local vLLM server via api_base). `claude-cli` and
`codex-cli` shell out to the local `claude` / `codex` CLIs (no API key — uses
each tool's own auth). WIMHF prompts expect free-text output, so json_mode
defaults to False.
"""
from __future__ import annotations

import os
import random
import subprocess
import tempfile
import time


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


class LLMClient:
    def __init__(self, *, backend: str = "openai", model: str = DEFAULT_MODEL,
                 api_base: str | None = DEFAULT_API_BASE,
                 api_key_env: str = "OPENROUTER_API_KEY",
                 temperature: float = 0.2, max_tokens: int = 512,
                 json_mode: bool = False, timeout: int = 180,
                 retries: int = 3, reasoning_effort: str | None = None,
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
        self._client = _client
        if backend == "openai" and self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=os.environ.get(api_key_env, "EMPTY"),
                                  base_url=api_base)

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
                    return out
                except Exception as e:
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
                    return out
                except Exception as e:
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
                out = _response_text(resp)
                if not out.strip():
                    # Distinguish TRUNCATION (reasoning ate the budget) from a real empty,
                    # so it's visible in logs/debug dumps instead of silently abstaining.
                    if _finish_reason(resp) == "length":
                        raise RuntimeError(
                            f"truncated (finish_reason=length) at max_tokens={mt} — the "
                            "model likely spent the budget reasoning; raise --max-tokens or "
                            "set --reasoning-effort minimal")
                    raise RuntimeError("empty response")
                return out
            except Exception as e:
                if _nonretryable(e):
                    raise                          # fast-fail: don't retry a dead key/credit
                last_exc = e
                if attempt < self.retries - 1:
                    _backoff(attempt)
        raise last_exc
