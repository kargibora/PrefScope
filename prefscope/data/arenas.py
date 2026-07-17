"""Per-arena adapters: map a raw HuggingFace arena dataset into the corpus schema.

Supported sources (label-free — we keep every battle, including ties):

- ``lmarena-100k`` / ``lmarena-140k`` — lmarena-ai pairwise preference dumps.
  ``conversation_a``/``conversation_b`` are ``[{role, content, ...}]`` message
  lists; the prompt is the user turn inside ``conversation_a``.
- ``comparia`` — ministere-culture/comparia-votes (gated; needs an HF token).
  ``opening_msg`` is the prompt; responses live in ``conversation_a/b``; models
  are ``model_a_name``/``model_b_name``.

The pure transform ``conversations_to_corpus`` is dataset-agnostic and unit
tested; ``load_arena`` adds the ``datasets`` I/O (optional ``[arena]`` extra).
Default policy: single-turn (first user turn + each model's first reply).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from prefscope.data.corpus import normalize

_USER_ROLES = {"user", "human"}
_ASSISTANT_ROLES = {"assistant", "model", "bot", "gpt", "ai"}

# source -> raw-column / HF config
SOURCES = {
    "lmarena-100k": {"hf_id": "lmarena-ai/arena-human-preference-100k"},
    "lmarena-140k": {"hf_id": "lmarena-ai/arena-human-preference-140k"},
    "comparia": {"hf_id": "ministere-culture/comparia-votes",
                 "model_a": "model_a_name", "model_b": "model_b_name",
                 "prompt_col": "opening_msg"},
}


def _winner_to_pref(winner) -> float:
    """lmarena 'winner' -> y = P(A preferred). tie/bothbad/unknown -> 0.5."""
    w = str(winner).strip().lower()
    if w == "model_a":
        return 1.0
    if w == "model_b":
        return 0.0
    return 0.5


def human_pref(df: pd.DataFrame, source: str) -> pd.Series:
    """Per-battle human preference as y = P(A preferred), from the arena's vote."""
    if source.startswith("lmarena"):
        return df["winner"].map(_winner_to_pref).astype(float)
    if source == "comparia":
        def _one(r):
            if bool(r.get("both_equal")):
                return 0.5
            chosen = str(r.get("chosen_model_name"))
            if chosen == str(r.get("model_a_name")):
                return 1.0
            if chosen == str(r.get("model_b_name")):
                return 0.0
            return 0.5
        return df.apply(_one, axis=1).astype(float)
    raise ValueError(f"no human-pref rule for source {source!r}")


def _to_text(content) -> str:
    """Coerce a message's content (str | list of parts | None) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, np.ndarray):
        content = content.tolist()
    if isinstance(content, (list, tuple)):
        parts = []
        for it in content:
            if isinstance(it, dict):
                parts.append(str(it.get("text", it.get("content", ""))))
            else:
                parts.append(str(it))
        return "\n".join(p for p in parts if p)
    return "" if content is None else str(content)


def _messages(conv):
    """Normalise a conversation field to a list of message dicts."""
    if isinstance(conv, np.ndarray):
        conv = conv.tolist()
    if isinstance(conv, dict):
        for k in ("messages", "conversation", "turns", "content"):
            if k in conv:
                conv = conv[k]
                break
    if isinstance(conv, np.ndarray):
        conv = conv.tolist()
    return list(conv) if isinstance(conv, (list, tuple)) else []


def _first(conv, roles) -> str:
    for m in _messages(conv):
        if isinstance(m, dict) and str(m.get("role", "")).lower() in roles:
            return _to_text(m.get("content"))
    return ""


def conversations_to_corpus(df: pd.DataFrame, source: str, *,
                            conv_a: str = "conversation_a",
                            conv_b: str = "conversation_b",
                            model_a: str = "model_a", model_b: str = "model_b",
                            prompt_col: str | None = None,
                            language_col: str = "language",
                            human_pref_vals=None) -> pd.DataFrame:
    """Map a raw arena frame to the normalized corpus schema (single-turn).

    prompt_col: if given, the prompt is read from that column directly
    (e.g. comparia ``opening_msg``); otherwise it is the first user turn of
    ``conv_a``. Model responses are each conversation's first assistant turn.
    human_pref_vals: optional per-row y=P(A preferred), carried into the corpus.
    """
    ca, cb = df[conv_a].tolist(), df[conv_b].tolist()
    out = pd.DataFrame({
        "model_a": df[model_a].astype(str),
        "model_b": df[model_b].astype(str),
        "completion_a": [_first(c, _ASSISTANT_ROLES) for c in ca],
        "completion_b": [_first(c, _ASSISTANT_ROLES) for c in cb],
    })
    if prompt_col is not None and prompt_col in df.columns:
        out["prompt"] = df[prompt_col].map(_to_text).tolist()
    else:
        out["prompt"] = [_first(c, _USER_ROLES) for c in ca]
    out["language"] = df[language_col].astype(str).tolist() if language_col in df.columns else ""
    if human_pref_vals is not None:
        out["human_pref"] = list(human_pref_vals)
    return normalize(out, source)


def load_arena(source: str, *, split: str = "train", limit: int | None = None,
               token: str | None = None, keep_labels: bool = False) -> pd.DataFrame:
    """Load one arena from HuggingFace and normalize it. Needs the ``[arena]`` extra.

    keep_labels: also carry the human vote as ``human_pref`` (y = P(A preferred)).
    """
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; known: {sorted(SOURCES)}")
    cfg = SOURCES[source]
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover - import guard
        raise ImportError(
            "loading arenas needs the 'datasets' library: uv sync --extra arena") from e
    ds = load_dataset(cfg["hf_id"], split=split, token=token)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    df = ds.to_pandas()
    hp = human_pref(df, source) if keep_labels else None
    return conversations_to_corpus(
        df, source, model_a=cfg.get("model_a", "model_a"),
        model_b=cfg.get("model_b", "model_b"), prompt_col=cfg.get("prompt_col"),
        human_pref_vals=hp)
