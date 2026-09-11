"""Small shared helpers: JSON sanitization, CSV reading, concept-name lookup."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _read_csv(p: Path):
    return pd.read_csv(p) if p.exists() else None


def _round(df: pd.DataFrame, n=5) -> list[dict]:
    return json.loads(df.round(n).to_json(orient="records"))


def _sanitize(o):
    """Recursively replace non-finite floats (NaN/Inf) with None. Python's json.dumps emits
    bare ``NaN``/``Infinity`` by default, which are INVALID JSON — the browser's JSON.parse
    rejects them and the whole file silently fails to load. (String content like a completion
    containing the text "NaN" is untouched — only float *values* are converted.)"""
    import math
    if isinstance(o, np.floating):          # np.float32 etc. aren't python-float subclasses
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    return o


def _dumps(obj, **kw) -> str:
    """json.dumps that always emits valid JSON (NaN/Inf -> null)."""
    return json.dumps(_sanitize(obj), **kw)


def _clean_label(value) -> str | None:
    """Return a non-blank display label, or ``None`` when it is missing.

    CSV round-trips can represent a missing label as ``None``, a pandas/NumPy null,
    whitespace, or the literal string ``"nan"``. Treat all of those as absent before
    any caller converts the value to ``str``.
    """
    if value is None or pd.isna(value):
        return None
    label = str(value).strip()
    if not label or label.casefold() == "nan":
        return None
    return label


def _label_or_id(value, feature_id: int, *, prefix: str = "feature") -> str:
    """Return a safe label, falling back to the feature ID when unnamed."""
    label = _clean_label(value)
    return label if label is not None else f"{prefix} {int(feature_id)}"


def _concept_or_none(m: dict, k):
    """Concept name for id ``k`` from map ``m``, or ``None`` when unnamed."""
    return _clean_label(m.get(k))


def _concept_or_id(m: dict, k, *, prefix: str = "feature") -> str:
    """Concept name for id ``k``, with a stable feature-ID fallback."""
    return _label_or_id(m.get(k), int(k), prefix=prefix)
