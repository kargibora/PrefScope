"""Corpus sanity summary for a normalized battle table."""
from __future__ import annotations

import pandas as pd


def summarize(battles: pd.DataFrame) -> dict:
    """Counts a human reviews before launching a heavy embed+train run.

    Works on labeled annotations, a label-free corpus, and single-response data:
    model counts appear only when model columns exist, the preference distribution
    only when ``y_judge`` is present, and the language column may be ``lang`` or
    ``language``.
    """
    model_cols = [c for c in ("model_a", "model_b") if c in battles.columns]
    out = {"n_battles": int(len(battles))}
    if model_cols:
        appearances = pd.concat([battles[c] for c in model_cols])
        out["n_models"] = int(appearances.nunique())
        out["model_counts"] = appearances.value_counts().to_dict()
    else:
        out["n_models"] = 0
        out["model_counts"] = {}
    out["paired"] = "completion_b" in battles.columns
    if "y_judge" in battles.columns:
        out["y_judge_dist"] = {float(k): int(v)
                               for k, v in battles["y_judge"].value_counts().items()}
    lang_col = next((c for c in ("lang", "language") if c in battles.columns), None)
    out["langs"] = battles[lang_col].value_counts().to_dict() if lang_col else {}
    return out
