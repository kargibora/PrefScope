"""Corpus sanity summary for a normalized battle table."""
from __future__ import annotations

import pandas as pd


def summarize(battles: pd.DataFrame) -> dict:
    """Counts a human reviews before launching a heavy embed+train run.

    Works on either a labeled annotation table or a label-free corpus: the
    preference distribution is included only when a ``y_judge`` column is present,
    and the language column may be named ``lang`` or ``language``.
    """
    appearances = pd.concat([battles["model_a"], battles["model_b"]])
    out = {
        "n_battles": int(len(battles)),
        "n_models": int(appearances.nunique()),
        "model_counts": appearances.value_counts().to_dict(),
    }
    if "y_judge" in battles.columns:
        out["y_judge_dist"] = {float(k): int(v)
                               for k, v in battles["y_judge"].value_counts().items()}
    lang_col = next((c for c in ("lang", "language") if c in battles.columns), None)
    out["langs"] = battles[lang_col].value_counts().to_dict() if lang_col else {}
    return out
