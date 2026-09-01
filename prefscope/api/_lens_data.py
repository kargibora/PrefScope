"""Torch-free dataset normalization used by the Lens facade."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

_REQUIRED_ITEM_COLS = ["prompt", "completion_a", "instruction_id"]

def pairs_to_battles(data, columns=None) -> pd.DataFrame:
    """Normalize preference data into a ``build_lens`` battles DataFrame.

    Accepts (a) a ``Dataset`` / iterable of ``PairItem`` (mapped
    ``x->prompt, y_a->completion_a, y_b->completion_b, id->instruction_id`` plus
    ``pref->human_pref`` / ``model_a`` / ``model_b`` when present), (b) a
    ``pd.DataFrame`` (the ``columns`` rename map is applied first, then required
    columns are validated), or (c) a ``str`` / ``Path`` parquet file (read, then
    treated as a DataFrame). ``completion_b`` is optional for homogeneous
    single-response data. Pure: no embedding, no torch.
    """
    if isinstance(data, (str, Path)):
        data = pd.read_parquet(data)

    if isinstance(data, pd.DataFrame):
        df = data.rename(columns=dict(columns)) if columns else data.copy()
        missing = [c for c in _REQUIRED_ITEM_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"battles missing required columns: {missing}")
        return df.reset_index(drop=True)

    # iterable of PairItem-like objects
    rows = []
    reserved = {
        "instruction_id", "prompt", "completion_a", "completion_b",
        "human_pref", "model_a", "model_b",
    }
    for it in data:
        if not isinstance(it.meta, dict):
            raise ValueError("PairItem.meta must be a mapping")
        collisions = reserved & set(it.meta)
        if collisions:
            raise ValueError(
                f"PairItem.meta collides with canonical fields: {sorted(collisions)}")
        rows.append({
            "instruction_id": it.id,
            "prompt": it.x,
            "completion_a": it.y_a,
            "completion_b": it.y_b,
            "human_pref": it.pref,
            "model_a": it.model_a,
            "model_b": it.model_b,
            **it.meta,
        })
    # The canonical keys above guarantee every required column exists. Custom scalar
    # metadata is retained for grouping and outcome analysis instead of being discarded.
    if not rows:
        return pd.DataFrame(columns=[
            "instruction_id", "prompt", "completion_a", "completion_b",
            "human_pref", "model_a", "model_b",
        ])
    return pd.DataFrame(rows)

__all__ = ["pairs_to_battles"]
