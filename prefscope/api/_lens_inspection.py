"""Internal accessors for lens annotations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def concept_names(lens):
    """Series mapping feature_id -> concept name, or None if unnamed.

    De-dups on ``feature_id`` so the index is unique (a duplicated id would
    make an ID lookup ambiguous.
    """
    if lens.names is not None and "concept" in lens.names.columns:
        names = lens.names.drop_duplicates("feature_id").set_index("feature_id")[
            "concept"
        ]
        if any(
            isinstance(value, str) and bool(value.strip())
            for value in names.tolist()
        ):
            return names
    return None


def feature_table(lens) -> pd.DataFrame:
    """One row per feature with every bundled annotation column available."""
    if lens.names is not None:
        return lens.names.copy()
    return pd.DataFrame(
        {"feature_id": np.arange(int(lens.projector.m_total), dtype=int)}
    )
