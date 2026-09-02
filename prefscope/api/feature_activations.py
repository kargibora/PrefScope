"""Long-form feature-activation table construction."""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np
import pandas as pd

from prefscope.api.feature_catalog import FeatureCatalog
from prefscope.core.features import FeatureMatrix


def feature_activation_table(
    matrix: FeatureMatrix,
    *,
    catalog: FeatureCatalog | None = None,
    active_only: bool = True,
    min_abs_activation: float = 0.0,
    top_k: int | None = None,
) -> pd.DataFrame:
    """Return one ranked long-form table, joining annotations by feature ID."""
    if not isinstance(matrix, FeatureMatrix):
        raise ValueError("matrix must be a FeatureMatrix")
    if not isinstance(active_only, bool):
        raise ValueError("active_only must be boolean")
    if (
        isinstance(min_abs_activation, bool)
        or not isinstance(min_abs_activation, Real)
        or not np.isfinite(float(min_abs_activation))
        or float(min_abs_activation) < 0
    ):
        raise ValueError("min_abs_activation must be a finite non-negative number")
    if top_k is not None and (
        isinstance(top_k, bool) or not isinstance(top_k, Integral) or top_k < 1
    ):
        raise ValueError("top_k must be a positive integer or None")

    records = []
    positions = np.arange(matrix.n_features)
    for row_id, values in zip(matrix.row_ids, matrix.values, strict=True):
        selected = positions[np.abs(values) >= float(min_abs_activation)]
        if active_only:
            selected = selected[values[selected] != 0]
        if len(selected):
            order = np.argsort(-np.abs(values[selected]), kind="stable")
            selected = selected[order]
        if top_k is not None:
            selected = selected[:top_k]
        for rank, position in enumerate(selected, start=1):
            activation = float(values[position])
            records.append(
                {
                    "row_id": row_id,
                    "rank": rank,
                    "feature_id": int(matrix.feature_ids[position]),
                    "activation": activation,
                    "abs_activation": abs(activation),
                    "feature_role": matrix.role,
                    "feature_orientation": matrix.orientation,
                    "activation_polarity": matrix.activation_polarity,
                    "code_semantics": matrix.code_semantics,
                }
            )
    table = pd.DataFrame.from_records(records)
    if table.empty:
        table = pd.DataFrame(
            {
                "row_id": pd.Series(dtype=object),
                "rank": pd.Series(dtype="int64"),
                "feature_id": pd.Series(dtype="int64"),
                "activation": pd.Series(dtype="float64"),
                "abs_activation": pd.Series(dtype="float64"),
                "feature_role": pd.Series(dtype=object),
                "feature_orientation": pd.Series(dtype=object),
                "activation_polarity": pd.Series(dtype=object),
                "code_semantics": pd.Series(dtype=object),
            }
        )
    if catalog is not None:
        if not isinstance(catalog, FeatureCatalog):
            raise ValueError("catalog must be a FeatureCatalog or None")
        catalog.validate_for(matrix)
        active_ids = tuple(dict.fromkeys(int(value) for value in table["feature_id"]))
        annotations = catalog.select(active_ids, strict=False).to_frame()
        table = table.merge(
            annotations,
            on="feature_id",
            how="left",
            sort=False,
            validate="many_to_one",
        )
    leading = [
        "row_id",
        "rank",
        "feature_id",
        "activation",
        "abs_activation",
        "feature_role",
        "feature_orientation",
        "activation_polarity",
        "code_semantics",
    ]
    return table[[*leading, *(column for column in table if column not in leading)]]


__all__ = ["feature_activation_table"]
