"""Small numerical helpers for feature matrices.

These functions do not interpret labels, choose statistical tests, or make claims about
causality, quality, or feature meaning.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from prefscope.core.features import FeatureMatrix


def _matrix(value, *, boolean: bool = False) -> FeatureMatrix:
    if not isinstance(value, FeatureMatrix):
        raise TypeError("analysis functions require a FeatureMatrix")
    if boolean and value.values.dtype != bool:
        raise ValueError("coactivation requires a boolean FeatureMatrix")
    return value


def activation_summary(matrix: FeatureMatrix) -> pd.DataFrame:
    """Return basic numerical activity summaries for each feature column."""
    matrix = _matrix(matrix)
    values = matrix.values
    numerical = values.astype(np.float64, copy=False)
    active = values != 0
    counts = active.sum(axis=0)
    sums = numerical.sum(axis=0)
    means_active = np.divide(
        sums,
        counts,
        out=np.full(values.shape[1], np.nan, dtype=float),
        where=counts > 0,
    )
    n_rows = values.shape[0]
    return pd.DataFrame(
        {
            "feature_id": matrix.feature_ids,
            "n_active": counts.astype(int),
            "activation_rate": counts / n_rows,
            "mean": numerical.mean(axis=0),
            "mean_when_active": means_active,
            "maximum": numerical.max(axis=0),
        }
    )


def coactivation_counts(presence: FeatureMatrix) -> pd.DataFrame:
    """Return the joint-presence count matrix for explicit boolean features."""
    presence = _matrix(presence, boolean=True)
    values = csr_matrix(presence.values, dtype=np.int64)
    result = pd.DataFrame(
        (values.T @ values).toarray(),
        index=presence.feature_ids,
        columns=presence.feature_ids,
    )
    result.index.name = result.columns.name = "feature_id"
    return result


def coactivation_pairs(
    presence: FeatureMatrix,
    *,
    min_count: int = 1,
) -> pd.DataFrame:
    """Return descriptive overlap measures for unordered feature pairs."""
    presence = _matrix(presence, boolean=True)
    if not isinstance(min_count, int) or isinstance(min_count, bool) or min_count < 1:
        raise ValueError("min_count must be a positive integer")
    counts = coactivation_counts(presence).to_numpy()
    first, second = np.triu_indices(len(presence.feature_ids), k=1)
    selected = counts[first, second] >= min_count
    first, second = first[selected], second[selected]
    order = sorted(
        range(len(first)),
        key=lambda index: (-counts[first[index], second[index]], first[index], second[index]),
    )
    rows = []
    for index in order:
        a, b = int(first[index]), int(second[index])
        count_a = int(counts[a, a])
        count_b = int(counts[b, b])
        count_both = int(counts[a, b])
        rows.append(
            {
                "feature_a": presence.feature_ids[a],
                "feature_b": presence.feature_ids[b],
                "count_a": count_a,
                "count_b": count_b,
                "count_both": count_both,
                "jaccard": count_both / (count_a + count_b - count_both),
            }
        )
    return pd.DataFrame(
        rows,
        columns=(
            "feature_a",
            "feature_b",
            "count_a",
            "count_b",
            "count_both",
            "jaccard",
        ),
    )


def cross_coactivation_counts(
    left: FeatureMatrix,
    right: FeatureMatrix,
) -> pd.DataFrame:
    """Return cross-counts for two row-aligned boolean feature matrices."""
    left = _matrix(left, boolean=True)
    right = _matrix(right, boolean=True)
    if left.row_ids != right.row_ids:
        raise ValueError("left and right must have identical ordered row_ids")
    left_values = csr_matrix(left.values, dtype=np.int64)
    right_values = csr_matrix(right.values, dtype=np.int64)
    result = pd.DataFrame(
        (left_values.T @ right_values).toarray(),
        index=left.feature_ids,
        columns=right.feature_ids,
    )
    result.index.name = "feature_left"
    result.columns.name = "feature_right"
    return result


def top_activating_rows(
    matrix: FeatureMatrix,
    *,
    k: int = 10,
) -> pd.DataFrame:
    """Return each feature's largest signed activations with stable row ties."""
    matrix = _matrix(matrix)
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer")
    take = min(k, len(matrix.row_ids))
    rows = []
    for column, feature_id in enumerate(matrix.feature_ids):
        values = matrix.values[:, column].astype(np.float32, copy=False)
        order = np.argsort(-values, kind="stable")[:take]
        rows.extend(
            {
                "feature_id": feature_id,
                "rank": rank,
                "row_id": matrix.row_ids[row],
                "activation": float(matrix.values[row, column]),
            }
            for rank, row in enumerate(order, start=1)
        )
    return pd.DataFrame(
        rows,
        columns=("feature_id", "rank", "row_id", "activation"),
    )


__all__ = [
    "activation_summary",
    "coactivation_counts",
    "coactivation_pairs",
    "cross_coactivation_counts",
    "top_activating_rows",
]
