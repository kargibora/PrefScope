"""Small typed conveniences for preference-labelled feature batches."""
from __future__ import annotations

import pandas as pd

from prefscope.core.features import FeatureBatch, FeatureMatrix


def preference_relevance(
    features: FeatureBatch | FeatureMatrix,
    *,
    preference_column: str = "pref",
    group_column: str | None = "group_id",
    feature_array: str = "z_diff",
) -> pd.DataFrame:
    """Return grouped descriptive preference relevance for A-minus-B features.

    Preference values must mean ``P(A preferred)``. The result is descriptive and
    dataset/judge-specific, not a causal estimate or a judgment that a feature is good
    or bad. Noncontiguous feature IDs are preserved.
    """
    from prefscope.pipeline.winrelevance import win_relevance

    if isinstance(features, FeatureBatch):
        matrix = features.matrix(feature_array)
    elif isinstance(features, FeatureMatrix):
        matrix = features
    else:
        raise ValueError("features must be a FeatureBatch or FeatureMatrix")
    if matrix.orientation != "a_minus_b":
        raise ValueError(
            "preference relevance requires an A-minus-B feature matrix")
    if preference_column not in matrix.metadata:
        raise ValueError(
            f"feature metadata has no preference column {preference_column!r}")
    labels = pd.to_numeric(
        pd.Series(matrix.metadata[preference_column]), errors="raise")
    usable = labels.notna()
    if not usable.any():
        raise ValueError(
            "preference relevance needs at least one nonmissing P(A preferred) label")
    if not labels[usable].between(0.0, 1.0).all():
        raise ValueError("preference labels must be P(A preferred) values in [0, 1]")
    groups = None
    if group_column is not None:
        if group_column not in matrix.metadata:
            raise ValueError(
                f"feature metadata has no group column {group_column!r}; pass "
                "group_column=None for row-level analysis")
        groups = matrix.metadata[group_column]
    table = win_relevance(
        matrix.values,
        labels.to_numpy(dtype=float),
        group_ids=groups,
    )
    positions = dict(enumerate(matrix.feature_ids))
    table["feature_id"] = table["feature_id"].map(positions)
    table["feature_role"] = matrix.role
    table["feature_orientation"] = matrix.orientation
    table["outcome_orientation"] = "p_a_preferred"
    table["causal_claim"] = "none_descriptive_dataset_specific"
    return table


__all__ = ["preference_relevance"]
