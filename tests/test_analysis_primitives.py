import numpy as np
import pandas as pd
import pytest

from prefscope import FeatureMatrix
from prefscope.analysis import (
    activation_summary,
    coactivation_counts,
    coactivation_pairs,
    cross_coactivation_counts,
    top_activating_rows,
)


def _matrix(values, *, row_ids=("a", "b", "c"), feature_ids=(4, 9)):
    return FeatureMatrix(values, row_ids=row_ids, feature_ids=feature_ids)


def test_activation_summary_is_only_numerical_activity():
    matrix = _matrix([[0.0, 2.0], [1.0, -2.0], [0.0, 0.0]])
    result = activation_summary(matrix)
    assert list(result.columns) == [
        "feature_id", "n_active", "activation_rate", "mean",
        "mean_when_active", "maximum",
    ]
    assert result["feature_id"].tolist() == [4, 9]
    assert result["n_active"].tolist() == [1, 2]
    assert result["mean"].tolist() == pytest.approx([1 / 3, 0.0])


def test_activation_summary_accumulates_canonical_values_in_float64():
    matrix = FeatureMatrix(
        np.array([[1e8], [1.0], [-1e8]], dtype=np.float32),
        row_ids=("a", "b", "c"),
        feature_ids=(5,),
    )
    result = activation_summary(matrix)
    assert result.loc[0, "mean"] == pytest.approx(1 / 3)

    large = FeatureMatrix(
        np.full((4, 1), 2e38, dtype=np.float32),
        row_ids=("a", "b", "c", "d"),
        feature_ids=(5,),
    )
    large_result = activation_summary(large)
    assert np.isfinite(large_result.loc[0, "mean"])
    assert np.isfinite(large_result.loc[0, "mean_when_active"])


def test_coactivation_requires_explicit_boolean_feature_matrix():
    presence = _matrix(
        [[True, True], [True, False], [False, True]], feature_ids=(2, 8)
    )
    counts = coactivation_counts(presence)
    expected = pd.DataFrame([[2, 1], [1, 2]], index=[2, 8], columns=[2, 8])
    expected.index.name = expected.columns.name = "feature_id"
    pd.testing.assert_frame_equal(counts, expected)

    pairs = coactivation_pairs(presence)
    assert list(pairs.columns) == [
        "feature_a", "feature_b", "count_a", "count_b", "count_both", "jaccard",
    ]
    assert pairs.to_dict(orient="records") == [{
        "feature_a": 2, "feature_b": 8, "count_a": 2, "count_b": 2,
        "count_both": 1, "jaccard": pytest.approx(1 / 3),
    }]

    with pytest.raises(ValueError, match="boolean FeatureMatrix"):
        coactivation_counts(_matrix([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))


def test_cross_counts_validate_row_identity():
    left = _matrix([[True, False], [True, True], [False, True]], feature_ids=(1, 2))
    right = _matrix([[False, True], [True, True], [True, False]], feature_ids=(7, 8))
    result = cross_coactivation_counts(left, right)
    expected = pd.DataFrame([[1, 2], [2, 1]], index=[1, 2], columns=[7, 8])
    expected.index.name = "feature_left"
    expected.columns.name = "feature_right"
    pd.testing.assert_frame_equal(result, expected)

    reordered = _matrix(
        [[False, True], [True, False], [True, True]],
        row_ids=("b", "a", "c"),
        feature_ids=(7, 8),
    )
    with pytest.raises(ValueError, match="ordered row_ids"):
        cross_coactivation_counts(left, reordered)


def test_top_activating_rows_accepts_boolean_values_with_stable_ties():
    matrix = FeatureMatrix(
        [[False], [True], [True]],
        row_ids=("a", "b", "c"),
        feature_ids=(7,),
    )
    result = top_activating_rows(matrix, k=3)
    assert result["row_id"].tolist() == ["b", "c", "a"]
    assert result["activation"].tolist() == [1.0, 1.0, 0.0]


def test_top_activating_rows_preserves_ids_and_stable_ties():
    matrix = _matrix([[1.0, 4.0], [3.0, 2.0], [3.0, 1.0]])
    result = top_activating_rows(matrix, k=2)
    assert result.to_dict(orient="records") == [
        {"feature_id": 4, "rank": 1, "row_id": "b", "activation": 3.0},
        {"feature_id": 4, "rank": 2, "row_id": "c", "activation": 3.0},
        {"feature_id": 9, "rank": 1, "row_id": "a", "activation": 4.0},
        {"feature_id": 9, "rank": 2, "row_id": "b", "activation": 2.0},
    ]
