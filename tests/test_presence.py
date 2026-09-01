import numpy as np
import pandas as pd
import pytest

from prefscope.analysis.presence import annotation_flag, concept_presence, feature_thresholds


def _annotations():
    return pd.DataFrame({
        "feature_id": [0, 1, 2],
        "semantic_threshold": [1.0, 2.0, np.nan],
        "presence_pass": [True, False, False],
    })


def test_calibrated_presence_omits_uncalibrated_features():
    codes = np.array([[0.5, 3.0, 4.0], [1.5, 0.0, 2.0]], dtype=np.float32)
    result = concept_presence(codes, _annotations(), policy="calibrated")

    assert result.feature_ids.tolist() == [0]
    assert result.values.tolist() == [[False], [True]]
    assert result.basis.tolist() == ["semantic_threshold"]


def test_mixed_presence_records_fallback_basis():
    codes = np.array([[0.5, 3.0, 0.0], [1.5, 0.0, 2.0]], dtype=np.float32)
    result = concept_presence(codes, _annotations(), policy="mixed")

    assert result.feature_ids.tolist() == [0, 1, 2]
    assert result.values.tolist() == [[False, True, False], [True, False, True]]
    assert result.basis.tolist() == [
        "semantic_threshold", "positive_nonzero", "positive_nonzero"]


def test_presence_rejects_out_of_range_features_and_unknown_policy():
    with pytest.raises(ValueError, match="inside"):
        concept_presence(np.zeros((2, 2)), _annotations(), feature_ids=[2])
    with pytest.raises(ValueError, match="policy"):
        concept_presence(np.zeros((2, 2)), policy="guess")


def test_feature_thresholds_requires_passing_calibration():
    threshold, calibrated = feature_thresholds(_annotations(), [0, 1, 2])
    assert threshold.tolist() == [1.0, 0.0, 0.0]
    assert calibrated.tolist() == [True, False, False]


def test_feature_thresholds_parses_persisted_booleans_and_partial_rows():
    annotations = pd.DataFrame({
        "feature_id": [0, 0, 1, 2],
        "concept": ["kept from names", np.nan, "x", "y"],
        "semantic_threshold": [np.nan, 1.25, 2.0, 3.0],
        "presence_pass": [np.nan, "True", "False", np.nan],
    })

    threshold, calibrated = feature_thresholds(annotations, [0, 1, 2])

    assert threshold.tolist() == [1.25, 0.0, 0.0]
    assert calibrated.tolist() == [True, False, False]


@pytest.mark.parametrize("value, expected", [
    (True, True), (False, False), ("True", True), ("False", False),
    ("yes", True), ("0", False), (np.nan, False), (None, False),
])
def test_annotation_flag_is_strict_for_persisted_values(value, expected):
    assert annotation_flag(value) is expected
