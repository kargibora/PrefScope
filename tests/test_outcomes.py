import numpy as np
import pandas as pd
import pytest

from prefscope.analysis.outcomes import (
    NormalizedOutcomes,
    OutcomeAssociationResult,
    associate_outcomes,
    associate_outcomes_by_group,
    normalize_outcomes,
)


def test_binary_outcome_validates_values_and_preserves_missingness():
    outcome = normalize_outcomes(
        [1, 0, None, np.nan], kind="binary", names=["accepted"])

    assert outcome.kind == "binary"
    assert outcome.names == ("accepted",)
    assert outcome.normalization == "none"
    assert outcome.observed[:, 0].tolist() == [True, True, False, False]
    assert outcome.values[:2, 0].tolist() == [1.0, 0.0]
    assert np.isnan(outcome.values[2:, 0]).all()

    with pytest.raises(ValueError, match="only 0, 1"):
        normalize_outcomes([0, 2, 1], kind="binary")


def test_probability_and_preference_require_closed_unit_interval():
    probability = normalize_outcomes(
        [0.0, 0.5, 1.0, None], kind="probability")
    preference = normalize_outcomes(
        [0.0, 0.5, 1.0, None], kind="preference")

    assert probability.values[:3, 0].tolist() == [0.0, 0.5, 1.0]
    assert preference.kind == "preference"
    with pytest.raises(ValueError, match=r"inside \[0, 1\]"):
        normalize_outcomes([0.2, 1.01], kind="probability")
    with pytest.raises(ValueError, match="numeric or missing"):
        normalize_outcomes([0.2, "winner"], kind="preference")


def test_continuous_outcome_zscores_observed_values_only():
    outcome = normalize_outcomes(
        [1.0, 2.0, np.nan, 5.0], kind="continuous", names=["rating"])
    observed = outcome.values[outcome.observed]

    assert outcome.normalization == "zscore"
    assert outcome.center.tolist() == [8 / 3]
    assert np.isclose(observed.mean(), 0.0)
    assert np.isclose(observed.std(ddof=0), 1.0)
    assert np.isnan(outcome.values[2, 0])


def test_multi_attribute_continuous_uses_dataframe_names_and_column_masks():
    raw = pd.DataFrame({
        "helpfulness": [1.0, 2.0, np.nan],
        "verbosity": [np.nan, 10.0, 14.0],
    })
    outcome = normalize_outcomes(raw, kind="multi_continuous")

    assert outcome.names == ("helpfulness", "verbosity")
    assert outcome.values.shape == (3, 2)
    assert outcome.observed.tolist() == [
        [True, False], [True, True], [False, True],
    ]
    assert np.allclose(outcome.center, [1.5, 12.0])
    assert np.allclose(outcome.scale, [0.5, 2.0])
    with pytest.raises(ValueError, match="2-D matrix"):
        normalize_outcomes([1.0, 2.0], kind="multi_continuous")


def test_normalization_validates_shape_names_and_infinities():
    with pytest.raises(ValueError, match="exactly one column"):
        normalize_outcomes(np.ones((3, 2)), kind="continuous")
    with pytest.raises(ValueError, match="unique non-empty"):
        normalize_outcomes(np.ones((3, 2)), kind="multi_continuous", names=["x", "x"])
    with pytest.raises(ValueError, match="not infinite"):
        normalize_outcomes([1.0, np.inf], kind="continuous")
    with pytest.raises(ValueError, match="normalization must"):
        normalize_outcomes([1.0], kind="continuous", normalization="rank")


def test_row_association_is_descriptive_and_uses_outcome_specific_missingness():
    outcome = normalize_outcomes(
        [0, 1, 1, np.nan, 0], kind="binary", names=["success"])
    features = np.column_stack([
        [0, 1, 1, 0, 0],
        np.ones(5),
    ])
    result = associate_outcomes(features, outcome, feature_ids=[10, 11])
    table = result.table.set_index("feature_id")

    assert result.grouped is False
    assert "not a causal effect" in result.estimand
    assert table.loc[10, "n_rows"] == 4
    assert table.loc[10, "n_units"] == 4
    assert table.loc[10, "analysis_unit"] == "row"
    assert table.loc[10, "correlation"] == 1.0
    assert table.loc[10, "slope"] == 1.0
    assert np.isnan(table.loc[11, "correlation"])
    assert np.isnan(table.loc[11, "q_value"])


def test_group_association_gives_each_observed_group_equal_weight():
    outcome = normalize_outcomes(
        [0, 0, 0, 0, 0, 0, 1, 0], kind="continuous", normalization="none")
    features = np.asarray([[0], [0], [0], [0], [0], [0], [1], [2]], dtype=float)
    groups = ["large"] * 6 + ["small-a", "small-b"]

    result = associate_outcomes_by_group(
        features, outcome, groups, feature_ids=[7], min_groups=3)
    row = result.table.iloc[0]

    assert result.grouped is True
    assert row["analysis_unit"] == "group"
    assert row["n_rows"] == 8
    assert row["n_units"] == 3
    assert row["feature_mean"] == 1.0
    assert row["outcome_mean"] == 1 / 3
    assert np.isclose(row["correlation"], 0.0)


def test_group_association_drops_missing_outcomes_before_aggregation():
    outcome = normalize_outcomes(
        [0.0, np.nan, 1.0, 0.0], kind="continuous", normalization="none")
    features = np.asarray([[0], [100], [1], [2]], dtype=float)
    groups = ["a", "a", "b", "c"]

    result = associate_outcomes_by_group(features, outcome, groups, min_groups=3)
    row = result.table.iloc[0]

    assert row["n_rows"] == 3
    assert row["n_units"] == 3
    assert row["feature_mean"] == 1.0


def test_multi_outcome_association_returns_one_row_per_feature_attribute():
    outcome = normalize_outcomes(
        pd.DataFrame({
            "quality": [1.0, 2.0, 3.0, 4.0],
            "style": [4.0, np.nan, 2.0, 1.0],
        }),
        kind="multi_continuous",
        normalization="none",
    )
    features = np.column_stack([np.arange(4), np.arange(4)[::-1]])

    result = associate_outcomes(features, outcome, feature_ids=[3, 8])

    assert len(result.table) == 4
    assert set(result.table["outcome"]) == {"quality", "style"}
    assert set(result.table["feature_id"]) == {3, 8}
    support = result.table.groupby("outcome")["n_rows"].first().to_dict()
    assert support == {"quality": 4, "style": 3}


def test_association_validates_alignment_groups_and_typed_result():
    outcome = normalize_outcomes([0.0, 1.0, 2.0], kind="continuous")
    with pytest.raises(ValueError, match="2-D matrix with 3 rows"):
        associate_outcomes(np.ones((2, 1)), outcome)
    with pytest.raises(ValueError, match="one entry per outcome row"):
        associate_outcomes(np.ones((3, 1)), outcome, group_ids=["a", "b"])
    with pytest.raises(ValueError, match="must not contain missing"):
        associate_outcomes(np.ones((3, 1)), outcome, group_ids=["a", None, "b"])
    with pytest.raises(ValueError, match="at least 3"):
        associate_outcomes(np.ones((3, 1)), outcome, min_units=2)
    with pytest.raises(ValueError, match="association table is missing"):
        OutcomeAssociationResult(pd.DataFrame(), outcome, grouped=False)


def test_normalized_outcome_contract_rejects_mismatched_missing_mask():
    with pytest.raises(ValueError, match="observed must exactly mark"):
        NormalizedOutcomes(
            kind="continuous",
            names=("rating",),
            raw_values=np.asarray([[1.0], [np.nan]]),
            values=np.asarray([[1.0], [np.nan]]),
            observed=np.asarray([[True], [True]]),
            normalization="none",
            center=np.asarray([0.0]),
            scale=np.asarray([1.0]),
        )


def test_outcome_contract_is_exported_from_analysis_namespace():
    from prefscope.analysis import (
        NormalizedOutcomes as ExportedOutcomes,
        associate_outcomes as exported_associate,
        normalize_outcomes as exported_normalize,
    )

    assert ExportedOutcomes is NormalizedOutcomes
    assert exported_associate is associate_outcomes
    assert exported_normalize is normalize_outcomes


def test_outcome_inference_is_withheld_for_thin_perfect_binary_cell():
    outcome = normalize_outcomes([1, 0, 0, 0, 0, 0], kind="binary")
    result = associate_outcomes(
        np.array([[1], [0], [0], [0], [0], [0]], dtype=float), outcome)
    row = result.table.iloc[0]
    assert np.isclose(row["correlation"], 1.0)
    assert not bool(row["inference_supported"])
    assert np.isnan(row["p_value"])
    assert np.isnan(row["q_value"])


def test_group_normalization_is_invariant_to_row_duplication():
    base_outcome = normalize_outcomes(
        [0.0, 1.0, 2.0], kind="continuous")
    base = associate_outcomes(
        np.array([[0.0], [1.0], [3.0]]), base_outcome,
        group_ids=["a", "b", "c"])
    repeated_outcome = normalize_outcomes(
        [0.0] * 20 + [1.0, 2.0], kind="continuous")
    repeated = associate_outcomes(
        np.array([[0.0]] * 20 + [[1.0], [3.0]]), repeated_outcome,
        group_ids=["a"] * 20 + ["b", "c"])

    left, right = base.table.iloc[0], repeated.table.iloc[0]
    assert np.isclose(left["slope"], right["slope"])
    assert np.isclose(
        left["association_outcome_scale"], right["association_outcome_scale"])


def test_binary_outcome_uses_exact_nonzero_inference():
    values = np.array([0.0] * 5 + [1.0] * 5)
    result = associate_outcomes(
        values[:, None], normalize_outcomes(values, kind="binary"))
    row = result.table.iloc[0]
    assert row["inference_test"] == "fisher_exact_range_midpoint_split"
    assert np.isclose(row["p_value"], 2.0 / 252.0)
    assert row["p_value"] > 0


def test_preference_outcomes_declare_that_ties_are_retained():
    features = np.arange(10, dtype=float)[:, None]
    outcomes = normalize_outcomes(
        [0.0, 0.5, 1.0, 0.5, 0.0, 1.0, 0.5, 1.0, 0.0, 0.5],
        kind="preference",
    )
    row = associate_outcomes(features, outcomes, min_units=3).table.iloc[0]
    assert row["tie_policy"] == "retained_as_0.5_neutral"
    assert row["missingness_policy"] == "per_outcome_attribute_complete_cases"


def test_associate_outcomes_rejects_lossy_feature_ids():
    outcomes = normalize_outcomes(
        [0.0, 1.0, 0.0], kind="binary", normalization="none")
    with pytest.raises(ValueError, match="non-boolean integers"):
        associate_outcomes(
            np.ones((3, 1)), outcomes, feature_ids=[1.9])
    with pytest.raises(ValueError, match="non-boolean integers"):
        associate_outcomes(
            np.ones((3, 1)), outcomes, feature_ids=[True])


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("raw_values", np.asarray([[1.0 + 1.0j]]), "real numeric"),
        ("values", np.asarray([[1.0 + 1.0j]]), "real numeric"),
        ("observed", np.asarray([[1]]), "boolean"),
    ],
)
def test_normalized_outcomes_rejects_unsafe_direct_array_types(
    field, replacement, message,
):
    kwargs = {
        "kind": "continuous",
        "names": ("rating",),
        "raw_values": np.asarray([[1.0]]),
        "values": np.asarray([[1.0]]),
        "observed": np.asarray([[True]]),
        "normalization": "none",
        "center": np.asarray([0.0]),
        "scale": np.asarray([1.0]),
    }
    kwargs[field] = replacement
    with pytest.raises(ValueError, match=message):
        NormalizedOutcomes(**kwargs)


def test_normalized_outcomes_rejects_inconsistent_direct_values():
    with pytest.raises(ValueError, match="must equal"):
        NormalizedOutcomes(
            kind="continuous", names=("rating",),
            raw_values=np.asarray([[1.0]]), values=np.asarray([[7.0]]),
            observed=np.asarray([[True]]), normalization="none",
            center=np.asarray([0.0]), scale=np.asarray([1.0]),
        )
