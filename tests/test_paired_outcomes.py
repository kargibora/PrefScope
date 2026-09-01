from __future__ import annotations

import numpy as np

from prefscope.analysis.outcomes import normalize_outcomes
from prefscope.analysis.paired_outcomes import (
    paired_outcome_shift,
    paired_outcome_shift_by_concept,
)
from prefscope.analysis.stats import benjamini_hochberg


def test_binary_paired_outcome_uses_exact_mcnemar_and_b_minus_a():
    a = normalize_outcomes(np.zeros(10), kind="binary", normalization="none")
    b = normalize_outcomes(
        np.array([1.0] * 5 + [0.0] * 5), kind="binary", normalization="none")
    row = paired_outcome_shift(a, b, min_units=5).iloc[0]
    assert row["delta_b_minus_a"] == 0.5
    assert row["mean_a"] == 0.0 and row["mean_b"] == 0.5
    assert row["inference_test"] == "exact_mcnemar_binomial"
    assert row["p_value"] == 0.0625
    assert row["discordant_pairs"] == 5


def test_grouped_probability_shift_is_equal_group_weighted_under_duplication():
    base_a = normalize_outcomes([0.0, 0.2, 0.8], kind="probability", normalization="none")
    base_b = normalize_outcomes([1.0, 0.4, 0.6], kind="probability", normalization="none")
    base = paired_outcome_shift(
        base_a, base_b, group_ids=["a", "b", "c"], min_units=2).iloc[0]

    repeated_a = normalize_outcomes(
        [0.0] * 20 + [0.2, 0.8], kind="probability", normalization="none")
    repeated_b = normalize_outcomes(
        [1.0] * 20 + [0.4, 0.6], kind="probability", normalization="none")
    repeated = paired_outcome_shift(
        repeated_a, repeated_b,
        group_ids=["a"] * 20 + ["b", "c"], min_units=2,
    ).iloc[0]
    assert base["n_units"] == repeated["n_units"] == 3
    assert np.isclose(base["delta_b_minus_a"], repeated["delta_b_minus_a"])
    assert base["inference_test"] == "hoeffding_bounded_paired_mean"


def test_continuous_paired_outcome_stays_descriptive_and_pairwise_complete():
    a = normalize_outcomes(
        np.array([[1.0, np.nan], [2.0, 2.0], [np.nan, 3.0]]),
        kind="multi_continuous", names=["x", "y"], normalization="none")
    b = normalize_outcomes(
        np.array([[2.0, 1.0], [4.0, 4.0], [5.0, 6.0]]),
        kind="multi_continuous", names=["x", "y"], normalization="none")
    table = paired_outcome_shift(a, b, min_units=2).set_index("outcome")
    assert table.loc["x", "n_rows"] == 2
    assert table.loc["y", "n_rows"] == 2
    assert table.loc["x", "delta_b_minus_a"] == 1.5
    assert table.loc["y", "delta_b_minus_a"] == 2.5
    assert not table["inference_supported"].any()
    assert table["p_value"].isna().all()


def test_prompt_concept_conditioned_shift_is_an_actual_heterogeneity_contrast():
    presence = np.array([[1]] * 10 + [[0]] * 10, dtype=bool)
    a = normalize_outcomes(np.zeros(20), kind="probability", normalization="none")
    b = normalize_outcomes(
        np.array([1.0] * 10 + [0.0] * 10),
        kind="probability", normalization="none")
    row = paired_outcome_shift_by_concept(
        presence, a, b, feature_ids=[7], basis=["semantic_threshold"]).iloc[0]
    assert row["shift_present_b_minus_a"] == 1.0
    assert row["shift_absent_b_minus_a"] == 0.0
    assert row["heterogeneity_present_minus_absent"] == 1.0
    assert row["n_present_units"] == row["n_absent_units"] == 10
    assert row["inference_test"] == (
        "hoeffding_bounded_difference_in_paired_shifts")


def test_prompt_conditioned_shift_rejects_within_group_presence_variation():
    presence = np.array([[1], [0], [1], [1]], dtype=bool)
    a = normalize_outcomes(np.zeros(4), kind="probability", normalization="none")
    b = normalize_outcomes(np.ones(4), kind="probability", normalization="none")
    with np.testing.assert_raises_regex(ValueError, "constant within"):
        paired_outcome_shift_by_concept(
            presence, a, b, group_ids=["a", "a", "b", "c"],
            min_units_per_arm=2)


def test_prompt_conditioned_group_estimand_is_duplication_invariant():
    base_presence = np.array([[1], [0], [0]], dtype=bool)
    base_a = normalize_outcomes([0.0, 0.0, 0.0], kind="probability", normalization="none")
    base_b = normalize_outcomes([1.0, 0.2, 0.4], kind="probability", normalization="none")
    base = paired_outcome_shift_by_concept(
        base_presence, base_a, base_b,
        group_ids=["a", "b", "c"], min_units_per_arm=2).iloc[0]

    repeated_presence = np.array([[1]] * 20 + [[0], [0]], dtype=bool)
    repeated_a = normalize_outcomes(
        np.zeros(22), kind="probability", normalization="none")
    repeated_b = normalize_outcomes(
        [1.0] * 20 + [0.2, 0.4], kind="probability", normalization="none")
    repeated = paired_outcome_shift_by_concept(
        repeated_presence, repeated_a, repeated_b,
        group_ids=["a"] * 20 + ["b", "c"], min_units_per_arm=2).iloc[0]
    assert np.isclose(
        base["heterogeneity_present_minus_absent"],
        repeated["heterogeneity_present_minus_absent"],
    )


def test_explicit_unique_groups_retain_exact_mcnemar():
    a = normalize_outcomes(np.zeros(10), kind="binary", normalization="none")
    b = normalize_outcomes(
        np.array([1.0] * 5 + [0.0] * 5), kind="binary", normalization="none")
    rows = paired_outcome_shift(
        a, b, group_ids=[f"g{i}" for i in range(10)], min_units=5)
    assert rows.loc[0, "inference_test"] == "exact_mcnemar_binomial"
    assert rows.loc[0, "p_value"] == 0.0625


def test_paired_outcome_reports_asymmetric_missingness_counts():
    a = normalize_outcomes(
        [1.0, np.nan, 0.0, np.nan], kind="probability", normalization="none")
    b = normalize_outcomes(
        [1.0, 0.0, np.nan, np.nan], kind="probability", normalization="none")
    row = paired_outcome_shift(a, b, min_units=2).iloc[0]
    assert row["n_rows_total"] == 4
    assert row["n_observed_a"] == row["n_observed_b"] == 2
    assert row["n_paired_rows"] == 1
    assert row["n_missing_a_only"] == 1
    assert row["n_missing_b_only"] == 1
    assert row["n_missing_both"] == 1
    assert not row["inference_supported"]
    assert row["support_reason"] == "fewer_than_min_independent_units"


def test_swapping_sides_negates_overall_and_heterogeneity_effects():
    presence = np.array([[1]] * 10 + [[0]] * 10, dtype=bool)
    a = normalize_outcomes(
        np.array([0.0] * 10 + [1.0] * 10),
        kind="probability", normalization="none")
    b = normalize_outcomes(
        np.array([1.0] * 10 + [0.0] * 10),
        kind="probability", normalization="none")
    forward = paired_outcome_shift(a, b).iloc[0]
    reverse = paired_outcome_shift(b, a).iloc[0]
    assert forward["delta_b_minus_a"] == 0.0
    assert reverse["delta_b_minus_a"] == -forward["delta_b_minus_a"]
    assert reverse["p_value"] == forward["p_value"]

    conditioned = paired_outcome_shift_by_concept(presence, a, b).iloc[0]
    conditioned_reverse = paired_outcome_shift_by_concept(presence, b, a).iloc[0]
    assert conditioned["shift_present_b_minus_a"] == 1.0
    assert conditioned["shift_absent_b_minus_a"] == -1.0
    assert conditioned["heterogeneity_present_minus_absent"] == 2.0
    assert conditioned_reverse["heterogeneity_present_minus_absent"] == -2.0
    assert conditioned_reverse["p_value"] == conditioned["p_value"]


def test_thin_prompt_stratum_withholds_heterogeneity_inference():
    presence = np.array([[1]] * 4 + [[0]] * 16, dtype=bool)
    a = normalize_outcomes(np.zeros(20), kind="probability", normalization="none")
    b = normalize_outcomes(np.ones(20), kind="probability", normalization="none")
    row = paired_outcome_shift_by_concept(
        presence, a, b, min_units_per_arm=5).iloc[0]
    assert not row["inference_supported"]
    assert np.isnan(row["p_value"]) and np.isnan(row["q_value"])
    assert row["support_reason"] == (
        "fewer_than_min_units_in_a_heterogeneity_stratum")


def test_conditioned_paired_outcome_rejects_untyped_outcomes_cleanly():
    with np.testing.assert_raises_regex(ValueError, "NormalizedOutcomes"):
        paired_outcome_shift_by_concept(
            np.ones((2, 1), dtype=bool), object(), object())


def test_conditioned_q_values_use_one_global_prompt_by_attribute_family():
    n = 40
    concepts = np.column_stack([
        np.arange(n) < 20,
        np.arange(n) % 2 == 0,
    ])
    a = normalize_outcomes(
        np.zeros(n), kind="binary", normalization="none", names=("quality",))
    b = normalize_outcomes(
        (np.arange(n) < 14).astype(float), kind="binary",
        normalization="none", names=("quality",),
    )
    table = paired_outcome_shift_by_concept(
        concepts, a, b, feature_ids=(10, 11), min_units_per_arm=8)
    expected = benjamini_hochberg(table["p_value"].to_numpy(dtype=float))
    np.testing.assert_allclose(table["q_value"], expected, equal_nan=True)
    assert table["multiplicity_family"].nunique() == 1
    assert "all prompt-feature × outcome-attribute" in table.loc[
        0, "multiplicity_family"]
