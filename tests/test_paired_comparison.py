import numpy as np
import pandas as pd
import pytest

from prefscope.analysis.paired import (
    paired_concept_shift,
    paired_concept_shift_by_region,
    summarize_response_scope,
)


def test_identical_pairs_have_zero_shift_and_no_evidence():
    a = np.array([[0, 1], [1, 0], [1, 1]], dtype=bool)
    result = paired_concept_shift(a, a, feature_ids=[10, 11])

    assert result["delta_b_minus_a"].tolist() == [0.0, 0.0]
    assert set(result["orientation"]) == {"delta_b_minus_a"}
    assert set(result["inference_test"]) == {"exact_mcnemar"}
    assert result["n_discordant"].tolist() == [0, 0]
    assert result["n_nonzero_groups"].tolist() == [0, 0]
    assert result["p_value"].tolist() == [1.0, 1.0]
    assert set(result["test"]) == {"exact_mcnemar"}


def test_swapping_sides_negates_effect_but_preserves_p_value():
    a = np.zeros((12, 1), dtype=bool)
    b = np.ones((12, 1), dtype=bool)
    ab = paired_concept_shift(a, b, feature_ids=[7])
    ba = paired_concept_shift(b, a, feature_ids=[7])

    assert ab.loc[0, "delta_b_minus_a"] == 1.0
    assert ba.loc[0, "delta_b_minus_a"] == -1.0
    assert ab.loc[0, "p_value"] == ba.loc[0, "p_value"]
    assert ab.loc[0, "b_only"] == ba.loc[0, "a_only"]
    assert ab.loc[0, "ci_low"] < 1.0  # finite-sample interval must not collapse at boundary
    assert ab.loc[0, "ci_method"] == "hoeffding"


def test_repeated_generations_use_group_level_hoeffding_mean_test():
    a = np.zeros((6, 1), dtype=bool)
    b = np.array([[1], [1], [1], [0], [1], [0]], dtype=bool)
    groups = ["p1", "p1", "p2", "p2", "p3", "p3"]
    result = paired_concept_shift(a, b, group_ids=groups)

    assert result.loc[0, "n_pairs"] == 6
    assert result.loc[0, "n_groups"] == 3
    assert result.loc[0, "n_nonzero_groups"] == 3
    assert result.loc[0, "test"] == "cluster_hoeffding"
    assert result.loc[0, "delta_b_minus_a"] == 2 / 3
    expected_p = min(1.0, 2.0 * np.exp(-3 * (2 / 3) ** 2 / 2))
    assert result.loc[0, "p_value"] == expected_p


def test_cluster_test_targets_mean_magnitude_not_majority_group_sign():
    # One small group moves fully toward B. Nine 100-row groups each move only 1%
    # toward A. The equal-group-weight mean remains positive even though most group
    # signs are negative, so a sign test would test a different estimand.
    group_sizes = [1] + [100] * 9
    groups = np.concatenate([
        np.full(size, group_id) for group_id, size in enumerate(group_sizes)
    ])
    a = np.zeros((len(groups), 1), dtype=bool)
    b = np.zeros_like(a)
    b[0, 0] = True
    offset = 1
    for size in group_sizes[1:]:
        a[offset, 0] = True
        offset += size

    result = paired_concept_shift(a, b, group_ids=groups)

    expected_delta = (1.0 - 9 * 0.01) / 10
    assert np.isclose(result.loc[0, "delta_b_minus_a"], expected_delta)
    assert result.loc[0, "n_nonzero_groups"] == 10
    assert result.loc[0, "p_value"] == 1.0


def test_clustered_scope_support_counts_nonzero_groups_not_rows():
    groups = np.repeat(np.arange(30), 10)
    contexts = np.repeat(np.arange(3), 100)
    a = np.zeros((len(groups), 1), dtype=bool)
    b = np.ones_like(a)
    membership = np.column_stack([contexts == k for k in range(3)])

    overall = paired_concept_shift(a, b, feature_ids=[0], group_ids=groups)
    conditional = paired_concept_shift_by_region(
        a, b, membership, feature_ids=[0], region_ids=[100, 101, 102],
        group_ids=groups, min_pairs=10,
    )
    annotations = pd.DataFrame({
        "feature_id": [0],
        "semantic_role": ["presentation"],
        "requested_share": [0.0],
    })
    scope = summarize_response_scope(
        overall, conditional, feature_annotations=annotations,
        min_discordant=20, min_contexts=3,
    )

    assert overall.loc[0, "n_discordant"] == 300
    assert overall.loc[0, "n_nonzero_groups"] == 30
    assert set(conditional["n_discordant"]) == {100}
    assert set(conditional["n_nonzero_groups"]) == {10}
    assert set(conditional["region_group_support"]) == {10}
    assert scope.loc[0, "n_supported_contexts"] == 0
    assert scope.loc[0, "response_scope"] == "context_specific_tendency"


def test_overlapping_regions_and_scope_separate_general_content_and_context():
    n = 120
    contexts = np.repeat(np.arange(3), 40)
    a = np.zeros((n, 3), dtype=bool)
    b = np.zeros_like(a)
    # f0: B expresses it on every prompt -> cross-context general tendency.
    b[:, 0] = True
    # f1: both sides express requested content in context 0 -> not a model shift.
    a[contexts == 0, 1] = True
    b[contexts == 0, 1] = True
    # f2: B differs only in context 0 -> context-specific tendency.
    b[contexts == 0, 2] = True
    membership = np.column_stack([contexts == k for k in range(3)])

    overall = paired_concept_shift(a, b, feature_ids=[0, 1, 2])
    conditional = paired_concept_shift_by_region(
        a, b, membership, feature_ids=[0, 1, 2], region_ids=[100, 101, 102],
        min_pairs=20)
    annotations = pd.DataFrame({
        "feature_id": [0, 1, 2],
        "semantic_role": ["response_policy", "requested_task", "presentation"],
        "requested_share": [0.0, 1.0, 0.0],
    })
    scope = summarize_response_scope(
        overall, conditional, feature_annotations=annotations,
        min_discordant=20, min_contexts=3)
    by_id = scope.set_index("feature_id")

    assert by_id.loc[0, "response_scope"] == "general_tendency"
    assert by_id.loc[1, "response_scope"] == "prompt_content"
    assert by_id.loc[2, "response_scope"] == "context_specific_tendency"
    assert conditional["region_id"].nunique() == 3


def test_unsupported_context_signal_cannot_promote_scope():
    overall = pd.DataFrame({
        "feature_id": [0], "delta_b_minus_a": [0.0], "q_value": [1.0],
        "n_nonzero_groups": [8], "n_discordant": [8],
    })
    conditional = pd.DataFrame({
        "feature_id": [0], "delta_b_minus_a": [1.0], "q_value": [0.001],
        "n_nonzero_groups": [8],
    })
    annotations = pd.DataFrame({
        "feature_id": [0], "semantic_role": ["presentation"],
        "requested_share": [0.0],
    })
    scope = summarize_response_scope(
        overall, conditional, feature_annotations=annotations,
        min_discordant=20,
    )
    assert scope.loc[0, "n_supported_contexts"] == 0
    assert scope.loc[0, "response_scope"] == "unclassified"


def test_paired_regions_require_constant_membership_within_group():
    a = np.zeros((4, 1), dtype=bool)
    b = np.ones((4, 1), dtype=bool)
    membership = np.array([[True], [False], [True], [False]])
    with np.testing.assert_raises_regex(ValueError, "constant within"):
        paired_concept_shift_by_region(
            a, b, membership, group_ids=["same", "same", "a", "b"],
            min_pairs=1,
        )


def test_paired_group_ids_preserve_type_identity_and_reject_missing():
    a = np.zeros((2, 1), dtype=bool)
    b = np.ones((2, 1), dtype=bool)
    result = paired_concept_shift(a, b, group_ids=[1, "1"])
    assert result.loc[0, "n_groups"] == 2
    with np.testing.assert_raises_regex(ValueError, "missing"):
        paired_concept_shift(a, b, group_ids=[1, None])


def test_paired_presence_rejects_nan_and_nonbinary_values():
    with pytest.raises(ValueError, match="finite boolean or numeric 0/1"):
        paired_concept_shift(
            np.array([[np.nan], [0.0]]), np.array([[1.0], [0.0]]))
    with pytest.raises(ValueError, match="finite boolean or numeric 0/1"):
        paired_concept_shift(
            np.array([[2.0], [0.0]]), np.array([[1.0], [0.0]]))
