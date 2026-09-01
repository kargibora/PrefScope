import numpy as np
import pandas as pd
import pytest

from prefscope.analysis.elicitation import (
    prompt_response_association,
    prompt_response_association_paired,
)


def test_paired_chunked_counts_equal_explicit_stacking():
    rng = np.random.default_rng(12)
    n = 37
    prompt = (rng.random((n, 5)) < 0.4).astype(np.float32)
    response_a = (rng.random((n, 7)) < 0.3).astype(np.float32)
    response_b = (rng.random((n, 7)) < 0.2).astype(np.float32)

    expected = prompt_response_association(
        np.vstack([prompt, prompt]),
        np.vstack([response_a, response_b]),
        prompt_features=[1, 3, 4], resp_features=[0, 2, 6],
        min_support=3, min_cooccur=2,
        group_ids=np.tile(np.arange(n), 2), chunk_size=9)
    actual = prompt_response_association_paired(
        prompt, response_a, response_b,
        prompt_features=[1, 3, 4], resp_features=[0, 2, 6],
        min_support=3, min_cooccur=2, chunk_size=8)

    pd.testing.assert_frame_equal(actual, expected)
    assert actual.attrs == expected.attrs


def test_repeated_groups_use_group_prevalence_without_changing_row_counts():
    # Ten response rows per independent prompt. Duplicating those rows must not create
    # extra inferential units or change the group-level p-value.
    group_ids = np.repeat(np.arange(12), 10)
    prompt_by_group = np.array([0] * 6 + [1] * 6, dtype=np.float32)
    response_by_group = np.array(
        [0.0, 0.1, 0.2, 0.1, 0.3, 0.2, 0.5, 0.6, 0.7, 0.8, 0.6, 0.7]
    )
    prompt = np.repeat(prompt_by_group, 10)[:, None]
    response = np.concatenate([
        np.r_[np.ones(round(rate * 10)), np.zeros(10 - round(rate * 10))]
        for rate in response_by_group
    ])[:, None]

    result = prompt_response_association(
        prompt, response, group_ids=group_ids, min_support=1, min_cooccur=1
    )
    duplicated = prompt_response_association(
        np.repeat(prompt, 2, axis=0),
        np.repeat(response, 2, axis=0),
        group_ids=np.repeat(group_ids, 2),
        min_support=1,
        min_cooccur=1,
    )

    row = result.iloc[0]
    duplicate_row = duplicated.iloc[0]
    assert row["n_x"] == 60
    assert duplicate_row["n_x"] == 120
    assert row["lift"] == duplicate_row["lift"]
    assert row["p_value"] == duplicate_row["p_value"]
    assert row["group_prevalence_difference"] == 0.5
    assert result.attrs["n_groups"] == 12
    assert result.attrs["inference_method"] == "two_sample_hoeffding_group_prevalence"
    assert "per-group response prevalence" in result.attrs["estimand"]


def test_repeated_groups_reject_varying_prompt_membership():
    prompt = np.array([[0], [1], [0], [0]], dtype=np.float32)
    response = np.ones((4, 1), dtype=np.float32)

    with pytest.raises(ValueError, match="prompt membership must be constant.*group 'a'"):
        prompt_response_association(
            prompt,
            response,
            group_ids=np.array(["a", "a", "b", "b"]),
            min_support=1,
            min_cooccur=1,
        )


def test_group_bonferroni_family_includes_unreported_pairs():
    group_ids = np.repeat(np.arange(8), 2)
    prompt_by_group = np.array(
        [[0, 0], [0, 1], [0, 0], [0, 1], [1, 0], [1, 1], [1, 0], [1, 1]],
        dtype=np.float32,
    )
    response_by_group = np.array(
        [[0, 0], [0, 1], [1, 0], [1, 1], [0, 0], [0, 1], [1, 0], [1, 1]],
        dtype=np.float32,
    )
    result = prompt_response_association(
        np.repeat(prompt_by_group, 2, axis=0),
        np.repeat(response_by_group, 2, axis=0),
        group_ids=group_ids,
        min_support=1,
        min_cooccur=5,
    )

    assert result.attrs["n_tested"] == 4
    assert len(result) < result.attrs["n_tested"]
    np.testing.assert_allclose(
        result["p_bonferroni"], np.minimum(1.0, 4 * result["p_value"])
    )


def test_group_inference_does_not_turn_zero_variance_into_p_zero():
    prompt = np.repeat([1.0, 1.0, 0.0, 0.0], 30)[:, None]
    response = prompt.copy()
    groups = np.repeat(["a", "b", "c", "d"], 30)

    result = prompt_response_association(
        prompt, response, group_ids=groups, min_support=2, min_cooccur=1)
    row = result.iloc[0]
    assert row["n_groups_x"] == 2
    assert 0.25 < row["p_value"] < 0.3
    assert not bool(row["significant"])


def test_elicitation_rejects_nonfinite_activations_and_invalid_threshold_options():
    prompt = np.array([[np.nan], [1.0]])
    response = np.ones((2, 1))
    with pytest.raises(ValueError, match="finite"):
        prompt_response_association(
            prompt, response, min_support=1, min_cooccur=1)
    with pytest.raises(ValueError, match="positive integer"):
        prompt_response_association(
            np.ones((2, 1)), response,
            min_support=True, min_cooccur=1)
    with pytest.raises(ValueError, match="finite"):
        prompt_response_association_paired(
            np.ones((2, 1)), response, np.array([[1.0], [np.nan]]),
            min_support=1, min_cooccur=1)


def test_elicitation_group_factorization_is_type_stable():
    prompt = np.ones((6, 1))
    response = np.ones((6, 1))
    result = prompt_response_association(
        prompt, response,
        group_ids=[1, True, "1", 1, True, "1"],
        min_support=1,
        min_cooccur=1,
    )
    assert result.loc[0, "n_groups_x"] == 3


def test_grouped_elicitation_names_hoeffding_statistic_without_calling_it_welch():
    prompt = np.array([[1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0]])
    response = prompt.copy()
    result = prompt_response_association(
        prompt, response,
        group_ids=["a", "a", "b", "b", "c", "c", "d", "d"],
        min_support=1, min_cooccur=1,
    )
    assert result.loc[0, "group_difference_statistic"] == 1.0
    assert np.isnan(result.loc[0, "welch_t"])
    assert result.loc[0, "inference_method"] == (
        "two_sample_hoeffding_group_prevalence")
