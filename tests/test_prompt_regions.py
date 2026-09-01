"""Prompt regions are overlapping concept presences, not a forced argmax label."""
import numpy as np
import pandas as pd
import pytest

from prefscope.analysis import (
    prompt_region_membership as public_prompt_region_membership,
    region_membership_contrast as public_region_membership_contrast,
)
from prefscope.analysis.dataset import region_membership_contrast
from prefscope.analysis.prompt_regions import prompt_region_membership
from prefscope.analysis.prompt_regions import regions_from_feature_presence


def test_prompt_region_membership_keeps_all_active_concepts_and_strengths():
    assert public_prompt_region_membership is prompt_region_membership
    assert public_region_membership_contrast is region_membership_contrast
    z = np.array([
        [3.0, 2.0, 0.0],
        [0.0, 4.0, 1.0],
        [0.0, 0.0, 0.0],
    ], dtype=np.float32)

    ids, membership, strength = prompt_region_membership(z, min_activation=0.5)

    assert list(ids) == [0, 1, 2]
    assert membership.tolist() == [
        [True, True, False],
        [False, True, True],
        [False, False, False],
    ]
    np.testing.assert_allclose(strength, z)


def test_prompt_cluster_membership_unions_every_active_member():
    z = np.array([
        [3.0, 0.0, 2.0],
        [0.0, 4.0, 0.0],
    ], dtype=np.float32)
    clusters = pd.DataFrame({
        "feature_id": [0, 1, 2],
        "cluster_id": [10, 10, 20],
    })

    ids, membership, strength = prompt_region_membership(z, clusters=clusters)

    assert list(ids) == [10, 20]
    assert membership.tolist() == [[True, True], [True, False]]
    np.testing.assert_allclose(strength, [[3.0, 2.0], [4.0, 0.0]])


def test_calibrated_presence_can_be_unioned_into_overlapping_clusters():
    presence = np.array([[True, False, True], [False, True, False]])
    clusters = pd.DataFrame({
        "feature_id": [4, 5, 6, 6],
        "cluster_id": [10, 10, 20, 30],
    })

    ids, membership = regions_from_feature_presence(
        presence, [4, 5, 6], clusters=clusters)

    assert ids.tolist() == [10, 20, 30]
    assert membership.tolist() == [[True, True, True], [True, False, False]]


def test_region_membership_contrast_finds_overlapping_region_signal():
    rng = np.random.default_rng(2)
    n = 600
    membership = np.zeros((n, 2), dtype=bool)
    membership[:300, 0] = True
    membership[150:450, 1] = True  # overlap on rows 150:300
    z = rng.normal(0, 0.1, (n, 3)).astype(np.float32)
    z[membership[:, 0], 0] += 2.0
    z[membership[:, 1], 1] -= 2.0

    out = region_membership_contrast(
        z, membership, region_ids=[10, 20], seed=0)

    hit0 = out[(out.region_id == 10) & (out.feature_id == 0)].iloc[0]
    hit1 = out[(out.region_id == 20) & (out.feature_id == 1)].iloc[0]
    assert hit0["delta"] > 0 and hit0["p_bonferroni"] < 0.05
    assert hit1["delta"] < 0 and hit1["p_bonferroni"] < 0.05
    assert bool(hit0["stable"]) and bool(hit1["stable"])
    assert {"n_inside", "n_outside", "cohens_d"} <= set(out.columns)


def test_region_membership_contrast_uses_equal_prompt_group_means():
    z = np.array([[1.0]] * 10 + [[-1.0], [-1.0], [-1.0]])
    membership = np.array([[True]] * 11 + [[False], [False]])
    groups = np.array(["repeat"] * 10 + ["inside-2", "outside-1", "outside-2"])

    result = region_membership_contrast(
        z, membership, group_ids=groups, min_inside=2, min_outside=2)

    row = result.iloc[0]
    assert row["n_inside"] == 11
    assert row["n_inside_groups"] == 2
    assert row["n_outside_groups"] == 2
    assert row["n_independent_groups"] == 4
    assert row["delta"] == 1.0
    assert row["analysis_unit"] == "prompt_group"


def test_region_membership_contrast_rejects_varying_membership_within_group():
    with np.testing.assert_raises_regex(ValueError, "constant within each group"):
        region_membership_contrast(
            np.ones((4, 1)),
            np.array([[True], [False], [True], [False]]),
            group_ids=np.array(["same", "same", "a", "b"]),
        )


def test_grouped_region_zero_variance_uses_bounded_inference():
    z = np.repeat([[1.0], [1.0], [-1.0], [-1.0]], 30, axis=0)
    membership = np.repeat([[True], [True], [False], [False]], 30, axis=0)
    groups = np.repeat(["a", "b", "c", "d"], 30)

    result = region_membership_contrast(
        z, membership, group_ids=groups, min_inside=2, min_outside=2)
    row = result.iloc[0]
    assert row["inference_test"] == "two_sample_hoeffding_bounded_signed_means"
    assert row["welch_p"] > 0.25


def test_region_membership_contrast_rejects_nan_or_nonbinary_membership():
    z = np.ones((4, 1))
    with pytest.raises(ValueError, match="finite boolean or numeric 0/1"):
        region_membership_contrast(
            z, np.array([[1.0], [0.0], [np.nan], [1.0]]))
    with pytest.raises(ValueError, match="finite boolean or numeric 0/1"):
        region_membership_contrast(
            z, np.array([[1.0], [0.0], [2.0], [1.0]]))


def test_regions_from_presence_rejects_nan_nonbinary_and_duplicate_features():
    with pytest.raises(ValueError, match="finite boolean or numeric 0/1"):
        regions_from_feature_presence(
            np.array([[np.nan], [1.0]]), [0])
    with pytest.raises(ValueError, match="finite boolean or numeric 0/1"):
        regions_from_feature_presence(
            np.array([[2.0], [1.0]]), [0])
    with pytest.raises(ValueError, match="feature_ids must be unique"):
        regions_from_feature_presence(
            np.ones((2, 2), dtype=bool), [0, 0])


def test_prompt_region_membership_rejects_nonfinite_codes_and_duplicate_selection():
    with pytest.raises(ValueError, match="finite"):
        prompt_region_membership(np.array([[np.nan]]))
    with pytest.raises(ValueError, match="unique"):
        prompt_region_membership(np.ones((2, 2)), feature_ids=[0, 0])
