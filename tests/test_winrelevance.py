import numpy as np
import pytest

from prefscope.pipeline.winrelevance import win_relevance, win_relevance_logistic


def test_win_relevance_detects_rewarded_feature():
    # feature 0: when A expresses it (z>0) A wins; when B (z<0) B wins -> strong +assoc
    z = np.array([[2.0, 0.0], [1.5, 0.0], [-2.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
    human = np.array([1.0, 1.0, 0.0, 0.0])      # P(A preferred)
    df = win_relevance(z, human).set_index("feature_id")
    f0 = df.loc[0]
    assert f0["win_rate_a_more"] == 1.0 and f0["win_rate_a_less"] == 0.0
    assert f0["win_assoc"] == 1.0
    assert f0["correlation"] > 0.99 and f0["sign"] == 1
    # feature 1 never fires -> undefined
    assert np.isnan(df.loc[1]["win_assoc"])


def test_chosen_always_a_still_has_winner_oriented_preference_summary():
    # Common HF chosen/rejected layout: the preferred response is always side A, so
    # outcome correlation is unidentified. Winner-oriented differences remain useful.
    z = np.array([[2.0], [1.0], [3.0], [-1.0]], dtype=np.float32)
    y = np.ones(4)
    row = win_relevance(z, y).iloc[0]
    assert np.isnan(row["correlation"])
    assert row["n_decisive_fire"] == 4
    assert row["preferred_side_rate"] == 0.75
    assert row["preferred_minus_rejected_mean"] == 1.25


def _logistic_data(seed=0, n=4000, beta=0.9):
    """A real (non-separable) rewarded feature plus a perfectly separable one."""
    rng = np.random.RandomState(seed)
    z_real = rng.randn(n)
    p = 1.0 / (1.0 + np.exp(-(beta * z_real)))
    y = (rng.rand(n) < p).astype(float)           # A preferred with prob p
    # separable feature: its sign perfectly determines the winner
    z_sep = np.where(y > 0.5, 1.0, -1.0) * (1.0 + rng.rand(n))
    z = np.column_stack([z_real, z_sep]).astype(np.float32)
    length = rng.randn(n)                          # nuisance length signal
    return z, y, length


def test_logistic_unpenalized_lrt_flags_separable_feature():
    z, y, length = _logistic_data()
    df = win_relevance_logistic(z, y, length).set_index("feature_id")
    # real rewarded feature: valid finite p, significant, positive Δwin-rate
    real = df.loc[0]
    assert not bool(real["separable"])
    assert np.isfinite(real["lr_p"])
    assert real["delta_win_rate"] > 0
    assert bool(real["delta_win_significant"])
    # separable feature: MLE diverges -> flagged, p is NaN, NOT called significant
    sep = df.loc[1]
    assert bool(sep["separable"])
    assert np.isnan(sep["lr_p"])
    assert not bool(sep["delta_win_significant"])
    # a stable point estimate is still reported for the separable feature
    assert np.isfinite(sep["delta_win_rate"])


def test_logistic_null_feature_not_significant():
    # feature uncorrelated with the outcome must not be flagged significant
    rng = np.random.RandomState(1)
    n = 3000
    y = (rng.rand(n) < 0.5).astype(float)
    z = rng.randn(n, 1).astype(np.float32)         # independent of y
    length = rng.randn(n)
    df = win_relevance_logistic(z, y, length).set_index("feature_id")
    assert not bool(df.loc[0]["separable"])
    assert not bool(df.loc[0]["delta_win_significant"])


def test_empty_feature_subset_returns_valid_empty_tables():
    z = np.zeros((10, 3), dtype=np.float32)
    y = np.tile([0.0, 1.0], 5)
    raw = win_relevance(z, y, features=[])
    controlled = win_relevance_logistic(z, y, np.zeros(10), features=[])
    assert raw.empty and {"feature_id", "significant"} <= set(raw.columns)
    assert controlled.empty and {"feature_id", "delta_win_significant"} <= set(
        controlled.columns)


def test_grouped_win_relevance_uses_equal_group_weight_and_group_bound():
    # One prompt contributes one aligned battle; another contributes nine opposed
    # repeats. Battle weighting gives 10%, while the prompt estimand gives each 50%.
    z = np.ones((10, 1), dtype=np.float32)
    z[1:, 0] = -1.0
    y = np.ones(10)
    groups = np.array(["one"] + ["repeated"] * 9)

    row = win_relevance(z, y, group_ids=groups).iloc[0]

    assert row["preferred_side_rate"] == 0.5
    assert row["fire_rate"] == 1.0
    assert row["n_groups"] == 2
    assert row["n_independent_groups"] == 2
    assert row["n_decisive_fire_groups"] == 2
    assert row["estimand"] == "equal_group_weight"
    assert row["preference_sign_test"] == "two_sided_hoeffding_bounded_group_mean"
    assert row["preference_sign_p"] == 1.0


def test_unique_groups_preserve_battle_weighted_win_relevance():
    z = np.array([[2.0], [1.0], [-1.0], [-2.0]], dtype=np.float32)
    y = np.array([1.0, 1.0, 0.0, 0.0])
    old = win_relevance(z, y).iloc[0]
    unique = win_relevance(z, y, group_ids=np.arange(len(y))).iloc[0]

    for column in ("fire_rate", "preferred_side_rate", "preference_sign_p",
                   "correlation", "p_value"):
        np.testing.assert_allclose(unique[column], old[column])
    assert unique["estimand"] == "battle_weighted"
    assert unique["n_groups"] == len(y)


def test_grouped_logistic_point_estimate_is_invariant_to_within_group_duplication():
    rng = np.random.default_rng(12)
    n_groups, per_group = 16, 30
    groups = np.repeat(np.arange(n_groups), per_group)
    z = rng.normal(size=(len(groups), 1))
    length = rng.normal(size=len(groups))
    group_effect = rng.normal(scale=0.7, size=n_groups)[groups]
    probability = 1.0 / (1.0 + np.exp(-(0.7 * z[:, 0] + group_effect)))
    y = (rng.random(len(groups)) < probability).astype(float)

    base = win_relevance_logistic(
        z, y, length, group_ids=groups).iloc[0]
    duplicate = groups == 0
    repeated = win_relevance_logistic(
        np.concatenate([z, np.tile(z[duplicate], (5, 1))]),
        np.concatenate([y, np.tile(y[duplicate], 5)]),
        np.concatenate([length, np.tile(length[duplicate], 5)]),
        group_ids=np.concatenate([groups, np.tile(groups[duplicate], 5)]),
    ).iloc[0]

    np.testing.assert_allclose(repeated["beta"], base["beta"], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        repeated["delta_win_rate"], base["delta_win_rate"], rtol=1e-5, atol=1e-6)
    assert repeated["n_groups"] == n_groups
    assert repeated["n_independent_groups"] == n_groups
    assert repeated["estimand"] == "equal_group_weight"
    assert repeated["inference_test"] == "cluster_robust_wald_t_g_minus_1_hc1"
    assert np.isfinite(repeated["lr_p"])


@pytest.mark.parametrize("bad_groups", [["a"], ["a", None, "b", "c"]])
def test_win_relevance_validates_group_alignment_and_missing_values(bad_groups):
    z = np.ones((4, 1), dtype=np.float32)
    y = np.array([0.0, 1.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="group_ids"):
        win_relevance(z, y, group_ids=bad_groups)


def test_logistic_validates_group_alignment_and_missing_values():
    z = np.ones((4, 1), dtype=np.float32)
    y = np.array([0.0, 1.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="group_ids"):
        win_relevance_logistic(z, y, np.zeros(4), group_ids=[0, 1, np.nan, 2])


def test_grouped_logistic_withholds_inference_for_too_few_independent_groups():
    z = np.array([[1.0], [1.0], [-1.0], [-1.0]])
    y = np.array([1.0, 1.0, 0.0, 0.0])
    groups = np.array(["a", "a", "b", "b"])
    row = win_relevance_logistic(
        z, y, np.zeros(4), group_ids=groups).iloc[0]
    assert row["n_independent_groups"] == 2
    assert not bool(row["inference_supported"])
    assert np.isnan(row["lr_p"])
    assert not bool(row["delta_win_significant"])


def test_group_correlation_uses_exact_nonzero_inference_and_is_sign_invariant():
    groups = np.arange(10)
    z = np.array([-1.0] * 5 + [1.0] * 5)[:, None]
    y = np.array([0.0] * 5 + [1.0] * 5)
    first = win_relevance(z, y, group_ids=groups).iloc[0]
    second = win_relevance(-z, 1.0 - y, group_ids=groups).iloc[0]
    assert first["correlation_test"] == (
        "fisher_exact_range_midpoint_split_across_rows")
    assert np.isclose(first["p_value"], 2.0 / 252.0)
    assert np.isclose(first["p_value"], second["p_value"])


def test_preference_estimators_declare_distinct_tie_policies():
    z = np.array([[-1.0], [0.5], [1.0], [-0.5], [0.25], [-0.25]])
    y = np.array([0.0, 0.5, 1.0, 0.5, 1.0, 0.0])
    descriptive = win_relevance(z, y)
    logistic = win_relevance_logistic(z, y, np.arange(len(y), dtype=float))
    assert descriptive.loc[0, "tie_policy"] == "retained_as_0.5_neutral"
    assert logistic.loc[0, "tie_policy"] == "dropped_from_binary_logistic"
