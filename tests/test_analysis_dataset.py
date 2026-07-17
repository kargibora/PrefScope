# tests/test_analysis_dataset.py
import numpy as np
import pandas as pd

from prefscope.analysis.dataset import dataset_reward, split_half_stable


def test_dataset_reward_is_mean_sign():
    # feature 0 always stronger in chosen (+1); feature 1: 3x rejected, 1x chosen
    z = np.array([[1.0, -1.0], [1.0, 1.0], [1.0, -1.0], [1.0, -1.0]])
    r = dataset_reward(z)
    assert r[0] == 1.0                     # systematically rewards feature 0
    assert r[1] == -0.5                    # mean(sign) = (-1+1-1-1)/4


def test_dataset_reward_empty():
    r = dataset_reward(np.empty((0, 3)))
    assert r.shape == (3,) and (r == 0).all()


def test_split_half_stable_flags_consistent_and_drops_dead():
    z = np.zeros((100, 3))
    z[:, 0] = 1.0       # always chosen-positive  -> stable
    z[:, 1] = -1.0      # always chosen-negative  -> stable
    z[:, 2] = 0.0       # never fires             -> not stable
    out = split_half_stable(z, dataset_reward, seed=0)
    assert list(out["feature_id"]) == [0, 1, 2]
    s = out.set_index("feature_id")["stable"]
    assert bool(s[0]) is True and bool(s[1]) is True
    assert bool(s[2]) is False              # zero effect on both halves -> filtered
    assert out.set_index("feature_id")["effect"][1] == -1.0


def test_split_half_stable_flags_sign_flip():
    # a feature that is +1 on half A and -1 on half B is genuinely unstable.
    # Reconstruct the same split split_half_stable uses (seed=0) so it's deterministic.
    n = 100
    perm = np.random.default_rng(0).permutation(n)
    a_idx, b_idx = perm[: n // 2], perm[n // 2:]
    z = np.zeros((n, 1))
    z[a_idx, 0] = 1.0
    z[b_idx, 0] = -1.0
    out = split_half_stable(z, dataset_reward, seed=0)
    assert bool(out.set_index("feature_id")["stable"][0]) is False


import pytest
from prefscope.analysis.dataset import spurious_share, label_inconsistency


def test_spurious_share():
    z = np.array([[10.0, 0.0, 0.0],     # all mass on the spurious feature 0
                  [1.0, 1.0, 1.0],      # 1/3 of |z| on feature 0
                  [0.0, 0.0, 5.0]])     # none on feature 0
    s = spurious_share(z, undesirable=[0])
    assert s[0] == 1.0
    assert abs(s[1] - 1.0 / 3.0) < 1e-9
    assert s[2] == 0.0


def test_spurious_share_empty_U_is_zero():
    z = np.array([[1.0, 2.0]])
    assert spurious_share(z, undesirable=[])[0] == 0.0


def test_label_inconsistency_sign():
    # U = {0}; quality features 1,2 both rewarded (+1). A pair whose chosen side is
    # WEAKER on both quality features is inconsistent (a_i < 0).
    reward = np.array([0.0, 1.0, 1.0])
    z_bad = np.array([[5.0, -1.0, -1.0]])      # chosen weaker on quality -> -2
    z_ok = np.array([[5.0, 1.0, 1.0]])         # chosen stronger -> +2
    assert label_inconsistency(z_bad, reward, undesirable=[0])[0] == -2.0
    assert label_inconsistency(z_ok, reward, undesirable=[0])[0] == 2.0


from prefscope.analysis.dataset import symmetric_activity


def test_symmetric_activity():
    z_a = np.array([[2.0, -4.0]])
    z_b = np.array([[-2.0, 0.0]])
    s = symmetric_activity(z_a, z_b)        # (|2|+|-2|)/2 = 2 ; (|-4|+|0|)/2 = 2
    np.testing.assert_allclose(s, [[2.0, 2.0]])
    assert (s >= 0).all()


from prefscope.analysis import region_behavior_contrast


def test_region_behavior_contrast_finds_region_signal():
    # region 0 (rows 0-49) rewards feature 0 (+), region 1 (50-99) penalizes it (-).
    # one flipped element per side gives nonzero variance (Welch well-defined).
    z = np.zeros((100, 2), dtype=np.float32)
    z[:50, 0] = 1.0;  z[0, 0] = -1.0
    z[50:, 0] = -1.0; z[50, 0] = 1.0
    z[:, 1] = np.tile([1.0, -1.0], 50)        # feature 1: balanced noise everywhere
    cluster_ids = np.array([0] * 50 + [1] * 50)

    df = region_behavior_contrast(z, cluster_ids, seed=0)
    assert {"cluster_id", "feature_id", "delta", "welch_p", "p_bonferroni",
            "stable"} <= set(df.columns)

    sig = df[(df.cluster_id == 0) & (df.feature_id == 0)].iloc[0]
    assert sig["delta"] > 0 and bool(sig["stable"]) and sig["p_bonferroni"] < 0.05

    noise = df[(df.cluster_id == 0) & (df.feature_id == 1)]
    # the balanced-noise feature is not a stable, significant region signal
    if len(noise):
        r = noise.iloc[0]
        assert not (bool(r["stable"]) and r["p_bonferroni"] < 0.05)


from prefscope.analysis import auto_undesirable, feature_confound_correlation
from prefscope.analysis import diagnose_dataset   # re-exported from the package


def test_diagnose_dataset_composes():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(200, 4))
    z[:, 0] = 3.0                       # feature 0: strong, consistent reward
    names = pd.DataFrame({"feature_id": [0, 1, 2, 3],
                          "concept": ["length", "clarity", "tone", "code"]})
    per_feature, per_sample = diagnose_dataset(
        z, undesirable=[0], ids=[f"ex{i}" for i in range(200)], names=names)

    # per-feature: reward + stability + attached concept name
    assert {"feature_id", "effect", "stable", "concept"} <= set(per_feature.columns)
    assert per_feature.set_index("feature_id").loc[0, "effect"] == 1.0
    assert per_feature.set_index("feature_id").loc[0, "concept"] == "length"

    # per-sample: id + both scores, one row per example
    assert {"id", "spurious_share", "label_inconsistency"} <= set(per_sample.columns)
    assert len(per_sample) == 200
    assert (per_sample["spurious_share"] >= 0).all()
    assert per_sample["id"].iloc[0] == "ex0"


def test_auto_undesirable_flags_surrogate_tracking_feature():
    rng = np.random.default_rng(0)
    length_diff = rng.normal(size=300)                 # per-example surrogate
    z = rng.normal(size=(300, 3))
    # feature 0's reward direction tracks the surrogate; 1 and 2 are independent
    z[:, 0] = np.where(length_diff > 0, 1.0, -1.0) * np.abs(z[:, 0])
    flagged = auto_undesirable(z, length_diff, threshold=0.3)
    assert 0 in flagged
    assert 1 not in flagged and 2 not in flagged
    corr = feature_confound_correlation(z, length_diff)
    assert int(corr.iloc[0]["feature_id"]) == 0        # strongest correlate ranked first
    assert abs(corr.iloc[0]["corr"]) > 0.3


def test_feature_confound_correlation_constant_surrogate_is_nan():
    z = np.sign(np.random.default_rng(1).normal(size=(50, 2)))
    out = feature_confound_correlation(z, np.zeros(50))   # no variance in surrogate
    assert out["corr"].isna().all()
    assert auto_undesirable(z, np.zeros(50)) == []


def test_region_behavior_contrast_bonferroni_counts_all_clusters():
    # 3 regions, one a singleton (degenerate -> dropped). Bonferroni must divide by
    # n_clusters * n_features (3), not by the count of surviving rows (2).
    z = np.zeros((99, 1), dtype=np.float32)
    z[:49, 0] = 1.0; z[0, 0] = -1.0          # region 0
    z[49:98, 0] = -1.0; z[49, 0] = 1.0       # region 1
    z[98, 0] = 1.0                            # region 2: singleton -> not testable
    cluster_ids = np.array([0] * 49 + [1] * 49 + [2])
    df = region_behavior_contrast(z, cluster_ids, seed=0)
    assert len(df) <= 2                       # the singleton region is dropped
    r = df.iloc[0]
    assert abs(r["p_bonferroni"] - min(r["welch_p"] * 3, 1.0)) < 1e-12
