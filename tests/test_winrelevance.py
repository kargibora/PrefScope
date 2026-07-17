import numpy as np

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
