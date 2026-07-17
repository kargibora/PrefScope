import numpy as np

from prefscope.pipeline.cluster import cluster_features, feature_mi


def _two_block_codes(n=500, seed=0):
    """Features {0,1,2} share one independent firing pattern; {3,4} another."""
    rng = np.random.default_rng(seed)
    fire_a = rng.random(n) < 0.5
    fire_b = rng.random(n) < 0.5            # independent of fire_a
    z = np.zeros((n, 5), dtype=np.float32)
    for f in (0, 1, 2):
        z[fire_a, f] = 1.0
    for f in (3, 4):
        z[fire_b, f] = 1.0
    return z


def test_feature_mi_high_within_low_across():
    mi = feature_mi(_two_block_codes())
    assert mi.shape == (5, 5)
    assert np.allclose(np.diag(mi), 0.0)
    # features sharing a pattern are highly dependent; cross-block ~ independent
    assert mi[0, 1] > 0.5
    assert mi[0, 1] > 10 * mi[0, 3]


def test_mi_leiden_recovers_the_two_blocks():
    clusters = cluster_features(_two_block_codes(), method="mi-leiden", resolution=1.0)
    cid = clusters.set_index("feature_id")["cluster_id"].to_dict()
    assert clusters["cluster_id"].nunique() == 2
    assert cid[0] == cid[1] == cid[2]        # block A together
    assert cid[3] == cid[4]                    # block B together
    assert cid[0] != cid[3]                    # the two blocks are distinct


def test_unknown_method_raises():
    import pytest
    with pytest.raises(ValueError, match="mi-leiden"):
        cluster_features(_two_block_codes(), method="nope")
