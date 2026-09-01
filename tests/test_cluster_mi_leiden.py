import numpy as np

from prefscope.pipeline.cluster import (
    _postprocess_small_communities,
    _sparsify_affinity,
    cluster_features,
    feature_cofire_affinity,
    feature_mi,
    partition_stability,
    cluster_run_diagnostics,
)


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


def test_positive_cofire_ignores_negative_pole_and_mutual_exclusion():
    z = np.zeros((100, 4), dtype=np.float32)
    z[:50, 0] = 1
    z[:50, 1] = 2
    z[:50, 2] = -3       # unnamed opposite pole: not presence
    z[50:, 3] = 1        # mutually exclusive, not a positive co-firing edge
    a = feature_cofire_affinity(z, metric="phi", min_cooccur=1)
    assert a.shape == (4, 4)
    assert np.allclose(a, a.T) and np.allclose(np.diag(a), 0)
    assert a[0, 1] > 0.99
    assert a[0, 2] == 0
    assert a[0, 3] == 0


def test_mutual_knn_drops_one_sided_and_zero_edges():
    w = np.array([[0.0, 0.9, 0.8],
                  [0.9, 0.0, 0.1],
                  [0.8, 0.1, 0.0]])
    mutual = _sparsify_affinity(w, 1, mode="mutual")
    union = _sparsify_affinity(w, 1, mode="union")
    assert mutual[0, 1] == mutual[1, 0] == 0.9
    assert mutual[0, 2] == mutual[2, 0] == 0
    assert union[0, 2] == union[2, 0] == 0.8
    assert not _sparsify_affinity(np.zeros((3, 3)), 1, mode="union").any()


def test_small_communities_are_preserved_not_pooled():
    labels = np.array([0, 0, 1, 2])
    out = _postprocess_small_communities(labels, min_size=3, policy="preserve")
    assert out[2] != out[3]
    assert len(np.unique(out)) == 3


def test_partition_stability_ignores_label_permutations():
    a = np.array([0, 0, 1, 1, 2, 2])
    b = np.array([7, 7, 3, 3, 9, 9])
    perfect = partition_stability([a, b])
    assert perfect["seed_ari_mean"] == 1.0
    c = np.array([0, 1, 0, 1, 2, 2])
    mixed = partition_stability([a, b, c])
    assert mixed["seed_ari_mean"] < 1.0


def test_cofire_leiden_emits_stability_diagnostics():
    clusters = cluster_features(
        _two_block_codes(), method="cofire-leiden", resolution=1.0,
        knn=2, knn_mode="mutual", min_cooccur=1, stability_runs=3)
    assert clusters["cluster_id"].nunique() == 2
    diag = cluster_run_diagnostics(clusters).iloc[0]
    assert diag["method"] == "cofire-leiden"
    assert diag["seed_ari_mean"] == 1.0
    assert diag["n_clusters"] == 2
