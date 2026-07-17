import numpy as np

from prefscope.pipeline.cluster import cluster_examples, cluster_features


def test_cluster_examples_recovers_row_groups():
    # two example groups: rows 0-49 active on features {0,1}; rows 50-99 on {2,3}
    z = np.zeros((100, 4), dtype=np.float32)
    z[:50, 0] = z[:50, 1] = 1.0
    z[50:, 2] = z[50:, 3] = 1.0
    out = cluster_examples(z, n_clusters=2)
    assert list(out.columns) == ["example_index", "cluster_id"]
    assert list(out["example_index"][:3]) == [0, 1, 2]
    cid = out["cluster_id"].to_numpy()
    assert out["cluster_id"].nunique() == 2
    assert len(set(cid[:50])) == 1 and len(set(cid[50:])) == 1   # each group homogeneous
    assert cid[0] != cid[50]                                      # groups distinct


def test_cluster_examples_clamps_k_and_rejects_unknown_method():
    import pytest
    z = np.ones((5, 2), dtype=np.float32)
    assert cluster_examples(z, n_clusters=10)["cluster_id"].nunique() <= 5
    with pytest.raises(ValueError, match="spherical-kmeans"):
        cluster_examples(z, n_clusters=2, method="mi-leiden")


def test_cluster_features_still_groups_columns():
    # the refactor must not change cluster_features: two co-firing feature blocks
    rng = np.random.default_rng(0)
    fire_a = (rng.random(200) < 0.5).astype(np.float32)
    fire_b = (rng.random(200) < 0.5).astype(np.float32)
    z = np.zeros((200, 4), dtype=np.float32)
    z[:, 0] = z[:, 1] = fire_a
    z[:, 2] = z[:, 3] = fire_b
    out = cluster_features(z, n_clusters=2, method="spherical-kmeans")
    cid = out.set_index("feature_id")["cluster_id"]
    assert cid[0] == cid[1] and cid[2] == cid[3] and cid[0] != cid[2]
