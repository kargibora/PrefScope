"""Specialized clustering recipes remain callable without registry dispatch."""
import numpy as np
import pandas as pd
import pytest
from prefscope.recipes.pipeline.cluster import (
    AgglomerativeClusterer, Clusterer, CofireLeidenClusterer, MiLeidenClusterer,
    SphericalKmeansClusterer,
)


def test_recipe_clusterer_wrappers_are_constructed_directly():
    assert isinstance(CofireLeidenClusterer(), CofireLeidenClusterer)
    assert isinstance(MiLeidenClusterer(resolution=1.5, knn=4), MiLeidenClusterer)


def test_spherical_kmeans_clusters_features_end_to_end():
    rng = np.random.default_rng(0)
    # 4 features in 2 correlated pairs -> 2 clusters
    base = rng.standard_normal((80, 2)).astype(np.float32)
    z = np.column_stack([base[:, 0], base[:, 0] + 0.01 * rng.standard_normal(80),
                         base[:, 1], base[:, 1] + 0.01 * rng.standard_normal(80)]).astype(np.float32)
    df = SphericalKmeansClusterer(n_clusters=2).cluster(z)
    assert set(df.columns) >= {"feature_id", "cluster_id"}
    assert len(df) == 4 and df["cluster_id"].nunique() == 2


def test_clusterer_is_abc_contract():
    assert issubclass(MiLeidenClusterer, Clusterer)
    assert issubclass(AgglomerativeClusterer, Clusterer)


def test_empty_feature_subset_returns_valid_empty_tables():
    from prefscope.recipes.pipeline.cluster import cluster_features, summarize_clusters
    clusters = cluster_features(np.zeros((10, 3)), features=[])
    summary = summarize_clusters(clusters)
    assert clusters.empty and list(clusters.columns) == ["feature_id", "cluster_id"]
    assert summary.empty and {"cluster_id", "behavior"} <= set(summary.columns)


def test_cluster_summary_never_uses_failed_opposite_pole_name():
    import pandas as pd
    from prefscope.recipes.pipeline.cluster import summarize_clusters
    clusters = pd.DataFrame({"feature_id": [0, 1], "cluster_id": [0, 0]})
    names = pd.DataFrame({
        "feature_id": [0, 1], "concept": ["flipped name", "verified name"],
        "correlation": [-0.99, 0.5], "fidelity_pass": [False, True],
    })
    row = summarize_clusters(clusters, names).iloc[0]
    assert row["behavior"] == "cluster 0"
    assert row["member_concepts"] == "verified name"

    names["fidelity_pass"] = False
    assert summarize_clusters(clusters, names).iloc[0]["behavior"] == "cluster 0"


def test_cluster_namer_allows_mixed_and_uses_representatives():
    import pandas as pd
    from prefscope.recipes.pipeline.cluster import name_clusters

    class Client:
        def __init__(self):
            self.prompts = []

        def raw(self, messages, **_):
            self.prompts.append(messages[0]["content"])
            return "MIXED"

    summary = pd.DataFrame([{
        "cluster_id": 4,
        "representative_concepts": "written in Russian | financial advice",
        "member_concepts": "ignored early member",
    }])
    client = Client()
    labels = name_clusters(summary, client)
    assert labels == {4: "Mixed / incoherent"}
    assert "written in Russian" in client.prompts[0]
    assert "Do not assume" in client.prompts[0]
    assert "ONE higher-level behavior" not in client.prompts[0]


@pytest.mark.parametrize("features", [[0, 0], [0.9], [True], [-1], [3], [10**30]])
def test_feature_selectors_are_strict_column_indices(features):
    from prefscope.recipes.analysis.feature_graph import feature_relationships
    from prefscope.recipes.pipeline.cluster import cluster_features, feature_cofire_affinity

    z = np.zeros((4, 3))
    for recipe in (feature_relationships, feature_cofire_affinity, cluster_features):
        with pytest.raises(ValueError):
            recipe(z, features=features)


def test_agglomerative_singleton_distance_and_membership():
    from prefscope.recipes.pipeline.cluster import cluster_features, feature_distance

    z = np.arange(12).reshape(4, 3)
    np.testing.assert_array_equal(feature_distance(z[:, [2]]), [[0.0]])
    result = cluster_features(z, features=[np.int64(2)], method="agglomerative")
    assert result.to_dict("list") == {"feature_id": [2], "cluster_id": [0]}


@pytest.mark.parametrize("features", [[], [0]])
def test_method_validation_precedes_empty_or_singleton_return(features):
    from prefscope.recipes.pipeline.cluster import cluster_features

    with pytest.raises(ValueError, match="method"):
        cluster_features(np.ones((3, 1)), features=features, method="unknown")


@pytest.mark.parametrize("bad", [np.nan, np.inf, 1j, "1"])
def test_dense_clustering_boundaries_reject_invalid_raw_values(bad):
    from prefscope.recipes.pipeline.cluster import cluster_features, feature_distance, feature_mi

    z = np.array([[bad], [0]])
    for recipe in (feature_distance, feature_mi):
        with pytest.raises(ValueError, match="finite real"):
            recipe(z)
    for method in ("agglomerative", "spherical-kmeans", "mi-leiden"):
        with pytest.raises(ValueError, match="finite real"):
            cluster_features(z, method=method)


@pytest.mark.parametrize("seed", [0, 1])
def test_spherical_kmeans_forwards_seed_to_native_estimator(seed):
    from sklearn.cluster import KMeans
    from prefscope.recipes.pipeline.cluster import cluster_features

    z = np.random.default_rng(0).normal(size=(8, 25)).astype(np.float32)
    points = z.T.astype(np.float64)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    expected = KMeans(n_clusters=5, n_init=10, random_state=seed).fit_predict(points)
    direct = cluster_features(z, n_clusters=5, seed=seed)
    wrapped = SphericalKmeansClusterer(n_clusters=5, seed=seed).cluster(z)
    np.testing.assert_array_equal(direct["cluster_id"], expected)
    np.testing.assert_array_equal(wrapped["cluster_id"], expected)
    if seed == 0:
        np.testing.assert_array_equal(cluster_features(z, n_clusters=5)["cluster_id"], expected)


def test_cluster_summary_rejects_duplicate_annotations():
    from prefscope.recipes.pipeline.cluster import summarize_clusters

    clusters = pd.DataFrame({"feature_id": [0, 1], "cluster_id": [0, 0]})
    names = pd.DataFrame({"feature_id": [0, 0, 1], "concept": ["old", "new", "other"]})
    with pytest.raises(pd.errors.MergeError, match="many-to-one"):
        summarize_clusters(clusters, names)
