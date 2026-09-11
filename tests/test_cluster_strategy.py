"""Specialized clustering recipes remain callable without registry dispatch."""
import numpy as np
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
