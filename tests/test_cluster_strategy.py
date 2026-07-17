"""Phase 3: clustering algorithms are swappable registry-resolved components."""
import numpy as np
import pytest

from prefscope.core import registry
from prefscope.pipeline.cluster import (
    AgglomerativeClusterer, Clusterer, MiLeidenClusterer, SphericalKmeansClusterer,
)


def test_clusterer_bucket_registered():
    assert {"mi-leiden", "spherical-kmeans", "agglomerative"} <= set(registry.available("clusterer"))
    assert isinstance(registry.make("clusterer", "mi-leiden"), MiLeidenClusterer)


def test_make_unknown_clusterer_raises_valueerror():
    with pytest.raises(ValueError, match="mi-leiden"):
        registry.make("clusterer", "nope")


def test_make_absorbs_irrelevant_params():
    # the CLI passes the union of params; each clusterer keeps only its own (via **_)
    km = registry.make("clusterer", "spherical-kmeans", n_clusters=3,
                       resolution=2.0, knn=5, min_cluster_size=2)
    assert isinstance(km, SphericalKmeansClusterer) and km.n_clusters == 3
    mil = registry.make("clusterer", "mi-leiden", n_clusters=99, resolution=1.5, knn=4)
    assert isinstance(mil, MiLeidenClusterer) and mil.resolution == 1.5 and mil.knn == 4


def test_spherical_kmeans_clusters_features_end_to_end():
    rng = np.random.default_rng(0)
    # 4 features in 2 correlated pairs -> 2 clusters
    base = rng.standard_normal((80, 2)).astype(np.float32)
    z = np.column_stack([base[:, 0], base[:, 0] + 0.01 * rng.standard_normal(80),
                         base[:, 1], base[:, 1] + 0.01 * rng.standard_normal(80)]).astype(np.float32)
    df = registry.make("clusterer", "spherical-kmeans", n_clusters=2).cluster(z)
    assert set(df.columns) >= {"feature_id", "cluster_id"}
    assert len(df) == 4 and df["cluster_id"].nunique() == 2


def test_clusterer_is_abc_contract():
    assert issubclass(MiLeidenClusterer, Clusterer)
    assert issubclass(AgglomerativeClusterer, Clusterer)


def test_empty_feature_subset_returns_valid_empty_tables():
    from prefscope.pipeline.cluster import cluster_features, summarize_clusters
    clusters = cluster_features(np.zeros((10, 3)), features=[])
    summary = summarize_clusters(clusters)
    assert clusters.empty and list(clusters.columns) == ["feature_id", "cluster_id"]
    assert summary.empty and {"cluster_id", "behavior"} <= set(summary.columns)


def test_cluster_summary_never_uses_failed_opposite_pole_name():
    import pandas as pd
    from prefscope.pipeline.cluster import summarize_clusters
    clusters = pd.DataFrame({"feature_id": [0, 1], "cluster_id": [0, 0]})
    names = pd.DataFrame({
        "feature_id": [0, 1], "concept": ["flipped name", "verified name"],
        "correlation": [-0.99, 0.5], "fidelity_pass": [False, True],
    })
    row = summarize_clusters(clusters, names).iloc[0]
    assert row["behavior"] == "verified name"
    assert row["member_concepts"] == "verified name"

    names["fidelity_pass"] = False
    assert summarize_clusters(clusters, names).iloc[0]["behavior"] == "cluster 0"
