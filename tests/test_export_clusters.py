import pandas as pd

from prefscope.viewer_export.clusters import export_feature_clusters


def test_cluster_export_is_self_contained_and_keeps_unclustered_axes():
    features = pd.DataFrame({
        "feature_id": [0, 1, 2, 3],
        "concept": ["broad Greek", "Greek code", None, "unclustered"],
        "fidelity_pass": [True, False, False, True],
        "semantic_family": ["behavioral", "prompt_specific", None, "behavioral"],
        # These may already have been attached by features.json export. Membership below
        # remains authoritative and must not create cluster_id_x/cluster_id_y columns.
        "cluster_id": [8, 8, 9, None],
        "behavior": ["cluster 8", "cluster 8", "Language", None],
    })
    membership = pd.DataFrame({
        "feature_id": [0, 1, 2],
        "cluster_id": [8, 8, 9],
        # Real cluster tables repeat labels for CSV readability. These must not create
        # merge suffixes or override the authoritative feature-table interpretation.
        "concept": ["stale zero", "stale one", "stale two"],
        "behavior": ["cluster 8", "cluster 8", "Language"],
    })
    summary = pd.DataFrame({
        "cluster_id": [8, 9],
        "behavior": ["cluster 8", "Language"],
        "representative_feature_ids": ["1,0", "2"],
        "representative_concepts": ["Greek code | broad Greek", ""],
        "within_affinity_mean": [0.4, None],
        "affinity_separation": [0.3, None],
    })
    diagnostics = pd.DataFrame({
        "method": ["cofire-leiden"], "seed_ari_mean": [0.91], "resolution": [5.0]
    })

    out = export_feature_clusters(
        membership, features, kind="response", summary=summary, diagnostics=diagnostics
    )
    assert out is not None
    assert out["n_clusters"] == 2
    assert out["n_clustered_features"] == 3
    assert out["unclustered_feature_ids"] == [3]
    assert out["diagnostics"]["method"] == "cofire-leiden"

    by_id = {row["cluster_id"]: row for row in out["clusters"]}
    # A mechanical "cluster N" string is not presented as a semantic label.
    assert by_id[8]["label"] is None
    assert by_id[8]["representative_feature_ids"] == [1, 0]
    assert [row["feature_id"] for row in by_id[8]["members"]] == [0, 1]
    assert [row["concept"] for row in by_id[8]["members"]] == ["broad Greek", "Greek code"]
    assert by_id[8]["n_named"] == 2
    assert by_id[8]["n_verified"] == 1
    assert by_id[9]["label"] == "Language"


def test_cluster_export_validates_kind_and_required_columns():
    import pytest

    features = pd.DataFrame({"feature_id": [0], "concept": ["x"]})
    with pytest.raises(ValueError, match="kind"):
        export_feature_clusters(
            pd.DataFrame({"feature_id": [0], "cluster_id": [0]}),
            features,
            kind="other",
        )
    with pytest.raises(ValueError, match="cluster table"):
        export_feature_clusters(pd.DataFrame({"feature_id": [0]}), features, kind="prompt")
