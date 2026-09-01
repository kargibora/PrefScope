import numpy as np
import pandas as pd

from prefscope.analysis.feature_graph import (
    feature_relationship_summary,
    feature_relationships,
)


def _codes_and_decoder():
    # f0: broad Greek; f1: Greek code, a strict subset of f0.
    # f2: exact activation/decoder duplicate of f1.
    # f3: disjoint Greek-labelled region (a naming collision, not redundancy).
    z = np.zeros((200, 4), dtype=np.float32)
    z[:100, 0] = 1
    z[:40, 1] = 1
    z[:40, 2] = 2
    z[140:180, 3] = 1
    decoder = np.asarray([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)
    names = pd.DataFrame({
        "feature_id": [0, 1, 2, 3],
        "concept": [
            "written in Greek",
            "provides code in Greek",
            "provides code in Greek",
            "is written in Greek",
        ],
    })
    return z, decoder, names


def test_relationships_distinguish_specialization_duplicates_and_name_collisions():
    z, decoder, names = _codes_and_decoder()
    out = feature_relationships(
        z,
        names=names,
        decoder=decoder,
        min_cooccur=1,
        min_jaccard=0.05,
    ).set_index(["feature_a", "feature_b"])

    # Feature 1 is wholly contained in broad feature 0, but not vice versa.
    assert out.loc[(0, 1), "relation"] == "b_specializes_a"
    assert out.loc[(0, 1), "containment_a_in_b"] == 0.4
    assert out.loc[(0, 1), "containment_b_in_a"] == 1.0

    # Same firing region alone is insufficient: aligned decoder directions are also needed.
    assert out.loc[(1, 2), "relation"] == "near_duplicate"
    assert bool(out.loc[(1, 2), "candidate_merge"])

    # Superficially equivalent names with disjoint activation regions need relabelling.
    assert out.loc[(0, 3), "relation"] == "same_name_collision"
    assert out.loc[(0, 3), "n_both"] == 0
    assert bool(out.loc[(0, 3), "needs_relabel"])
    assert not bool(out.loc[(0, 3), "candidate_merge"])


def test_names_never_create_merge_candidates_without_decoder_and_overlap():
    z, _, names = _codes_and_decoder()
    out = feature_relationships(z, names=names, min_cooccur=1)
    assert not out["candidate_merge"].any()
    assert ((out.feature_a == 0) & (out.feature_b == 3)).any()


def test_feature_subset_preserves_global_ids_and_empty_contract():
    z, decoder, names = _codes_and_decoder()
    out = feature_relationships(
        z, names=names, decoder=decoder, features=[1, 2], min_cooccur=1
    )
    assert out[["feature_a", "feature_b"]].values.tolist() == [[1, 2]]
    empty = feature_relationships(z, features=[0])
    assert empty.empty
    assert {"feature_a", "feature_b", "relation"} <= set(empty.columns)


def test_relationship_summary_reports_actionable_totals():
    z, decoder, names = _codes_and_decoder()
    out = feature_relationships(z, names=names, decoder=decoder, min_cooccur=1)
    summary = feature_relationship_summary(out).set_index("relation")["n_pairs"]
    assert summary["candidate_merge_total"] == 1
    assert summary["needs_relabel_total"] >= 1
