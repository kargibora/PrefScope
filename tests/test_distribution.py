import numpy as np
import pytest

from prefscope.analysis.distribution import concept_coactivation, concept_distribution


def _codes():
    # f0 always fires, f1 fires with f0 on the first half, f2 never fires.
    z = np.zeros((10, 3), dtype=np.float32)
    z[:, 0] = 1.0
    z[:5, 1] = 2.0
    return z


def test_distribution_reports_prevalence_and_coverage():
    out = concept_distribution(_codes())
    by_id = {f["feature_id"]: f for f in out["features"]}
    assert by_id[0]["fire_rate"] == 1.0 and by_id[0]["n_active"] == 10
    assert by_id[1]["fire_rate"] == 0.5
    assert by_id[2]["n_active"] == 0
    assert out["dead_features"] == [2]
    assert out["coverage"] == 1.0
    assert out["concepts_per_row"]["mean"] == pytest.approx(1.5)


def test_distribution_is_chunk_invariant():
    z = np.random.default_rng(0).normal(size=(97, 5)).astype(np.float32)
    z[np.abs(z) < 0.5] = 0.0
    whole = concept_distribution(z, chunk_rows=1000)
    chunked = concept_distribution(z, chunk_rows=7)
    for a, b in zip(whole["features"], chunked["features"]):
        assert a["n_active"] == b["n_active"]
        assert a["mean_activation"] == pytest.approx(b["mean_activation"])
    assert whole["concepts_per_row"] == chunked["concepts_per_row"]


def test_distribution_uses_positive_concept_pole_and_selected_columns():
    z = np.array([
        [2.0, -9.0, 1.0],
        [-3.0, -8.0, 4.0],
    ], dtype=np.float32)
    out = concept_distribution(z, columns=[0, 2], feature_ids=[10, 12])
    by_id = {row["feature_id"]: row for row in out["features"]}
    assert set(by_id) == {10, 12}
    assert by_id[10]["n_active"] == 1
    assert by_id[12]["n_active"] == 2
    assert out["concepts_per_row"]["mean"] == pytest.approx(1.5)


def test_distribution_groups_give_per_subset_fire_rates():
    groups = ["de"] * 5 + ["cs"] * 5
    out = concept_distribution(_codes(), groups=groups)
    by_id = {f["feature_id"]: f for f in out["features"]}
    assert out["groups"] == ["cs", "de"]
    assert by_id[1]["group_fire_rate"] == {"de": 1.0, "cs": 0.0}
    assert out["group_totals"] == {"cs": 5, "de": 5}
    assert by_id[1]["max_activation"] == 2.0


def test_distribution_payload_does_not_grow_with_rows():
    small = concept_distribution(np.ones((10, 4), dtype=np.float32))
    large = concept_distribution(np.ones((10_000, 4), dtype=np.float32))
    assert len(small["features"]) == len(large["features"]) == 4


def test_coactivation_ranks_by_lift_and_keeps_examples():
    out = concept_coactivation(_codes(), min_pair_count=1, n_examples=3)
    assert out["pairs"], "expected at least the f0/f1 pair"
    top = out["pairs"][0]
    assert {top["a"], top["b"]} == {0, 1}
    assert top["count"] == 5
    assert top["rows"] and all(r < 5 for r in top["rows"])


def test_coactivation_examples_are_strong_joint_activators_not_first_matches():
    z = np.zeros((10, 2), dtype=np.float32)
    z[:2] = 0.01
    z[2:4] = [[2.0, 3.0], [3.0, 2.0]]
    out = concept_coactivation(
        z, min_pair_count=1, n_examples=2, example_pairs=1)
    assert set(out["pairs"][0]["rows"]) == {2, 3}


def test_coactivation_does_not_treat_negative_pole_as_named_concept():
    z = np.array([[1.0, -2.0], [0.0, -3.0], [0.0, 0.0], [1.0, 1.0]],
                 dtype=np.float32)
    out = concept_coactivation(z, min_pair_count=1, n_examples=1)
    pair = out["pairs"][0]
    assert pair["count"] == 1
    assert pair["rows"] == [3]


def test_coactivation_lift_is_one_for_independent_features():
    rng = np.random.default_rng(1)
    z = (rng.random((4000, 2)) < 0.5).astype(np.float32)
    out = concept_coactivation(z, min_pair_count=1, n_examples=0)
    assert out["pairs"][0]["lift"] == pytest.approx(1.0, abs=0.1)


def test_coactivation_respects_min_count_and_cap():
    assert concept_coactivation(_codes(), min_pair_count=99)["pairs"] == []
    capped = concept_coactivation(_codes(), min_pair_count=1, max_pairs=1)
    assert len(capped["pairs"]) == 1 and capped["truncated"] is True


def test_coactivation_is_chunk_invariant():
    z = (np.random.default_rng(2).random((83, 6)) < 0.4).astype(np.float32)
    a = concept_coactivation(z, min_pair_count=1, chunk_rows=1000, n_examples=2)
    b = concept_coactivation(z, min_pair_count=1, chunk_rows=9, n_examples=2)
    assert a["pairs"] == b["pairs"]


def test_bad_shapes_are_refused():
    with pytest.raises(ValueError):
        concept_distribution(np.zeros(5))
    with pytest.raises(ValueError):
        concept_distribution(np.zeros((5, 2)), groups=["a"])
    with pytest.raises(ValueError):
        concept_coactivation(np.zeros((5, 2)), top_k=0)


def test_distribution_rejects_lossy_column_and_feature_ids():
    codes = np.ones((3, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="non-boolean integers"):
        concept_distribution(codes, columns=[0.9])
    with pytest.raises(ValueError, match="non-boolean integers"):
        concept_distribution(codes, columns=[0], feature_ids=[True])
