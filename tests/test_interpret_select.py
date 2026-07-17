import numpy as np

from prefscope.interpret.select import (
    name_verify_split, top_pairs, holdout_buckets,
)


def test_name_verify_split_disjoint_and_deterministic():
    ids = [str(i) for i in range(200)]
    nm1, vm1 = name_verify_split(ids, verify_frac=0.2)
    nm2, vm2 = name_verify_split(ids, verify_frac=0.2)
    assert np.array_equal(nm1, nm2) and np.array_equal(vm1, vm2)
    assert not (nm1 & vm1).any()
    assert np.array_equal(nm1 | vm1, np.ones(200, bool))
    assert 20 < vm1.sum() < 60


def test_top_pairs_picks_highest_positive_and_zeros_from_pool():
    z = np.array([5.0, -3.0, 0.0, 2.0, 0.0, 9.0, 0.0])
    pool = np.array([0, 2, 3, 4, 5, 6])
    rng = np.random.default_rng(0)
    out = top_pairs(z, pool, n_active=2, n_zero=2, rng=rng)
    assert list(out["active"]) == [5, 0]
    assert set(out["zero"]).issubset({2, 4, 6})
    assert len(out["zero"]) == 2


def test_holdout_buckets_pos_neg_tie():
    z = np.array([5.0, -3.0, 0.0, 2.0, 0.0, -9.0])
    pool = np.arange(6)
    rng = np.random.default_rng(0)
    out = holdout_buckets(z, pool, n_per_bucket=1, rng=rng)
    assert out["pos"][0] == 0
    assert out["neg"][0] == 5
    assert out["tie"][0] in (2, 4)


def test_holdout_buckets_stratified_random_spans_non_extreme_rows():
    z = np.concatenate([np.arange(1, 51), -np.arange(1, 51), np.zeros(50)])
    out = holdout_buckets(z, np.arange(len(z)), n_per_bucket=10,
                          rng=np.random.default_rng(0), sampling="stratified-random")
    assert len(out["pos"]) == len(out["neg"]) == len(out["tie"]) == 10
    # A uniform draw is not merely the ten largest magnitudes in either sign bucket.
    assert set(out["pos"]) != set(range(40, 50))
    assert set(out["neg"]) != set(range(90, 100))


def test_close_silent_order_prefers_similar_other_concepts():
    from prefscope.interpret.select import close_silent_order
    # feature 0 is being named; its activators also express concept 1.
    active = np.array([[5., 4., 0., 0.], [6., 3., 0., 0.]])
    silent = np.array([[0., 4., 0., 0.],   # shares concept 1 -> close (rank first)
                       [0., 0., 0., 5.]])   # shares nothing -> far
    order = close_silent_order(active, silent, f=0)
    assert order[0] == 0                    # the concept-1-sharing silent is closest
    # feature 0 is removed before ranking, so a huge f0 in a silent row can't fool it
    silent2 = np.array([[9., 0., 0., 0.], [0., 4., 0., 0.]])
    assert close_silent_order(active, silent2, f=0)[0] == 1


def test_close_negatives_sampler_is_registered():
    """The pipeline/CLI use --negatives close; verify resolves it via the registry. It must
    be a registered negative_sampler (was KeyError: only random/similar) — the blocking bug."""
    import numpy as np
    from prefscope.interpret.select import select_negatives
    z = np.array([0., 0., 1., 2., 0., 0.], np.float32)
    emb = np.eye(6, dtype=np.float32)
    out = select_negatives(z, np.arange(6), 2, strategy="close",
                           active_idx=[2, 3], embeddings=emb, rng=np.random.default_rng(0))
    assert len(out) == 2
    assert all(z[i] == 0 for i in out)      # negatives are silent, and it did not crash
