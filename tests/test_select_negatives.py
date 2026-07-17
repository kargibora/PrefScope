import numpy as np
import pytest

from prefscope.core import registry
from prefscope.interpret.select import select_negatives, top_pairs
import prefscope.interpret.verify  # noqa: F401  (ensures samplers are registered)


def test_random_matches_top_pairs_zero_bucket():
    z = np.array([2., 2., 2., 0., 0., 0., 0., 0.])
    pool = np.arange(8)
    sel = top_pairs(z, pool, 2, 3, np.random.default_rng(0))
    neg = select_negatives(z, pool, 3, strategy="random", rng=np.random.default_rng(0))
    np.testing.assert_array_equal(np.sort(neg), np.sort(sel["zero"]))


def test_random_returns_all_when_few_silent():
    z = np.array([1., 0., 0.])
    neg = select_negatives(z, np.arange(3), 10, strategy="random",
                           rng=np.random.default_rng(0))
    assert np.array_equal(np.sort(neg), np.array([1, 2]))


def test_similar_picks_nearest_silent():
    z = np.array([1., 1., 0., 0., 0., 0.])
    E = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [-1, 0], [1, 0.1]], dtype=float)
    neg = select_negatives(z, np.arange(6), 1, strategy="similar",
                           active_idx=[0, 1], embeddings=E)
    assert list(neg) == [5]                       # row 5 is nearest the active centroid


def test_similar_requires_embeddings_and_active():
    with pytest.raises(ValueError):
        select_negatives(np.array([1., 0.]), np.arange(2), 1, strategy="similar",
                         active_idx=[0])           # no embeddings


def test_negative_sampler_registry():
    assert {"random", "similar"} <= set(registry.available("negative_sampler"))
    assert callable(registry.get("negative_sampler", "similar"))


def test_interpreter_and_verifier_registry():
    import prefscope.interpret.strategy      # noqa: F401  (registers all interpreter classes,
    #                                          including single-text wrapping prompt_name)
    assert {"pairwise", "individual", "single-text"} <= set(registry.available("interpreter"))
    assert {"pairwise", "individual", "prompt"} <= set(registry.available("verifier"))
