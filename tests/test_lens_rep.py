"""Behavior-preservation tests for the lens_rep strategies.

Uses a REAL trained (nonlinear, BatchTopK-thresholded) SAE projector so the
orientation assertions aren't vacuous — for a linear map project(b-a) would equal
-project(a-b) and a negation bug would slip through.
"""
import numpy as np
import pytest
import torch

from prefscope.encode.sae import SAEProjector
from prefscope.pipeline.lens_rep import get_lens_rep
from prefscope.sae.train import train_sae


@pytest.fixture(scope="module")
def proj(tmp_path_factory):
    rng = np.random.default_rng(0)
    d = 8
    X = (rng.standard_normal((400, 3)) @ rng.standard_normal((3, d))).astype(np.float32)
    model, config, _ = train_sae(X[:300], X[300:], m_total=8, k=2,
                                 matryoshka_prefix=(4,), n_epochs=4, min_epochs=4,
                                 patience=4, batch=64, device="cpu", seed=0)
    ckpt = tmp_path_factory.mktemp("lens") / "sae_model.pt"
    torch.save({"state_dict": model.state_dict(), "config": config}, ckpt)
    return SAEProjector(ckpt, device="cpu")


@pytest.fixture
def ab():
    rng = np.random.default_rng(1)
    return (rng.standard_normal((20, 8)).astype(np.float32),
            rng.standard_normal((20, 8)).astype(np.float32))


class _Counting:
    def __init__(self, p):
        self.p = p
        self.calls = 0

    def project(self, x):
        self.calls += 1
        return self.p.project(x)


def test_difference_matches_explicit_formula(proj, ab):
    e_a, e_b = ab
    rep = get_lens_rep("difference")
    np.testing.assert_array_equal(rep.training_matrix(e_a, e_b), e_a - e_b)
    np.testing.assert_array_equal(rep.contrast_codes(proj, e_a, e_b), proj.project(e_a - e_b))
    za_self, zb_self = rep.oriented_codes(proj, e_a, e_b)
    np.testing.assert_array_equal(za_self, proj.project(e_a - e_b))
    np.testing.assert_array_equal(zb_self, proj.project(e_b - e_a))   # genuine 2nd pass
    out = rep.output_arrays(proj, e_a, e_b)
    assert list(out) == ["z_diff"]
    np.testing.assert_array_equal(out["z_diff"], proj.project(e_a - e_b))


def test_nonlinearity_guard_fixture_is_not_linear(proj, ab):
    """The fixture must be nonlinear, else the orientation test is meaningless."""
    e_a, e_b = ab
    assert not np.allclose(proj.project(e_b - e_a), -proj.project(e_a - e_b))


def test_individual_matches_explicit_formula(proj, ab):
    e_a, e_b = ab
    rep = get_lens_rep("individual")
    np.testing.assert_array_equal(rep.training_matrix(e_a, e_b), np.vstack([e_a, e_b]))
    np.testing.assert_array_equal(rep.contrast_codes(proj, e_a, e_b),
                                  proj.project(e_a) - proj.project(e_b))
    za_self, zb_self = rep.oriented_codes(proj, e_a, e_b)
    np.testing.assert_array_equal(za_self, proj.project(e_a) - proj.project(e_b))
    np.testing.assert_array_equal(zb_self, proj.project(e_b) - proj.project(e_a))
    out = rep.output_arrays(proj, e_a, e_b)
    assert list(out) == ["z_a", "z_b", "z_diff"]
    np.testing.assert_array_equal(out["z_diff"], out["z_a"] - out["z_b"])


def test_individual_oriented_projects_each_side_once(proj, ab):
    """Perf guard: the bank path must not re-project (2 calls for N battles, not 4)."""
    e_a, e_b = ab
    c = _Counting(proj)
    get_lens_rep("individual").oriented_codes(c, e_a, e_b)
    assert c.calls == 2
    c2 = _Counting(proj)
    get_lens_rep("difference").oriented_codes(c2, e_a, e_b)
    assert c2.calls == 2   # project(a-b) and project(b-a)


def test_prompt_rep_raises_clear_contrast_error(proj, ab):
    e_a, e_b = ab
    with pytest.raises(ValueError, match="prompt lens has no"):
        get_lens_rep("prompt").contrast_codes(proj, e_a, e_b)


def test_unknown_input_rep_lists_available():
    with pytest.raises(ValueError, match="difference"):   # message lists available reps
        get_lens_rep("bogus")


def test_contrastive_flags():
    assert get_lens_rep("difference").contrastive is True
    assert get_lens_rep("individual").contrastive is True
    assert get_lens_rep("prompt").contrastive is False
