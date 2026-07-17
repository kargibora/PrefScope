"""JumpReLU SAE (arXiv:2407.14435): trains, gates per-feature, projects one-sided."""
import math

import numpy as np
import torch

from prefscope.encode.sae import SAEProjector
from prefscope.sae.model import JumpReLUSAE
from prefscope.sae.train import train_sae


def _data(n=200, d=16, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n, 4)) @ rng.standard_normal((4, d))).astype(np.float32)


def test_jumprelu_trains_and_projects_one_sided(tmp_path):
    X = _data()
    model, config, log = train_sae(
        X[:160], X[160:], m_total=24, k=4, matryoshka_prefix=(),
        sae_type="jumprelu", sparsity_coef=1e-3, bandwidth=1e-3,
        n_epochs=5, min_epochs=1, batch=32, device="cpu", seed=0)

    assert isinstance(model, JumpReLUSAE)
    assert config["sae_type"] == "jumprelu"
    assert "sparsity_coef" in config and "bandwidth" in config

    # round-trip through the frozen projector (the inference path used downstream)
    ckpt = tmp_path / "sae_model.pt"
    torch.save({"state_dict": model.state_dict(), "config": config}, ckpt)
    proj = SAEProjector(ckpt, device="cpu")
    assert proj.sae_type == "jumprelu" and proj.feature_threshold is not None

    z = proj.project(X[160:])
    assert z.shape == (40, 24)
    assert (z >= 0).all()                       # JumpReLU codes are one-sided non-negative
    active = float((z != 0).mean())
    assert 0.0 < active < 1.0                    # gated: neither all-dead nor all-active


def test_jumprelu_thresholds_learn():
    # the STE on log-threshold must actually move the thresholds off their init
    X = _data(seed=1)
    init = math.log(1e-3)
    model, *_ = train_sae(
        X[:160], X[160:], m_total=24, k=4, matryoshka_prefix=(),
        sae_type="jumprelu", sparsity_coef=5e-2, bandwidth=0.5,   # wide kernel -> visible grad
        n_epochs=10, min_epochs=10, batch=32, device="cpu", seed=1)
    moved = (model.log_threshold.detach() - init).abs().max().item()
    assert moved > 1e-4


def test_unknown_sae_type_still_rejected():
    import pytest
    with pytest.raises(ValueError, match="jumprelu"):
        train_sae(_data()[:160], _data()[160:], sae_type="nope",
                  n_epochs=1, device="cpu")
