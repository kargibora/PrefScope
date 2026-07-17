"""Artifact-loading safety: SAE checkpoints must load without executing pickles.

``torch.load(weights_only=False)`` unpickles arbitrary objects, i.e. arbitrary code —
unacceptable for lenses that users download and share. The production loaders must use
``weights_only=True`` (torch's safe unpickler) and still round-trip our
``{"state_dict": ..., "config": ...}`` checkpoint.
"""
from __future__ import annotations

import inspect

import numpy as np

from prefscope.encode.sae import SAEProjector
from prefscope.sae.train import train_sae


def _tiny_lens(tmp_path):
    """Train a tiny BatchTopK SAE and save it exactly as the pipeline does."""
    import torch
    rng = np.random.RandomState(0)
    X = rng.randn(200, 8).astype(np.float32)
    model, config, _ = train_sae(X[:160], X[160:], m_total=8, k=2,
                                 matryoshka_prefix=(4,), n_epochs=3, min_epochs=3,
                                 patience=3, batch=32, device="cpu", seed=0)
    ckpt = tmp_path / "sae_model.pt"
    torch.save({"state_dict": model.state_dict(), "config": config}, ckpt)
    return ckpt


def test_projector_loads_checkpoint_with_safe_unpickler(tmp_path):
    ckpt = _tiny_lens(tmp_path)
    proj = SAEProjector(ckpt, device="cpu")           # must not need weights_only=False
    z = proj.project(np.random.RandomState(1).randn(5, 8).astype(np.float32))
    assert z.shape == (5, 8)


def test_no_unsafe_torch_load_in_loaders():
    """The projector's source must not opt out of the safe unpickler."""
    src = inspect.getsource(SAEProjector.__init__)
    assert "weights_only=False" not in src
    assert "weights_only=True" in src
