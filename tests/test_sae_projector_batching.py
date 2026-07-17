"""SAEProjector.project/reconstruct process rows in chunks (bounded GPU memory)
and are chunk-invariant — the encoder is row-independent, so batching only changes
results by float-reduction noise, never the selected support."""
import numpy as np
import torch

import prefscope.adapters  # noqa: F401  (registers the SAE classes)
from prefscope.encode.sae import SAEProjector
from prefscope.sae.train import train_sae


def _projector(tmp_path, n=500, d=16, m=8):
    X = np.random.randn(n, d).astype("float32")
    model, config, _ = train_sae(X[: n - 50], X[n - 50:], m_total=m, k=4,
                                 n_epochs=2, device="cpu")
    torch.save({"state_dict": model.state_dict(), "config": config},
               tmp_path / "sae_model.pt")
    return SAEProjector(tmp_path / "sae_model.pt"), X


def test_project_is_chunk_invariant(tmp_path):
    p, X = _projector(tmp_path)
    z_one = p.project(X, batch=10_000)        # single chunk
    z_many = p.project(X, batch=13)           # many chunks
    assert np.array_equal(z_one != 0, z_many != 0)     # identical selected support
    assert np.allclose(z_one, z_many, atol=1e-5)       # identical up to float noise


def test_reconstruct_is_chunk_invariant(tmp_path):
    p, X = _projector(tmp_path)
    z = p.project(X)
    assert np.allclose(p.reconstruct(z, batch=10_000),
                       p.reconstruct(z, batch=11), atol=1e-5)


def test_project_handles_empty_and_large_batch(tmp_path):
    p, X = _projector(tmp_path, n=300)
    assert p.project(np.empty((0, 16), dtype="float32")).shape == (0, p.m_total)
    assert p.project(X, batch=1).shape == (300, p.m_total)   # batch smaller than N
