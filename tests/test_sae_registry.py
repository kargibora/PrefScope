"""SAE architecture as a registry component (kind "sae").

Mirrors the interpreter/verifier/clusterer registry pattern: built-ins are
registered, unknown names raise a friendly ValueError, and a user can plug in a
custom nn.Module SAE end-to-end. Also guards the SAEProjector refactor against
numerical drift vs. calling the model's own ``encode``.
"""
import numpy as np
import pytest
import torch

import prefscope.sae.model  # noqa: F401  (registers built-in "sae" components)
from prefscope.core import registry
from prefscope.encode.sae import SAEProjector
from prefscope.sae.model import BatchTopKSAE
from prefscope.sae.train import train_sae


def _data(n=240, d=8, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n, 3)) @ rng.standard_normal((3, d))
    return (base + 0.05 * rng.standard_normal((n, d))).astype(np.float32)


def test_builtins_registered():
    avail = set(registry.available("sae"))
    assert {"batchtopk", "simple-topk", "jumprelu"} <= avail


def test_unknown_sae_type_lists_options():
    X = _data()
    with pytest.raises(ValueError) as ei:
        train_sae(X[:200], X[200:], m_total=8, k=2, n_epochs=1, device="cpu",
                  sae_type="does-not-exist")
    # the friendly error enumerates the registered alternatives
    assert "batchtopk" in str(ei.value)


# A user-defined SAE that is NOT one of the built-ins: identical math to the base
# class but registered under a new name, proving the plug-in path works end to end.
@registry.register("sae", "test-custom")
class _CustomSAE(BatchTopKSAE):
    pass


def test_custom_sae_plugs_in_end_to_end(tmp_path):
    X = _data()
    model, config, log = train_sae(
        X[:200], X[200:], m_total=8, k=2, matryoshka_prefix=(4,),
        n_epochs=4, min_epochs=4, patience=4, batch=64, device="cpu", seed=0,
        sae_type="test-custom")
    assert isinstance(model, _CustomSAE)
    assert config["sae_type"] == "test-custom"

    ckpt = tmp_path / "sae_model.pt"
    torch.save({"state_dict": model.state_dict(), "config": config}, ckpt)
    proj = SAEProjector(ckpt, device="cpu")
    assert proj.sae_type == "test-custom"
    z = proj.project(X[:10])
    assert z.shape == (10, 8)
    assert (z != 0).any()                       # the custom lens actually produces codes


def test_projector_matches_model_encode(tmp_path):
    """Refactor guard: SAEProjector.project must equal the model's own encode()."""
    X = _data(seed=3)
    model, config, _ = train_sae(
        X[:200], X[200:], m_total=8, k=2, matryoshka_prefix=(4,),
        n_epochs=4, min_epochs=4, patience=4, batch=64, device="cpu", seed=0,
        sae_type="batchtopk")
    ckpt = tmp_path / "sae_model.pt"
    torch.save({"state_dict": model.state_dict(), "config": config}, ckpt)
    proj = SAEProjector(ckpt, device="cpu")

    xq = X[:16]
    z_proj = proj.project(xq)                    # no whitener in this lens
    with torch.no_grad():
        z_ref = model.eval().encode(torch.from_numpy(xq)).numpy()
    np.testing.assert_allclose(z_proj, z_ref, atol=1e-6)
