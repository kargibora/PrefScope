import numpy as np
import torch

from prefscope.encode.sae import SAEProjector


def _fake_checkpoint(path):
    # D=3, M=2. encoder maps (x - b_in) to 2 neurons; threshold prunes small acts.
    state_dict = {
        "input_bias": torch.zeros(3),
        "neuron_bias": torch.zeros(2),
        "threshold": torch.tensor(0.5),
        "encoder.weight": torch.tensor([[1.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0]]),
        "decoder.weight": torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),  # (D=3, M=2)
        "steps_since_activation": torch.zeros(2),
    }
    torch.save({"state_dict": state_dict,
                "config": {"input_dim": 3, "m_total_neurons": 2,
                           "k_active_neurons": 1}}, path)


def test_projector_threshold_select(tmp_path):
    ckpt = tmp_path / "sae_model.pt"
    _fake_checkpoint(ckpt)
    proj = SAEProjector(ckpt)
    assert proj.input_dim == 3
    assert proj.m_total == 2
    # x = [2.0, 0.3, 9.0]; pre = [2.0, 0.3]; threshold 0.5 -> [2.0, 0.0]
    x = np.array([[2.0, 0.3, 9.0]], dtype=np.float32)
    z = proj.project(x)
    assert z.shape == (1, 2)
    np.testing.assert_allclose(z, [[2.0, 0.0]], atol=1e-6)


def test_projector_reconstruct_and_residual(tmp_path):
    ckpt = tmp_path / "sae_model.pt"
    _fake_checkpoint(ckpt)
    proj = SAEProjector(ckpt)
    # reconstruct: z=[2,0] -> z @ W_dec.T = [2,0,0]
    rec = proj.reconstruct(np.array([[2.0, 0.0]], dtype=np.float32))
    np.testing.assert_allclose(rec, [[2.0, 0.0, 0.0]], atol=1e-6)
    # residual: x=[2,0.3,9] -> project -> z=[2,0] -> recon=[2,0,0] -> resid=[0,0.3,9]
    resid = proj.residual_norm(np.array([[2.0, 0.3, 9.0]], dtype=np.float32))
    np.testing.assert_allclose(resid, [np.sqrt(0.3**2 + 9.0**2)], atol=1e-5)


def test_projector_loads_from_dir(tmp_path):
    _fake_checkpoint(tmp_path / "sae_model.pt")
    proj = SAEProjector(tmp_path)  # directory accepted
    assert proj.m_total == 2


import pytest
from prefscope.config import CONFIG


@pytest.mark.slow
def test_real_frozen_sae_shapes():
    if not (CONFIG.frozen_sae_dir / "sae_model.pt").exists():
        pytest.skip("frozen SAE checkpoint not present")
    proj = SAEProjector(CONFIG.frozen_sae_dir)
    assert proj.m_total == 128
    assert proj.input_dim == 1024
    z = proj.project(np.zeros((4, 1024), dtype=np.float32))
    assert z.shape == (4, 128)
