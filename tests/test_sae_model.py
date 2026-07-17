import numpy as np
import pytest
import torch

from prefscope.sae.model import BatchTopKSAE, SimpleTopKSAE, encode_in_batches

REQUIRED_KEYS = {"encoder.weight", "decoder.weight", "input_bias",
                 "neuron_bias", "threshold", "steps_since_activation"}


def _model(seed=0):
    torch.manual_seed(seed)
    return BatchTopKSAE(input_dim=8, m_total_neurons=8, k_active_neurons=2,
                        matryoshka_prefix_lengths=[4])


def test_state_dict_has_model_keys():
    sd = _model().state_dict()
    assert REQUIRED_KEYS.issubset(set(sd.keys()))
    assert sd["encoder.weight"].shape == (8, 8)
    assert sd["decoder.weight"].shape == (8, 8)
    assert sd["input_bias"].shape == (8,)
    assert sd["neuron_bias"].shape == (8,)
    assert sd["threshold"].ndim == 0


def test_forward_train_uses_batch_topk_budget():
    m = _model().train()
    x = torch.randn(5, 8)
    recon, info = m(x)
    assert recon.shape == (5, 8)
    assert int((info["activations"] != 0).sum()) <= 2 * 5


def test_forward_eval_threshold_select_shape():
    m = _model().eval()
    x = torch.randn(5, 8)
    recon, info = m(x)
    assert recon.shape == (5, 8)
    assert info["activations"].shape == (5, 8)


def test_encode_in_batches_matches_threshold_select():
    m = _model().eval()
    X = np.random.randn(7, 8).astype(np.float32)
    z = encode_in_batches(m, X, batch=3, device=torch.device("cpu"))
    assert z.shape == (7, 8)
    with torch.no_grad():
        xt = torch.from_numpy(X)
        z_ref = m._threshold_select(m.encode_pre(xt)).numpy()
    np.testing.assert_allclose(z, z_ref, atol=1e-6)


def test_simple_topk_sae_smoke():
    torch.manual_seed(42)
    m = SimpleTopKSAE(input_dim=8, m_total_neurons=8, k_active_neurons=2)
    x = torch.randn(5, 8)

    m.train()
    recon_train, info_train = m(x)
    assert recon_train.shape == (5, 8)
    assert info_train["activations"].shape == (5, 8)

    m.eval()
    with torch.no_grad():
        recon_eval, info_eval = m(x)
    assert recon_eval.shape == (5, 8)
    assert info_eval["activations"].shape == (5, 8)


def test_matryoshka_norm_mse_averages_prefix_levels():
    torch.manual_seed(0)
    m = _model()                       # prefixes become [4, 8]
    x = torch.randn(6, 8)
    recon, info = m(x)
    activ = info["activations"]
    partial = activ.clone(); partial[:, 4:] = 0
    prefix4 = float(m._normalized_mse(m.decoder(partial) + m.input_bias, x))
    full = float(m._normalized_mse(recon, x))
    matry = float(m.matryoshka_norm_mse(recon, activ, x))
    # objective is the mean of the prefix-4 and full-code reconstruction levels
    assert matry == pytest.approx((prefix4 + full) / 2, rel=1e-5)
    assert matry >= full - 1e-6        # coarse prefix never improves on the full code


def test_matryoshka_norm_mse_mode_independent():
    torch.manual_seed(0)
    m = _model()
    x = torch.randn(6, 8)
    recon, info = m(x)
    activ = info["activations"]
    m.train(); a = float(m.matryoshka_norm_mse(recon, activ, x))
    m.eval();  b = float(m.matryoshka_norm_mse(recon, activ, x))
    assert a == pytest.approx(b)       # no longer gated on self.training (the bug)


def test_no_prefix_matryoshka_is_plain_norm_mse():
    torch.manual_seed(0)
    m = BatchTopKSAE(input_dim=8, m_total_neurons=8, k_active_neurons=2,
                     matryoshka_prefix_lengths=None)
    x = torch.randn(6, 8)
    recon, info = m(x)
    assert float(m.matryoshka_norm_mse(recon, info["activations"], x)) == \
        pytest.approx(float(m._normalized_mse(recon, x)))
