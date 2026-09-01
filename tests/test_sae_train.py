import numpy as np
import pytest
import torch

from prefscope.sae.train import train_sae
from prefscope.encode.sae import SAEProjector
from prefscope.sae.model import BatchTopKSAE, NonnegativeBatchTopKSAE


def _data(n=240, d=8, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n, 3)) @ rng.standard_normal((3, d))
    return (base + 0.05 * rng.standard_normal((n, d))).astype(np.float32)


def test_train_returns_model_config_log():
    X = _data()
    Xtr, Xva = X[:200], X[200:]
    model, config, log = train_sae(
        Xtr, Xva, m_total=8, k=2, matryoshka_prefix=(4,),
        n_epochs=8, min_epochs=8, patience=8, batch=64, device="cpu", seed=0)
    assert config["input_dim"] == 8
    assert config["m_total_neurons"] == 8
    assert config["k_active_neurons"] == 2
    assert config["optimizer"] == "adam"
    assert config["weight_decay"] == 0.0
    assert config["activation_polarity"] == "signed"
    assert len(log) == 8
    assert log[-1]["val_norm_mse"] < log[0]["val_norm_mse"]
    assert log[-1]["val_ev"] == pytest.approx(1.0 - log[-1]["val_norm_mse"])
    assert config["calibration_l0"] == pytest.approx(2.0, rel=0.05)
    assert config["deployment_val_explained_variance"] == pytest.approx(
        1.0 - config["deployment_val_norm_mse"])


def test_auto_individual_uses_nonnegative_batchtopk_without_matryoshka():
    X = _data()
    model, config, _ = train_sae(
        X[:200], X[200:], m_total=8, k=2, input_rep="individual",
        n_epochs=2, min_epochs=2, patience=2, batch=64, device="cpu", seed=0)
    assert isinstance(model, NonnegativeBatchTopKSAE)
    assert config["sae_type"] == "batchtopk-relu"
    assert config["activation_polarity"] == "nonnegative"
    assert config["code_semantics"] == "presence"
    assert config["matryoshka_prefix_lengths"] == []
    assert (model.encode(torch.from_numpy(X[:12])) >= 0).all()


def test_train_per_batch_loop_is_numerically_identical():
    """The CPU-resident-matrix + per-batch-transfer refactor must not change
    numerics: two runs on the same seed give bit-identical weights and logs."""
    X = _data()
    Xtr, Xva = X[:200], X[200:]
    kw = dict(m_total=8, k=2, matryoshka_prefix=(4,), n_epochs=8, min_epochs=8,
              patience=8, batch=64, device="cpu", seed=0)
    m1, c1, l1 = train_sae(Xtr.copy(), Xva.copy(), **kw)
    m2, c2, l2 = train_sae(Xtr.copy(), Xva.copy(), **kw)
    for (k1, v1), (k2, v2) in zip(m1.state_dict().items(), m2.state_dict().items()):
        assert k1 == k2
        assert torch.equal(v1, v2)
    assert l1 == l2


def test_max_train_rows_caps_training_matrix(monkeypatch):
    """With max_train_rows < N the trainer must see at most that many rows."""
    import prefscope.sae.train as train_mod
    X = _data(n=300)
    Xtr, Xva = X[:260], X[260:]
    cap = 100

    seen = {}
    orig_from_numpy = torch.from_numpy

    # spy on the tensor built from the (already-capped) X_train inside train_sae
    cap_done = {"first": True}

    def spy(arr):
        # first from_numpy call inside train_sae is X_train, second is X_val
        if cap_done["first"]:
            seen["train_rows"] = arr.shape[0]
            cap_done["first"] = False
        return orig_from_numpy(arr)

    monkeypatch.setattr(train_mod.torch, "from_numpy", spy)
    train_sae(Xtr, Xva, m_total=8, k=2, matryoshka_prefix=(4,), n_epochs=2,
              min_epochs=2, patience=2, batch=32, device="cpu", seed=0,
              max_train_rows=cap)
    assert seen["train_rows"] == cap


def test_max_train_rows_none_uses_all_rows(monkeypatch):
    """Default (None) caps nothing — full train set is used."""
    import prefscope.sae.train as train_mod
    X = _data(n=300)
    Xtr, Xva = X[:260], X[260:]
    seen = {}
    orig_from_numpy = torch.from_numpy
    cap_done = {"first": True}

    def spy(arr):
        if cap_done["first"]:
            seen["train_rows"] = arr.shape[0]
            cap_done["first"] = False
        return orig_from_numpy(arr)

    monkeypatch.setattr(train_mod.torch, "from_numpy", spy)
    train_sae(Xtr, Xva, m_total=8, k=2, matryoshka_prefix=(4,), n_epochs=2,
              min_epochs=2, patience=2, batch=32, device="cpu", seed=0)
    assert seen["train_rows"] == 260


def test_train_rejects_unknown_sae_type():
    import pytest
    X = np.random.randn(40, 8).astype(np.float32)
    with pytest.raises(ValueError):
        train_sae(X[:30], X[30:], m_total=8, k=2, n_epochs=1, device="cpu",
                  sae_type="batch_topk")


def test_trained_checkpoint_loads_in_projector(tmp_path):
    X = _data()
    model, config, log = train_sae(
        X[:200], X[200:], m_total=8, k=2, matryoshka_prefix=(4,),
        n_epochs=4, min_epochs=4, patience=4, batch=64, device="cpu", seed=0)
    ckpt = tmp_path / "sae_model.pt"
    torch.save({"state_dict": model.state_dict(), "config": config}, ckpt)
    proj = SAEProjector(ckpt, device="cpu")
    z = proj.project(X[:10])
    assert z.shape == (10, 8)

    # dim guard: feeding embeddings of the wrong width must fail loudly (not a
    # cryptic torch matmul error) — this is the embedder/lens-mismatch case.
    import numpy as np
    import pytest
    bad = np.zeros((4, proj.input_dim + 1), dtype=np.float32)
    with pytest.raises(ValueError, match="input_dim"):
        proj.project(bad)


def test_legacy_checkpoint_without_sae_type_remains_signed(tmp_path):
    """The v1 fallback must preserve the numerical meaning of old artifacts."""
    torch.manual_seed(3)
    model = BatchTopKSAE(input_dim=8, m_total_neurons=8, k_active_neurons=2)
    model.threshold.fill_(0.25)
    config = {"input_dim": 8, "m_total_neurons": 8, "k_active_neurons": 2}
    ckpt = tmp_path / "sae_model.pt"
    torch.save({"state_dict": model.state_dict(), "config": config}, ckpt)
    proj = SAEProjector(ckpt)
    X = _data(n=20)
    with torch.no_grad():
        expected = model.eval().encode(torch.from_numpy(X)).numpy()
    np.testing.assert_array_equal(proj.project(X), expected)
    assert proj.sae_type == "batchtopk"
    assert proj.activation_polarity == "signed"
