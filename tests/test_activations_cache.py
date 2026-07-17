import numpy as np
import pandas as pd
import pytest

from prefscope.activations.cache import ActivationCache


def test_append_finalize_reopen_roundtrip(tmp_path):
    c = ActivationCache(tmp_path, hidden_dim=4)
    a = np.arange(8, dtype=np.float32).reshape(2, 4)
    b = np.arange(12, dtype=np.float32).reshape(3, 4) + 100
    c.append(a, [{"battle_id": "x", "span": "prompt", "token_idx": i} for i in range(2)])
    c.append(b, [{"battle_id": "x", "span": "a", "token_idx": i} for i in range(3)])
    c.finalize(extra_manifest={"model_id": "m", "layer": 7})

    r = ActivationCache.open(tmp_path)
    assert r.n_tokens == 5
    assert r.hidden_dim == 4
    np.testing.assert_allclose(np.asarray(r.acts[:2]), a, atol=1e-2)
    np.testing.assert_allclose(np.asarray(r.acts[2:]), b, atol=1e-1)
    assert list(r.index["span"]) == ["prompt", "prompt", "a", "a", "a"]
    assert r.manifest["model_id"] == "m" and r.manifest["layer"] == 7
    assert r.manifest["n_tokens"] == 5 and r.manifest["hidden_dim"] == 4


def test_open_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ActivationCache.open(tmp_path)


def test_open_zero_token_cache(tmp_path):
    c = ActivationCache(tmp_path, hidden_dim=4)
    c.finalize(extra_manifest={"model_id": "m"})
    r = ActivationCache.open(tmp_path)
    assert r.n_tokens == 0
    assert r.acts.shape == (0, 4)


def test_append_after_finalize_rejected(tmp_path):
    import numpy as np
    c = ActivationCache(tmp_path, hidden_dim=2)
    c.finalize()
    with pytest.raises(RuntimeError):
        c.append(np.zeros((1, 2), dtype=np.float32), [{"battle_id": "x", "span": "a", "token_idx": 0}])


def test_write_methods_rejected_on_opened_cache(tmp_path):
    import numpy as np
    c = ActivationCache(tmp_path, hidden_dim=2)
    c.append(np.zeros((1, 2), dtype=np.float32), [{"battle_id": "x", "span": "a", "token_idx": 0}])
    c.finalize()
    r = ActivationCache.open(tmp_path)
    with pytest.raises(RuntimeError):
        r.append(np.zeros((1, 2), dtype=np.float32), [{"battle_id": "x", "span": "a", "token_idx": 0}])


from prefscope.activations.cache import train_val_row_indices


def test_train_val_split_disjoint_and_deterministic():
    tr, va = train_val_row_indices(1000, val_frac=0.1, max_train_tokens=None, seed=0)
    assert len(va) == 100
    assert len(tr) == 900
    assert set(tr).isdisjoint(set(va))
    assert sorted(tr.tolist() + va.tolist()) == list(range(1000))
    tr2, va2 = train_val_row_indices(1000, val_frac=0.1, max_train_tokens=None, seed=0)
    np.testing.assert_array_equal(tr, tr2)
    np.testing.assert_array_equal(va, va2)


def test_reservoir_cap_limits_train_only():
    tr, va = train_val_row_indices(1000, val_frac=0.1, max_train_tokens=300, seed=1)
    assert len(va) == 100          # val is never capped
    assert len(tr) == 300          # train capped to the reservoir budget
    assert set(tr).isdisjoint(set(va))
