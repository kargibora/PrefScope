import numpy as np
import pandas as pd
import pytest

from prefscope.api.loaded_lens import LoadedLens
from prefscope.core.dataset import Dataset
from prefscope.core.types import PairItem
from prefscope import analysis


class _PairData(Dataset):
    def __iter__(self):
        yield PairItem(id="1", x="q", y_a="aaaa", y_b="b", pref=1.0, model_a="m1", model_b="m2")
        yield PairItem(id="2", x="q", y_a="a", y_b="bbbb", pref=0.0, model_a="m1", model_b="m2")


class _SingleData(Dataset):
    def __iter__(self):
        yield PairItem(id="1", x="q", y_a="a")            # y_b is None


class FakeEmbedder:
    """Deterministic: each row is filled with the completion's length."""
    def encode(self, prompts, completions):
        return np.array([[float(len(c))] * 4 for c in completions], dtype=np.float32)


class FakeProjector:
    m_total = 3
    def project(self, x):
        x = np.asarray(x, dtype=np.float32)
        # codes: [col0, -col0, 0] — deterministic, sign-carrying
        return np.stack([x[:, 0], -x[:, 0], np.zeros(len(x))], axis=1).astype(np.float32)


def _names():
    return pd.DataFrame({"feature_id": [0, 1, 2], "concept": ["a", "b", "c"],
                         "fidelity_pass": [True, False, True]})


def _lens(manifest=None, names=None):
    return LoadedLens(FakeProjector(), FakeEmbedder(),
                      names=names, manifest=manifest or {"input_rep": "difference"})


def test_project_shapes_and_meta():
    lens = _lens()
    codes, meta = lens.project(_PairData())
    assert codes.shape == (2, 3)
    assert list(meta.columns) == ["id", "pref", "model_a", "model_b"]
    assert list(meta["pref"]) == [1.0, 0.0]
    assert list(meta["model_a"]) == ["m1", "m1"]


def test_project_uses_difference_contrast():
    lens = _lens()
    codes, _ = lens.project(_PairData())
    # row1: e_a(len 4) - e_b(len 1) = 3 -> col0=3 ; row2: 1 - 4 = -3 -> col0=-3
    np.testing.assert_allclose(codes[:, 0], [3.0, -3.0])


def test_project_token_granularity_raises():
    lens = _lens(manifest={"input_rep": "difference", "granularity": "token"})
    with pytest.raises(ValueError, match="token-granularity"):
        lens.project(_PairData())


def test_project_single_response_raises():
    lens = _lens()
    with pytest.raises(ValueError, match="encode_pairs.*requires y_b"):
        lens.project(_SingleData())


def test_diagnose_delegates_to_analysis():
    lens = _lens(names=_names())
    codes, meta = lens.project(_PairData())
    got = lens.diagnose(codes, meta)
    exp = analysis.diagnose(codes, meta, names=_names())
    pd.testing.assert_frame_equal(got, exp)


def test_evaluate_preference_delegates():
    lens = _lens()
    codes = np.random.default_rng(0).normal(size=(60, 3)).astype(np.float32)
    meta = pd.DataFrame({"pref": (codes[:, 0] > 0).astype(float)})
    out = lens.evaluate_preference(codes, meta, seed=0)
    assert out["accuracy"] > 0.8 and "top_features" in out


def test_fidelity_feature_ids():
    assert _lens(names=_names()).fidelity_feature_ids == [0, 2]
    assert _lens().fidelity_feature_ids is None


import json as _json
import torch


def _write_synthetic_lens(tmp_path, m=3, d=4):
    """A minimal sae_model.pt matching SAEProjector's expected state_dict keys."""
    sd = {
        "encoder.weight": torch.zeros(m, d),
        "input_bias": torch.zeros(d),
        "neuron_bias": torch.zeros(m),
        "threshold": torch.tensor(0.1),
        "decoder.weight": torch.zeros(d, m),
    }
    torch.save({"state_dict": sd, "config": {"m_total": m, "k": 2}},
               tmp_path / "sae_model.pt")
    pd.DataFrame({"feature_id": list(range(m)), "concept": ["x"] * m,
                  "fidelity_pass": [True] * m}).to_csv(
        tmp_path / "feature_names.csv", index=False)
    (tmp_path / "manifest.json").write_text(_json.dumps(
        {"input_rep": "difference", "embed_model_id": "Qwen/Qwen3-Embedding-0.6B"}))


def test_from_dir_loads_projector_names_manifest(tmp_path):
    _write_synthetic_lens(tmp_path)
    lens = LoadedLens.from_dir(tmp_path, device="cpu")
    assert lens.projector.m_total == 3 and lens.projector.input_dim == 4
    assert lens.input_rep == "difference"
    assert lens.names is not None and lens.fidelity_feature_ids == [0, 1, 2]
    assert lens.embedder is not None        # constructed lazily; no model download
