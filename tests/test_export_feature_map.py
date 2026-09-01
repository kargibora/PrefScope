import json

import numpy as np
import pandas as pd
import torch

from prefscope.viewer_export import export_feature_map


def _lens(tmp_path, directions: np.ndarray):
    lens = tmp_path / "lens"
    lens.mkdir()
    # decoder.weight is (embedding dimension, feature count)
    torch.save(
        {"state_dict": {"decoder.weight": torch.tensor(directions.T)}, "config": {}},
        lens / "sae_model.pt",
    )
    (lens / "manifest.json").write_text(json.dumps({
        "schema_version": 2, "m_total": int(len(directions)), "k": 1,
        "input_dim": int(directions.shape[1]), "input_rep": "individual",
    }))
    return lens


def test_feature_map_contains_every_decoder_axis_and_does_not_need_names(tmp_path):
    directions = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ], dtype=np.float32)
    lens = _lens(tmp_path, directions)
    features = pd.DataFrame({
        "feature_id": [0, 2],
        "concept": ["first", np.nan],
        "fidelity_pass": [True, False],
    })

    out = export_feature_map(lens, features, seed=3)

    assert out is not None
    assert out["n_total"] == 4
    assert out["n_named"] == 1
    assert out["n_verified"] == 1
    assert {p["feature_id"] for p in out["points"]} == set(range(4))
    assert all(np.isfinite([p["x"], p["y"]]).all() for p in out["points"])
    assert out["basis"] == "sae_decoder_direction"


def test_feature_map_retains_zero_decoder_axes_but_marks_them(tmp_path):
    directions = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
    ], dtype=np.float32)
    out = export_feature_map(_lens(tmp_path, directions), pd.DataFrame({
        "feature_id": [0, 1, 2],
    }))

    assert out is not None and out["n_zero_decoder"] == 1
    point = next(p for p in out["points"] if p["feature_id"] == 2)
    assert point["zero_decoder"] is True
