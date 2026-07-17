import json

import numpy as np
import pandas as pd
import pytest

from prefscope.interpret.io import load_lens_battles


def _make_lens(tmp_path):
    pd.DataFrame({"instruction_id": ["2", "0", "1"],
                  "model_a": ["X", "X", "Y"]}).to_parquet(tmp_path / "battles.parquet")
    np.save(tmp_path / "z_diff.npy", np.arange(3 * 4, dtype=np.float32).reshape(3, 4))
    (tmp_path / "manifest.json").write_text(json.dumps({"input_rep": "difference"}))
    ann = {"per_sample": [
        {"instruction_id": "0", "model_a": "X", "model_b": "Y", "instruction": "p0",
         "completion_a": "a0", "completion_b": "b0", "judge_pref": 1.0},
        {"instruction_id": "1", "model_a": "Y", "model_b": "X", "instruction": "p1",
         "completion_a": "a1", "completion_b": "b1", "judge_pref": 0.0},
        {"instruction_id": "2", "model_a": "X", "model_b": "Y", "instruction": "p2",
         "completion_a": "a2", "completion_b": "b2", "judge_pref": 0.5},
    ]}
    apath = tmp_path / "ann.json"
    apath.write_text(json.dumps(ann))
    return apath


def test_load_aligns_battles_to_zdiff_order(tmp_path):
    ann = _make_lens(tmp_path)
    battles, z_diff, manifest = load_lens_battles(tmp_path, ann)
    assert list(battles["instruction_id"]) == ["2", "0", "1"]
    assert list(battles["completion_a"]) == ["a2", "a0", "a1"]
    assert z_diff.shape == (3, 4)
    assert manifest["input_rep"] == "difference"


def test_load_raises_when_annotation_missing_a_lens_battle(tmp_path):
    ann = _make_lens(tmp_path)
    data = json.loads(ann.read_text())
    data["per_sample"] = [s for s in data["per_sample"] if s["instruction_id"] != "2"]
    ann.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="missing"):
        load_lens_battles(tmp_path, ann)


def test_load_from_corpus_aligns_to_zdiff_order(tmp_path):
    _make_lens(tmp_path)  # writes battles.parquet (ids 2,0,1) + z_diff + manifest
    from prefscope.data.corpus import normalize, write_corpus
    raw = pd.DataFrame({
        "prompt": ["p0", "p1", "p2"], "model_a": ["X", "Y", "X"],
        "model_b": ["Y", "X", "Y"], "completion_a": ["a0", "a1", "a2"],
        "completion_b": ["b0", "b1", "b2"],
    })
    corp = normalize(raw, "lmarena-100k")
    corp["battle_id"] = ["0", "1", "2"]   # match the lens instruction_ids
    write_corpus(corp, tmp_path / "corp.parquet")

    battles, z_diff, _ = load_lens_battles(tmp_path, corpus=str(tmp_path / "corp.parquet"))
    assert list(battles["instruction_id"]) == ["2", "0", "1"]
    assert list(battles["completion_a"]) == ["a2", "a0", "a1"]


def test_load_requires_exactly_one_source(tmp_path):
    _make_lens(tmp_path)
    with pytest.raises(ValueError, match="exactly one"):
        load_lens_battles(tmp_path)
