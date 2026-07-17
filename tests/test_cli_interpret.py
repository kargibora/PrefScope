import json

import numpy as np
import pandas as pd

from prefscope import __main__ as cli


def _lens_and_ann(tmp_path):
    pd.DataFrame({"instruction_id": ["0", "1"], "model_a": ["X", "Y"]}
                 ).to_parquet(tmp_path / "battles.parquet")
    np.save(tmp_path / "z_diff.npy", np.array([[1.0, 0.0], [-1.0, 0.0]], np.float32))
    (tmp_path / "manifest.json").write_text(json.dumps({"input_rep": "difference"}))
    ann = {"per_sample": [
        {"instruction_id": "0", "model_a": "X", "model_b": "Y", "instruction": "p0",
         "completion_a": "a0", "completion_b": "b0", "judge_pref": 1.0},
        {"instruction_id": "1", "model_a": "Y", "model_b": "X", "instruction": "p1",
         "completion_a": "a1", "completion_b": "b1", "judge_pref": 0.0},
    ]}
    apath = tmp_path / "ann.json"
    apath.write_text(json.dumps(ann))
    return apath


def test_interpret_name_writes_csv(tmp_path, monkeypatch):
    ann = _lens_and_ann(tmp_path)
    captured = {}

    def fake_name_features(battles, z_diff, client, **kw):
        captured["n_rows"] = len(battles)
        captured["abbreviate"] = kw.get("abbreviate")
        return pd.DataFrame({"feature_id": [0, 1], "concept": ["x", "y"],
                             "concept_abbrev": ["", ""],
                             "n_active": [1, 1], "n_zero": [0, 0]})

    # dispatch now goes CLI -> registry strategy -> name_features (looked up lazily)
    monkeypatch.setattr("prefscope.interpret.name.name_features", fake_name_features)
    monkeypatch.setattr(cli, "LLMClient", lambda **kw: object())
    out_csv = tmp_path / "feature_names.csv"
    rc = cli.main(["interpret", "name", "--lens-dir", str(tmp_path),
                   "--annotations", str(ann), "--out", str(out_csv),
                   "--model", "deepseek/deepseek-v3.2"])
    assert rc == 0
    assert out_csv.exists()
    assert captured["n_rows"] == 2
