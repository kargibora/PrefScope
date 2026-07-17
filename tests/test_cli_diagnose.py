import json

import numpy as np
import pandas as pd

from prefscope import __main__ as cli


def _lens_and_ann(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"input_rep": "difference"}))
    ann = {"per_sample": [
        {"instruction_id": "0", "model_a": "M", "model_b": "Y", "instruction": "p0",
         "completion_a": "a0", "completion_b": "b0", "judge_pref": 1.0},
        {"instruction_id": "1", "model_a": "Y", "model_b": "M", "instruction": "p1",
         "completion_a": "a1", "completion_b": "b1", "judge_pref": 0.0},
    ]}
    apath = tmp_path / "ann.json"
    apath.write_text(json.dumps(ann))
    return apath


def test_diagnose_writes_csv(tmp_path, monkeypatch):
    ann = _lens_and_ann(tmp_path)
    captured = {}

    def fake_run_diagnose(battles, model, embedder, projector, **kw):
        captured["model"] = model
        captured["input_rep"] = kw.get("input_rep")
        df = pd.DataFrame({"feature_id": [0, 1], "concept": ["a", "b"],
                           "concept_abbrev": ["", ""],
                           "net_direction": [0.4, -0.3],
                           "fire_rate": [0.6, 0.5],
                           "outcome_assoc": [0.1, float("nan")]})
        return df, {"model": model, "n_battles": 2, "win_rate": 1.0, "n_features": 2}

    monkeypatch.setattr("prefscope.pipeline.diagnose.run_diagnose", fake_run_diagnose)
    monkeypatch.setattr(cli, "Embedder", lambda *a, **k: object())
    monkeypatch.setattr("prefscope.encode.sae.SAEProjector", lambda *a, **k: object())

    out_csv = tmp_path / "diagnosis.csv"
    rc = cli.main(["diagnose", "--lens-dir", str(tmp_path),
                   "--annotations", str(ann), "--model", "M",
                   "--out", str(out_csv), "--device", "cpu"])
    assert rc == 0
    assert out_csv.exists()
    assert captured["model"] == "M"
    assert captured["input_rep"] == "difference"
    written = pd.read_csv(out_csv)
    assert list(written["feature_id"]) == [0, 1]
