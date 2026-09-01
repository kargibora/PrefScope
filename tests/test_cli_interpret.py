import json

import numpy as np
import pandas as pd

from prefscope import __main__ as cli
from prefscope.cli import common as cli_common


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
    monkeypatch.setattr(cli_common, "LLMClient", lambda **kw: object())
    out_csv = tmp_path / "feature_names.csv"
    rc = cli.main(["interpret", "name", "--lens-dir", str(tmp_path),
                   "--annotations", str(ann), "--out", str(out_csv),
                   "--model", "deepseek/deepseek-v3.2"])
    assert rc == 0
    assert out_csv.exists()
    assert captured["n_rows"] == 2


def test_interpret_name_checkpoints_and_resumes_after_interruption(tmp_path, monkeypatch):
    ann = _lens_and_ann(tmp_path)
    out_csv = tmp_path / "resumable_names.csv"
    first_features = []

    def interrupted_name(battles, z_diff, client, **kw):
        first_features.extend(kw["features"])
        row = {"feature_id": int(kw["features"][0]), "concept": "saved",
               "status": "ok", "confidence": "high"}
        kw["on_result"](row)
        raise RuntimeError("connection lost")

    monkeypatch.setattr("prefscope.interpret.name.name_features", interrupted_name)
    monkeypatch.setattr(cli_common, "LLMClient", lambda **kw: object())
    argv = ["interpret", "name", "--lens-dir", str(tmp_path),
            "--annotations", str(ann), "--out", str(out_csv), "--model", "m"]
    import pytest
    with pytest.raises(RuntimeError, match="connection lost"):
        cli.main(argv)

    saved = pd.read_csv(out_csv)
    assert saved["feature_id"].tolist() == [0]
    assert (tmp_path / "resumable_names.resume.json").exists()
    assert first_features == [0, 1]

    resumed_features = []

    def resumed_name(battles, z_diff, client, **kw):
        resumed_features.extend(kw["features"])
        rows = [{"feature_id": int(f), "concept": f"feature {f}",
                 "status": "ok", "confidence": "high"} for f in kw["features"]]
        for row in rows:
            kw["on_result"](row)
        return pd.DataFrame(rows)

    monkeypatch.setattr("prefscope.interpret.name.name_features", resumed_name)
    assert cli.main(argv) == 0
    assert resumed_features == [1]
    assert pd.read_csv(out_csv)["feature_id"].tolist() == [0, 1]


def test_interpret_name_fresh_restarts_completed_output(tmp_path, monkeypatch):
    ann = _lens_and_ann(tmp_path)
    out_csv = tmp_path / "names.csv"
    calls = []

    def fake_name(battles, z_diff, client, **kw):
        calls.append(list(kw["features"]))
        rows = [{"feature_id": int(f), "concept": "x"} for f in kw["features"]]
        for row in rows:
            kw["on_result"](row)
        return pd.DataFrame(rows)

    monkeypatch.setattr("prefscope.interpret.name.name_features", fake_name)
    monkeypatch.setattr(cli_common, "LLMClient", lambda **kw: object())
    argv = ["interpret", "name", "--lens-dir", str(tmp_path),
            "--annotations", str(ann), "--out", str(out_csv), "--model", "m"]

    assert cli.main(argv) == 0
    assert cli.main(argv) == 0
    assert cli.main(argv + ["--fresh"]) == 0
    assert calls == [[0, 1], [0, 1]]


def test_interpret_verify_resumes_completed_rows(tmp_path, monkeypatch):
    ann = _lens_and_ann(tmp_path)
    names = tmp_path / "names_input.csv"
    pd.DataFrame({"feature_id": [0, 1], "concept": ["x", "y"],
                  "status": ["ok", "ok"]}).to_csv(names, index=False)
    out = tmp_path / "fidelity.csv"
    calls = []

    def fake_verify(battles, z_diff, names_df, client, **kw):
        calls.append(list(kw["features"]))
        rows = []
        for f in kw["features"]:
            row = {"feature_id": int(f), "concept": "x", "correlation": 0.9,
                   "p_bonferroni": 0.01, "fidelity_pass": True}
            rows.append(row)
            kw["on_result"](row)
        return pd.DataFrame(rows)

    monkeypatch.setattr("prefscope.interpret.verify.verify_features", fake_verify)
    monkeypatch.setattr(cli_common, "LLMClient", lambda **kw: object())
    argv = ["interpret", "verify", "--lens-dir", str(tmp_path),
            "--annotations", str(ann), "--names", str(names), "--out", str(out),
            "--model", "m"]

    assert cli.main(argv) == 0
    assert cli.main(argv) == 0
    assert calls == [[0, 1]]
    assert pd.read_csv(out)["feature_id"].tolist() == [0, 1]
