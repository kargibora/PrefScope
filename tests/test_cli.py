import json

import pandas as pd

from prefscope import __main__ as cli


def _write_annotations(tmp_path):
    data = {"per_sample": [
        {"instruction_id": "1", "model_a": "X", "model_b": "Y",
         "instruction": "p1", "completion_a": "a1", "completion_b": "b1",
         "judge_pref": 1.0},
        {"instruction_id": "2", "model_a": "Y", "model_b": "X",
         "instruction": "p2", "completion_a": "a2", "completion_b": "b2",
         "judge_pref": 0.0},
    ]}
    path = tmp_path / "ann.json"
    path.write_text(json.dumps(data))
    return path


def test_inspect_command_returns_summary(tmp_path, capsys):
    path = _write_annotations(tmp_path)
    rc = cli.main(["inspect", "--annotations", str(path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "n_battles" in out
    assert "2" in out


def test_inspect_command_accepts_corpus(tmp_path, capsys):
    corpus = tmp_path / "corpus.parquet"
    pd.DataFrame({
        "battle_id": ["a", "b"], "source": ["s", "s"], "language": ["en", "de"],
        "prompt": ["p1", "p2"], "model_a": ["X", "Y"], "model_b": ["Y", "X"],
        "completion_a": ["a1", "a2"], "completion_b": ["b1", "b2"],
    }).to_parquet(corpus)
    rc = cli.main(["inspect", "--corpus", str(corpus)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "n_battles" in out and "langs" in out


def test_inspect_command_requires_exactly_one_source(tmp_path):
    assert cli.main(["inspect"]) == 2   # neither --corpus nor --annotations


def test_build_lens_command_dispatches(tmp_path, monkeypatch):
    path = _write_annotations(tmp_path)
    captured = {}

    def fake_build_lens(battles, embedder, out_dir, **kw):
        captured["n"] = len(battles)
        captured["out_dir"] = str(out_dir)
        captured["m_total"] = kw.get("m_total")
        captured["matryoshka_prefix"] = kw.get("matryoshka_prefix")
        captured["input_rep"] = kw.get("input_rep")
        return {"n_battles": len(battles)}

    monkeypatch.setattr("prefscope.pipeline.build_lens.build_lens", fake_build_lens)
    rc = cli.main(["build-lens", "--annotations", str(path),
                   "--out", str(tmp_path / "lens"), "--m-total", "32",
                   "--matryoshka-prefix", "8", "16", "--device", "cpu"])
    assert rc == 0
    assert captured["n"] == 2
    assert captured["m_total"] == 32
    assert captured["matryoshka_prefix"] == (8, 16)
    assert captured["out_dir"].endswith("lens")
    # default input_rep must be "difference"
    assert captured["input_rep"] == "difference"


def test_build_lens_input_rep_individual_flows_through(tmp_path, monkeypatch):
    path = _write_annotations(tmp_path)
    captured = {}

    def fake_build_lens(battles, embedder, out_dir, **kw):
        captured["input_rep"] = kw.get("input_rep")
        return {"n_battles": len(battles)}

    monkeypatch.setattr("prefscope.pipeline.build_lens.build_lens", fake_build_lens)
    rc = cli.main(["build-lens", "--annotations", str(path),
                   "--out", str(tmp_path / "lens"),
                   "--input-rep", "individual", "--device", "cpu"])
    assert rc == 0
    assert captured["input_rep"] == "individual"


def test_build_lens_matryoshka_prefix_defaults_to_8(tmp_path, monkeypatch):
    path = _write_annotations(tmp_path)
    captured = {}

    def fake_build_lens(battles, embedder, out_dir, **kw):
        captured["matryoshka_prefix"] = kw.get("matryoshka_prefix")
        return {"n_battles": len(battles)}

    monkeypatch.setattr("prefscope.pipeline.build_lens.build_lens", fake_build_lens)
    rc = cli.main(["build-lens", "--annotations", str(path),
                   "--out", str(tmp_path / "lens"), "--device", "cpu"])
    assert rc == 0
    assert captured["matryoshka_prefix"] == (8,)
