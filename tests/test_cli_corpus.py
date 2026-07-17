import numpy as np
import pandas as pd

from prefscope import __main__ as cli
from prefscope.data.corpus import normalize


def _norm(source):
    raw = pd.DataFrame({
        "prompt": ["q1", "q2"], "model_a": ["A", "A"], "model_b": ["B", "B"],
        "completion_a": ["ca1", "ca2"], "completion_b": ["cb1", "cb2"],
    })
    return normalize(raw, source)


def test_build_corpus_merges_sources(tmp_path, monkeypatch):
    calls = []

    def fake_load_arena(source, **kw):
        calls.append((source, kw.get("limit")))
        return _norm(source)

    monkeypatch.setattr("prefscope.data.arenas.load_arena", fake_load_arena)
    out = tmp_path / "corp.parquet"
    rc = cli.main(["build-corpus", "--source", "lmarena-100k", "lmarena-140k",
                   "--out", str(out), "--limit", "5"])
    assert rc == 0
    assert out.exists()
    # both arenas have identical content -> deduped to 2 rows
    written = pd.read_parquet(out)
    assert len(written) == 2
    assert {c[0] for c in calls} == {"lmarena-100k", "lmarena-140k"}
    assert calls[0][1] == 5


def test_build_corpus_unknown_source(tmp_path, monkeypatch):
    rc = cli.main(["build-corpus", "--source", "nope",
                   "--out", str(tmp_path / "x.parquet")])
    assert rc == 2


def test_build_lens_reads_corpus(tmp_path, monkeypatch):
    from prefscope.data.corpus import write_corpus
    write_corpus(_norm("lmarena-100k"), tmp_path / "corp.parquet")

    captured = {}

    def fake_build_lens(battles, embedder, out, **kw):
        captured["n"] = len(battles)
        captured["has_iid"] = "instruction_id" in battles.columns
        return {"ok": True}

    monkeypatch.setattr("prefscope.pipeline.build_lens.build_lens", fake_build_lens)
    monkeypatch.setattr(cli, "Embedder", lambda *a, **k: object())
    monkeypatch.setattr(cli, "NpyCache", lambda *a, **k: object())

    rc = cli.main(["build-lens", "--corpus", str(tmp_path / "corp.parquet"),
                   "--out", str(tmp_path / "lens"), "--device", "cpu"])
    assert rc == 0
    assert captured["n"] == 2
    assert captured["has_iid"] is True


def test_build_lens_rejects_both_inputs(tmp_path):
    rc = cli.main(["build-lens", "--corpus", "c.parquet",
                   "--annotations", "a.json", "--out", str(tmp_path / "l")])
    assert rc == 2
