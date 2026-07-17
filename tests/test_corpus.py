import pandas as pd

from prefscope.data.corpus import (
    CORPUS_COLS, load_corpus, make_battle_id, merge_corpora, normalize, write_corpus,
)


def _raw():
    return pd.DataFrame({
        "prompt": ["q1", "q2", "  ", "q4"],
        "model_a": ["A", "A", "A", "A"],
        "model_b": ["B", "B", "B", "B"],
        "completion_a": ["ca1", "ca2", "ca3", ""],   # row3 empty completion
        "completion_b": ["cb1", "cb2", "cb3", "cb4"],
    })


def test_normalize_drops_invalid_and_adds_fields():
    out = normalize(_raw(), "lmsys")
    assert list(out.columns) == CORPUS_COLS
    # row with blank prompt and row with empty completion are dropped
    assert list(out["prompt"]) == ["q1", "q2"]
    assert (out["source"] == "lmsys").all()
    assert out["battle_id"].str.len().eq(16).all()
    assert out["battle_id"].nunique() == 2


def test_normalize_carries_language():
    df = _raw().iloc[:2].copy()
    df["language"] = ["English", "French"]
    out = normalize(df, "lmsys")
    assert list(out["language"]) == ["English", "French"]
    # absent language defaults to empty string, not NaN
    out2 = normalize(_raw().iloc[:2], "lmsys")
    assert (out2["language"] == "").all()


def test_normalize_keeps_explicit_battle_id():
    df = _raw().iloc[:2].copy()
    df["battle_id"] = ["x", "y"]
    out = normalize(df, "lmsys")
    assert list(out["battle_id"]) == ["x", "y"]


def test_make_battle_id_is_content_stable_and_source_independent():
    r = {"source": "lmsys", "prompt": "q", "model_a": "A", "model_b": "B",
         "completion_a": "ca", "completion_b": "cb"}
    assert make_battle_id(r) == make_battle_id(dict(r))
    r_other = dict(r, source="comparia")  # different source, same content
    assert make_battle_id(r) == make_battle_id(r_other)
    r2 = dict(r); r2["completion_a"] = "different"
    assert make_battle_id(r) != make_battle_id(r2)


def test_merge_dedups_identical_battles_across_arenas():
    a = normalize(_raw(), "lmarena-100k")
    b = normalize(_raw(), "lmarena-140k")   # same content, different arena
    merged = merge_corpora([a, b])
    assert len(merged) == 2                  # overlap collapsed
    assert merged["battle_id"].is_unique


def test_merge_empty():
    assert list(merge_corpora([]).columns) == CORPUS_COLS


def test_roundtrip_and_load_exposes_instruction_id(tmp_path):
    out = normalize(_raw(), "lmsys")
    p = tmp_path / "corp" / "merged.parquet"
    write_corpus(out, p)
    loaded = load_corpus(p)
    assert (loaded["instruction_id"] == loaded["battle_id"]).all()
    assert set(CORPUS_COLS) <= set(loaded.columns)


def test_load_rejects_non_corpus(tmp_path):
    import pytest
    p = tmp_path / "bad.parquet"
    pd.DataFrame({"foo": [1]}).to_parquet(p)
    with pytest.raises(ValueError):
        load_corpus(p)


def test_normalize_carries_human_pref():
    from prefscope.data.corpus import normalize
    df = _raw().iloc[:2].copy()
    df["human_pref"] = [1.0, 0.5]
    out = normalize(df, "lmarena-100k")
    assert "human_pref" in out.columns
    assert list(out["human_pref"]) == [1.0, 0.5]
    # label-free corpus has no such column (backward compatible)
    assert "human_pref" not in normalize(_raw().iloc[:2], "x").columns
