import json
from pathlib import Path

from prefscope.data.ingest import load_battles


def _write_json(tmp_path: Path) -> Path:
    samples = [
        {  # good row, judge prefers A: OpenJury pref=0.0 => y_judge flips to 1.0
            "instruction_id": "x1", "model_a": "A", "model_b": "B",
            "instruction": "p1", "completion_a": "ca1", "completion_b": "cb1",
            "judge_pref": 0.0, "judge_label": "A",
            "scores_a": {"clarity": 8.0}, "scores_b": {"clarity": 5.0},
            "len_a": 10, "len_b": 20, "parse_error": False,
            "instruction_metadata": {"lang": "English"},
        },
        {  # quarter-tie collapses to 0.5 (flip leaves 0.5 unchanged)
            "instruction_id": "x2", "model_a": "A", "model_b": "C",
            "instruction": "p2", "completion_a": "ca2", "completion_b": "cb2",
            "preference": 0.75, "judge_label": "tie",
            "len_a": 5, "len_b": 5, "parse_error": False,
            "instruction_metadata": {"lang": "German"},
        },
        {  # dropped: parse_error
            "instruction_id": "x3", "model_a": "A", "model_b": "B",
            "instruction": "p3", "completion_a": "ca3", "completion_b": "cb3",
            "judge_pref": 0.0, "parse_error": True,
        },
        {  # dropped: missing completion
            "instruction_id": "x4", "model_a": "A", "model_b": "B",
            "instruction": "p4", "completion_a": None, "completion_b": "cb4",
            "judge_pref": 0.0, "parse_error": False,
        },
        {  # duplicate instruction_id of x1, dropped by dedup
            "instruction_id": "x1", "model_a": "A", "model_b": "B",
            "instruction": "p1", "completion_a": "ca1", "completion_b": "cb1",
            "judge_pref": 0.0, "parse_error": False,
        },
    ]
    p = tmp_path / "ann.json"
    p.write_text(json.dumps({"per_sample": samples}))
    return p


def test_load_battles_filters_and_collapses(tmp_path):
    df = load_battles(_write_json(tmp_path))
    assert sorted(df["instruction_id"]) == ["x1", "x2"]
    r1 = df.set_index("instruction_id").loc["x1"]
    assert r1["y_judge"] == 1.0  # pref 0.0 (A) flipped to P(A)=1.0
    assert r1["lang"] == "English"
    r2 = df.set_index("instruction_id").loc["x2"]
    assert r2["y_judge"] == 0.5  # 0.75 collapsed, flip(0.5)=0.5
    assert r2["lang"] == "German"


def test_load_battles_accepts_bare_list(tmp_path):
    p = tmp_path / "bare.json"
    p.write_text(json.dumps([
        {"instruction_id": "z1", "model_a": "A", "model_b": "B",
         "instruction": "p", "completion_a": "c", "completion_b": "d",
         "preference": 0.0, "parse_error": False},
    ]))
    df = load_battles(p)
    assert list(df["instruction_id"]) == ["z1"]
    assert df.iloc[0]["y_judge"] == 1.0  # pref 0.0 (A) -> P(A)=1.0


def test_load_battles_reads_matches_key(tmp_path):
    """OpenJury annotate output stores battles under 'matches', not 'per_sample'."""
    p = tmp_path / "annotate.json"
    p.write_text(json.dumps({"matches": [
        {"instruction_id": "m1", "model_a": "A", "model_b": "B",
         "instruction": "p", "completion_a": "ca", "completion_b": "cb",
         "preference": 1.0, "parse_error": False},   # B preferred -> y_judge 0.0
    ]}))
    df = load_battles(p)
    assert list(df["instruction_id"]) == ["m1"]
    assert df.iloc[0]["y_judge"] == 0.0
    assert df.iloc[0]["completion_a"] == "ca"


def test_load_battles_rejects_dict_without_per_sample(tmp_path):
    import pytest
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"results": [], "metadata": {}}))
    with pytest.raises(ValueError):
        load_battles(p)


def test_lang_fallback_to_top_level_when_metadata_lacks_language(tmp_path):
    """metadata dict with no lang/language key should fall back to top-level lang."""
    p = tmp_path / "fallback.json"
    p.write_text(json.dumps({"per_sample": [
        {
            "instruction_id": "f1",
            "model_a": "A", "model_b": "B",
            "instruction": "p", "completion_a": "ca", "completion_b": "cb",
            "judge_pref": 1.0, "parse_error": False,
            "lang": "French",
            "instruction_metadata": {"source": "x"},
        },
    ]}))
    df = load_battles(p)
    assert list(df["instruction_id"]) == ["f1"]
    assert df.iloc[0]["lang"] == "French"
