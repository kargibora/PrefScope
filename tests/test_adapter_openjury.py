# tests/test_adapter_openjury.py
import json

from prefscope.core import registry
from prefscope.core.types import PairItem
import prefscope.adapters  # noqa: F401


def _write_openjury(tmp_path):
    payload = {"per_sample": [
        {"instruction_id": "i1", "instruction": "Q1", "model_a": "ma", "model_b": "mb",
         "completion_a": "A1", "completion_b": "B1", "judge_pref": 0.0,
         "scores_a": {"clarity": 5}, "scores_b": {"clarity": 3}},
        {"instruction_id": "i2", "instruction": "Q2", "model_a": "ma", "model_b": "mb",
         "completion_a": "A2", "completion_b": "B2", "judge_pref": 1.0},
        {"instruction_id": "i3", "parse_error": True},
    ]}
    p = tmp_path / "ann.json"
    p.write_text(json.dumps(payload))
    return p


def test_openjury_yields_pairitems(tmp_path):
    p = _write_openjury(tmp_path)
    cls = registry.get("dataset", "openjury")
    ds = cls(p)
    items = list(ds)
    assert all(isinstance(it, PairItem) for it in items)
    assert [it.id for it in items] == ["i1", "i2"]   # parse_error row dropped
    first = items[0]
    assert first.x == "Q1" and first.y_a == "A1" and first.y_b == "B1"
    # judge_pref=0.0 means A wins -> ingest flips to P(A)=1.0
    assert first.pref == 1.0
    assert first.model_a == "ma" and first.model_b == "mb"
    assert first.meta["scores_a"] == {"clarity": 5}


def test_openjury_len(tmp_path):
    p = _write_openjury(tmp_path)
    ds = registry.get("dataset", "openjury")(p)
    assert len(ds) == 2
