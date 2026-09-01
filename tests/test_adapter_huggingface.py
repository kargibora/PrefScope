import pandas as pd

from prefscope.data.datasets import HuggingFaceDataset
from prefscope.core import registry
from prefscope.data import datasets as dataset_adapters


def test_huggingface_adapter_uses_shared_mapping(monkeypatch):
    seen = {}

    def fake_load(dataset_id, **kwargs):
        seen.update(dataset_id=dataset_id, **kwargs)
        return pd.DataFrame({
            "question": ["q1", "q2"],
            "answer_a": ["a1", "a2"],
            "answer_b": ["b1", "b2"],
            "winner": ["A", "B"],
        })

    monkeypatch.setattr(dataset_adapters, "load_hf_table", fake_load)
    data = HuggingFaceDataset(
        "owner/data",
        prompt="question",
        a="answer_a",
        b="answer_b",
        pref="winner",
        label_mode="winner",
        a_values=("A",),
        b_values=("B",),
        split="test",
        streaming=True,
        limit=2,
    )
    items = list(data)
    assert [item.pref for item in items] == [1.0, 0.0]
    assert seen["dataset_id"] == "owner/data" and seen["split"] == "test"
    assert registry.get("dataset", "huggingface") is HuggingFaceDataset
