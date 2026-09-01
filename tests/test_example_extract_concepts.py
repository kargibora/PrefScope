import numpy as np
import pandas as pd

from prefscope.analysis.presence import concept_presence
from prefscope.pipeline.text_concepts import (
    extract_present_concepts, extract_text_concepts, resolve_device,
)


class _FakeLens:
    def __init__(self):
        self.feature_table = pd.DataFrame({
            "feature_id": [0, 1, 2],
            "concept": ["uses headings", "discusses sports", "unverified"],
            "fidelity_pass": [True, True, False],
            "semantic_threshold": [2.0, np.nan, 1.0],
            "presence_pass": [True, False, True],
            "semantic_role": ["presentation", "topic_content", "presentation"],
        })

    def presence(self, codes, *, feature_ids, policy):
        return concept_presence(
            codes, self.feature_table, feature_ids=feature_ids, policy=policy)


def test_example_extracts_verified_concepts_and_reports_presence_basis():
    rows = extract_present_concepts(
        _FakeLens(), np.array([3.0, 1.0, 4.0]), policy="mixed", top=20)

    assert [row["feature_id"] for row in rows] == [0, 1]
    assert [row["presence_basis"] for row in rows] == [
        "semantic_threshold", "positive_nonzero"]


def test_example_strict_calibration_omits_uncalibrated_and_honors_top():
    rows = extract_present_concepts(
        _FakeLens(), np.array([3.0, 1.0, 4.0]), policy="calibrated", top=1)

    assert len(rows) == 1
    assert rows[0]["concept"] == "uses headings"


def test_example_explicit_device_is_unchanged():
    assert resolve_device("cpu") == "cpu"


class _FakeLoadedLens(_FakeLens):
    input_rep = "individual"

    def __init__(self, model_id="embed"):
        super().__init__()
        self.embedder = type("EmbedderContract", (), {
            "model_id": model_id,
            "model_revision": "v1",
            "max_tokens": 128,
            "pooling": "last-token",
            "normalization": "l2",
            "prompt_embed_instruction": "prompt instruction",
            "effective_dtype_name": lambda self: "float32",
        })()

    def encode_one(self, prompt, completion=None):
        return np.array([3.0, 0.0, 0.0])


def test_extract_supports_completion_lens_without_prompt_lens(monkeypatch):
    monkeypatch.setattr(
        "prefscope.pipeline.text_concepts._load_source",
        lambda *args, **kwargs: _FakeLoadedLens(),
    )
    result = extract_text_concepts(
        "question", "answer", completion_lens="response-lens", device="cpu",
        presence_policy="calibrated",
    )
    assert "prompt" not in result
    assert result["completion"][0]["concept"] == "uses headings"


def test_extract_rejects_incompatible_two_lens_contracts(monkeypatch):
    lenses = iter([_FakeLoadedLens("prompt-embed"), _FakeLoadedLens("response-embed")])
    monkeypatch.setattr(
        "prefscope.pipeline.text_concepts._load_source",
        lambda *args, **kwargs: next(lenses),
    )
    import pytest
    with pytest.raises(ValueError, match="incompatible embedding contracts"):
        extract_text_concepts(
            "question", "answer", prompt_lens="prompt-lens",
            completion_lens="response-lens", device="cpu",
        )
