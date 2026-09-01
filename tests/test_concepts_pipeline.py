"""Long-form concept export keeps every active feature and streams BYO data."""
from __future__ import annotations

import numpy as np
import pandas as pd

from prefscope.api.loaded_lens import Lens
from prefscope.pipeline.concepts import export_concepts


class _Projector:
    m_total = 3

    def project(self, x):
        return np.asarray(x, dtype=np.float32)[:, :3]


class _Embedder:
    def encode(self, prompts, completions):
        return np.asarray([
            [len(text), text.count("a"), -text.count("z")]
            for text in completions
        ], dtype=np.float32)

    def encode_prompts(self, prompts):
        return np.asarray([
            [len(text), text.count("?"), 0.0]
            for text in prompts
        ], dtype=np.float32)


def _names():
    return pd.DataFrame({
        "feature_id": [0, 1, 2],
        "concept": ["length", "contains a", "contains z"],
        "fidelity_pass": [True, True, False],
        "semantic_threshold": [3.0, 1.0, 1.0],
        "presence_pass": [True, True, True],
    })


def _lens(kind="individual"):
    return Lens(
        _Projector(), _Embedder(), names=_names(),
        manifest={"input_rep": kind})


def test_export_concepts_single_and_second_response_all_activations(tmp_path):
    data = tmp_path / "input.csv"
    pd.DataFrame({
        "prompt": ["p0", "p1"],
        "answer": ["aa", "z"],
        "other": ["a", "zz"],
    }).to_csv(data, index=False)
    out = tmp_path / "concepts.parquet"

    result = export_concepts(
        _lens(), data, out, response_col="answer", response2_col="other",
        batch_size=1, include_text=True, log=lambda *_: None)

    concepts = pd.read_parquet(out)
    assert result["encoded_items"] == 4
    assert set(concepts["side"]) == {"a", "b"}
    assert set(concepts["row_id"]) == {0, 1}
    assert {"feature_id", "concept", "activation", "abs_activation",
            "rank", "fidelity_pass", "semantic_threshold"} <= set(concepts.columns)
    # Negative signed activations remain visible rather than disappearing.
    z_rows = concepts[concepts["feature_id"] == 2]
    assert len(z_rows) == 2 and (z_rows["activation"] < 0).all()
    assert {"prompt", "completion"} <= set(concepts.columns)


def test_export_concepts_filters_and_prompt_lens(tmp_path):
    data = tmp_path / "input.jsonl"
    pd.DataFrame({
        "prompt": ["why?", "plain"],
        "response": ["aaaa", "zzzz"],
    }).to_json(data, orient="records", lines=True)

    filtered = tmp_path / "filtered.csv"
    export_concepts(
        _lens(), data, filtered, fidelity_only=True,
        semantic_presence_only=True, log=lambda *_: None)
    frame = pd.read_csv(filtered)
    assert set(frame["feature_id"]) <= {0, 1}
    assert frame["semantic_present"].all()

    prompts = tmp_path / "prompts.parquet"
    export_concepts(
        _lens("prompt"), data, prompts, top_k=1, log=lambda *_: None)
    prompt_frame = pd.read_parquet(prompts)
    assert set(prompt_frame["side"]) == {"prompt"}
    assert prompt_frame.groupby("row_id").size().max() == 1


def test_export_concepts_empty_after_filter_still_writes_schema(tmp_path):
    data = tmp_path / "input.csv"
    pd.DataFrame({"prompt": ["p"], "response": [""]}).to_csv(data, index=False)
    out = tmp_path / "empty.parquet"

    result = export_concepts(_lens(), data, out, log=lambda *_: None)

    frame = pd.read_parquet(out)
    assert result["concept_rows"] == 0
    assert {"row_id", "side", "feature_id", "activation", "concept"} <= set(frame.columns)
