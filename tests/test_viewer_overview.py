import json

import numpy as np
import pandas as pd

from prefscope.viewer_export.overview import (
    export_coactivation,
    export_concept_distribution,
    export_prompt_coactivation,
    export_prompt_concept_distribution,
)


def _paired_lens(tmp_path):
    lens = tmp_path / "lens"
    lens.mkdir()
    ids = [f"id{i}" for i in range(6)]
    np.save(lens / "z_a.npy", np.zeros((6, 2), dtype=np.float32))
    np.save(lens / "z_b.npy", np.ones((6, 2), dtype=np.float32))
    pd.DataFrame({"instruction_id": ids}).to_parquet(lens / "battles.parquet")
    (lens / "manifest.json").write_text(json.dumps({
        "schema_version": 2, "m_total": 2, "k": 2, "input_dim": 4,
        "input_rep": "individual", "dataset_mode": "paired",
    }))
    corpus = tmp_path / "corpus.parquet"
    pd.DataFrame({
        "battle_id": ids,
        "prompt": [f"prompt {i}" for i in range(6)],
        "completion_a": [f"A response {i}" for i in range(6)],
        "completion_b": [f"B response {i}" for i in range(6)],
    }).to_parquet(corpus)
    return lens, corpus


def test_paired_response_summaries_include_both_sides_and_all_axes(tmp_path):
    lens, corpus = _paired_lens(tmp_path)
    # Feature 1 is unnamed and failed, but activation-level co-activation must still
    # retain it so every feature-atlas point can have empirical neighbors.
    features = pd.DataFrame({
        "feature_id": [0, 1],
        "concept": ["named", np.nan],
        "fidelity_pass": [True, False],
    })

    distribution = export_concept_distribution(lens, features)
    coactivation = export_coactivation(lens, features, corpus_path=str(corpus))

    assert distribution is not None and distribution["n_rows"] == 12
    assert coactivation is not None and coactivation["n_rows"] == 12
    assert coactivation["selection"] == "all_axes"
    pair = next(p for p in coactivation["pairs"] if {p["a"], p["b"]} == {0, 1})
    assert pair["count"] == 6
    assert pair["rows"] and min(pair["rows"]) >= 6
    assert all(coactivation["examples"][str(row)]["response"].startswith("B response")
               for row in pair["rows"])
    assert all(
        coactivation["examples"][str(row)]["activations"] == {"0": 1.0, "1": 1.0}
        for row in pair["rows"]
    )


def test_prompt_coactivation_includes_prompts_and_both_activation_values(tmp_path):
    lens, corpus = _paired_lens(tmp_path)
    (lens / "z_a.npy").unlink()
    (lens / "z_b.npy").unlink()
    z_prompt = np.array([
        [1.0, 2.0], [0.5, 1.0], [3.0, 4.0],
        [1.5, 0.5], [2.5, 3.5], [0.25, 0.75],
    ], dtype=np.float32)
    np.save(lens / "z_prompt.npy", z_prompt)
    features = pd.DataFrame({
        "feature_id": [0, 1],
        "concept": ["creative request", "short story request"],
    })

    result = export_prompt_coactivation(
        lens, features, corpus_path=str(corpus), n_examples=3
    )

    assert result is not None
    pair = next(p for p in result["pairs"] if {p["a"], p["b"]} == {0, 1})
    assert pair["rows"]
    for row in pair["rows"]:
        example = result["examples"][str(row)]
        assert example["prompt"].startswith("prompt ")
        assert example["activations"]["0"] == float(z_prompt[row, 0])
        assert example["activations"]["1"] == float(z_prompt[row, 1])


def test_prompt_distribution_uses_prompt_codes_and_prompt_labels(tmp_path):
    lens, _ = _paired_lens(tmp_path)
    # A prompt lens can coexist with response arrays in a working directory.  The
    # prompt-specific exporter must still use z_prompt.npy exclusively.
    z_prompt = np.array([
        [1.0, 0.0], [0.0, 2.0], [3.0, 4.0],
        [0.0, 0.0], [5.0, 0.0], [0.0, 6.0],
    ], dtype=np.float32)
    np.save(lens / "z_prompt.npy", z_prompt)
    features = pd.DataFrame({
        "feature_id": [0, 1],
        "concept": ["asks for an explanation", "requests code"],
        "fidelity_pass": [True, True],
    })

    result = export_prompt_concept_distribution(lens, features, chunk_rows=2)

    assert result is not None
    assert result["code_array"] == "z_prompt.npy"
    assert result["n_rows"] == 6
    assert result["coverage"] == 5 / 6
    assert result["concepts_per_row"]["mean"] == 1.0
    assert [row["concept"] for row in result["features"]] == [
        "asks for an explanation", "requests code",
    ]
    assert [row["n_active"] for row in result["features"]] == [3, 3]
