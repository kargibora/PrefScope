import json

import numpy as np
import pandas as pd
import pytest

from prefscope.interpret.io import load_lens_battles
from prefscope.viewer_export.examples import (
    export_examples,
    export_joint_examples,
    export_prompt_examples,
)
from prefscope.viewer_export.overview import export_prompt_coactivation


def _single_lens(tmp_path, n=6, m=3):
    lens = tmp_path / "lens"
    lens.mkdir()
    z = np.zeros((n, m), dtype=np.float32)
    z[:, 0] = np.arange(n, dtype=np.float32)
    np.save(lens / "z_a.npy", z)
    (lens / "manifest.json").write_text(json.dumps(
        {"dataset_mode": "single", "m_total": m, "k": 2, "input_rep": "individual"}))
    ids = [f"id{i}" for i in range(n)]
    pd.DataFrame({"instruction_id": ids, "language": ["de"] * n}).to_parquet(
        lens / "battles.parquet", index=False)
    corpus = tmp_path / "corpus.parquet"
    pd.DataFrame({
        "battle_id": ids,
        "prompt": [f"p{i}" for i in range(n)],
        "completion_a": [f"a{i}" for i in range(n)],
        "source": ["s"] * n, "language": ["de"] * n,
    }).to_parquet(corpus, index=False)
    return lens, corpus


def test_load_lens_battles_reads_z_a_when_no_z_diff(tmp_path):
    lens, corpus = _single_lens(tmp_path)
    battles, z, manifest = load_lens_battles(lens, corpus=corpus)
    assert z.shape == (6, 3)
    assert len(battles) == 6
    assert manifest["dataset_mode"] == "single"
    assert list(battles["completion_a"]) == [f"a{i}" for i in range(6)]


def test_load_lens_battles_prefers_z_diff_when_present(tmp_path):
    lens, corpus = _single_lens(tmp_path)
    np.save(lens / "z_diff.npy", np.ones((6, 3), dtype=np.float32) * 7)
    _, z, _ = load_lens_battles(lens, corpus=corpus)
    assert float(z[0, 0]) == 7.0


def test_load_lens_battles_errors_without_any_codes(tmp_path):
    lens, corpus = _single_lens(tmp_path)
    (lens / "z_a.npy").unlink()
    with pytest.raises(FileNotFoundError, match="z_diff.npy|z_a.npy"):
        load_lens_battles(lens, corpus=corpus)


def test_export_examples_on_single_response_lens(tmp_path):
    lens, corpus = _single_lens(tmp_path)
    features = pd.DataFrame({"feature_id": [0], "concept": ["uses headings"]})
    out = export_examples(lens, str(corpus), features, 3)
    assert set(out) == {"0"}
    rows = out["0"]
    assert rows and all(r["prompt"] and r["completion_a"] for r in rows)
    assert all(r["completion_b"] == "" for r in rows)
    assert all(r["group"] == "de" and r["group_column"] == "language" for r in rows)


def test_export_examples_keeps_language_specific_evidence(tmp_path):
    lens, corpus = _single_lens(tmp_path)
    battles = pd.read_parquet(lens / "battles.parquet")
    battles["language"] = ["de", "cs", "de", "cs", "de", "cs"]
    battles.to_parquet(lens / "battles.parquet", index=False)
    corpus_frame = pd.read_parquet(corpus)
    corpus_frame["language"] = battles["language"]
    corpus_frame.to_parquet(corpus, index=False)
    features = pd.DataFrame({"feature_id": [0], "concept": ["uses headings"]})

    rows = export_examples(lens, str(corpus), features, n_per=1, n_per_group=1)["0"]

    assert {row["group"] for row in rows} == {"cs", "de"}
    assert len(rows) == 3  # one global maximum plus one additional row per language


def test_export_examples_adds_percentile_and_sampling_modes(tmp_path):
    lens, corpus = _single_lens(tmp_path, n=12)
    features = pd.DataFrame({
        "feature_id": [0], "concept": ["uses headings"],
        "semantic_threshold": [3.0], "presence_pass": [True],
    })

    rows = export_examples(
        lens, str(corpus), features, n_per=2, n_per_group=0,
        n_random=2, n_boundary=2, seed=7,
    )["0"]

    assert {row["selection_kind"] for row in rows} == {
        "strongest", "random_present", "near_threshold",
    }
    assert all(0 < row["activation_percentile"] <= 100 for row in rows)
    assert all(row["activation_reference"] == "positive_activation" for row in rows)
    strongest = [row for row in rows if row["selection_kind"] == "strongest"]
    assert max(row["activation_percentile"] for row in strongest) == 100.0


def test_export_prompt_examples_adds_random_and_boundary_modes(tmp_path):
    _, corpus = _single_lens(tmp_path, n=12)
    prompt_lens = tmp_path / "prompt_sampling_lens"
    prompt_lens.mkdir()
    values = np.arange(12, dtype=np.float32).reshape(-1, 1)
    np.save(prompt_lens / "z_prompt.npy", values)
    pd.DataFrame({
        "instruction_id": [f"id{i}" for i in range(12)],
        "prompt": [f"prompt {i}" for i in range(12)],
        "language": ["de"] * 12,
    }).to_parquet(prompt_lens / "battles.parquet", index=False)
    features = pd.DataFrame({"feature_id": [0], "concept": ["counting"]})

    rows = export_prompt_examples(
        prompt_lens, str(corpus), features, n_per=2, n_per_group=0,
        n_random=2, n_boundary=2, seed=7,
    )["0"]

    assert {row["selection_kind"] for row in rows} == {
        "strongest", "random_present", "near_boundary",
    }
    assert all(0 < row["activation_percentile"] <= 100 for row in rows)


def test_export_joint_examples_on_single_response_lens(tmp_path):
    lens, corpus = _single_lens(tmp_path)
    prompt_lens = tmp_path / "prompt_lens"
    prompt_lens.mkdir()
    zp = np.zeros((6, 2), dtype=np.float32)
    zp[2:, 1] = np.arange(1, 5, dtype=np.float32)
    np.save(prompt_lens / "z_prompt.npy", zp)
    pd.DataFrame({
        "instruction_id": [f"id{i}" for i in range(6)],
        "prompt": [f"p{i}" for i in range(6)],
    }).to_parquet(prompt_lens / "battles.parquet", index=False)

    out = export_joint_examples(lens, str(corpus), prompt_lens, [(1, 0)], per_pair=2)

    assert out is not None
    rows = out["1"]["examples"]["0"]
    assert len(rows) == 2
    assert all(row["side"] == "a" for row in rows)
    assert all(row["prompt"] and row["response"] for row in rows)


def test_export_prompt_examples_covers_unverified_and_silent_axes(tmp_path):
    _, corpus = _single_lens(tmp_path)
    prompt_lens = tmp_path / "prompt_lens"
    prompt_lens.mkdir()
    z = np.zeros((6, 3), dtype=np.float32)
    z[:, 0] = np.arange(6, dtype=np.float32)
    z[1:4, 1] = [0.5, 2.0, 1.0]
    np.save(prompt_lens / "z_prompt.npy", z)
    pd.DataFrame({
        "instruction_id": [f"id{i}" for i in range(6)],
        "prompt": [f"prompt {i}" for i in range(6)],
        "language": ["de", "cs", "de", "cs", "de", "cs"],
    }).to_parquet(prompt_lens / "battles.parquet", index=False)
    features = pd.DataFrame({
        "feature_id": [0, 1, 2],
        "concept": ["counting", "middle", "silent"],
        "fidelity_pass": [True, False, np.nan],
    })

    examples = export_prompt_examples(prompt_lens, str(corpus), features, n_per=2)
    coactivation = export_prompt_coactivation(prompt_lens, features, top_k=2)

    assert set(examples) == {"0", "1", "2"}
    assert [row["prompt"] for row in examples["0"][:2]] == ["prompt 5", "prompt 4"]
    assert {row["group"] for row in examples["0"]} == {"cs", "de"}
    assert all(row["group_column"] == "language" for row in examples["0"])
    assert examples["2"] == []
    assert coactivation is not None
    assert coactivation["code_array"] == "z_prompt.npy"
    assert coactivation["n_total_features"] == 3


def test_response_map_reads_prepared_corpus_ids(tmp_path):
    from prefscope.viewer_export.maps import _corpus_frame
    corpus = tmp_path / "prepared.parquet"
    pd.DataFrame({
        "row_id": [0, 1], "prompt": ["p0", "p1"], "completion_a": ["a0", "a1"],
        "item_id": ["0", "1"], "source": ["s", "s"], "language": ["de", "cs"],
    }).to_parquet(corpus, index=False)
    frame = _corpus_frame(corpus)
    assert "battle_id" in frame.columns and "instruction_id" in frame.columns
