import json

import numpy as np
import pandas as pd

from prefscope.__main__ import main


def test_prepare_dataset_cli_maps_explicit_winner_labels(tmp_path):
    raw = tmp_path / "raw.csv"
    pd.DataFrame({
        "question": ["q1", "q2", "q3"],
        "left": ["a1", "a2", "a3"],
        "right": ["b1", "b2", "b3"],
        "choice": ["left", "right", "tie"],
    }).to_csv(raw, index=False)
    out = tmp_path / "canonical.parquet"
    assert main([
        "prepare-dataset",
        "--data", str(raw),
        "--out", str(out),
        "--prompt-col", "question",
        "--response-col", "left",
        "--response-2-col", "right",
        "--label-col", "choice",
        "--label-mode", "winner",
        "--a-wins-value", "left",
        "--b-wins-value", "right",
        "--tie-value", "tie",
    ]) == 0
    frame = pd.read_parquet(out)
    assert list(frame["human_pref"]) == [1.0, 0.0, 0.5]


def test_win_relevance_reads_encode_dataset_bundle_and_writes_summary(tmp_path):
    encoded = tmp_path / "encoded"
    encoded.mkdir()
    meta = pd.DataFrame({
        "prompt": [f"q{i}" for i in range(7)],
        "completion_a": ["short", "a much longer answer", "aa", "long answer here",
                         "a", "another longer answer", "ignored"],
        "completion_b": ["a longer answer", "short", "bbb", "x",
                         "long response", "tiny", "ignored"],
        "human_pref": [1.0, 0.0, 1.0, 0.0, 0.5, 1.0, np.nan],
        "model_a": ["A", "A", "A", "B", "B", "A", "A"],
        "model_b": ["B", "B", "B", "A", "A", "B", "B"],
    })
    meta.to_parquet(encoded / "meta.parquet", index=False)
    z = np.array([
        [1, 0], [-1, 1], [1, -1], [-1, 1], [0, -1], [1, 0], [99, 99],
    ], dtype=np.float32)
    np.save(encoded / "z_diff.npy", z)
    out = tmp_path / "win.csv"

    assert main([
        "win-relevance",
        "--encoded-dir", str(encoded),
        "--all-features",
        "--out", str(out),
    ]) == 0
    result = pd.read_csv(out)
    summary = json.loads((tmp_path / "win_summary.json").read_text())
    assert set(result["feature_id"]) == {0, 1}
    assert summary["n_labeled"] == 6
    assert summary["n_ties"] == 1
    assert summary["label_semantics"] == "human_pref = P(A preferred)"
    models = pd.read_csv(tmp_path / "win_models.csv").set_index("model")
    assert set(models.index) == {"A", "B"}
    assert models.loc["A", "n_battles"] == 6
