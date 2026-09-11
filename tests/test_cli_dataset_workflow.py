
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
