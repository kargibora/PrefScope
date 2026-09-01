from __future__ import annotations

import numpy as np
import pandas as pd

from prefscope.cli.parser import build_parser


def test_win_relevance_cli_uses_prompt_groups_by_default(tmp_path):
    encoded = tmp_path / "encoded"
    encoded.mkdir()
    pd.DataFrame({
        "prompt": ["repeat", "repeat", "other", "third"],
        "completion_a": ["a"] * 4,
        "completion_b": ["b"] * 4,
        "human_pref": [1.0, 1.0, 0.0, 1.0],
    }).to_parquet(encoded / "meta.parquet", index=False)
    np.save(encoded / "z_diff.npy", np.array([[1.0], [1.0], [-1.0], [1.0]], np.float32))
    out = tmp_path / "win.csv"
    args = build_parser().parse_args([
        "win-relevance", "--encoded-dir", str(encoded), "--out", str(out),
    ])

    assert args.func(args) == 0
    row = pd.read_csv(out).iloc[0]
    assert row["estimand"] == "equal_group_weight"
    assert row["n_groups"] == 3
    assert row["delta_win_inference_test"] == "cluster_robust_wald_t_g_minus_1_hc1"


def test_win_relevance_cli_rejects_missing_explicit_group_column(tmp_path):
    encoded = tmp_path / "encoded"
    encoded.mkdir()
    pd.DataFrame({
        "prompt": ["p", "q"], "completion_a": ["a", "a"],
        "completion_b": ["b", "b"], "human_pref": [1.0, 0.0],
    }).to_parquet(encoded / "meta.parquet", index=False)
    np.save(encoded / "z_diff.npy", np.ones((2, 1), np.float32))
    args = build_parser().parse_args([
        "win-relevance", "--encoded-dir", str(encoded),
        "--group-col", "missing", "--out", str(tmp_path / "out.csv"),
    ])

    assert args.func(args) == 2
