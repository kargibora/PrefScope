from __future__ import annotations

import json

import numpy as np
import pandas as pd

from prefscope.cli.parser import build_parser


def test_associate_outcomes_cli_exports_grouped_multi_attribute_table(tmp_path):
    encoded = tmp_path / "encoded"
    encoded.mkdir()
    pd.DataFrame({
        "prompt": ["same", "same", "other", "third"],
        "helpfulness": [1.0, 3.0, 2.0, 4.0],
        "correctness": [0.0, 2.0, 1.0, 4.0],
    }).to_parquet(encoded / "meta.parquet", index=False)
    np.save(encoded / "z_a.npy", np.array([
        [0.0, 1.0], [2.0, 1.0], [1.0, 0.0], [3.0, 1.0],
    ], dtype=np.float32))
    out = tmp_path / "associations.csv"
    args = build_parser().parse_args([
        "associate-outcomes", "--encoded-dir", str(encoded),
        "--outcome-col", "helpfulness", "--outcome-col", "correctness",
        "--outcome-kind", "multi_continuous", "--out", str(out),
    ])

    assert args.func(args) == 0
    table = pd.read_csv(out)
    assert set(table["outcome"]) == {"helpfulness", "correctness"}
    assert set(table["analysis_unit"]) == {"group"}
    assert set(table["n_units"]) == {3}
    sidecar = json.loads((tmp_path / "associations_outcomes.json").read_text())
    assert sidecar["grouped"] is True
    assert sidecar["normalization"] == "zscore"


def test_associate_outcomes_cli_rejects_missing_columns(tmp_path):
    encoded = tmp_path / "encoded"
    encoded.mkdir()
    pd.DataFrame({"prompt": ["p"]}).to_parquet(encoded / "meta.parquet", index=False)
    np.save(encoded / "z_a.npy", np.ones((1, 1), dtype=np.float32))
    args = build_parser().parse_args([
        "associate-outcomes", "--encoded-dir", str(encoded),
        "--outcome-col", "missing", "--outcome-kind", "continuous",
        "--out", str(tmp_path / "out.csv"),
    ])

    assert args.func(args) == 2
