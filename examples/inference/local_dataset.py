#!/usr/bin/env python
"""Featurize the bundled local preference table and print array summaries."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from prefscope import Lens, TableDataset, save_feature_batch
from prefscope.observability import observe_run

# Edit these constants, then run: python examples/inference/local_dataset.py
LENS_CONFIG = Path(__file__).with_name("saelens.yaml")
DATA = Path(__file__).parents[1] / "assets" / "sample_corpus.parquet"
LIMIT = 12
OUTPUT = Path("example-output/inference/local-dataset")


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    dataset = TableDataset(
        DATA,
        prompt="prompt",
        a="completion_a",
        b="completion_b",
        pref="human_pref",
        id="battle_id",
        group_id="battle_id",
    )
    rows = list(dataset)[:LIMIT]

    with observe_run(OUTPUT.with_suffix(".events.jsonl"), pretty=True):
        lens = Lens.from_config(LENS_CONFIG)
        features = lens.featurize(rows)
        save_feature_batch(features, OUTPUT, overwrite=True)

    print()
    print(f"Featurized {len(rows)} preference pairs:")
    for view, values in features.arrays.items():
        mean_active = float(np.count_nonzero(values, axis=1).mean())
        matrix = features.matrix(view)
        print(
            f"  {view}: shape={values.shape}, role={matrix.role}, "
            f"orientation={matrix.orientation}, mean active={mean_active:.1f}"
        )
    print()
    print(f"Feature bundle: {OUTPUT}")


if __name__ == "__main__":
    main()
