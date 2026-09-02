#!/usr/bin/env python
"""Stream a small Hugging Face split, featurize it, and print summaries."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from prefscope import HuggingFaceDataset, Lens, save_feature_batch
from prefscope.observability import observe_run

# Edit these constants, then run: python examples/inference/huggingface_dataset.py
LENS_CONFIG = Path(__file__).with_name("saelens.yaml")
DATASET = "lmsys/lmsys-arena-human-preference-55k"
REVISION = None  # Optional. Set an exact commit SHA for reproducible research.
SPLIT = "train"
LIMIT = 10
OUTPUT = Path("example-output/inference/huggingface-dataset")


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    dataset = HuggingFaceDataset(
        DATASET,
        split=SPLIT,
        revision=REVISION,
        streaming=True,
        limit=LIMIT,
        prompt="prompt",
        a="response_a",
        b="response_b",
    )
    rows = list(dataset)

    with observe_run(OUTPUT.with_suffix(".events.jsonl"), pretty=True):
        lens = Lens.from_config(LENS_CONFIG)
        features = lens.featurize(rows)
        save_feature_batch(features, OUTPUT, overwrite=True)

    print()
    print(f"Featurized {len(rows)} streamed rows:")
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
