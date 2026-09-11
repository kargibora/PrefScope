#!/usr/bin/env python
"""Use an optional recipe to rank preference associations."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from prefscope import Lens, TableDataset
from prefscope.recipes.analysis.preference_relevance import preference_relevance

LENS_CONFIG = Path(__file__).parents[1] / "inference" / "saelens.yaml"
DATA = Path(__file__).parents[1] / "assets" / "sample_corpus.parquet"
LIMIT = 25
N_FEATURES = 64


def main() -> None:
    sample = list(
        TableDataset(
            DATA,
            prompt="prompt",
            a="completion_a",
            b="completion_b",
            pref="human_pref",
            id="battle_id",
            group_id="battle_id",
        )
    )[:LIMIT]
    probe_rows, rows = sample[:1], sample[1:]
    lens = Lens.from_config(LENS_CONFIG)
    probe = lens.featurize(probe_rows, views=("response_difference",))
    activity = np.abs(probe.arrays["z_diff"][0])
    feature_ids = np.argsort(activity)[-N_FEATURES:][::-1]
    features = lens.featurize(rows, feature_ids=feature_ids)
    relevance = preference_relevance(features)

    relevance = relevance.assign(
        absolute_correlation=relevance["correlation"].abs()
    ).sort_values("absolute_correlation", ascending=False)
    columns = [
        "feature_id",
        "n_fire",
        "correlation",
        "n_independent_groups",
        "estimand",
    ]
    print("Preference relevance:")
    print(relevance[columns].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
