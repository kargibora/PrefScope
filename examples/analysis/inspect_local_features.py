#!/usr/bin/env python
"""Print the strongest codes for several responses in the bundled local table."""

from __future__ import annotations

from pathlib import Path

from prefscope import Lens, TableDataset, feature_activation_table
from prefscope.integrations import NeuronpediaProvider
from prefscope.observability import observe_run
from prefscope.presentation import FeatureTableRenderer

LENS_CONFIG = Path(__file__).parents[1] / "inference" / "saelens.yaml"
DATA = Path(__file__).parents[1] / "assets" / "sample_corpus.parquet"
LIMIT = 3
TOP_K = 3
EVENTS = Path("example-output/analysis/inspect-local-features.events.jsonl")


def main() -> None:
    rows = list(
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

    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    with observe_run(EVENTS, pretty=True):
        lens = Lens.from_config(LENS_CONFIG)
        features = lens.featurize(rows, views=("response_a",))
        matrix = features.matrix("z_a")
        catalog = lens.feature_catalog
        top = feature_activation_table(matrix, top_k=TOP_K)
        provider = NeuronpediaProvider.from_lens(lens)
        if not catalog.labels and provider is not None:
            ids = tuple(dict.fromkeys(int(value) for value in top["feature_id"]))
            catalog = catalog.merge(provider.fetch(ids, strict=False))
        table = feature_activation_table(matrix, catalog=catalog, top_k=TOP_K)

    FeatureTableRenderer(max_rows=LIMIT * TOP_K).print(table)


if __name__ == "__main__":
    main()
