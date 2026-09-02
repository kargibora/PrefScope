#!/usr/bin/env python
"""Print the strongest SAE codes and available feature descriptions for one response."""

from __future__ import annotations

from pathlib import Path

from prefscope import Lens, PairItem, feature_activation_table
from prefscope.integrations import NeuronpediaProvider
from prefscope.observability import observe_run
from prefscope.presentation import FeatureTableRenderer

LENS_CONFIG = Path(__file__).with_name("saelens.yaml")
PROMPT = "Why do leaves often look green?"
RESPONSE = "Chlorophyll absorbs red and blue light and reflects more green light."
TOP_K = 5
EVENTS = Path("example-output/inference/single-item.events.jsonl")


def main() -> None:
    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    item = PairItem("example-0", PROMPT, RESPONSE)

    with observe_run(EVENTS, pretty=True):
        lens = Lens.from_config(LENS_CONFIG)
        features = lens.featurize([item], views=("response_a",))
        matrix = features.matrix("z_a")
        catalog = lens.feature_catalog
        top = feature_activation_table(matrix, top_k=TOP_K)
        provider = NeuronpediaProvider.from_lens(lens)
        if not catalog.labels and provider is not None:
            catalog = catalog.merge(provider.fetch(top["feature_id"], strict=False))
        table = feature_activation_table(matrix, catalog=catalog, top_k=TOP_K)

    FeatureTableRenderer(max_rows=TOP_K).print(table)


if __name__ == "__main__":
    main()
