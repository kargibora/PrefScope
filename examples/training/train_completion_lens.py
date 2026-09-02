#!/usr/bin/env python
"""Train a small completion lens on deterministic toy preference pairs."""

from __future__ import annotations

from pathlib import Path

from prefscope import Lens, PairItem, SAEConfig, TrainConfig
from prefscope.observability import observe_run

# This downloads the public reader model. CPU works; MPS/CUDA is faster.
DEVICE = "cpu"
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
OUTPUT = Path("example-output/training/completion-lens")
FACTS = (
    (
        "Why is the sky blue?",
        "Short wavelengths scatter more strongly.",
        "The ocean paints it.",
    ),
    (
        "Why are leaves green?",
        "Chlorophyll reflects more green light.",
        "Soil dyes the leaves.",
    ),
    (
        "What is binary search?",
        "It repeatedly halves a sorted search range.",
        "It checks every item.",
    ),
    ("What causes tides?", "Mostly the Moon's gravity.", "Clouds pull the oceans."),
    (
        "Why does ice float?",
        "Solid water is less dense than liquid water.",
        "Ice has no mass.",
    ),
    (
        "What is photosynthesis?",
        "Plants store light energy in chemical bonds.",
        "Plants consume soil light.",
    ),
)


def make_pairs() -> list[PairItem]:
    pairs = []
    for index, (prompt, accurate, inaccurate) in enumerate(FACTS):
        pairs.append(PairItem(f"{index}-a", prompt, accurate, inaccurate, pref=1.0))
        pairs.append(PairItem(f"{index}-b", prompt, inaccurate, accurate, pref=0.0))
    return pairs


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(
            f"remove the previous example lens before rerunning: {OUTPUT}"
        )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    config = TrainConfig(
        sae=SAEConfig(m=16, k=4, input_rep="individual"),
        embed_model_id=EMBED_MODEL,
        device=DEVICE,
        val_frac=0.2,
        train_kwargs={"n_epochs": 5, "min_epochs": 2, "patience": 2, "batch": 8},
    )

    with observe_run(OUTPUT.with_suffix(".events.jsonl"), pretty=True):
        lens = Lens.train(make_pairs(), config=config, out=OUTPUT)

    print()
    print(f"Trained {lens.input_rep} lens with {len(lens.feature_table)} features.")
    print(f"Lens directory: {OUTPUT}")


if __name__ == "__main__":
    main()
