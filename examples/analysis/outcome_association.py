#!/usr/bin/env python
"""Measure one descriptive feature-outcome association on synthetic data."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from prefscope import FeatureMatrix, OutcomeSpec, analyze_dataset
from prefscope.observability import observe_run

N_ROWS = 24
EVENTS = Path("example-output/analysis/outcome-association.events.jsonl")


def main() -> None:
    row_ids = tuple(f"row-{i}" for i in range(N_ROWS))
    group_ids = tuple(f"prompt-{i // 2}" for i in range(N_ROWS))
    signal = np.linspace(-1.0, 1.0, N_ROWS)
    features = FeatureMatrix(
        np.column_stack([signal, signal**2]),
        row_ids,
        role="response",
        orientation="absolute",
        feature_ids=(0, 1),
        activation_polarity="signed",
        code_semantics="synthetic numerical activity",
    )
    outcome = OutcomeSpec(
        0.5 + 0.4 * signal,
        row_ids,
        kind="probability",
        names=("score",),
    )

    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    with observe_run(EVENTS, pretty=True):
        result = analyze_dataset(
            {"features": features}, {"score": outcome}, group_ids=group_ids
        )

    artifact = result.artifact("outcome_associations")
    columns = ["feature_id", "n_units", "correlation", "slope", "q_value"]
    print()
    print("Outcome associations:")
    print(artifact.table[columns].to_string(index=False))


if __name__ == "__main__":
    main()
