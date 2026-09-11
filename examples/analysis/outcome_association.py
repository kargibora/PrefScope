#!/usr/bin/env python
"""Use an optional recipe for a project-specific outcome association."""

from __future__ import annotations

import numpy as np

from prefscope import FeatureMatrix
from prefscope.recipes.analysis.outcomes import associate_outcomes, normalize_outcomes

N_ROWS = 24


def main() -> None:
    row_ids = tuple(f"row-{i}" for i in range(N_ROWS))
    group_ids = tuple(f"prompt-{i // 2}" for i in range(N_ROWS))
    signal = np.linspace(-1.0, 1.0, N_ROWS)
    features = FeatureMatrix(
        np.column_stack([signal, signal**2]),
        row_ids,
        feature_ids=(0, 1),
    )
    outcome = normalize_outcomes(
        0.5 + 0.4 * signal,
        kind="probability",
        names=("score",),
        normalization="none",
    )
    result = associate_outcomes(
        features.values,
        outcome,
        feature_ids=features.feature_ids,
        group_ids=group_ids,
    )
    columns = ["feature_id", "n_units", "correlation", "slope", "q_value"]
    print("Outcome associations:")
    print(result.table[columns].to_string(index=False))


if __name__ == "__main__":
    main()
