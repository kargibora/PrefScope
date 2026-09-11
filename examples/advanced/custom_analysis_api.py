#!/usr/bin/env python
"""A custom analysis is an ordinary function over a FeatureMatrix."""

from __future__ import annotations

import numpy as np
import pandas as pd

from prefscope import FeatureMatrix, Report, activation_summary


def feature_magnitude(matrix: FeatureMatrix) -> pd.DataFrame:
    """Example project-specific analysis with no registry or framework base class."""
    values = np.abs(matrix.values)
    return pd.DataFrame(
        {
            "feature_id": matrix.feature_ids,
            "mean_absolute_value": values.mean(axis=0),
        }
    )


def main() -> None:
    matrix = FeatureMatrix(
        [[0.0, 1.0], [2.0, -1.0]],
        row_ids=("row-0", "row-1"),
        feature_ids=(3, 8),
    )
    report = Report(
        title="Custom analysis",
        metrics={"n_rows": len(matrix.row_ids)},
        tables={
            "activity": activation_summary(matrix),
            "magnitude": feature_magnitude(matrix),
        },
    )
    print(report.tables["magnitude"].to_string(index=False))


if __name__ == "__main__":
    main()
