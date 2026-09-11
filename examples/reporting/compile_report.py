#!/usr/bin/env python
"""Collect caller-computed metrics and tables in a small report."""

from pathlib import Path

import pandas as pd

from prefscope import Report

OUTPUT = Path("example-output/reporting/report")


def main() -> None:
    feature_activity = pd.DataFrame(
        {
            "feature_id": [2, 7],
            "activation_rate": [0.35, 0.18],
            "mean_activation": [0.42, 0.21],
        }
    )
    report = Report(
        title="Example analysis",
        metrics={"n_rows": 100, "mean_l0": 3.4},
        metadata={"note": "Metrics were computed by the caller."},
    )
    report.add_table("feature_activity", feature_activity)
    directory = report.save(OUTPUT)
    print(f"Report: {directory}")


if __name__ == "__main__":
    main()
