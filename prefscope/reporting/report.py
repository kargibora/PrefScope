"""A small container for metrics and tables produced by a PrefScope workflow."""
from __future__ import annotations

from collections.abc import Mapping
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import pandas as pd


def _name(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("report names must be non-empty strings")
    return value


def _metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    result = dict(value or {})
    try:
        json.dumps(result, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError("report metadata must contain JSON values") from exc
    return result


class Report:
    """Collect caller-defined numerical metrics and pandas tables.

    Report does not interpret columns, select statistics, apply privacy policy, or
    generate narrative. The caller owns those decisions.
    """

    def __init__(
        self,
        title: str | None = None,
        *,
        metrics: Mapping[str, int | float] | None = None,
        tables: Mapping[str, pd.DataFrame] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if title is not None and (not isinstance(title, str) or not title.strip()):
            raise ValueError("title must be a non-empty string or None")
        self.title = title
        self.metrics: dict[str, int | float] = {}
        self.tables: dict[str, pd.DataFrame] = {}
        self.metadata = _metadata(metadata)
        for name, value in dict(metrics or {}).items():
            self.add_metric(name, value)
        for name, table in dict(tables or {}).items():
            self.add_table(name, table)

    def add_metric(self, name: str, value: int | float) -> None:
        """Add or replace one caller-named finite numerical metric."""
        name = _name(name)
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("report metrics must be finite real numbers")
        number = int(value) if isinstance(value, Integral) else float(value)
        if not math.isfinite(number):
            raise TypeError("report metrics must be finite real numbers")
        self.metrics[name] = number

    def add_table(self, name: str, table: pd.DataFrame) -> None:
        """Add or replace one caller-defined table without interpreting its columns."""
        name = _name(name)
        if not isinstance(table, pd.DataFrame):
            raise TypeError("report tables must be pandas DataFrames")
        self.tables[name] = table.copy(deep=True)

    def save(self, directory: str | Path) -> Path:
        """Write ``report.json`` and one CSV per table to a new directory."""
        directory = Path(directory).expanduser().resolve()
        directory.mkdir(parents=True)
        table_directory = directory / "tables"
        table_directory.mkdir()
        table_paths = {}
        for index, (name, table) in enumerate(self.tables.items()):
            relative = Path("tables") / f"table_{index}.csv"
            table.to_csv(directory / relative)
            table_paths[name] = relative.as_posix()
        payload = {
            "title": self.title,
            "metrics": self.metrics,
            "metadata": self.metadata,
            "tables": table_paths,
        }
        (directory / "report.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return directory


__all__ = ["Report"]
