"""Versioned logical schemas for public pandas result tables."""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping
import re

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

_COLUMN_KINDS = frozenset({
    "string", "integer", "float", "boolean", "nullable_integer", "numeric",
})
_SCHEMA_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


def _portable_names(values, *, where: str) -> tuple[str, ...]:
    result = tuple(values)
    if len(set(result)) != len(result) or any(
        not isinstance(value, str) or not value for value in result
    ):
        raise ValueError(f"{where} must contain unique non-empty strings")
    return result


@dataclass(frozen=True)
class TableContract:
    """Logical, versioned contract for one public pandas result table.

    Logical kinds deliberately avoid exact pandas dtype strings, which vary between
    supported pandas versions. Validation never casts, reorders, or mutates a table.
    """

    schema_name: str
    schema_version: int
    required_columns: tuple[str, ...]
    dtypes: Mapping[str, str]
    unique_key: tuple[str, ...]
    orientation: str
    units: Mapping[str, str] = field(default_factory=dict)
    allow_extra_columns: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.schema_name, str) or not _SCHEMA_NAME.fullmatch(
            self.schema_name
        ):
            raise ValueError("table schema_name must use portable lower_snake_case")
        if (
            not isinstance(self.schema_version, int)
            or isinstance(self.schema_version, bool)
            or self.schema_version <= 0
        ):
            raise ValueError("table schema_version must be a positive integer")
        columns = _portable_names(
            self.required_columns, where="table required_columns")
        keys = _portable_names(self.unique_key, where="table unique_key")
        if not keys:
            raise ValueError("table unique_key must contain at least one column")
        if not set(keys).issubset(columns):
            raise ValueError("table unique_key columns must be required columns")
        dtypes = dict(self.dtypes)
        if set(dtypes) != set(columns):
            raise ValueError("table dtypes must declare every required column exactly once")
        unknown_kinds = {kind for kind in dtypes.values() if kind not in _COLUMN_KINDS}
        if unknown_kinds:
            raise ValueError(f"unknown logical table dtype(s): {sorted(unknown_kinds)}")
        units = dict(self.units)
        if not set(units).issubset(columns) or any(
            not isinstance(value, str) or not value for value in units.values()
        ):
            raise ValueError("table units must map required columns to non-empty strings")
        if not isinstance(self.orientation, str) or not self.orientation:
            raise ValueError("table orientation must be a non-empty string")
        if not isinstance(self.allow_extra_columns, bool):
            raise ValueError("table allow_extra_columns must be a boolean")
        object.__setattr__(self, "required_columns", columns)
        object.__setattr__(self, "unique_key", keys)
        object.__setattr__(self, "dtypes", MappingProxyType(dtypes))
        object.__setattr__(self, "units", MappingProxyType(units))

    @property
    def identifier(self) -> str:
        return f"{self.schema_name}/v{self.schema_version}"

    def _validate_series(self, column: str, series: pd.Series, kind: str) -> None:
        non_null = series.dropna()
        valid = True
        if kind == "string":
            valid = (
                ptypes.is_string_dtype(series.dtype)
                or ptypes.is_object_dtype(series.dtype)
            ) and all(isinstance(value, str) for value in non_null.tolist())
        elif kind == "boolean":
            valid = not series.isna().any() and (
                ptypes.is_bool_dtype(series.dtype)
                or bool(len(non_null))
                and all(
                    isinstance(value, (bool, np.bool_))
                    for value in non_null.tolist()
                )
            )
        elif kind == "integer":
            valid = not series.isna().any() and (
                ptypes.is_integer_dtype(series.dtype)
                or bool(len(non_null))
                and all(
                    isinstance(value, (int, np.integer))
                    and not isinstance(value, (bool, np.bool_))
                    for value in non_null.tolist()
                )
            )
        elif kind == "nullable_integer":
            valid = (
                ptypes.is_integer_dtype(series.dtype)
                or not len(non_null) and ptypes.is_float_dtype(series.dtype)
                or bool(len(non_null))
                and all(
                    (
                        isinstance(value, (int, np.integer))
                        and not isinstance(value, (bool, np.bool_))
                    )
                    or (
                        isinstance(value, (float, np.floating))
                        and float(value).is_integer()
                    )
                    for value in non_null.tolist()
                )
            )
        elif kind == "float":
            valid = ptypes.is_float_dtype(series.dtype) or (
                ptypes.is_object_dtype(series.dtype)
                and bool(len(non_null))
                and all(
                    isinstance(value, (float, np.floating))
                    for value in non_null.tolist()
                )
            )
        elif kind == "numeric":
            valid = ptypes.is_numeric_dtype(series.dtype) and not ptypes.is_bool_dtype(
                series.dtype)
        if not bool(valid):
            raise ValueError(
                f"table {self.identifier} column {column!r} must have logical "
                f"dtype {kind!r}; observed {series.dtype}")

    def validate(self, table: pd.DataFrame) -> None:
        if not isinstance(table, pd.DataFrame):
            raise ValueError(f"table {self.identifier} must be a pandas DataFrame")
        observed = tuple(table.columns)
        if len(set(observed)) != len(observed) or any(
            not isinstance(column, str) or not column for column in observed
        ):
            raise ValueError(
                f"table {self.identifier} columns must be unique non-empty strings")
        missing = [column for column in self.required_columns if column not in observed]
        if missing:
            raise ValueError(
                f"table {self.identifier} is missing required columns: {missing}")
        extras = [column for column in observed if column not in self.required_columns]
        if extras and not self.allow_extra_columns:
            raise ValueError(f"table {self.identifier} has unexpected columns: {extras}")
        declared_observed = tuple(
            column for column in observed if column in self.required_columns)
        if declared_observed != self.required_columns:
            raise ValueError(
                f"table {self.identifier} columns are not in canonical order")
        for column, kind in self.dtypes.items():
            self._validate_series(column, table[column], kind)
        if table.loc[:, list(self.unique_key)].isna().any().any():
            raise ValueError(f"table {self.identifier} unique key contains missing values")
        if table.duplicated(list(self.unique_key)).any():
            raise ValueError(f"table {self.identifier} unique key contains duplicates")

    def empty_frame(self) -> pd.DataFrame:
        pandas_dtypes = {
            "string": "string",
            "integer": "int64",
            "nullable_integer": "Int64",
            "float": "float64",
            "numeric": "float64",
            "boolean": "bool",
        }
        return pd.DataFrame({
            column: pd.Series(dtype=pandas_dtypes[self.dtypes[column]])
            for column in self.required_columns
        })

    def to_manifest(self) -> dict[str, object]:
        return {
            "name": self.schema_name,
            "version": self.schema_version,
            "required_columns": list(self.required_columns),
            "dtypes": dict(self.dtypes),
            "unique_key": list(self.unique_key),
            "orientation": self.orientation,
            "units": dict(self.units),
            "allow_extra_columns": self.allow_extra_columns,
        }

    @classmethod
    def from_manifest(cls, value: object) -> "TableContract":
        """Parse the exact portable representation emitted by :meth:`to_manifest`."""
        expected = {
            "name", "version", "required_columns", "dtypes", "unique_key",
            "orientation", "units", "allow_extra_columns",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError(
                f"table contract fields must be exactly {sorted(expected)}")
        if not isinstance(value["required_columns"], list) or not isinstance(
            value["unique_key"], list
        ):
            raise ValueError("table contract columns and unique_key must be arrays")
        if not isinstance(value["dtypes"], Mapping) or not isinstance(
            value["units"], Mapping
        ):
            raise ValueError("table contract dtypes and units must be objects")
        try:
            return cls(
                schema_name=value["name"],
                schema_version=value["version"],
                required_columns=tuple(value["required_columns"]),
                dtypes=value["dtypes"],
                unique_key=tuple(value["unique_key"]),
                orientation=value["orientation"],
                units=value["units"],
                allow_extra_columns=value["allow_extra_columns"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid portable TableContract") from exc


__all__ = ["TableContract"]
