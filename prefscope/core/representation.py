"""Torch-free contracts for interchangeable representation sources.

A representation source maps normalized :class:`PairItem` rows to one or more
aligned fixed-width matrices.  Text embedders, residual-activation extractors,
precomputed arrays, and user-defined functions can all implement this contract.
The downstream lens and analysis layers depend on the batch contract, not on the
source implementation.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
from numbers import Integral, Real
import os
import re
from types import MappingProxyType
from typing import Callable, Iterable, Mapping
from urllib.parse import parse_qsl, urlsplit

import numpy as np

from prefscope.core.types import PairItem


class _FrozenDict(dict):
    """JSON-serializable dict whose mutation methods fail."""

    def _immutable(self, *args, **kwargs):
        raise TypeError("portable provenance is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable


def _deep_freeze_json(value):
    if isinstance(value, dict):
        return _FrozenDict({key: _deep_freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze_json(item) for item in value)
    return value


def validate_portable_mapping(
    value: Mapping | None, *, where: str
) -> MappingProxyType:
    resolved = dict(value or {})
    try:
        json.dumps(resolved, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where} must be JSON-serializable") from exc
    sensitive = {
        "token", "accesstoken", "hftoken", "apikey", "password", "secret",
        "clientsecret", "authorization", "cookie", "credentials", "credential",
    }

    def normalized_key(key) -> str:
        return re.sub(r"[^a-z0-9]", "", str(key).casefold())

    def key_is_sensitive(key) -> bool:
        normalized = normalized_key(key)
        return (
            normalized in sensitive
            or normalized.endswith(("token", "password", "secret", "credential"))
            or normalized.endswith("apikey")
        )

    def sensitive_keys(item):
        found = set()
        if isinstance(item, dict):
            for key, nested in item.items():
                if key_is_sensitive(key):
                    found.add(str(key).casefold())
                found.update(sensitive_keys(nested))
        elif isinstance(item, (list, tuple)):
            for nested in item:
                found.update(sensitive_keys(nested))
        elif isinstance(item, str):
            parsed = urlsplit(item)
            if parsed.username is not None or parsed.password is not None:
                found.add("url_userinfo")
            for key, _ in parse_qsl(parsed.query, keep_blank_values=True):
                if key_is_sensitive(key):
                    found.add(f"url_query:{key.casefold()}")
        return found

    def local_paths(item):
        found = []
        if isinstance(item, dict):
            for key, nested in item.items():
                found.extend(local_paths(str(key)))
                found.extend(local_paths(nested))
        elif isinstance(item, (list, tuple)):
            for nested in item:
                found.extend(local_paths(nested))
        elif isinstance(item, str):
            if (
                os.path.isabs(item)
                or bool(re.match(r"^[A-Za-z]:[\\/]", item))
                or item.startswith(("\\", "//"))
                or item.casefold().startswith("file://")
            ):
                found.append(item)
        return found

    def validate_keys(item):
        if isinstance(item, dict):
            for key, nested in item.items():
                if not isinstance(key, str) or not key:
                    raise ValueError(
                        f"{where} keys must be non-empty strings")
                validate_keys(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                validate_keys(nested)

    validate_keys(resolved)
    found = sensitive_keys(resolved)
    if found:
        raise ValueError(
            f"{where} must not contain credential values: {sorted(found)}")
    paths = local_paths(resolved)
    if paths:
        raise ValueError(f"{where} must not contain absolute local paths")
    frozen = {key: _deep_freeze_json(item) for key, item in resolved.items()}
    return MappingProxyType(frozen)


# Private alias retained for the first alpha API wave.
_portable_mapping = validate_portable_mapping


def _aligned_metadata(value: Mapping | None, n_rows: int) -> MappingProxyType:
    resolved = {}
    for name, column in dict(value or {}).items():
        if not isinstance(name, str) or not name:
            raise ValueError("metadata column names must be non-empty strings")
        values = tuple(column)
        if len(values) != n_rows:
            raise ValueError(
                f"metadata column {name!r} has {len(values)} rows; expected {n_rows}")
        canonical = []
        kinds = set()
        for item in values:
            item_type = type(item)
            is_missing = item is None
            if isinstance(item, (float, np.floating)) and np.isnan(item):
                is_missing = True
            if (
                item_type.__module__.startswith("pandas.")
                and item_type.__name__ in {"NAType", "NaTType"}
            ):
                is_missing = True
            if is_missing:
                canonical.append(None)
                continue
            if isinstance(item, (bool, np.bool_)):
                canonical.append(bool(item))
                kinds.add("bool")
            elif isinstance(item, str):
                canonical.append(str(item))
                kinds.add("str")
            elif isinstance(item, Integral):
                numeric = int(item)
                if not -(2 ** 63) <= numeric < 2 ** 63:
                    raise ValueError(
                        f"metadata column {name!r} integers must fit signed int64")
                canonical.append(numeric)
                kinds.add("int")
            elif isinstance(item, Real):
                numeric = float(item)
                if not np.isfinite(numeric):
                    raise ValueError(
                        f"metadata column {name!r} must contain finite values")
                canonical.append(numeric)
                kinds.add("float")
            else:
                raise ValueError(
                    f"metadata column {name!r} must contain portable scalar values")
        if len(kinds) > 1:
            raise ValueError(
                f"metadata column {name!r} must use one portable scalar type")
        resolved[name] = tuple(canonical)
    return MappingProxyType(resolved)


def validate_row_ids(values) -> tuple[str, ...]:
    raw = tuple(values)
    if not raw:
        raise ValueError("row_ids must contain at least one identifier")
    missing = []
    for value in raw:
        is_missing = value is None
        if isinstance(value, (float, np.floating)) and np.isnan(value):
            is_missing = True
        value_type = type(value)
        if (
            value_type.__module__.startswith("pandas.")
            and value_type.__name__ in {"NAType", "NaTType"}
        ):
            is_missing = True
        missing.append(is_missing)
    if any(missing):
        raise ValueError("row_ids must not contain missing identifiers")
    ids = tuple(str(value) for value in raw)
    if any(not value.strip() for value in ids):
        raise ValueError("row_ids must contain non-empty identifiers")
    if len(set(ids)) != len(ids):
        raise ValueError("row_ids must be unique")
    return ids


# Internal alias retained for compatibility with the first alpha API wave.
_row_ids = validate_row_ids


@dataclass(frozen=True, eq=False)
class RepresentationBatch:
    """Aligned fixed-width vectors produced by one representation source.

    ``arrays`` uses semantic names rather than a source-specific class. Built-in
    text sources publish ``prompt``, ``response_a``, and optionally
    ``response_b``. A custom source may add other arrays, but every array must be
    a finite numeric ``(n_rows, width)`` matrix aligned to ``row_ids``.
    """

    row_ids: tuple[str, ...]
    arrays: Mapping[str, np.ndarray]
    metadata: Mapping[str, tuple[object, ...]] = field(default_factory=dict)
    granularity: str = "item"
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ids = validate_row_ids(self.row_ids)
        arrays = dict(self.arrays)
        if self.granularity != "item":
            raise ValueError(
                "representation granularity must currently be 'item'; pool token or "
                "sequence activations in the source before constructing a batch")
        if not arrays:
            raise ValueError("representation batch must contain at least one array")
        checked: dict[str, np.ndarray] = {}
        for name, value in arrays.items():
            if not isinstance(name, str) or not name or "/" in name or "\\" in name:
                raise ValueError(
                    "representation array names must be non-empty portable strings")
            source = np.asarray(value)
            if source.ndim != 2 or source.shape[0] != len(ids):
                raise ValueError(
                    f"representation array {name!r} must have shape "
                    f"({len(ids)}, width), got {source.shape}")
            if (
                source.shape[1] <= 0
                or not np.issubdtype(source.dtype, np.number)
                or np.issubdtype(source.dtype, np.complexfloating)
            ):
                raise ValueError(
                    f"representation array {name!r} must be a non-empty real numeric "
                    "matrix")
            with np.errstate(over="ignore", invalid="ignore"):
                matrix = np.array(source, dtype=np.float32, order="C", copy=True)
            if not np.isfinite(matrix).all():
                raise ValueError(
                    f"representation array {name!r} contains non-finite values or "
                    "cannot be represented as float32")
            matrix = np.frombuffer(
                matrix.tobytes(order="C"), dtype=matrix.dtype
            ).reshape(matrix.shape)
            checked[name] = matrix
        object.__setattr__(self, "row_ids", ids)
        object.__setattr__(self, "arrays", MappingProxyType(checked))
        object.__setattr__(self, "metadata", _aligned_metadata(self.metadata, len(ids)))
        object.__setattr__(
            self, "provenance", validate_portable_mapping(self.provenance, where="provenance"))

    @property
    def n_rows(self) -> int:
        return len(self.row_ids)

    def array(self, name: str) -> np.ndarray:
        try:
            return self.arrays[name]
        except KeyError:
            available = ", ".join(sorted(self.arrays))
            raise ValueError(
                f"representation batch has no array {name!r}; available: {available}")                 from None

    def subset(self, indices) -> "RepresentationBatch":
        index = np.asarray(indices)
        if index.ndim != 1:
            raise ValueError("subset indices must be one-dimensional")
        if index.dtype == bool:
            if len(index) != self.n_rows:
                raise ValueError("boolean subset mask must have one entry per row")
            positions = np.flatnonzero(index)
        else:
            if not np.issubdtype(index.dtype, np.integer):
                raise ValueError("subset indices must be integers or a boolean mask")
            positions = index.astype(int)
            if ((positions < 0) | (positions >= self.n_rows)).any():
                raise ValueError("subset index is outside the representation batch")
        if len(np.unique(positions)) != len(positions):
            raise ValueError("subset indices must not duplicate rows")
        return RepresentationBatch(
            row_ids=tuple(self.row_ids[position] for position in positions),
            arrays={name: value[positions] for name, value in self.arrays.items()},
            metadata={
                name: tuple(values[position] for position in positions)
                for name, values in self.metadata.items()
            },
            granularity=self.granularity,
            provenance=dict(self.provenance),
        )


class RepresentationSource(ABC):
    """Pluggable producer of aligned fixed-width representations."""

    @abstractmethod
    def encode(self, items: Iterable[PairItem]) -> RepresentationBatch:
        """Encode normalized items into one aligned representation batch."""


class CallableRepresentationSource(RepresentationSource):
    """Adapt a user function without requiring a framework-specific subclass."""

    def __init__(
        self,
        function: Callable[[list[PairItem]], RepresentationBatch],
        *,
        name: str = "callable",
        provenance: Mapping[str, object] | None = None,
    ) -> None:
        if not callable(function):
            raise ValueError("function must be callable")
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty string")
        self.function = function
        self.name = name
        self.source_provenance = validate_portable_mapping(
            provenance, where="source provenance")

    def encode(self, items: Iterable[PairItem]) -> RepresentationBatch:
        materialized = list(items)
        result = self.function(materialized)
        if not isinstance(result, RepresentationBatch):
            raise ValueError("representation function must return RepresentationBatch")
        expected_ids = validate_row_ids(item.id for item in materialized)
        if result.row_ids != expected_ids:
            raise ValueError(
                "representation function row_ids must exactly match input item order")
        provenance = {
            **dict(result.provenance),
            **dict(self.source_provenance),
            "source_type": "callable",
            "source_name": self.name,
        }
        return RepresentationBatch(
            row_ids=result.row_ids,
            arrays=result.arrays,
            metadata=result.metadata,
            granularity=result.granularity,
            provenance=provenance,
        )


__all__ = [
    "RepresentationBatch", "RepresentationSource", "CallableRepresentationSource",
]
