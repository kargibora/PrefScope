"""Independent prompt-group identifiers for grouped statistical inference."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def normalized_prompt(value) -> str:
    """Normalize only transport-level text differences, not semantic content."""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def prompt_group_hash(value) -> str:
    return hashlib.sha256(normalized_prompt(value).encode("utf-8")).hexdigest()[:16]


def validate_group_ids(group_ids, n_rows: int, *, name: str = "group_ids") -> np.ndarray:
    """Validate aligned nonmissing scalar IDs without changing public identities."""
    try:
        materialized = list(group_ids)
    except TypeError as exc:
        raise ValueError(f"{name} must be a one-dimensional iterable") from exc
    values = np.empty(len(materialized), dtype=object)
    values[:] = materialized
    if len(values) != int(n_rows):
        raise ValueError(f"{name} must have one entry per row")
    if pd.isna(values).any():
        raise ValueError(f"{name} must not contain missing values")
    for value in values:
        try:
            hash(value)
        except TypeError as exc:
            raise ValueError(f"{name} must contain hashable scalar values") from exc
    return values


def typed_group_keys(group_ids, n_rows: int | None = None) -> np.ndarray:
    """Return hashable type-stable keys, distinguishing ``1``, ``True``, and ``1.0``."""
    if n_rows is None:
        try:
            n_rows = len(group_ids)
        except TypeError:
            group_ids = tuple(group_ids)
            n_rows = len(group_ids)
    values = validate_group_ids(group_ids, n_rows)
    keys = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        kind = type(value)
        keys[index] = (kind.__module__, kind.__qualname__, value)
    return keys


def factorize_group_ids(group_ids, n_rows: int | None = None) -> tuple[np.ndarray, int]:
    """Factorize type-stable independent-group IDs in first-seen order."""
    keys = typed_group_keys(group_ids, n_rows)
    codes, labels = pd.factorize(keys, sort=False)
    if bool((codes < 0).any()):
        raise ValueError("group_ids must not contain missing values")
    return codes.astype(int), int(len(labels))


def resolve_group_ids(
    frame: pd.DataFrame,
    *,
    group_col: str | None = None,
    prompt_col: str = "prompt",
) -> np.ndarray | None:
    """Return aligned independent-group ids, deriving them from prompt text when possible.

    An explicit ``group_col`` fails closed. Without one, a canonical ``group_id`` column
    is preferred; otherwise identical normalized prompts receive the same stable hash.
    ``None`` means the table carries no defensible grouping information, so callers must
    retain their documented row-independent compatibility behavior.
    """
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("group resolution needs a pandas DataFrame")
    selected = group_col
    if selected is None and "group_id" in frame.columns:
        selected = "group_id"
    if selected is not None:
        if selected not in frame.columns:
            raise ValueError(
                f"group column {selected!r} is absent; available: {list(frame.columns)}")
        values = frame[selected]
        if values.isna().any():
            raise ValueError(f"group column {selected!r} contains missing values")
        array = values.to_numpy(dtype=object)
        return validate_group_ids(
            array, len(frame), name=f"group column {selected!r}")
    if prompt_col in frame.columns:
        prompts = frame[prompt_col]
        if prompts.isna().any():
            raise ValueError(f"prompt column {prompt_col!r} contains missing values")
        return prompts.map(prompt_group_hash).to_numpy(dtype=str)
    return None


__all__ = [
    "normalized_prompt", "prompt_group_hash", "validate_group_ids",
    "typed_group_keys", "factorize_group_ids", "resolve_group_ids",
]
