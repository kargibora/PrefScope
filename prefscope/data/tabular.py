"""Load and canonicalize arbitrary local or Hugging Face text datasets.

This module is deliberately upstream of lenses and analyses.  A source is loaded
once and mapped into PrefScope's canonical table:

``prompt · completion_a [· completion_b] [· human_pref] [· model_a · model_b]``

``human_pref`` always means P(A preferred).  Winner labels are never guessed:
callers must either provide probabilities, declare explicit winner tokens, or
state that A is the known chosen response.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
import hashlib
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd

from prefscope.data import pair_schema

SUPPORTED_SUFFIXES = (".parquet", ".csv", ".jsonl", ".json")
_HF_SOURCE_ATTR = "prefscope_hf_source"
_COMMIT_SHA = re.compile(r"^[0-9a-fA-F]{40}$")
_CANONICAL_HASH_VERSION = "prefscope-canonical-table-v1"


def _commit_sha(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text.lower() if _COMMIT_SHA.fullmatch(text) else None


def _resolve_hf_revision(dataset_id: str, revision: str | None, token=None) -> str:
    """Resolve a mutable Hub ref to the exact dataset repository commit."""
    explicit = _commit_sha(revision)
    if explicit is not None:
        return explicit
    try:
        from huggingface_hub import HfApi

        info = HfApi().dataset_info(
            repo_id=dataset_id, revision=revision, token=token)
    except Exception as exc:
        raise ValueError(
            f"could not resolve Hugging Face dataset {dataset_id!r} revision "
            f"{revision!r} to an immutable commit SHA; connect to the Hub or pass "
            "an exact 40-character commit revision") from exc
    resolved = _commit_sha(getattr(info, "sha", None))
    if resolved is None:
        raise ValueError(
            f"Hugging Face returned no immutable commit SHA for dataset "
            f"{dataset_id!r} revision {revision!r}")
    return resolved


def _hash_field(hasher, marker: bytes, payload: bytes = b"") -> None:
    hasher.update(marker)
    hasher.update(len(payload).to_bytes(8, "big"))
    hasher.update(payload)


def _hash_scalar(hasher, value) -> None:
    try:
        missing = bool(pd.isna(value))
    except (TypeError, ValueError):
        missing = False
    if missing:
        _hash_field(hasher, b"n")
    elif isinstance(value, (bool, np.bool_)):
        _hash_field(hasher, b"b", b"1" if bool(value) else b"0")
    elif isinstance(value, (int, np.integer)):
        _hash_field(hasher, b"i", str(int(value)).encode("ascii"))
    elif isinstance(value, (float, np.floating)):
        _hash_field(hasher, b"f", float(value).hex().encode("ascii"))
    elif isinstance(value, (pd.Timestamp, np.datetime64, datetime, date)):
        timestamp = pd.Timestamp(value)
        _hash_field(hasher, b"t", timestamp.isoformat().encode("utf-8"))
    elif isinstance(value, (pd.Timedelta, np.timedelta64)):
        duration = pd.Timedelta(value)
        _hash_field(hasher, b"d", str(duration.value).encode("ascii"))
    elif isinstance(value, Decimal):
        _hash_field(hasher, b"m", format(value, "f").encode("ascii"))
    elif isinstance(value, str):
        _hash_field(hasher, b"s", value.encode("utf-8"))
    else:
        raise ValueError(
            f"canonical table contains unsupported value type "
            f"{type(value).__name__}; canonical fields must be scalar")


def canonical_table_hash(frame: pd.DataFrame) -> str:
    """Hash ordered canonical content, independent of the source location.

    The versioned, length-delimited encoding binds column names, row order, IDs,
    text, labels, models, language, and explicitly retained scalar metadata. The mutable ``source`` location is excluded
    because it is recorded separately as provenance and is not dataset content.
    """
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("canonical table hash needs a pandas DataFrame")
    columns = [str(column) for column in frame.columns if column != "source"]
    hasher = hashlib.sha256()
    _hash_field(hasher, b"v", _CANONICAL_HASH_VERSION.encode("ascii"))
    for column in columns:
        _hash_field(hasher, b"c", column.encode("utf-8"))
    _hash_field(hasher, b"r", str(len(frame)).encode("ascii"))
    for row in frame[columns].itertuples(index=False, name=None):
        _hash_field(hasher, b"[")
        for value in row:
            _hash_scalar(hasher, value)
        _hash_field(hasher, b"]")
    return f"sha256:{hasher.hexdigest()}"


def hf_revision_provenance(
    frame: pd.DataFrame, requested_revision: str | None,
) -> dict:
    """Read the immutable Hub revision attached by :func:`load_hf_table`.

    A 40-character requested revision is already immutable and also supports
    lightweight offline fakes that return an ordinary DataFrame.
    """
    metadata = frame.attrs.get(_HF_SOURCE_ATTR, {}) if isinstance(frame, pd.DataFrame) else {}
    resolved = _commit_sha(metadata.get("resolved_revision"))
    if resolved is None:
        resolved = _commit_sha(requested_revision)
    if resolved is None:
        raise ValueError(
            "Hugging Face table has no resolved immutable dataset commit SHA; "
            "load it with load_hf_table or pass an exact 40-character revision")
    return {
        "requested_revision": requested_revision,
        "resolved_revision": resolved,
    }


def load_local_table(path) -> pd.DataFrame:
    """Read a supported local tabular file."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".jsonl", ".json"}:
        return pd.read_json(path, lines=suffix == ".jsonl")
    raise ValueError(
        f"unsupported data format {suffix!r}; use one of {list(SUPPORTED_SUFFIXES)}")


def load_hf_table(
    dataset_id: str,
    *,
    name: str | None = None,
    split: str = "train",
    revision: str | None = None,
    token=None,
    streaming: bool = False,
    limit: int | None = None,
) -> pd.DataFrame:
    """Load one Hugging Face dataset split as a DataFrame.

    ``datasets`` is an optional dependency (``pip install prefscope[arena]``).
    Streaming is useful for inspecting or analyzing a bounded sample and therefore
    requires ``limit``; silently exhausting an unbounded iterable would defeat the
    point of streaming.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Hugging Face dataset loading needs the optional 'arena' extra: "
            "pip install 'prefscope[arena]'") from exc
    if not dataset_id or "/" not in str(dataset_id):
        raise ValueError("dataset_id must be a Hugging Face repository id such as owner/name")
    if limit is not None and int(limit) <= 0:
        raise ValueError("limit must be positive")
    if streaming and limit is None:
        raise ValueError(
            "streaming=True requires a finite limit; otherwise conversion to a table "
            "would consume the entire stream")

    requested_revision = revision
    resolved_revision = _resolve_hf_revision(dataset_id, revision, token=token)
    # Load the resolved SHA rather than the mutable requested ref. This avoids a
    # time-of-check/time-of-use race if a branch moves between resolution and load.
    dataset = load_dataset(
        dataset_id,
        name=name,
        split=split,
        revision=resolved_revision,
        token=token,
        streaming=bool(streaming),
    )
    if streaming:
        frame = pd.DataFrame(list(dataset.take(int(limit))))
    else:
        if limit is not None:
            dataset = dataset.select(range(min(int(limit), len(dataset))))
        frame = dataset.to_pandas()
    frame.attrs[_HF_SOURCE_ATTR] = {
        "requested_revision": requested_revision,
        "resolved_revision": resolved_revision,
    }
    return frame


def write_table(frame: pd.DataFrame, path) -> None:
    """Write a canonical table in a format accepted by the inference commands."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        frame.to_parquet(path, index=False)
    elif suffix == ".csv":
        frame.to_csv(path, index=False)
    elif suffix == ".jsonl":
        frame.to_json(path, orient="records", lines=True)
    elif suffix == ".json":
        frame.to_json(path, orient="records")
    else:
        raise ValueError(
            f"unsupported output format {suffix!r}; use one of "
            f"{list(SUPPORTED_SUFFIXES)}")


@dataclass(frozen=True)
class ColumnMapping:
    """Map arbitrary source columns and structured-message selectors.

    A role selector has the form ``ROLE:first`` or ``ROLE:last``.  For example,
    preference datasets whose ``chosen`` and ``rejected`` columns are chat-message
    lists can use the same ``chosen`` column twice:

    ``prompt="chosen", prompt_role="user:first"`` and
    ``response_a="chosen", response_a_role="assistant:last"``.
    """

    prompt: str = "prompt"
    response_a: str = "response"
    response_b: str | None = None
    label: str | None = None
    model_a: str | None = None
    model_b: str | None = None
    item_id: str | None = None
    language: str | None = None
    metadata: tuple[str, ...] = field(default_factory=tuple)
    prompt_role: str | None = None
    response_a_role: str | None = None
    response_b_role: str | None = None
    label_mode: str | None = None
    a_values: tuple[str, ...] = field(default_factory=tuple)
    b_values: tuple[str, ...] = field(default_factory=tuple)
    tie_values: tuple[str, ...] = field(default_factory=tuple)
    auto_pair: bool = True
    group_id: str | None = None


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, dict, np.ndarray)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _content_text(value) -> str:
    if _is_missing(value):
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if "text" in value:
            return _content_text(value["text"])
        if "content" in value:
            return _content_text(value["content"])
        raise ValueError(f"message content object has no text/content field: {value!r}")
    if isinstance(value, (list, tuple)):
        return "".join(_content_text(part) for part in value)
    return str(value)


def _message_text(value, selector: str) -> str:
    try:
        role, occurrence = selector.rsplit(":", 1)
    except ValueError as exc:
        raise ValueError(
            f"role selector {selector!r} must be ROLE:first or ROLE:last") from exc
    role = role.strip().casefold()
    occurrence = occurrence.strip().casefold()
    if not role or occurrence not in {"first", "last"}:
        raise ValueError(
            f"role selector {selector!r} must be ROLE:first or ROLE:last")

    if isinstance(value, dict) and "messages" in value:
        value = value["messages"]
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            f"role selector {selector!r} needs a list of message objects, got "
            f"{type(value).__name__}")
    matches = [
        message for message in value
        if isinstance(message, dict)
        and str(message.get("role", message.get("from", ""))).strip().casefold() == role
    ]
    if not matches:
        return ""
    message = matches[0] if occurrence == "first" else matches[-1]
    content = message.get("content", message.get("value", message.get("text", "")))
    return _content_text(content)


def _text_series(series: pd.Series, selector: str | None, *, column: str) -> pd.Series:
    def one(value):
        if _is_missing(value):
            return pd.NA
        if selector is not None:
            text = _message_text(value, selector)
        else:
            if isinstance(value, (list, tuple, dict, np.ndarray)):
                raise ValueError(
                    f"column {column!r} contains structured data; add a role selector "
                    "such as user:first or assistant:last")
            text = str(value)
        text = text.strip()
        return text if text else pd.NA

    return series.map(one).astype("string")


def _label_key(value) -> str:
    return str(value).strip().casefold()


def normalize_preference_labels(
    labels: pd.Series | None,
    *,
    mode: str | None,
    n_rows: int,
    a_values: Iterable = (),
    b_values: Iterable = (),
    tie_values: Iterable = (),
) -> pd.Series | None:
    """Normalize labels to P(A preferred), preserving missing labels as NaN."""
    if mode is None:
        if labels is None:
            return None
        raise ValueError(
            "a label column was mapped but label_mode is unset; choose "
            "'probability', 'winner', or 'a-wins'")
    mode = str(mode).strip().casefold()
    if mode not in {"probability", "winner", "a-wins"}:
        raise ValueError("label_mode must be one of: probability, winner, a-wins")
    if mode == "a-wins":
        if labels is not None:
            raise ValueError("label_mode='a-wins' does not take a label column")
        return pd.Series(np.ones(int(n_rows), dtype=float))
    if labels is None:
        raise ValueError(f"label_mode={mode!r} requires a label column")

    missing = labels.map(_is_missing)
    if mode == "probability":
        try:
            result = pd.to_numeric(labels, errors="raise").astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "probability labels must be numeric values in [0, 1]") from exc
        result[missing] = np.nan
        bad = result.notna() & ~result.between(0.0, 1.0)
        if bad.any():
            raise ValueError(
                f"probability labels outside [0, 1]: "
                f"{result[bad].drop_duplicates().head(10).tolist()}")
        return result

    groups = {
        1.0: {_label_key(value) for value in a_values},
        0.0: {_label_key(value) for value in b_values},
        0.5: {_label_key(value) for value in tie_values},
    }
    if not groups[1.0] or not groups[0.0]:
        raise ValueError(
            "winner labels require explicit a_values and b_values; this prevents "
            "silently reversing a dataset's preference convention")
    overlap = (groups[1.0] & groups[0.0]) | (groups[1.0] & groups[0.5]) | (
        groups[0.0] & groups[0.5])
    if overlap:
        raise ValueError(f"winner token sets overlap: {sorted(overlap)}")
    lookup = {key: value for value, keys in groups.items() for key in keys}
    keys = labels.map(lambda value: None if _is_missing(value) else _label_key(value))
    unknown = sorted({key for key in keys if key is not None and key not in lookup})
    if unknown:
        raise ValueError(
            f"unmapped winner label(s) {unknown[:10]}; declare their A/B/tie meaning")
    return keys.map(lambda key: np.nan if key is None else lookup[key]).astype(float)


def _resolve_column(
    frame: pd.DataFrame,
    requested: str | None,
    *,
    aliases: tuple[str, ...] = (),
) -> str | None:
    if requested is not None and requested in frame.columns:
        return requested
    if requested is not None:
        canonical = pair_schema.ENCODE_ALIASES.get(requested)
        if canonical in frame.columns:
            return canonical
        if requested not in aliases:
            return requested
    return next((column for column in aliases if column in frame.columns), None)


def canonicalize_table(
    frame: pd.DataFrame,
    mapping: ColumnMapping,
    *,
    source: str = "table",
    drop_empty: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Map a raw table into the canonical PrefScope inference schema."""
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)
    raw = frame.reset_index(drop=True)
    prompt = _resolve_column(raw, mapping.prompt, aliases=(pair_schema.PROMPT,))
    response_a = _resolve_column(
        raw, mapping.response_a, aliases=("response", pair_schema.RESPONSE_A))
    response_b = _resolve_column(
        raw, mapping.response_b,
        aliases=("response_2", pair_schema.RESPONSE_B)
        if mapping.response_b is None and mapping.auto_pair else ())
    # A generic single-response classification dataset may legitimately have a column
    # named "label"; only auto-detect preference labels when the table is paired.
    if mapping.label_mode == "a-wins" and mapping.label is None:
        label = None
    else:
        label = _resolve_column(
            raw, mapping.label,
            aliases=("label", pair_schema.LABEL)
            if mapping.label is None and response_b is not None else ())
    model_a = _resolve_column(
        raw, mapping.model_a,
        aliases=("model", pair_schema.MODEL_A) if mapping.model_a is None else ())
    model_b = _resolve_column(
        raw, mapping.model_b,
        aliases=("model_2", pair_schema.MODEL_B) if mapping.model_b is None else ())
    group_id = _resolve_column(
        raw, mapping.group_id,
        aliases=("group_id",) if mapping.group_id is None else ())

    unresolved = [
        name for name, column in (("prompt", prompt), ("response_a", response_a))
        if column is None
    ]
    if unresolved:
        raise ValueError(
            f"could not resolve required field(s) {unresolved}; configure ColumnMapping "
            f"for available columns: {list(raw.columns)}")
    required = [prompt, response_a]
    optional = [
        response_b, label, model_a, model_b, mapping.item_id, group_id,
        mapping.language,
    ]
    missing = [column for column in required + optional
               if column is not None and column not in raw.columns]
    metadata = tuple(
        column for column in mapping.metadata if column != group_id
    )
    if any(not isinstance(column, str) or not column for column in metadata):
        raise ValueError("metadata columns must be non-empty strings")
    if len(set(metadata)) != len(metadata):
        raise ValueError("metadata columns must be unique")
    missing_metadata = [column for column in metadata if column not in raw.columns]
    if missing_metadata:
        raise ValueError(
            f"metadata column(s) not found: {missing_metadata}; available: "
            f"{list(raw.columns)}")
    if missing:
        raise ValueError(
            f"mapped column(s) not found: {missing}; available: {list(raw.columns)}")
    if response_b is None and mapping.label_mode == "a-wins":
        raise ValueError("label_mode='a-wins' requires a second response")

    out = pd.DataFrame({
        "row_id": np.arange(len(raw), dtype=int),
        pair_schema.PROMPT: _text_series(
            raw[prompt], mapping.prompt_role, column=prompt),
        pair_schema.RESPONSE_A: _text_series(
            raw[response_a], mapping.response_a_role, column=response_a),
    })
    if mapping.item_id is not None:
        out["item_id"] = raw[mapping.item_id].astype("string")
    else:
        out["item_id"] = out["row_id"].astype("string")
    invalid_ids = out["item_id"].isna() | (out["item_id"].str.strip().str.len() == 0)
    if invalid_ids.any():
        raise ValueError("item IDs must be nonmissing, non-empty identifiers")
    if out["item_id"].duplicated().any():
        duplicated = out.loc[out["item_id"].duplicated(False), "item_id"].head(10)
        raise ValueError(f"item IDs must be unique; duplicates: {duplicated.tolist()}")
    if group_id is not None:
        out["group_id"] = raw[group_id].astype("string")
        invalid_groups = (
            out["group_id"].isna()
            | (out["group_id"].str.strip().str.len() == 0)
        )
        if invalid_groups.any():
            raise ValueError("group IDs must be nonmissing, non-empty identifiers")
    if response_b is not None:
        out[pair_schema.RESPONSE_B] = _text_series(
            raw[response_b], mapping.response_b_role, column=response_b)

    resolved_mode = mapping.label_mode
    if label == pair_schema.LABEL and resolved_mode is None:
        resolved_mode = "probability"
    normalized = normalize_preference_labels(
        raw[label] if label is not None else None,
        mode=resolved_mode,
        n_rows=len(raw),
        a_values=mapping.a_values,
        b_values=mapping.b_values,
        tie_values=mapping.tie_values,
    )
    if normalized is not None:
        out[pair_schema.LABEL] = normalized

    for src, dst in ((model_a, pair_schema.MODEL_A), (model_b, pair_schema.MODEL_B)):
        if src is not None:
            out[dst] = raw[src].astype("string")
    out["source"] = str(source)
    if mapping.language is not None:
        out["language"] = raw[mapping.language].astype("string")
    collisions = sorted(set(metadata).intersection(out.columns))
    if collisions:
        raise ValueError(
            f"metadata columns collide with canonical fields: {collisions}")
    for column in metadata:
        out[column] = raw[column].reset_index(drop=True)

    required_text = [pair_schema.PROMPT, pair_schema.RESPONSE_A]
    if response_b is not None:
        required_text.append(pair_schema.RESPONSE_B)
    good = pd.Series(True, index=out.index)
    for column in required_text:
        good &= out[column].notna() & (out[column].str.len() > 0)
    n_dropped = int((~good).sum())
    if n_dropped and not drop_empty:
        bad_rows = out.loc[~good, "row_id"].head(10).tolist()
        raise ValueError(f"empty required text in source row(s) {bad_rows}")
    out = out.loc[good].reset_index(drop=True)
    if out.empty:
        raise ValueError("no rows remain after mapping required prompt/response text")

    summary = {
        "source": str(source),
        "input_rows": int(len(raw)),
        "output_rows": int(len(out)),
        "dropped_empty_rows": n_dropped,
        "mode": "paired" if response_b is not None else "single",
        "has_preference": bool(
            pair_schema.LABEL in out.columns and out[pair_schema.LABEL].notna().any()),
        "columns": list(out.columns),
        "canonical_table_hash": canonical_table_hash(out),
        "canonical_table_hash_version": _CANONICAL_HASH_VERSION,
        "mapping": {
            "prompt": prompt,
            "response_a": response_a,
            "response_b": response_b,
            "label": label,
            "label_mode": resolved_mode,
            "model_a": model_a,
            "model_b": model_b,
            "group_id": group_id,
        },
    }
    return out, summary
