"""Materialize local or Hugging Face data in PrefScope's canonical schema."""
from __future__ import annotations

import json
import os
from dataclasses import fields, replace
from pathlib import Path

import yaml

from prefscope.data.tabular import (
    ColumnMapping,
    canonicalize_table,
    hf_revision_provenance,
    load_hf_table,
    load_local_table,
    write_table,
)


def load_dataset_spec(path) -> dict:
    """Read a reusable YAML/JSON dataset-source and column-mapping specification."""
    path = Path(path)
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: dataset spec must be a mapping")
    return payload


def mapping_from_spec(spec: dict | None) -> ColumnMapping:
    """Build ``ColumnMapping`` from a compact reusable dataset spec."""
    spec = spec or {}
    columns = dict(spec.get("columns") or {})
    text = dict(spec.get("text") or {})
    label = dict(spec.get("label") or {})
    values = {
        "prompt": columns.get("prompt", "prompt"),
        "response_a": columns.get("response_a", "response"),
        "response_b": columns.get("response_b"),
        "label": columns.get("label"),
        "model_a": columns.get("model_a"),
        "model_b": columns.get("model_b"),
        "item_id": columns.get("item_id"),
        "group_id": columns.get("group_id"),
        "language": columns.get("language"),
        "metadata": tuple(columns.get("metadata") or ()),
        "prompt_role": text.get("prompt_role"),
        "response_a_role": text.get("response_a_role"),
        "response_b_role": text.get("response_b_role"),
        "label_mode": label.get("mode"),
        "a_values": tuple(label.get("a_values") or ()),
        "b_values": tuple(label.get("b_values") or ()),
        "tie_values": tuple(label.get("tie_values") or ()),
        "auto_pair": str(spec.get("mode", "auto")).casefold() != "single",
    }
    return ColumnMapping(**values)


def override_mapping(mapping: ColumnMapping, **overrides) -> ColumnMapping:
    """Apply non-None CLI overrides to a spec-derived mapping."""
    known = {item.name for item in fields(ColumnMapping)}
    clean = {key: value for key, value in overrides.items()
             if key in known and value is not None}
    for key in ("a_values", "b_values", "tie_values", "metadata"):
        if key in clean:
            clean[key] = tuple(clean[key])
    return replace(mapping, **clean)


def prepare_dataset(
    out,
    *,
    data=None,
    hf_dataset: str | None = None,
    hf_name: str | None = None,
    split: str = "train",
    revision: str | None = None,
    resolved_revision: str | None = None,
    token=None,
    token_env: str | None = None,
    streaming: bool = False,
    limit: int | None = None,
    mapping: ColumnMapping | None = None,
    drop_empty: bool = True,
) -> dict:
    """Load, map, validate, and write a canonical dataset plus provenance."""
    if (data is None) == (hf_dataset is None):
        raise ValueError("provide exactly one of data or hf_dataset")
    if token is not None and token_env is not None:
        raise ValueError("pass token or token_env, not both")
    if token_env is not None:
        token = os.environ.get(token_env)
    mapping = mapping or ColumnMapping()

    if hf_dataset is not None:
        raw = load_hf_table(
            hf_dataset,
            name=hf_name,
            split=split,
            revision=resolved_revision or revision,
            token=token,
            streaming=streaming,
            limit=limit,
        )
        source = f"hf://datasets/{hf_dataset}"
        loaded_revision = hf_revision_provenance(
            raw, resolved_revision or revision)["resolved_revision"]
        if resolved_revision is not None and loaded_revision != resolved_revision.lower():
            raise ValueError(
                "loaded Hugging Face dataset revision does not match the resolved "
                "analysis fingerprint")
        revision_info = {
            "requested_revision": revision,
            "resolved_revision": loaded_revision,
        }
        source_info = {
            "type": "huggingface",
            "dataset_id": hf_dataset,
            "name": hf_name,
            "split": split,
            # Keep the historical key as the user-requested ref. New consumers
            # should use the explicit requested/resolved pair below.
            "revision": revision,
            **revision_info,
            "streaming": bool(streaming),
            "limit": int(limit) if limit is not None else None,
        }
    else:
        raw = load_local_table(data)
        source = str(Path(data))
        if limit is not None:
            if int(limit) <= 0:
                raise ValueError("limit must be positive")
            raw = raw.iloc[:int(limit)].copy()
        source_info = {
            "type": "local",
            "path": str(Path(data)),
            "limit": int(limit) if limit is not None else None,
        }

    canonical, summary = canonicalize_table(
        raw, mapping, source=source, drop_empty=drop_empty)
    write_table(canonical, out)
    summary["output"] = str(Path(out))
    summary["source_spec"] = source_info
    manifest_path = Path(out).with_name(f"{Path(out).stem}.prefscope.json")
    manifest_path.write_text(json.dumps(summary, indent=2))
    summary["manifest"] = str(manifest_path)
    return summary
