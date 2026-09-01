"""Validated configuration for the published-lens analysis workflow."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

import yaml

from prefscope.analysis.presence import PRESENCE_POLICIES
from prefscope.artifacts import WIN_RELEVANCE
from prefscope.config import VIEWER_EXPORT_DEFAULTS

_TOP_KEYS = {
    "version", "lenses", "data", "out_dir", "device", "embedding",
    "concepts", "analysis", "viewer",
}
_LENS_KEYS = {
    "repo", "revision", "completion", "prompt", "completion_subfolder",
    "prompt_subfolder", "completion_annotations", "prompt_annotations",
    "token_env", "cache_dir", "local_files_only",
}
_SOURCE_KEYS = {
    "type", "path", "dataset_id", "name", "split", "revision", "token_env",
    "streaming", "limit",
}
_DATA_KEYS = {"source", "columns", "text", "label", "mode", "drop_empty"}
_COLUMN_KEYS = {
    "prompt", "response_a", "response_b", "label", "model_a", "model_b",
    "item_id", "group_id", "language", "metadata",
}
_TEXT_KEYS = {"prompt_role", "response_a_role", "response_b_role"}
_LABEL_KEYS = {"mode", "a_values", "b_values", "tie_values"}
_EMBEDDING_KEYS = {"backend", "batch_size", "cache_dir"}
_CONCEPT_KEYS = {
    "presence_policy", "fidelity_only", "named_only", "top_k", "include_text",
    "chunk_size",
}
_ANALYSIS_KEYS = {
    "relationships", "comparison", "preference", "outcomes", "min_support",
    "min_cooccur", "min_context_pairs", "group_col", "examples_per_direction",
    "side_a_name", "side_b_name",
}
_OUTCOME_ANALYSIS_KEYS = {
    "columns", "kind", "normalization", "code_array", "min_units", "output",
}
_RESERVED_ANALYSIS_FILENAMES = {
    "analysis_state.json", "analyze.resolved.yaml", "dataset.parquet",
    "response_concepts.parquet", "prompt_concepts.parquet",
    "prompt_response_relations.csv", WIN_RELEVANCE,
}
_VIEWER_KEYS = {
    "enabled", "output_dir", "examples_per_feature", "examples_per_group", "examples_random",
    "examples_boundary", "prompt_examples_per_feature", "prompt_examples_per_group",
    "prompt_examples_random", "prompt_examples_boundary", "joint_examples",
    "feature_map", "prompt_feature_map", "response_map", "map_sample",
    "map_sample_mode", "coactivation_top_k", "coactivation_max_pairs",
}

EMBEDDING_DEFAULTS = MappingProxyType({
    "backend": "hf",
    "batch_size": None,
})
CONCEPT_DEFAULTS = MappingProxyType({
    "presence_policy": "mixed",
    "fidelity_only": True,
    "named_only": True,
    "top_k": None,
    "include_text": False,
    "chunk_size": 4096,
})
ANALYSIS_DEFAULTS = MappingProxyType({
    "relationships": "auto",
    "comparison": "auto",
    "preference": "auto",
    "min_support": 30,
    "min_cooccur": 5,
    "min_context_pairs": 30,
    "examples_per_direction": 3,
    "side_a_name": "A",
    "side_b_name": "B",
})
VIEWER_DEFAULTS = MappingProxyType({
    **VIEWER_EXPORT_DEFAULTS,
    "enabled": True,
    "joint_examples": True,
    "feature_map": True,
    "prompt_feature_map": True,
    "response_map": True,
})


def _check_keys(raw: dict, allowed: set[str], where: str) -> None:
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(
            f"unknown {where} key(s): {', '.join(sorted(unknown))}; "
            f"allowed: {', '.join(sorted(allowed))}")


def _mapping(raw, where: str) -> dict:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{where} must be a mapping")
    return dict(raw)


def _require_bool(values: dict, keys, where: str) -> None:
    for key in keys:
        if key in values and not isinstance(values[key], bool):
            raise ValueError(f"{where}.{key} must be a boolean")


def _require_positive_int(values: dict, keys, where: str, *, nullable=()) -> None:
    nullable = set(nullable)
    for key in keys:
        if key not in values or (values[key] is None and key in nullable):
            continue
        value = values[key]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{where}.{key} must be a positive integer")


def _require_nonnegative_int(values: dict, keys, where: str) -> None:
    for key in keys:
        if key not in values:
            continue
        value = values[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{where}.{key} must be a non-negative integer")


def _local_path(value, base_dir: Path) -> str | None:
    if value is None:
        return None
    text = str(value)
    if text.startswith("hf://"):
        return text
    path = Path(text).expanduser()
    return str(path if path.is_absolute() else (base_dir / path).resolve())


def set_config_value(raw: dict, dotted_key: str, value) -> None:
    """Set one dotted config key, creating intermediate mappings."""
    parts = [part for part in str(dotted_key).split(".") if part]
    if not parts:
        raise ValueError("config override key cannot be empty")
    cursor = raw
    for part in parts[:-1]:
        current = cursor.get(part)
        if current is None:
            current = {}
            cursor[part] = current
        if not isinstance(current, dict):
            raise ValueError(
                f"cannot set {dotted_key!r}: {part!r} is not a mapping")
        cursor = current
    cursor[parts[-1]] = value


def apply_set_overrides(raw: dict, overrides) -> dict:
    """Apply repeatable ``path.to.key=YAML_VALUE`` overrides to a config mapping."""
    result = deepcopy(raw)
    for expression in overrides or ():
        if "=" not in expression:
            raise ValueError(
                f"invalid --set {expression!r}; expected path.to.key=value")
        key, text = expression.split("=", 1)
        set_config_value(result, key.strip(), yaml.safe_load(text))
    return result


@dataclass(frozen=True)
class LensSource:
    source: str
    revision: str | None = None
    subfolder: str | None = None
    annotations: str | None = None


@dataclass
class AnalyzeConfig:
    """Validated configuration for :func:`run_analysis`."""

    completion_lens: LensSource
    prompt_lens: LensSource | None
    data: dict
    out_dir: str
    device: str = "cpu"
    embedding: dict = field(default_factory=dict)
    concepts: dict = field(default_factory=dict)
    analysis: dict = field(default_factory=dict)
    viewer: dict = field(default_factory=dict)
    lens_options: dict = field(default_factory=dict)
    version: int = 1

    @classmethod
    def from_dict(cls, raw: dict, *, base_dir=".") -> "AnalyzeConfig":
        raw = _mapping(raw, "config root")
        _check_keys(raw, _TOP_KEYS, "config")
        version = int(raw.get("version", 1))
        if version != 1:
            raise ValueError(f"unsupported analyze config version {version}; expected 1")
        base = Path(base_dir).expanduser().resolve()

        lenses = _mapping(raw.get("lenses"), "lenses")
        _check_keys(lenses, _LENS_KEYS, "lenses")
        _require_bool(lenses, ("local_files_only",), "lenses")
        repo = lenses.get("repo")
        direct_completion = lenses.get("completion")
        direct_prompt = lenses.get("prompt")
        if repo or direct_completion:
            if not direct_completion and not lenses.get("completion_subfolder"):
                raise ValueError("lenses.repo requires completion_subfolder")
            if repo:
                base_source = str(repo)
                if not base_source.startswith("hf://"):
                    base_source = f"hf://{base_source}"
            else:
                base_source = None
            completion_source = (
                _local_path(direct_completion, base)
                if direct_completion else base_source)
            completion = LensSource(
                completion_source,
                lenses.get("revision") if completion_source.startswith("hf://") else None,
                None if direct_completion else str(lenses["completion_subfolder"]),
                _local_path(lenses.get("completion_annotations"), base),
            )
            prompt_source = (
                _local_path(direct_prompt, base)
                if direct_prompt else base_source)
            prompt = (
                LensSource(
                    prompt_source,
                    lenses.get("revision") if prompt_source.startswith("hf://") else None,
                    None if direct_prompt else str(lenses["prompt_subfolder"]),
                    _local_path(lenses.get("prompt_annotations"), base),
                )
                if direct_prompt or (repo and lenses.get("prompt_subfolder")) else None
            )
        else:
            raise ValueError(
                "configure lenses.repo + completion_subfolder, or lenses.completion")

        data = _mapping(raw.get("data"), "data")
        _check_keys(data, _DATA_KEYS, "data")
        source = _mapping(data.get("source"), "data.source")
        _check_keys(source, _SOURCE_KEYS, "data.source")
        source_type = str(source.get("type", "local")).strip().casefold()
        if source_type in {"local", "file"}:
            if not source.get("path"):
                raise ValueError("local data.source requires path")
            source["type"] = "local"
            source["path"] = _local_path(source["path"], base)
        elif source_type in {"huggingface", "hf", "hub"}:
            dataset_id = source.get("dataset_id") or source.get("path")
            if not dataset_id:
                raise ValueError("Hugging Face data.source requires dataset_id")
            source["type"] = "huggingface"
            source["dataset_id"] = str(dataset_id)
            source.pop("path", None)
        else:
            raise ValueError("data.source.type must be local or huggingface")
        _require_bool(source, ("streaming",), "data.source")
        _require_positive_int(source, ("limit",), "data.source", nullable=("limit",))
        data["source"] = source
        for key, allowed in (
            ("columns", _COLUMN_KEYS), ("text", _TEXT_KEYS), ("label", _LABEL_KEYS)
        ):
            nested = _mapping(data.get(key), f"data.{key}")
            _check_keys(nested, allowed, f"data.{key}")
            if nested:
                data[key] = nested
        metadata_columns = data.get("columns", {}).get("metadata", [])
        if not isinstance(metadata_columns, list) or any(
            not isinstance(column, str) or not column for column in metadata_columns
        ):
            raise ValueError("data.columns.metadata must be a list of non-empty strings")
        mode = str(data.get("mode", "auto")).casefold()
        if mode not in {"auto", "single", "paired"}:
            raise ValueError("data.mode must be auto, single, or paired")
        data["mode"] = mode
        _require_bool(data, ("drop_empty",), "data")

        embedding = _mapping(raw.get("embedding"), "embedding")
        _check_keys(embedding, _EMBEDDING_KEYS, "embedding")
        if embedding.get("cache_dir") is not None:
            embedding["cache_dir"] = _local_path(embedding["cache_dir"], base)
        for key, value in EMBEDDING_DEFAULTS.items():
            embedding.setdefault(key, value)
        _require_positive_int(
            embedding, ("batch_size",), "embedding", nullable=("batch_size",))

        concepts = _mapping(raw.get("concepts"), "concepts")
        _check_keys(concepts, _CONCEPT_KEYS, "concepts")
        for key, value in CONCEPT_DEFAULTS.items():
            concepts.setdefault(key, value)
        _require_bool(
            concepts, ("fidelity_only", "named_only", "include_text"), "concepts")
        _require_positive_int(
            concepts, ("top_k", "chunk_size"), "concepts", nullable=("top_k",))
        if concepts["presence_policy"] not in PRESENCE_POLICIES:
            raise ValueError(
                f"concepts.presence_policy must be one of {list(PRESENCE_POLICIES)}")

        analysis = _mapping(raw.get("analysis"), "analysis")
        _check_keys(analysis, _ANALYSIS_KEYS, "analysis")
        for key in ("relationships", "comparison", "preference"):
            analysis.setdefault(key, ANALYSIS_DEFAULTS[key])
            if not (isinstance(analysis[key], bool) or analysis[key] == "auto"):
                raise ValueError(f"analysis.{key} must be true, false, or auto")
        outcome_spec = analysis.get("outcomes")
        if outcome_spec is not None:
            outcome_spec = _mapping(outcome_spec, "analysis.outcomes")
            _check_keys(outcome_spec, _OUTCOME_ANALYSIS_KEYS, "analysis.outcomes")
            columns = outcome_spec.get("columns")
            if not isinstance(columns, list) or not columns or any(
                not isinstance(column, str) or not column for column in columns
            ):
                raise ValueError(
                    "analysis.outcomes.columns must be a non-empty list of strings")
            kind = outcome_spec.get("kind")
            allowed_kinds = {
                "binary", "probability", "preference", "continuous",
                "multi_continuous",
            }
            if kind not in allowed_kinds:
                raise ValueError(
                    f"analysis.outcomes.kind must be one of {sorted(allowed_kinds)}")
            if kind != "multi_continuous" and len(columns) != 1:
                raise ValueError(
                    f"analysis.outcomes.kind={kind!r} requires exactly one column")
            outcome_spec.setdefault("normalization", "auto")
            if outcome_spec["normalization"] not in {"auto", "none", "zscore"}:
                raise ValueError(
                    "analysis.outcomes.normalization must be auto, none, or zscore")
            outcome_spec.setdefault("code_array", "z_a")
            if outcome_spec["code_array"] not in {"z_a", "z_diff", "z_prompt"}:
                raise ValueError(
                    "analysis.outcomes.code_array must be z_a, z_diff, or z_prompt")
            outcome_spec.setdefault("min_units", 3)
            if not isinstance(outcome_spec["min_units"], int) or outcome_spec["min_units"] < 3:
                raise ValueError("analysis.outcomes.min_units must be an integer >= 3")
            outcome_spec.setdefault("output", "outcome_associations.csv")
            output_name = outcome_spec["output"]
            if (
                not isinstance(output_name, str)
                or output_name in {"", ".", ".."}
                or Path(output_name).name != output_name
                or Path(output_name).suffix.casefold() != ".csv"
                or output_name.casefold() in _RESERVED_ANALYSIS_FILENAMES
            ):
                raise ValueError(
                    "analysis.outcomes.output must be a safe, unreserved CSV filename")
            retained = set(data.get("columns", {}).get("metadata", [])) | {
                "human_pref", "item_id", "language", "source", "prompt",
                "completion_a", "completion_b", "model_a", "model_b",
            }
            absent = [column for column in columns if column not in retained]
            if absent:
                raise ValueError(
                    f"analysis.outcomes columns {absent} will not be retained; add "
                    "them to data.columns.metadata")
            analysis["outcomes"] = outcome_spec
        group_col = analysis.get("group_col")
        if group_col is not None:
            if not isinstance(group_col, str) or not group_col:
                raise ValueError("analysis.group_col must be a non-empty string or null")
            retained_groups = set(data.get("columns", {}).get("metadata", [])) | {
                "item_id", "language", "source", "prompt", "model_a", "model_b",
            }
            if group_col not in retained_groups:
                raise ValueError(
                    f"analysis.group_col {group_col!r} will not be retained; add it "
                    "to data.columns.metadata")
        for key, value in ANALYSIS_DEFAULTS.items():
            analysis.setdefault(key, value)
        _require_positive_int(
            analysis,
            ("min_support", "min_cooccur", "min_context_pairs"),
            "analysis")
        _require_nonnegative_int(
            analysis, ("examples_per_direction",), "analysis")

        viewer = _mapping(raw.get("viewer"), "viewer")
        _check_keys(viewer, _VIEWER_KEYS, "viewer")
        for key, value in VIEWER_DEFAULTS.items():
            viewer.setdefault(key, value)
        if viewer.get("output_dir") is not None:
            viewer["output_dir"] = _local_path(viewer["output_dir"], base)
        _require_bool(
            viewer,
            ("enabled", "joint_examples", "feature_map", "prompt_feature_map",
             "response_map"),
            "viewer")
        _require_nonnegative_int(
            viewer,
            ("examples_per_feature", "examples_per_group", "examples_random",
             "examples_boundary", "prompt_examples_per_feature",
             "prompt_examples_per_group", "prompt_examples_random",
             "prompt_examples_boundary"),
            "viewer")
        _require_positive_int(
            viewer, ("map_sample", "coactivation_top_k", "coactivation_max_pairs"),
            "viewer")
        if viewer["map_sample_mode"] not in {"random", "top-activating", "hybrid"}:
            raise ValueError(
                "viewer.map_sample_mode must be random, top-activating, or hybrid")

        out_dir = raw.get("out_dir")
        if not out_dir:
            raise ValueError("config missing required key: out_dir")
        lens_options = {
            "token_env": lenses.get("token_env"),
            "cache_dir": _local_path(lenses.get("cache_dir"), base),
            "local_files_only": bool(lenses.get("local_files_only", False)),
        }
        return cls(
            completion_lens=completion,
            prompt_lens=prompt,
            data=data,
            out_dir=_local_path(out_dir, base),
            device=str(raw.get("device", "cpu")),
            embedding=embedding,
            concepts=concepts,
            analysis=analysis,
            viewer=viewer,
            lens_options=lens_options,
            version=version,
        )

    @classmethod
    def load(cls, path, *, overrides=()) -> "AnalyzeConfig":
        path = Path(path).expanduser().resolve()
        raw = yaml.safe_load(path.read_text())
        raw = apply_set_overrides(_mapping(raw, "config root"), overrides)
        return cls.from_dict(raw, base_dir=path.parent)

    def to_dict(self) -> dict:
        def source(spec: LensSource) -> str:
            return (
                f"{spec.source.rstrip('/')}/{spec.subfolder.strip('/')}"
                if spec.subfolder else spec.source)

        lenses = {"completion": source(self.completion_lens)}
        if self.prompt_lens is not None:
            lenses["prompt"] = source(self.prompt_lens)
        revisions = {
            spec.revision for spec in (self.completion_lens, self.prompt_lens)
            if spec is not None and spec.revision is not None
        }
        if len(revisions) == 1:
            lenses["revision"] = revisions.pop()
        if self.completion_lens.annotations is not None:
            lenses["completion_annotations"] = self.completion_lens.annotations
        if self.prompt_lens is not None and self.prompt_lens.annotations is not None:
            lenses["prompt_annotations"] = self.prompt_lens.annotations
        lenses.update({
            key: value for key, value in self.lens_options.items()
            if value not in (None, False)
        })
        result = {
            "version": self.version,
            "lenses": lenses,
            "data": self.data,
            "out_dir": self.out_dir,
            "device": self.device,
            "embedding": self.embedding,
            "concepts": self.concepts,
            "analysis": self.analysis,
            "viewer": self.viewer,
        }
        return result

# Preserve the historical public module identity after the implementation split.
for _public_object in (AnalyzeConfig, LensSource, apply_set_overrides, set_config_value):
    _public_object.__module__ = "prefscope.pipeline.analyze"
del _public_object


__all__ = [
    "AnalyzeConfig", "LensSource", "apply_set_overrides", "set_config_value",
    "EMBEDDING_DEFAULTS", "CONCEPT_DEFAULTS", "ANALYSIS_DEFAULTS",
    "VIEWER_DEFAULTS",
]
