"""Torch-free contracts for PrefScope report compilation and bundle v3."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from types import MappingProxyType
from typing import Mapping

import pandas as pd

from prefscope.api.feature_catalog import FeatureCatalog

from prefscope.api.analysis_contracts import AnalysisDataset, AnalysisArtifact
from prefscope.api.analysis_execution import DatasetAnalysisResult
from prefscope.api.analysis_io import AnalysisDatasetReference, LoadedAnalysisResult
from prefscope.analysis.grouping import factorize_group_ids
from prefscope.core.features import validate_feature_ids
from prefscope.core.representation import validate_portable_mapping
from prefscope.core.table_schema import TableContract
from prefscope.reporting.privacy import PrivacyPolicy, normalize_field_name, validate_privacy_safe

REPORT_BUNDLE_VERSION = 3
JSON_TABLE_FORMAT = "prefscope.json_table"
JSON_TABLE_VERSION = 1
_PORTABLE_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_SCHEMA_NAME = re.compile(r"^[a-z][a-z0-9_.-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MEDIA = re.compile(r"^[a-z0-9][a-z0-9.+-]*/[a-z0-9][a-z0-9.+-]*$")
_ARTIFACT_PATH = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")



class _StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class SectionKind(_StringEnum):
    NOTICE = "notice"
    METRIC_CARDS = "metric_cards"
    TEXT_CARDS = "text_cards"
    TYPED_TABLE = "typed_table"
    DISTRIBUTION = "distribution"
    FEATURE_RANKING = "feature_ranking"
    RELATIONSHIP = "relationship"
    BATTLE_INDEX = "battle_index"
    BATTLE_DETAIL = "battle_detail"
    PROVENANCE = "provenance"


class EvidenceLayer(_StringEnum):
    METADATA = "metadata"
    RAW_AXIS = "raw_axis"
    PROPOSED_NAME = "proposed_name"
    EXTREME_FIDELITY = "extreme_fidelity"
    SEMANTIC_PRESENCE = "semantic_presence"
    FEATURE_ROLE = "feature_role"
    RESPONSE_SCOPE = "response_scope"
    CONTEXT = "context"
    MODEL_TENDENCY = "model_tendency"
    OUTCOME_ASSOCIATION = "outcome_association"
    DESCRIPTIVE = "descriptive"
    INFERENTIAL = "inferential"
    PROVENANCE = "provenance"


class SectionOrientation(_StringEnum):
    NONE = "none"
    AS_DECLARED = "as_declared"
    A_MINUS_B = "a_minus_b"
    B_MINUS_A = "b_minus_a"
    PRESENT_MINUS_ABSENT = "present_minus_absent"
    PRESENT_MINUS_ABSENT_OF_B_MINUS_A = "present_minus_absent_of_b_minus_a"
    FEATURE_ACTIVATION_TO_DECLARED_OUTCOME = "feature_activation_to_declared_outcome"
    PER_FEATURE_SET_AS_DECLARED = "per_feature_set_as_declared"
    A_MINUS_B_FEATURES_AND_LENGTH_P_A_PREFERRED = (
        "a_minus_b_features_and_length__p_a_preferred")


class SectionStatus(_StringEnum):
    READY = "ready"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


class ArtifactStatus(_StringEnum):
    READY = "ready"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


class ReportStatus(_StringEnum):
    READY = "ready"
    PARTIAL = "partial"
    FAILED = "failed"


class StatusReason(_StringEnum):
    NOT_APPLICABLE = "not_applicable"
    INPUT_ABSENT = "input_absent"
    INSUFFICIENT_SUPPORT = "insufficient_support"
    PROCESSING_ERROR = "processing_error"


class ArtifactPrivacy(_StringEnum):
    PUBLIC = "public"
    AGGREGATE = "aggregate"
    OPAQUE_ROWS = "opaque_rows"
    TEXT_SNIPPETS = "text_snippets"
    LOCAL_FULL_TEXT = "local_full_text"


class ReportMode(_StringEnum):
    SINGLE_PAIR = "single_pair"
    CORPUS = "corpus"
    PAIRED_BATTLES = "paired_battles"
    TABLE_ONLY = "table_only"


def _enum(value, cls, where: str):
    try:
        return value if isinstance(value, cls) else cls(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where} must be one of {[item.value for item in cls]}") from exc


def _nonempty(value: object, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where} must be a non-empty string")
    return value


def _positive_int(value: object, where: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{where} must be a positive integer")
    return value


def _portable_name(value: object, where: str) -> str:
    if not isinstance(value, str) or not _PORTABLE_NAME.fullmatch(value):
        raise ValueError(f"{where} must use portable lower_snake_case")
    return value


def _safe_mapping(value: object, *, where: str, nonempty: bool = False):
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be an object")
    resolved = dict(value)
    if nonempty and not resolved:
        raise ValueError(f"{where} must not be empty")
    validate_privacy_safe(resolved, where=where)
    return validate_portable_mapping(resolved, where=where)


def _coordinates(value: object, *, where: str):
    resolved = _safe_mapping(value, where=where, nonempty=True)
    for key, item in resolved.items():
        if not isinstance(key, str) or normalize_field_name(key) != key:
            raise ValueError(f"{where} keys must use lower_snake_case")
        if isinstance(item, bool) or not isinstance(item, (str, int)):
            raise ValueError(f"{where} values must be strings or integers")
        if isinstance(item, str) and not item:
            raise ValueError(f"{where} string values must not be empty")
    return resolved


def table_contract_from_manifest(value: object) -> TableContract:
    """Parse the exact portable representation emitted by ``TableContract``."""
    contract = TableContract.from_manifest(value)
    validate_privacy_safe(contract.to_manifest(), where="table contract")
    return contract


@dataclass(frozen=True)
class ReportError:
    """Sanitized structured error suitable for a report or artifact manifest."""

    code: str
    message: str
    retryable: bool = False
    detail: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _portable_name(self.code, "report error code")
        _nonempty(self.message, "report error message")
        if not isinstance(self.retryable, bool):
            raise ValueError("report error retryable must be a boolean")
        validate_privacy_safe(self.message, where="report error message")
        object.__setattr__(
            self, "detail", _safe_mapping(self.detail, where="report error detail"))

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code, "message": self.message,
            "retryable": self.retryable, "detail": dict(self.detail),
        }

    @classmethod
    def from_dict(cls, value: object) -> "ReportError":
        expected = {"code", "message", "retryable", "detail"}
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError(f"report error fields must be exactly {sorted(expected)}")
        return cls(**dict(value))


@dataclass(frozen=True)
class DatasetLineage:
    dataset_sha256: str
    row_ids_sha256: str
    group_partition_sha256: str
    group_source: str
    n_rows: int
    n_groups: int

    def __post_init__(self) -> None:
        for name in ("dataset_sha256", "row_ids_sha256", "group_partition_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or not _SHA256.fullmatch(value):
                raise ValueError(f"dataset lineage {name} must be a SHA-256 digest")
        _portable_name(self.group_source, "dataset lineage group_source")
        for name in ("n_rows", "n_groups"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"dataset lineage {name} must be non-negative")
        if self.n_rows == 0 and self.n_groups != 0:
            raise ValueError("empty dataset lineage cannot have groups")
        if self.n_rows > 0 and not 1 <= self.n_groups <= self.n_rows:
            raise ValueError("dataset lineage n_groups must be within row support")

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_sha256": self.dataset_sha256,
            "row_ids_sha256": self.row_ids_sha256,
            "group_partition_sha256": self.group_partition_sha256,
            "group_source": self.group_source,
            "n_rows": self.n_rows,
            "n_groups": self.n_groups,
        }

    @classmethod
    def from_dict(cls, value: object) -> "DatasetLineage":
        expected = {
            "dataset_sha256", "row_ids_sha256", "group_partition_sha256",
            "group_source", "n_rows", "n_groups",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError(f"dataset lineage fields must be exactly {sorted(expected)}")
        return cls(**dict(value))


@dataclass(frozen=True)
class SourceArtifactReference:
    source_id: str
    artifact_type: str
    schema_version: int
    sha256: str

    def __post_init__(self) -> None:
        _portable_name(self.source_id, "source artifact source_id")
        if not isinstance(self.artifact_type, str) or not _SCHEMA_NAME.fullmatch(
            self.artifact_type
        ):
            raise ValueError("source artifact_type must be portable")
        _positive_int(self.schema_version, "source artifact schema_version")
        if not isinstance(self.sha256, str) or not _SHA256.fullmatch(self.sha256):
            raise ValueError("source artifact sha256 must be a SHA-256 digest")

    def to_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id, "artifact_type": self.artifact_type,
            "schema_version": self.schema_version, "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> "SourceArtifactReference":
        expected = {"source_id", "artifact_type", "schema_version", "sha256"}
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError(f"source reference fields must be exactly {sorted(expected)}")
        return cls(**dict(value))


@dataclass(frozen=True)
class CompilerProvenance:
    compiler_name: str
    compiler_version: str
    report_spec_name: str
    report_spec_version: int
    report_spec_sha256: str

    def __post_init__(self) -> None:
        _portable_name(self.compiler_name, "compiler_name")
        _nonempty(self.compiler_version, "compiler_version")
        validate_privacy_safe(self.compiler_version, where="compiler_version")
        _portable_name(self.report_spec_name, "report_spec_name")
        _positive_int(self.report_spec_version, "report_spec_version")
        if not isinstance(self.report_spec_sha256, str) or not _SHA256.fullmatch(
            self.report_spec_sha256
        ):
            raise ValueError("report_spec_sha256 must be a SHA-256 digest")

    def to_dict(self) -> dict[str, object]:
        return {
            "compiler_name": self.compiler_name,
            "compiler_version": self.compiler_version,
            "report_spec_name": self.report_spec_name,
            "report_spec_version": self.report_spec_version,
            "report_spec_sha256": self.report_spec_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> "CompilerProvenance":
        expected = {
            "compiler_name", "compiler_version", "report_spec_name",
            "report_spec_version", "report_spec_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError(f"compiler provenance fields must be exactly {sorted(expected)}")
        return cls(**dict(value))


@dataclass(frozen=True)
class SamplingProvenance:
    method: str
    sampling_frame_sha256: str
    seed: int | None
    population_count: int
    sampled_count: int
    max_examples_per_feature: int

    def __post_init__(self) -> None:
        _portable_name(self.method, "sampling method")
        if not isinstance(self.sampling_frame_sha256, str) or not _SHA256.fullmatch(
            self.sampling_frame_sha256
        ):
            raise ValueError("sampling_frame_sha256 must be a SHA-256 digest")
        if self.seed is not None and (
            not isinstance(self.seed, int) or isinstance(self.seed, bool)
            or not -(2 ** 53 - 1) <= self.seed <= 2 ** 53 - 1
        ):
            raise ValueError("sampling seed must be a browser-safe integer or null")
        for name in ("population_count", "sampled_count", "max_examples_per_feature"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"sampling {name} must be non-negative")
        if self.sampled_count > self.population_count:
            raise ValueError("sampling sampled_count cannot exceed population_count")

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "sampling_frame_sha256": self.sampling_frame_sha256,
            "seed": self.seed, "population_count": self.population_count,
            "sampled_count": self.sampled_count,
            "max_examples_per_feature": self.max_examples_per_feature,
        }

    @classmethod
    def from_dict(cls, value: object) -> "SamplingProvenance":
        expected = {
            "method", "sampling_frame_sha256", "seed", "population_count",
            "sampled_count", "max_examples_per_feature",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError(f"sampling provenance fields must be exactly {sorted(expected)}")
        return cls(**dict(value))


@dataclass(frozen=True)
class ReportLineage:
    dataset: DatasetLineage
    sources: tuple[SourceArtifactReference, ...]
    compiler: CompilerProvenance
    sampling: SamplingProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, DatasetLineage):
            raise ValueError("report lineage dataset must be DatasetLineage")
        sources = tuple(self.sources)
        if not sources or not all(isinstance(item, SourceArtifactReference) for item in sources):
            raise ValueError("report lineage needs source artifact references")
        ids = [item.source_id for item in sources]
        if len(set(ids)) != len(ids):
            raise ValueError("report lineage source_id values must be unique")
        if not isinstance(self.compiler, CompilerProvenance):
            raise ValueError("report lineage compiler must be CompilerProvenance")
        if not isinstance(self.sampling, SamplingProvenance):
            raise ValueError("report lineage sampling must be SamplingProvenance")
        if self.sampling.population_count != self.dataset.n_rows:
            raise ValueError("sampling population_count must match dataset lineage n_rows")
        object.__setattr__(self, "sources", sources)

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset.to_dict(),
            "sources": [item.to_dict() for item in self.sources],
            "compiler": self.compiler.to_dict(),
            "sampling": self.sampling.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> "ReportLineage":
        expected = {"dataset", "sources", "compiler", "sampling"}
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError(f"report lineage fields must be exactly {sorted(expected)}")
        if not isinstance(value["sources"], list):
            raise ValueError("report lineage sources must be an array")
        return cls(
            dataset=DatasetLineage.from_dict(value["dataset"]),
            sources=tuple(SourceArtifactReference.from_dict(item) for item in value["sources"]),
            compiler=CompilerProvenance.from_dict(value["compiler"]),
            sampling=SamplingProvenance.from_dict(value["sampling"]),
        )


def _validate_state(status, reason, error, *, status_cls, where: str):
    resolved_status = _enum(status, status_cls, f"{where} status")
    resolved_reason = (
        None if reason is None else _enum(reason, StatusReason, f"{where} reason"))
    if error is not None and not isinstance(error, ReportError):
        raise ValueError(f"{where} error must be a ReportError or null")
    if resolved_status.value == "ready":
        if resolved_reason is not None or error is not None:
            raise ValueError(f"ready {where} must not have a reason or error")
    elif resolved_status.value == "unavailable":
        if resolved_reason is None or resolved_reason is StatusReason.PROCESSING_ERROR:
            raise ValueError(
                f"unavailable {where} needs a non-processing status reason")
        if error is not None:
            raise ValueError(f"unavailable {where} must not have an error")
    else:
        if resolved_reason is not StatusReason.PROCESSING_ERROR or error is None:
            raise ValueError(
                f"error {where} needs processing_error and a typed ReportError")
    return resolved_status, resolved_reason


@dataclass(frozen=True)
class SectionContract:
    """One ordered, versioned report section and its availability state."""

    section_id: str
    kind: SectionKind
    version: int
    title: str
    evidence_layer: EvidenceLayer
    orientation: SectionOrientation
    coordinates: Mapping[str, object]
    status: SectionStatus = SectionStatus.READY
    reason: StatusReason | None = None
    error: ReportError | None = None

    def __post_init__(self) -> None:
        _portable_name(self.section_id, "section_id")
        object.__setattr__(self, "kind", _enum(self.kind, SectionKind, "section kind"))
        _positive_int(self.version, "section version")
        _nonempty(self.title, "section title")
        validate_privacy_safe(self.title, where="section title")
        object.__setattr__(
            self, "evidence_layer",
            _enum(self.evidence_layer, EvidenceLayer, "section evidence_layer"),
        )
        object.__setattr__(
            self, "orientation",
            _enum(self.orientation, SectionOrientation, "section orientation"),
        )
        object.__setattr__(
            self, "coordinates",
            _coordinates(self.coordinates, where="section coordinates"),
        )
        status, reason = _validate_state(
            self.status, self.reason, self.error,
            status_cls=SectionStatus, where="section",
        )
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "reason", reason)

    def to_dict(self) -> dict[str, object]:
        return {
            "section_id": self.section_id, "kind": self.kind.value,
            "version": self.version, "title": self.title,
            "evidence_layer": self.evidence_layer.value,
            "orientation": self.orientation.value,
            "coordinates": dict(self.coordinates), "status": self.status.value,
            "reason": None if self.reason is None else self.reason.value,
            "error": None if self.error is None else self.error.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> "SectionContract":
        expected = {
            "section_id", "kind", "version", "title", "evidence_layer",
            "orientation", "coordinates", "status", "reason", "error",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError(f"section fields must be exactly {sorted(expected)}")
        data = dict(value)
        if data["error"] is not None:
            data["error"] = ReportError.from_dict(data["error"])
        return cls(**data)


@dataclass(frozen=True)
class ReportCapabilities:
    """Capability banner inputs persisted in every report manifest."""

    mode: ReportMode
    n_rows: int
    n_groups: int | None = None
    feature_views: tuple[str, ...] = field(default_factory=tuple)
    evidence_layers: tuple[EvidenceLayer, ...] = field(default_factory=tuple)
    table_only: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", _enum(self.mode, ReportMode, "report mode"))
        if not isinstance(self.n_rows, int) or isinstance(self.n_rows, bool) or self.n_rows < 0:
            raise ValueError("report capability n_rows must be a non-negative integer")
        if self.n_groups is not None and (
            not isinstance(self.n_groups, int)
            or isinstance(self.n_groups, bool)
            or self.n_groups < 0
        ):
            raise ValueError("report capability n_groups must be non-negative or null")
        views = tuple(self.feature_views)
        if len(set(views)) != len(views) or any(
            not isinstance(item, str) or not _PORTABLE_NAME.fullmatch(item) for item in views
        ):
            raise ValueError("feature_views must contain unique portable names")
        layers = tuple(
            _enum(item, EvidenceLayer, "capability evidence layer")
            for item in self.evidence_layers)
        if len(set(layers)) != len(layers):
            raise ValueError("capability evidence_layers must be unique")
        if not isinstance(self.table_only, bool):
            raise ValueError("report capability table_only must be a boolean")
        if (self.mode is ReportMode.TABLE_ONLY) != self.table_only:
            raise ValueError("table_only must agree with report mode")
        object.__setattr__(self, "feature_views", views)
        object.__setattr__(self, "evidence_layers", layers)

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode.value, "n_rows": self.n_rows,
            "n_groups": self.n_groups, "feature_views": list(self.feature_views),
            "evidence_layers": [item.value for item in self.evidence_layers],
            "table_only": self.table_only,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ReportCapabilities":
        expected = {
            "mode", "n_rows", "n_groups", "feature_views", "evidence_layers",
            "table_only",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError(f"capability fields must be exactly {sorted(expected)}")
        if not isinstance(value["feature_views"], list) or not isinstance(
            value["evidence_layers"], list
        ):
            raise ValueError("capability feature_views/evidence_layers must be arrays")
        return cls(
            mode=value["mode"], n_rows=value["n_rows"], n_groups=value["n_groups"],
            feature_views=tuple(value["feature_views"]),
            evidence_layers=tuple(value["evidence_layers"]),
            table_only=value["table_only"],
        )


@dataclass(frozen=True, eq=False)
class ReportDataset:
    """Thin presentation wrapper over one explicit supported analysis source."""

    analysis: AnalysisDataset | DatasetAnalysisResult | LoadedAnalysisResult
    feature_catalogs: Mapping[str, pd.DataFrame | FeatureCatalog] = field(default_factory=dict)
    row_metadata: pd.DataFrame | None = None
    _dataset: AnalysisDataset | None = field(init=False, repr=False, compare=False)
    _dataset_reference: AnalysisDatasetReference | None = field(
        init=False, repr=False, compare=False)
    _row_ids: tuple[str, ...] = field(init=False, repr=False, compare=False)
    _group_codes: tuple[int, ...] = field(init=False, repr=False, compare=False)
    _artifacts: Mapping[str, AnalysisArtifact] = field(
        init=False, repr=False, compare=False)
    _table_only: bool = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        dataset = None
        reference = None
        artifacts = {}
        if isinstance(self.analysis, AnalysisDataset):
            dataset = self.analysis
        elif isinstance(self.analysis, DatasetAnalysisResult):
            if not isinstance(self.analysis.dataset, AnalysisDataset):
                raise ValueError(
                    "DatasetAnalysisResult must retain a live AnalysisDataset; "
                    "use LoadedAnalysisResult for detached tables")
            dataset = self.analysis.dataset
            artifacts = dict(self.analysis.artifacts)
        elif isinstance(self.analysis, LoadedAnalysisResult):
            reference = self.analysis.dataset_reference
            artifacts = dict(self.analysis.artifacts)
        else:
            raise ValueError(
                "analysis must be AnalysisDataset, DatasetAnalysisResult, "
                "or LoadedAnalysisResult")

        if dataset is not None:
            rows = dataset.row_ids
            if dataset.group_ids is None:
                group_codes = tuple(range(dataset.n_rows))
            else:
                codes, _ = factorize_group_ids(dataset.group_ids)
                group_codes = tuple(int(value) for value in codes)
        else:
            rows = reference.row_ids
            group_codes = reference.group_codes
        if artifacts and not all(
            isinstance(name, str) and isinstance(item, AnalysisArtifact)
            and name == item.name for name, item in artifacts.items()
        ):
            raise ValueError("analysis artifacts must be a name-aligned mapping")

        catalogs = {}
        raw_catalogs = dict(self.feature_catalogs)
        if reference is not None and raw_catalogs:
            raise ValueError(
                "detached feature catalogs are forbidden without a feature identity proof")
        if dataset is not None and not set(raw_catalogs).issubset(dataset.features):
            unknown = sorted(set(raw_catalogs) - set(dataset.features))
            raise ValueError(f"feature catalogs refer to unknown feature sets: {unknown}")
        for name, catalog in raw_catalogs.items():
            _portable_name(name, "feature catalog name")
            if isinstance(catalog, FeatureCatalog):
                catalog.validate_for(dataset.features[name])
                observed = catalog.feature_ids
            else:
                if not isinstance(catalog, pd.DataFrame):
                    raise ValueError(
                        f"feature catalog {name!r} must be a DataFrame or FeatureCatalog"
                    )
                if (
                    "feature_id" not in catalog.columns
                    or catalog.columns.duplicated().any()
                ):
                    raise ValueError(
                        f"feature catalog {name!r} needs unique columns including feature_id"
                    )
                if (
                    catalog["feature_id"].isna().any()
                    or catalog["feature_id"].duplicated().any()
                ):
                    raise ValueError(
                        f"feature catalog {name!r} feature_id must be unique and present"
                    )
                try:
                    observed = validate_feature_ids(
                        tuple(catalog["feature_id"].tolist())
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"feature catalog {name!r} has invalid feature_id values"
                    ) from exc
            if observed != dataset.features[name].feature_ids:
                raise ValueError(
                    f"feature catalog {name!r} must match analysis feature order")
            catalogs[name] = (
                catalog
                if isinstance(catalog, FeatureCatalog)
                else catalog.copy(deep=True).reset_index(drop=True)
            )

        if self.row_metadata is None:
            metadata = pd.DataFrame({"row_id": rows})
        elif not isinstance(self.row_metadata, pd.DataFrame):
            raise ValueError("row_metadata must be a pandas DataFrame with row_id")
        else:
            metadata = self.row_metadata.copy(deep=True).reset_index(drop=True)
            if "row_id" not in metadata.columns or metadata.columns.duplicated().any():
                raise ValueError("row_metadata needs unique columns including row_id")
            if tuple(metadata["row_id"].tolist()) != rows:
                raise ValueError(
                    "row_metadata row_id must exactly prove the analysis row order")
        object.__setattr__(self, "feature_catalogs", MappingProxyType(catalogs))
        object.__setattr__(self, "row_metadata", metadata)
        object.__setattr__(self, "_dataset", dataset)
        object.__setattr__(self, "_dataset_reference", reference)
        object.__setattr__(self, "_row_ids", rows)
        object.__setattr__(self, "_group_codes", group_codes)
        object.__setattr__(self, "_artifacts", MappingProxyType(artifacts))
        object.__setattr__(self, "_table_only", reference is not None)

    @classmethod
    def from_analysis(cls, analysis, *, feature_catalogs=None, row_metadata=None):
        return cls(
            analysis=analysis, feature_catalogs=dict(feature_catalogs or {}),
            row_metadata=row_metadata,
        )

    @property
    def dataset(self) -> AnalysisDataset | None:
        return self._dataset

    @property
    def dataset_reference(self) -> AnalysisDatasetReference | None:
        return self._dataset_reference

    @property
    def group_codes(self) -> tuple[int, ...]:
        return self._group_codes

    @property
    def artifacts(self) -> Mapping[str, AnalysisArtifact]:
        return self._artifacts

    @property
    def row_ids(self) -> tuple[str, ...]:
        return self._row_ids

    @property
    def n_rows(self) -> int:
        return len(self._row_ids)

    @property
    def table_only(self) -> bool:
        return self._table_only


@dataclass(frozen=True)
class ReportSpec:
    """Declarative report selection; it does not compile or render sections."""

    title: str
    sections: str | tuple[SectionKind, ...] = "auto"
    max_examples_per_feature: int = 8
    name: str = "report"
    version: int = 1
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _nonempty(self.title, "report title")
        validate_privacy_safe(self.title, where="report title")
        _portable_name(self.name, "report spec name")
        _positive_int(self.version, "report spec version")
        if (
            not isinstance(self.max_examples_per_feature, int)
            or isinstance(self.max_examples_per_feature, bool)
            or self.max_examples_per_feature < 0
        ):
            raise ValueError("max_examples_per_feature must be a non-negative integer")
        if self.sections == "auto":
            sections = "auto"
        elif isinstance(self.sections, (list, tuple)):
            sections = tuple(
                _enum(item, SectionKind, "report spec section") for item in self.sections)
            if not sections or len(set(sections)) != len(sections):
                raise ValueError("explicit report spec sections must be non-empty and unique")
        else:
            raise ValueError("report spec sections must be 'auto' or section kinds")
        object.__setattr__(self, "sections", sections)
        object.__setattr__(
            self, "metadata", _safe_mapping(self.metadata, where="report spec metadata"))


@dataclass(frozen=True)
class ReportArtifact:
    """One typed artifact descriptor; instance identity is separate from table schema."""

    artifact_id: str
    schema_name: str
    schema_version: int
    section_id: str
    evidence_layer: EvidenceLayer
    orientation: str
    coordinates: Mapping[str, object]
    status: ArtifactStatus
    reason: StatusReason | None
    error: ReportError | None
    source_refs: tuple[str, ...]
    path: str | None
    media_type: str | None
    sha256: str | None
    table_contract: TableContract | None
    estimand: str
    units: Mapping[str, str]
    support: Mapping[str, object]
    missing: str
    tie: str
    test: str
    multiplicity: str
    privacy: ArtifactPrivacy

    def __post_init__(self) -> None:
        _portable_name(self.artifact_id, "artifact_id")
        if not isinstance(self.schema_name, str) or not _SCHEMA_NAME.fullmatch(
            self.schema_name
        ):
            raise ValueError("artifact schema_name must be portable")
        _positive_int(self.schema_version, "artifact schema_version")
        _portable_name(self.section_id, "artifact section_id")
        object.__setattr__(
            self, "evidence_layer",
            _enum(self.evidence_layer, EvidenceLayer, "artifact evidence_layer"),
        )
        orientation = _nonempty(self.orientation, "artifact orientation")
        validate_privacy_safe(orientation, where="artifact orientation")
        object.__setattr__(
            self, "coordinates",
            _coordinates(self.coordinates, where="artifact coordinates"),
        )
        status, reason = _validate_state(
            self.status, self.reason, self.error,
            status_cls=ArtifactStatus, where="artifact",
        )
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "reason", reason)
        source_refs = tuple(self.source_refs)
        if not source_refs or len(set(source_refs)) != len(source_refs) or any(
            not isinstance(item, str) or not _PORTABLE_NAME.fullmatch(item)
            for item in source_refs
        ):
            raise ValueError("artifact source_refs must contain unique portable source IDs")
        object.__setattr__(self, "source_refs", source_refs)
        if status is ArtifactStatus.READY:
            if not all(
                isinstance(item, str) and item
                for item in (self.path, self.media_type, self.sha256)
            ):
                raise ValueError("ready artifact needs path, media_type, and sha256")
            path_parts = self.path.split("/")
            if (
                not _ARTIFACT_PATH.fullmatch(self.path)
                or self.path.startswith(("/", "\\"))
                or any(part in {"", ".", ".."} for part in path_parts)
                or "\\" in self.path
                or (len(self.path) >= 2 and self.path[0].isalpha() and self.path[1] == ":")
            ):
                raise ValueError("artifact path must be a safe relative POSIX path")
            if not _MEDIA.fullmatch(self.media_type):
                raise ValueError("artifact media_type must be a portable media type")
            if not _SHA256.fullmatch(self.sha256):
                raise ValueError("artifact sha256 must be a lowercase SHA-256 digest")
        elif any(item is not None for item in (self.path, self.media_type, self.sha256)):
            raise ValueError(
                "unready artifact must not declare path, media_type, or sha256")
        if self.table_contract is not None:
            if not isinstance(self.table_contract, TableContract):
                raise ValueError("artifact table_contract must be a TableContract or null")
            if (
                self.table_contract.schema_name != self.schema_name
                or self.table_contract.schema_version != self.schema_version
            ):
                raise ValueError("artifact schema identity must match TableContract")
            if self.table_contract.orientation != orientation:
                raise ValueError("artifact orientation must match TableContract exactly")
        for name in ("estimand", "missing", "tie", "test", "multiplicity"):
            _nonempty(getattr(self, name), f"artifact {name}")
            validate_privacy_safe(getattr(self, name), where=f"artifact {name}")
        units = dict(self.units)
        if any(
            not isinstance(key, str) or not key
            or not isinstance(value, str) or not value for key, value in units.items()
        ):
            raise ValueError("artifact units must map strings to non-empty strings")
        validate_privacy_safe(units, where="artifact units")
        if self.table_contract is not None and units != dict(self.table_contract.units):
            raise ValueError("artifact units must match TableContract exactly")
        object.__setattr__(self, "units", MappingProxyType(units))
        object.__setattr__(
            self, "support", _safe_mapping(self.support, where="artifact support"))
        object.__setattr__(
            self, "privacy", _enum(self.privacy, ArtifactPrivacy, "artifact privacy"))

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id, "schema_name": self.schema_name,
            "schema_version": self.schema_version, "section_id": self.section_id,
            "evidence_layer": self.evidence_layer.value,
            "orientation": self.orientation, "coordinates": dict(self.coordinates),
            "status": self.status.value,
            "reason": None if self.reason is None else self.reason.value,
            "error": None if self.error is None else self.error.to_dict(),
            "source_refs": list(self.source_refs),
            "path": self.path, "media_type": self.media_type, "sha256": self.sha256,
            "table_contract": (
                None if self.table_contract is None else self.table_contract.to_manifest()),
            "estimand": self.estimand, "units": dict(self.units),
            "support": dict(self.support), "missing": self.missing, "tie": self.tie,
            "test": self.test, "multiplicity": self.multiplicity,
            "privacy": self.privacy.value,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ReportArtifact":
        expected = {
            "artifact_id", "schema_name", "schema_version", "section_id",
            "evidence_layer", "orientation", "coordinates", "status", "reason",
            "error", "source_refs", "path", "media_type", "sha256",
            "table_contract", "estimand", "units",
            "support", "missing", "tie", "test", "multiplicity", "privacy",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError(f"artifact fields must be exactly {sorted(expected)}")
        data = dict(value)
        if not isinstance(data["source_refs"], list):
            raise ValueError("artifact source_refs must be an array")
        data["source_refs"] = tuple(data["source_refs"])
        if data["error"] is not None:
            data["error"] = ReportError.from_dict(data["error"])
        if data["table_contract"] is not None:
            data["table_contract"] = table_contract_from_manifest(data["table_contract"])
        return cls(**data)


@dataclass(frozen=True)
class ReportManifest:
    """Portable v3 completion record, including failed and unavailable reports."""

    name: str
    title: str
    status: ReportStatus
    sections: tuple[SectionContract, ...]
    capabilities: ReportCapabilities
    lineage: ReportLineage
    artifacts: tuple[ReportArtifact, ...]
    errors: tuple[ReportError, ...]
    privacy: Mapping[str, object]
    schema: str = "prefscope.report.bundle"
    version: int = REPORT_BUNDLE_VERSION

    def __post_init__(self) -> None:
        if self.schema != "prefscope.report.bundle":
            raise ValueError("report manifest schema must be 'prefscope.report.bundle'")
        if self.version != REPORT_BUNDLE_VERSION:
            raise ValueError(f"report manifest version must be {REPORT_BUNDLE_VERSION}")
        _portable_name(self.name, "report manifest name")
        _nonempty(self.title, "report manifest title")
        validate_privacy_safe(self.title, where="report manifest title")
        status = _enum(self.status, ReportStatus, "report status")
        object.__setattr__(self, "status", status)
        sections = tuple(self.sections)
        if not sections or not all(isinstance(item, SectionContract) for item in sections):
            raise ValueError("report manifest needs ordered SectionContract values")
        section_ids = [item.section_id for item in sections]
        if len(set(section_ids)) != len(section_ids):
            raise ValueError("report manifest section_id values must be unique")
        if not isinstance(self.capabilities, ReportCapabilities):
            raise ValueError("report manifest capabilities must be ReportCapabilities")
        if not isinstance(self.lineage, ReportLineage):
            raise ValueError("report manifest lineage must be ReportLineage")
        if self.capabilities.n_rows != self.lineage.dataset.n_rows:
            raise ValueError("report capabilities n_rows must match dataset lineage")
        if self.capabilities.n_groups != self.lineage.dataset.n_groups:
            raise ValueError("report capabilities n_groups must match dataset lineage")
        artifacts = tuple(self.artifacts)
        if not all(isinstance(item, ReportArtifact) for item in artifacts):
            raise ValueError("report manifest artifacts must be ReportArtifact values")
        artifact_ids = [item.artifact_id for item in artifacts]
        if len(set(artifact_ids)) != len(artifact_ids):
            raise ValueError("report artifact_id values must be unique")
        known_sources = {item.source_id for item in self.lineage.sources}
        unknown_sources = sorted({
            source_id for artifact in artifacts for source_id in artifact.source_refs
            if source_id not in known_sources
        })
        if unknown_sources:
            raise ValueError(
                f"report artifacts refer to unknown lineage sources: {unknown_sources}")
        unknown_sections = sorted(
            {item.section_id for item in artifacts} - set(section_ids))
        if unknown_sections:
            raise ValueError(
                f"report artifacts refer to unknown sections: {unknown_sections}")
        section_by_id = {item.section_id: item for item in sections}
        children = {section_id: [] for section_id in section_ids}
        for artifact in artifacts:
            section = section_by_id[artifact.section_id]
            if artifact.evidence_layer is not section.evidence_layer:
                raise ValueError(
                    f"artifact {artifact.artifact_id!r} evidence layer does not "
                    "match its section")
            children[artifact.section_id].append(artifact)
        for section in sections:
            owned = children[section.section_id]
            has_ready = any(item.status is ArtifactStatus.READY for item in owned)
            has_error = any(item.status is ArtifactStatus.ERROR for item in owned)
            if section.status is SectionStatus.READY and (not has_ready or has_error):
                raise ValueError(
                    f"ready section {section.section_id!r} needs a ready artifact "
                    "and no error artifacts")
            if section.status is SectionStatus.UNAVAILABLE and any(
                item.status is not ArtifactStatus.UNAVAILABLE for item in owned
            ):
                raise ValueError(
                    f"unavailable section {section.section_id!r} may own only "
                    "unavailable artifacts")
            if section.status is SectionStatus.ERROR and (not has_error or has_ready):
                raise ValueError(
                    f"error section {section.section_id!r} needs an error artifact "
                    "and no ready artifacts")
        ready_layers = {
            item.evidence_layer for item in sections
            if item.status is SectionStatus.READY
        } | {
            item.evidence_layer for item in artifacts
            if item.status is ArtifactStatus.READY
        }
        if ready_layers != set(self.capabilities.evidence_layers):
            raise ValueError(
                "report capability evidence_layers must exactly match ready content")
        errors = tuple(self.errors)
        if not all(isinstance(item, ReportError) for item in errors):
            raise ValueError("report manifest errors must be ReportError values")
        has_processing_error = bool(errors) or any(
            item.status is SectionStatus.ERROR for item in sections
        ) or any(item.status is ArtifactStatus.ERROR for item in artifacts)
        has_ready_content = any(
            item.status is SectionStatus.READY for item in sections
        ) or any(item.status is ArtifactStatus.READY for item in artifacts)
        if status is ReportStatus.READY and has_processing_error:
            raise ValueError("ready report must not contain processing errors")
        if status is ReportStatus.PARTIAL and (
            not has_processing_error or not has_ready_content
        ):
            raise ValueError(
                "partial report needs both ready content and a processing error")
        if status is ReportStatus.FAILED and (
            not has_processing_error or has_ready_content
        ):
            raise ValueError(
                "failed report needs a processing error and no ready content")
        if not isinstance(self.privacy, Mapping):
            raise ValueError("report manifest privacy must be an object")
        policy = PrivacyPolicy.from_manifest(self.privacy)
        if any(
            item.privacy is ArtifactPrivacy.OPAQUE_ROWS for item in artifacts
        ) and not policy.opaque_ids:
            raise ValueError("opaque-row artifacts require opaque IDs in report privacy")
        if policy.profile_name.value == "shareable" and any(
            item.privacy is ArtifactPrivacy.LOCAL_FULL_TEXT for item in artifacts
        ):
            raise ValueError("shareable report must not declare local full text artifacts")
        if policy.text.value == "none" and any(
            item.privacy in {ArtifactPrivacy.TEXT_SNIPPETS, ArtifactPrivacy.LOCAL_FULL_TEXT}
            for item in artifacts
        ):
            raise ValueError("text-none report must not declare text-bearing artifacts")
        if policy.text.value == "snippets" and any(
            item.privacy is ArtifactPrivacy.LOCAL_FULL_TEXT for item in artifacts
        ):
            raise ValueError("snippet report must not declare full text artifacts")
        for section in sections:
            policy.validate_sanitized(
                section.coordinates,
            )
            if section.error is not None:
                policy.validate_sanitized(
                    section.error.detail,
                )
        for artifact in artifacts:
            policy.validate_sanitized(
                artifact.coordinates,
            )
            policy.validate_sanitized(
                artifact.support,
            )
            if artifact.error is not None:
                policy.validate_sanitized(
                    artifact.error.detail,
                )
        for error in errors:
            policy.validate_sanitized(
                error.detail,
            )
        validate_privacy_safe(dict(self.privacy), where="report manifest privacy")
        object.__setattr__(self, "sections", sections)
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(self, "errors", errors)
        object.__setattr__(
            self, "privacy",
            validate_portable_mapping(dict(self.privacy), where="report manifest privacy"),
        )

    def artifact(self, artifact_id: str) -> ReportArtifact:
        for artifact in self.artifacts:
            if artifact.artifact_id == artifact_id:
                return artifact
        raise ValueError(f"unknown report artifact {artifact_id!r}")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema, "name": self.name, "version": self.version,
            "title": self.title, "status": self.status.value,
            "sections": [item.to_dict() for item in self.sections],
            "capabilities": self.capabilities.to_dict(),
            "lineage": self.lineage.to_dict(),
            "artifacts": [item.to_dict() for item in self.artifacts],
            "errors": [item.to_dict() for item in self.errors],
            "privacy": dict(self.privacy),
        }

    @classmethod
    def from_dict(cls, value: object) -> "ReportManifest":
        expected = {
            "schema", "name", "version", "title", "status", "sections",
            "capabilities", "lineage", "artifacts", "errors", "privacy",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError(f"report manifest fields must be exactly {sorted(expected)}")
        for name in ("sections", "artifacts", "errors"):
            if not isinstance(value[name], list):
                raise ValueError(f"report manifest {name} must be an array")
        return cls(
            schema=value["schema"], name=value["name"], version=value["version"],
            title=value["title"], status=value["status"],
            sections=tuple(SectionContract.from_dict(item) for item in value["sections"]),
            capabilities=ReportCapabilities.from_dict(value["capabilities"]),
            lineage=ReportLineage.from_dict(value["lineage"]),
            artifacts=tuple(ReportArtifact.from_dict(item) for item in value["artifacts"]),
            errors=tuple(ReportError.from_dict(item) for item in value["errors"]),
            privacy=value["privacy"],
        )


def table_to_json_table(
    table: pd.DataFrame, contract: TableContract, privacy: PrivacyPolicy,
) -> dict[str, object]:
    """Create the canonical v1 JSON-table wire form used by report bundles."""
    if not isinstance(contract, TableContract):
        raise ValueError("contract must be a TableContract")
    if not isinstance(privacy, PrivacyPolicy):
        raise ValueError("privacy must be a PrivacyPolicy")
    contract.validate(table)
    if contract.allow_extra_columns:
        raise ValueError("canonical JSON tables do not permit extra columns")
    if any(normalize_field_name(column) != column for column in table.columns):
        raise ValueError("JSON-table columns must already use normalized field names")
    validate_privacy_safe(contract.to_manifest(), where="JSON-table contract")
    records = []
    for raw in table.to_dict(orient="records"):
        sanitized = privacy.sanitize(raw)
        if not isinstance(sanitized, Mapping) or tuple(sanitized) != tuple(table.columns):
            raise ValueError("privacy sanitation changed JSON-table columns")
        records.append(dict(sanitized))
    wire = {
        "format": JSON_TABLE_FORMAT,
        "version": JSON_TABLE_VERSION,
        "schema": contract.to_manifest(),
        "records": records,
    }
    # Publication helpers must never return a privacy-sanitized payload that no
    # longer satisfies its declared logical schema.
    parse_json_table(wire, expected_contract=contract, privacy=privacy)
    return wire


def _frame_from_json_records(
    records: list[Mapping[str, object]], contract: TableContract,
) -> pd.DataFrame:
    if not records:
        return contract.empty_frame()
    columns = {}
    for column in contract.required_columns:
        values = [record[column] for record in records]
        kind = contract.dtypes[column]
        non_null = [value for value in values if value is not None]
        if kind == "string":
            valid = all(isinstance(value, str) for value in non_null)
            dtype = "string"
        elif kind == "integer":
            valid = len(non_null) == len(values) and all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in non_null)
            dtype = "int64"
        elif kind == "nullable_integer":
            valid = all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in non_null)
            dtype = "Int64"
        elif kind == "float":
            valid = all(isinstance(value, float) for value in non_null)
            dtype = "float64"
        elif kind == "numeric":
            valid = all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in non_null)
            dtype = (
                "int64"
                if len(non_null) == len(values)
                and all(isinstance(value, int) for value in non_null)
                else "float64"
            )
        else:
            valid = len(non_null) == len(values) and all(
                isinstance(value, bool) for value in non_null)
            dtype = "bool"
        if not valid:
            raise ValueError(
                f"JSON-table column {column!r} has invalid values for {kind!r}")
        try:
            columns[column] = pd.Series(values, dtype=dtype)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"JSON-table column {column!r} cannot be represented as {kind!r}") from exc
    return pd.DataFrame(columns)


def _normalize_json_table_schema(value: object) -> dict[str, object]:
    """Validate that a JSON-table schema uses canonical field names."""
    if not isinstance(value, Mapping):
        raise ValueError("JSON-table schema must be an object")
    schema = dict(value)
    for name in ("required_columns", "unique_key"):
        raw = schema.get(name)
        if not isinstance(raw, list):
            raise ValueError(f"JSON-table schema {name} must be an array")
        normalized = [normalize_field_name(item) for item in raw]
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"JSON-table schema {name} collides after normalization")
        if normalized != raw:
            raise ValueError(
                f"JSON-table schema {name} must already use normalized field names")
    for name in ("dtypes", "units"):
        raw = schema.get(name)
        if not isinstance(raw, Mapping):
            raise ValueError(f"JSON-table schema {name} must be an object")
        normalized = [normalize_field_name(key) for key in raw]
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"JSON-table schema {name} collides after normalization")
        if normalized != list(raw):
            raise ValueError(
                f"JSON-table schema {name} must already use normalized field names")
    return schema


def _normalize_json_record(value: object) -> dict[str, object]:
    """Validate that one JSON-table record uses canonical field names."""
    if not isinstance(value, Mapping):
        raise ValueError("each JSON-table record must be an object")
    validate_privacy_safe(value, where="JSON-table raw record")
    keys = list(value)
    normalized = [normalize_field_name(key) for key in keys]
    if len(set(normalized)) != len(normalized):
        raise ValueError("JSON-table record keys collide after normalization")
    if normalized != keys:
        raise ValueError("JSON-table record keys must already be normalized")
    return dict(value)


def parse_json_table(
    value: object, *, expected_contract: TableContract | None = None,
    privacy: PrivacyPolicy | None = None,
) -> tuple[pd.DataFrame, TableContract]:
    """Parse and validate the canonical JSON-table wire form."""
    expected = {"format", "version", "schema", "records"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"JSON-table fields must be exactly {sorted(expected)}")
    if value["format"] != JSON_TABLE_FORMAT or value["version"] != JSON_TABLE_VERSION:
        raise ValueError("unsupported JSON-table format or version")
    normalized_schema = _normalize_json_table_schema(value["schema"])
    contract = table_contract_from_manifest(normalized_schema)
    if contract.allow_extra_columns:
        raise ValueError("canonical JSON tables do not permit extra columns")
    if expected_contract is not None:
        if not isinstance(expected_contract, TableContract) or contract != expected_contract:
            raise ValueError("JSON-table contract does not match expected TableContract")
    raw_records = value["records"]
    if not isinstance(raw_records, list):
        raise ValueError("JSON-table records must be an array")
    records = [_normalize_json_record(record) for record in raw_records]
    columns = contract.required_columns
    for record in records:
        if set(record) != set(columns):
            raise ValueError("each JSON-table record must declare every schema column")
        if privacy is not None:
            privacy.validate_sanitized(record)
        else:
            validate_privacy_safe(record, where="JSON-table record")
    table = _frame_from_json_records(records, contract)
    contract.validate(table)
    return table, contract


__all__ = [
    "REPORT_BUNDLE_VERSION", "JSON_TABLE_FORMAT", "JSON_TABLE_VERSION",
    "ArtifactPrivacy", "ArtifactStatus", "CompilerProvenance", "DatasetLineage",
    "EvidenceLayer", "ReportArtifact", "ReportCapabilities", "ReportDataset",
    "ReportError", "ReportLineage", "ReportManifest", "ReportMode", "ReportSpec",
    "ReportStatus", "SamplingProvenance", "SectionContract", "SectionKind",
    "SourceArtifactReference",
    "SectionOrientation", "SectionStatus", "StatusReason", "parse_json_table",
    "table_contract_from_manifest", "table_to_json_table",
]
