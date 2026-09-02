"""Typed, privacy-aware foundations for PrefScope report bundles."""
from __future__ import annotations

from prefscope.reporting.contracts import (
    JSON_TABLE_FORMAT,
    JSON_TABLE_VERSION,
    REPORT_BUNDLE_VERSION,
    ArtifactPrivacy,
    ArtifactStatus,
    CompilerProvenance,
    DatasetLineage,
    EvidenceLayer,
    ReportArtifact,
    ReportCapabilities,
    ReportDataset,
    ReportError,
    ReportLineage,
    ReportManifest,
    ReportMode,
    ReportSpec,
    ReportStatus,
    SamplingProvenance,
    SectionContract,
    SectionKind,
    SectionOrientation,
    SectionStatus,
    SourceArtifactReference,
    StatusReason,
    parse_json_table,
    table_contract_from_manifest,
    table_to_json_table,
)
from prefscope.reporting.io import (
    MANIFEST_FILENAME,
    PathPayload,
    ReportBundle,
    artifact_sha256,
    canonical_json_bytes,
    json_payload,
    load_report_bundle,
    write_report_bundle,
)
from prefscope.reporting.privacy import (
    PrivacyPolicy,
    PrivacyProfile,
    TextPolicy,
    html_neutral_text,
    normalize_field_name,
    sanitize_json,
    validate_html_neutral_snippet,
    validate_privacy_safe,
)
from prefscope.reporting.source import FeatureBundleReader, FeatureChunk, FeatureSource

__all__ = [
    "REPORT_BUNDLE_VERSION", "JSON_TABLE_FORMAT", "JSON_TABLE_VERSION",
    "MANIFEST_FILENAME", "PathPayload", "ArtifactPrivacy", "ArtifactStatus",
    "CompilerProvenance", "DatasetLineage", "EvidenceLayer",
    "FeatureBundleReader", "FeatureChunk", "FeatureSource", "PrivacyPolicy",
    "PrivacyProfile", "ReportArtifact", "ReportBundle", "ReportCapabilities",
    "ReportDataset", "ReportError", "ReportLineage", "ReportManifest", "ReportMode",
    "ReportSpec", "ReportStatus", "SamplingProvenance", "SectionContract",
    "SectionKind", "SectionOrientation", "SectionStatus", "SourceArtifactReference",
    "StatusReason", "TextPolicy", "artifact_sha256",
    "canonical_json_bytes", "html_neutral_text", "json_payload",
    "load_report_bundle", "normalize_field_name", "parse_json_table",
    "sanitize_json", "table_contract_from_manifest", "table_to_json_table",
    "validate_html_neutral_snippet", "validate_privacy_safe", "write_report_bundle",
]
