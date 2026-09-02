from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from prefscope.api.analysis_contracts import AnalysisArtifact, AnalysisDataset
from prefscope.api.analysis_execution import DatasetAnalysisResult
from prefscope.api.analysis_io import AnalysisDatasetReference, LoadedAnalysisResult
from prefscope.api.feature_catalog import FeatureCatalog
from prefscope.core.features import FeatureMatrix
from prefscope.core.table_schema import TableContract
from prefscope.reporting.contracts import (
    ArtifactPrivacy,
    ArtifactStatus,
    CompilerProvenance,
    DatasetLineage,
    EvidenceLayer,
    JSON_TABLE_FORMAT,
    ReportArtifact,
    ReportCapabilities,
    ReportDataset,
    ReportError,
    ReportLineage,
    ReportManifest,
    ReportMode,
    ReportSpec,
    ReportStatus,
    SectionContract,
    SectionKind,
    SectionOrientation,
    SectionStatus,
    SourceArtifactReference,
    StatusReason,
    SamplingProvenance,
    parse_json_table,
    table_to_json_table,
)
from prefscope.reporting.privacy import (
    PrivacyPolicy,
    PrivacyProfile,
    TextPolicy,
    html_neutral_text,
    normalize_field_name,
    validate_html_neutral_snippet,
)


def _dataset() -> AnalysisDataset:
    matrix = FeatureMatrix(
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        row_ids=("row-a", "row-b"),
        role="response",
        feature_ids=(4, 8),
    )
    return AnalysisDataset(features={"response": matrix}, outcomes={})


def _table_contract() -> TableContract:
    return TableContract(
        schema_name="feature_summary",
        schema_version=1,
        required_columns=("feature_id", "score"),
        dtypes={"feature_id": "integer", "score": "float"},
        unique_key=("feature_id",),
        orientation="as_declared",
        units={"score": "unitless"},
    )


def _analysis_artifact() -> AnalysisArtifact:
    return AnalysisArtifact(
        name="feature_summary",
        table=pd.DataFrame({"feature_id": [4, 8], "score": [0.2, 0.4]}),
        estimand="mean feature score",
        table_contract=_table_contract(),
    )


def _ready_section() -> SectionContract:
    return SectionContract(
        section_id="activity",
        kind=SectionKind.TYPED_TABLE,
        version=1,
        title="Activity",
        evidence_layer=EvidenceLayer.RAW_AXIS,
        orientation=SectionOrientation.AS_DECLARED,
        coordinates={"scope": "dataset", "view": "response"},
    )


def _ready_artifact(*, artifact_id="activity_table") -> ReportArtifact:
    return ReportArtifact(
        artifact_id=artifact_id,
        schema_name="feature_summary",
        schema_version=1,
        section_id="activity",
        evidence_layer=EvidenceLayer.RAW_AXIS,
        orientation="as_declared",
        coordinates={"scope": "dataset", "view": "response"},
        status=ArtifactStatus.READY,
        reason=None,
        error=None,
        source_refs=("analysis",),
        path=f"data/{artifact_id}.json",
        media_type="application/json",
        sha256="a" * 64,
        table_contract=_table_contract(),
        estimand="mean feature score",
        units={"score": "unitless"},
        support={"n_rows": 5},
        missing="complete_case",
        tie="not_applicable",
        test="not_applicable",
        multiplicity="not_applicable",
        privacy=ArtifactPrivacy.AGGREGATE,
    )


def _privacy(**overrides) -> PrivacyPolicy:
    options = {
        "text_fields": ("prompt_text",),
        "id_fields": ("row_id", "group_id"),
        "allow_fields": ("feature_id", "score"),
        "cell_count_fields": ("n_rows",),
        "categorical_fields": {
            "label": ("safe", "four", "eight"),
            "scope": ("dataset", "report"),
            "view": ("response",),
            "stage": ("compile",),
        },
    }
    options.update(overrides)
    return PrivacyPolicy.shareable(**options)


def _lineage(*, n_rows=2, n_groups=2) -> ReportLineage:
    return ReportLineage(
        dataset=DatasetLineage(
            dataset_sha256="d" * 64,
            row_ids_sha256="a" * 64,
            group_partition_sha256="b" * 64,
            group_source="explicit",
            n_rows=n_rows,
            n_groups=n_groups,
        ),
        sources=(
            SourceArtifactReference(
                source_id="analysis",
                artifact_type="prefscope.analysis_result",
                schema_version=1,
                sha256="c" * 64,
            ),
        ),
        compiler=CompilerProvenance(
            compiler_name="report_compiler",
            compiler_version="1.0",
            report_spec_name="report",
            report_spec_version=1,
            report_spec_sha256="e" * 64,
        ),
        sampling=SamplingProvenance(
            method="deterministic",
            sampling_frame_sha256="f" * 64,
            seed=7,
            population_count=n_rows,
            sampled_count=n_rows,
            max_examples_per_feature=8,
        ),
    )


def test_report_dataset_keeps_live_alignment_and_requires_exact_row_metadata():
    dataset = _dataset()
    catalog = pd.DataFrame({"feature_id": [4, 8], "label": ["four", "eight"]})
    metadata = pd.DataFrame({"row_id": ["row-a", "row-b"], "split": ["a", "b"]})
    wrapped = ReportDataset.from_analysis(
        dataset, feature_catalogs={"response": catalog}, row_metadata=metadata
    )
    assert wrapped.dataset is dataset
    assert wrapped.row_ids == dataset.row_ids
    assert wrapped.table_only is False
    assert wrapped.row_metadata["row_id"].tolist() == list(dataset.row_ids)

    typed = FeatureCatalog(
        pd.DataFrame({"feature_id": [4, 8], "name": ["four", "eight"]})
    )
    typed_wrapped = ReportDataset.from_analysis(
        dataset, feature_catalogs={"response": typed}
    )
    assert typed_wrapped.feature_catalogs["response"] is typed
    assert dict(typed_wrapped.feature_catalogs["response"].labels) == {
        4: "four",
        8: "eight",
    }

    with pytest.raises(ValueError, match="exactly prove"):
        ReportDataset(dataset, row_metadata=metadata.iloc[::-1].reset_index(drop=True))
    with pytest.raises(ValueError, match="DataFrame with row_id"):
        ReportDataset(dataset, row_metadata={"row_id": dataset.row_ids})
    with pytest.raises(ValueError, match="feature order"):
        ReportDataset(
            dataset,
            feature_catalogs={"response": catalog.iloc[::-1].reset_index(drop=True)},
        )
    with pytest.raises(ValueError, match="invalid feature_id"):
        ReportDataset(
            dataset,
            feature_catalogs={"response": pd.DataFrame({"feature_id": [4.0, 8.0]})},
        )
    with pytest.raises(ValueError, match="invalid feature_id"):
        ReportDataset(
            dataset,
            feature_catalogs={"response": pd.DataFrame({"feature_id": [False, True]})},
        )


def test_report_dataset_accepts_attached_and_explicit_detached_results():
    dataset = _dataset()
    artifact = _analysis_artifact()
    attached = DatasetAnalysisResult(dataset, {artifact.name: artifact})
    attached_report = ReportDataset(attached)
    assert attached_report.dataset is dataset
    assert attached_report.dataset_reference is None
    assert attached_report.group_codes == (0, 1)
    assert attached_report.artifacts[artifact.name] is artifact
    assert not attached_report.table_only

    row_digest = hashlib.sha256()
    row_digest.update(b"prefscope-analysis-rows-v1\0")
    for row_id in dataset.row_ids:
        encoded = row_id.encode("utf-8")
        row_digest.update(len(encoded).to_bytes(8, "big"))
        row_digest.update(encoded)
    group_digest = hashlib.sha256()
    group_digest.update(b"prefscope-analysis-groups-v1\0")
    group_digest.update((2).to_bytes(8, "big"))
    for code in (0, 1):
        group_digest.update(code.to_bytes(8, "big", signed=True))
    reference = AnalysisDatasetReference(
        row_ids=dataset.row_ids,
        group_source="row",
        group_codes=(0, 1),
        row_ids_sha256=row_digest.hexdigest(),
        group_partition_sha256=group_digest.hexdigest(),
    )
    loaded = LoadedAnalysisResult(
        dataset_reference=reference, artifacts={artifact.name: artifact}
    )
    detached = ReportDataset(loaded)
    assert detached.dataset is None
    assert detached.dataset_reference is reference
    assert detached.row_ids == dataset.row_ids
    assert detached.group_codes == reference.group_codes
    assert detached.table_only
    assert detached.artifacts[artifact.name] is artifact
    with pytest.raises(ValueError, match="feature identity proof"):
        ReportDataset(
            loaded,
            feature_catalogs={"response": pd.DataFrame({"feature_id": [4, 8]})},
        )

    malformed = SimpleNamespace(
        dataset_reference=reference, artifacts={artifact.name: artifact}
    )
    with pytest.raises(ValueError, match="LoadedAnalysisResult"):
        ReportDataset(malformed)


def test_report_spec_matches_public_design_shape():
    spec = ReportSpec(
        title="Model A vs Model B", sections="auto", max_examples_per_feature=8
    )
    assert spec.name == "report"
    assert spec.sections == "auto"
    explicit = replace(spec, sections=(SectionKind.NOTICE, SectionKind.TYPED_TABLE))
    assert explicit.sections == (SectionKind.NOTICE, SectionKind.TYPED_TABLE)
    with pytest.raises(ValueError, match="unique"):
        replace(spec, sections=(SectionKind.NOTICE, SectionKind.NOTICE))


def test_section_states_have_typed_reason_and_error_contracts():
    unavailable = SectionContract(
        section_id="outcomes",
        kind=SectionKind.NOTICE,
        version=1,
        title="Outcomes",
        evidence_layer=EvidenceLayer.OUTCOME_ASSOCIATION,
        orientation=SectionOrientation.NONE,
        coordinates={"scope": "dataset"},
        status=SectionStatus.UNAVAILABLE,
        reason=StatusReason.INPUT_ABSENT,
    )
    assert SectionContract.from_dict(unavailable.to_dict()) == unavailable
    with pytest.raises(ValueError, match="non-processing"):
        replace(unavailable, reason=StatusReason.PROCESSING_ERROR)

    error = ReportError(
        code="analysis_failed", message="Analysis failed", detail={"stage": "compile"}
    )
    failed = replace(
        unavailable,
        status=SectionStatus.ERROR,
        reason=StatusReason.PROCESSING_ERROR,
        error=error,
    )
    assert failed.error is error
    with pytest.raises(ValueError, match="typed ReportError"):
        replace(failed, error=None)
    with pytest.raises(ValueError, match="canonical HTML-neutral"):
        replace(error, message="<script>unsafe</script>")


def test_artifact_instance_identity_is_separate_from_schema_and_status_controls_files():
    first = _ready_artifact(artifact_id="activity_overall")
    second = _ready_artifact(artifact_id="activity_by_group")
    assert first.schema_name == second.schema_name == "feature_summary"
    assert first.artifact_id != second.artifact_id

    unavailable = replace(
        first,
        status=ArtifactStatus.UNAVAILABLE,
        reason=StatusReason.INSUFFICIENT_SUPPORT,
        path=None,
        media_type=None,
        sha256=None,
    )
    assert unavailable.path is None
    with pytest.raises(ValueError, match="must not declare"):
        replace(unavailable, path="data/leak.json")
    with pytest.raises(ValueError, match="needs path"):
        replace(first, path=None)
    with pytest.raises(ValueError, match="safe relative"):
        replace(first, path="data/<unsafe>.json")

    error = ReportError(code="serialization_failed", message="Serialization failed")
    failed = replace(
        unavailable,
        status=ArtifactStatus.ERROR,
        reason=StatusReason.PROCESSING_ERROR,
        error=error,
    )
    assert failed.to_dict()["error"]["code"] == "serialization_failed"
    with pytest.raises(ValueError, match="support"):
        replace(first, support={"source_path": "/private/data.csv"})


def test_artifact_exact_orientation_is_separate_from_coarse_section_direction():
    contract = TableContract(
        schema_name="outcome_associations",
        schema_version=1,
        required_columns=("feature_id", "estimate"),
        dtypes={"feature_id": "integer", "estimate": "float"},
        unique_key=("feature_id",),
        orientation="feature_activation_to_declared_outcome",
        units={"estimate": "probability_points"},
    )
    artifact = replace(
        _ready_artifact(),
        schema_name="outcome_associations",
        table_contract=contract,
        evidence_layer=EvidenceLayer.OUTCOME_ASSOCIATION,
        orientation="feature_activation_to_declared_outcome",
        units={"estimate": "probability_points"},
    )
    restored = ReportArtifact.from_dict(artifact.to_dict())
    assert restored.orientation == "feature_activation_to_declared_outcome"
    assert SectionOrientation.AS_DECLARED.value == "as_declared"
    assert (
        len(
            {
                EvidenceLayer.FEATURE_ROLE,
                EvidenceLayer.RESPONSE_SCOPE,
                EvidenceLayer.CONTEXT,
                EvidenceLayer.MODEL_TENDENCY,
            }
        )
        == 4
    )
    with pytest.raises(ValueError, match="match TableContract exactly"):
        replace(artifact, orientation="as_declared")


def test_manifest_persists_title_sections_capabilities_artifacts_errors_and_status():
    section = _ready_section()
    artifact = _ready_artifact()
    capabilities = ReportCapabilities(
        mode=ReportMode.PAIRED_BATTLES,
        n_rows=2,
        n_groups=2,
        feature_views=("response",),
        evidence_layers=(EvidenceLayer.RAW_AXIS,),
    )
    policy = _privacy()
    manifest = ReportManifest(
        name="comparison",
        title="Model A vs Model B",
        status=ReportStatus.READY,
        sections=(section,),
        capabilities=capabilities,
        lineage=_lineage(),
        artifacts=(artifact,),
        errors=(),
        privacy=policy.to_manifest(),
    )
    restored = ReportManifest.from_dict(json.loads(json.dumps(manifest.to_dict())))
    assert restored.title == manifest.title
    assert restored.sections == manifest.sections
    assert restored.capabilities == capabilities
    assert restored.artifact("activity_table").schema_name == "feature_summary"

    with pytest.raises(ValueError, match="unknown sections"):
        replace(manifest, artifacts=(replace(artifact, section_id="missing"),))
    with pytest.raises(ValueError, match="needs a ready artifact"):
        replace(manifest, artifacts=())
    with pytest.raises(ValueError, match="evidence layer does not match"):
        replace(
            manifest,
            artifacts=(
                replace(artifact, evidence_layer=EvidenceLayer.SEMANTIC_PRESENCE),
            ),
        )
    processing_error = ReportError(code="compile_failed", message="Compile failed")
    with pytest.raises(ValueError, match="must not contain processing errors"):
        replace(manifest, errors=(processing_error,))
    partial = replace(manifest, status=ReportStatus.PARTIAL, errors=(processing_error,))
    assert partial.status is ReportStatus.PARTIAL
    with pytest.raises(ValueError, match="both ready content and a processing error"):
        replace(manifest, status=ReportStatus.PARTIAL)
    with pytest.raises(ValueError, match="no ready content"):
        replace(partial, status=ReportStatus.FAILED)
    unavailable_section = replace(
        section, status=SectionStatus.UNAVAILABLE, reason=StatusReason.INPUT_ABSENT
    )
    with pytest.raises(ValueError, match="only unavailable artifacts"):
        replace(manifest, sections=(unavailable_section,))

    for support, match in (
        ({"n_rows": 5, "row_id": "raw-row"}, "wrong type tag"),
        ({"n_rows": 5, "prompt_text": "raw text"}, "must be null"),
        ({"n_rows": 5, "unknown_field": 1}, "unknown shareable"),
        ({"n_rows": 4}, "below minimum_cell_count"),
    ):
        with pytest.raises(ValueError, match=match):
            replace(manifest, artifacts=(replace(artifact, support=support),))


def test_manifest_requires_strict_lineage_and_validates_artifact_sources():
    section = _ready_section()
    artifact = _ready_artifact()
    capabilities = ReportCapabilities(
        mode=ReportMode.CORPUS,
        n_rows=2,
        n_groups=2,
        evidence_layers=(EvidenceLayer.RAW_AXIS,),
    )
    manifest = ReportManifest(
        name="lineage",
        title="Lineage",
        status=ReportStatus.READY,
        sections=(section,),
        capabilities=capabilities,
        lineage=_lineage(),
        artifacts=(artifact,),
        errors=(),
        privacy=_privacy().to_manifest(),
    )
    restored = ReportManifest.from_dict(json.loads(json.dumps(manifest.to_dict())))
    assert restored.lineage == manifest.lineage
    assert restored.artifacts[0].source_refs == ("analysis",)
    with pytest.raises(ValueError, match="unknown lineage sources"):
        replace(
            manifest,
            artifacts=(replace(artifact, source_refs=("unknown_source",)),),
        )
    with pytest.raises(ValueError, match="n_rows must match dataset lineage"):
        replace(
            manifest,
            capabilities=replace(capabilities, n_rows=3),
        )
    with pytest.raises(ValueError, match="n_groups must match dataset lineage"):
        replace(
            manifest,
            capabilities=replace(capabilities, n_groups=None),
        )
    with pytest.raises(ValueError, match="n_groups must match dataset lineage"):
        replace(
            manifest,
            capabilities=replace(capabilities, n_groups=1),
        )
    sampling_wire = manifest.lineage.sampling.to_dict()
    assert set(sampling_wire) == {
        "method",
        "sampling_frame_sha256",
        "seed",
        "population_count",
        "sampled_count",
        "max_examples_per_feature",
    }
    del sampling_wire["sampling_frame_sha256"]
    with pytest.raises(ValueError, match="fields must be exactly"):
        SamplingProvenance.from_dict(sampling_wire)
    raw = manifest.to_dict()
    del raw["lineage"]
    with pytest.raises(ValueError, match="fields must be exactly"):
        ReportManifest.from_dict(raw)
    with pytest.raises(ValueError, match="population_count must match"):
        replace(
            manifest.lineage,
            sampling=replace(manifest.lineage.sampling, population_count=3),
        )
    with pytest.raises(ValueError, match="SHA-256"):
        replace(manifest.lineage.sources[0], sha256="not-a-hash")
    with pytest.raises(ValueError, match="sampling_frame_sha256"):
        replace(manifest.lineage.sampling, sampling_frame_sha256="missing")


def test_manifest_allows_all_unavailable_or_failed_without_ready_artifacts():
    unavailable = SectionContract(
        section_id="outcomes",
        kind=SectionKind.NOTICE,
        version=1,
        title="Outcomes unavailable",
        evidence_layer=EvidenceLayer.OUTCOME_ASSOCIATION,
        orientation=SectionOrientation.NONE,
        coordinates={"scope": "dataset"},
        status=SectionStatus.UNAVAILABLE,
        reason=StatusReason.INPUT_ABSENT,
    )
    capabilities = ReportCapabilities(
        mode=ReportMode.TABLE_ONLY,
        n_rows=2,
        n_groups=2,
        feature_views=(),
        evidence_layers=(),
        table_only=True,
    )
    policy = _privacy()
    completed = ReportManifest(
        name="unavailable_report",
        title="Unavailable report",
        status=ReportStatus.READY,
        sections=(unavailable,),
        capabilities=capabilities,
        lineage=_lineage(),
        artifacts=(),
        errors=(),
        privacy=policy.to_manifest(),
    )
    assert completed.artifacts == ()

    error = ReportError(code="load_failed", message="Load failed")
    with pytest.raises(ValueError, match="both ready content and a processing error"):
        replace(completed, status=ReportStatus.PARTIAL, errors=(error,))
    with pytest.raises(ValueError, match="processing error and no ready content"):
        replace(completed, status=ReportStatus.FAILED)
    failed_section = replace(
        unavailable,
        status=SectionStatus.ERROR,
        reason=StatusReason.PROCESSING_ERROR,
        error=error,
    )
    failed_artifact = replace(
        _ready_artifact(),
        artifact_id="outcome_error",
        section_id="outcomes",
        evidence_layer=EvidenceLayer.OUTCOME_ASSOCIATION,
        status=ArtifactStatus.ERROR,
        reason=StatusReason.PROCESSING_ERROR,
        error=error,
        path=None,
        media_type=None,
        sha256=None,
    )
    failed = replace(
        completed,
        status=ReportStatus.FAILED,
        sections=(failed_section,),
        artifacts=(failed_artifact,),
        errors=(error,),
    )
    assert ReportManifest.from_dict(failed.to_dict()).status is ReportStatus.FAILED


def test_manifest_rejects_artifact_privacy_beyond_top_level_policy():
    section = _ready_section()
    capabilities = ReportCapabilities(
        mode=ReportMode.CORPUS,
        n_rows=2,
        n_groups=2,
        evidence_layers=(EvidenceLayer.RAW_AXIS,),
    )
    with pytest.raises(ValueError, match="full text"):
        ReportManifest(
            name="unsafe",
            title="Unsafe",
            status=ReportStatus.READY,
            sections=(section,),
            capabilities=capabilities,
            lineage=_lineage(),
            artifacts=(
                replace(_ready_artifact(), privacy=ArtifactPrivacy.LOCAL_FULL_TEXT),
            ),
            errors=(),
            privacy=_privacy().to_manifest(),
        )


def test_privacy_rejects_shareable_full_and_persists_explicit_field_contracts():
    with pytest.raises(ValueError, match="does not permit full"):
        _privacy(text=TextPolicy.FULL)
    policy = _privacy()
    restored = PrivacyPolicy.from_manifest(policy.to_manifest())
    assert restored.profile_name is PrivacyProfile.SHAREABLE
    assert restored.allow_fields == ("feature_id", "score")
    assert dict(restored.categorical_fields) == {
        "label": ("safe", "four", "eight"),
        "scope": ("dataset", "report"),
        "view": ("response",),
        "stage": ("compile",),
    }
    assert restored.text_fields == ("prompt_text",)
    assert restored.id_fields == ("row_id", "group_id")
    assert restored.cell_count_fields == ("n_rows",)
    assert restored.sanitize({"label": "safe"}) == {"label": "safe"}
    with pytest.raises(ValueError, match="declared enum"):
        restored.sanitize({"label": "unpersisted"})
    with pytest.raises(ValueError, match="non-empty"):
        _privacy(categorical_fields={"label": ()})
    with pytest.raises(ValueError, match="unique"):
        _privacy(categorical_fields={"label": ("safe", "safe")})
    assert normalize_field_name("promptText") == "prompt_text"
    assert normalize_field_name("row-ID") == "row_id"
    with pytest.raises(ValueError, match="field roles must be disjoint"):
        PrivacyPolicy.shareable(text_fields=("row_id",), id_fields=("row_id",))
    with pytest.raises(ValueError, match="field roles must be disjoint"):
        _privacy(redact_fields=("score",))
    with pytest.raises(ValueError, match="secret keys"):
        PrivacyPolicy.shareable(redact_fields=("api_key",), id_fields=())
    with pytest.raises(ValueError, match="PII-bearing keys"):
        PrivacyPolicy.shareable(redact_fields=("email_address",), id_fields=())


def test_shareable_privacy_normalizes_keys_redacts_and_type_tags_ids():
    policy = _privacy(text=TextPolicy.SNIPPETS, snippet_chars=20)
    clean = policy.sanitize(
        {
            "rowID": "raw-row",
            "promptText": "<b>hello</b>",
            "Score": 0.5,
            "label": "safe",
            "nRows": 5,
        }
    )
    assert clean["row_id"].startswith("opaque:row_id:")
    assert "raw-row" not in json.dumps(clean)
    assert clean["prompt_text"].startswith("&lt;b&gt;")
    assert clean["label"] == "safe"
    policy.validate_sanitized(clean)
    # One canonical helper validates already escaped snippets idempotently.
    escaped = html_neutral_text("<safe>")
    assert validate_html_neutral_snippet(escaped) == "&lt;safe&gt;"
    with pytest.raises(ValueError, match="canonical HTML-neutral"):
        validate_html_neutral_snippet("<safe>")
    with pytest.raises(ValueError, match="bounded snippet length"):
        validate_html_neutral_snippet("&lt;&lt;&lt;", max_chars=2)
    assert validate_html_neutral_snippet("&lt;&lt;…", max_chars=2) == "&lt;&lt;…"
    with pytest.raises(ValueError, match="canonical HTML-neutral"):
        validate_html_neutral_snippet("&LT;", max_chars=2)
    policy.validate_sanitized(clean)
    with pytest.raises(ValueError, match="wrong type tag"):
        policy.validate_sanitized({**clean, "row_id": "opaque:group_id:" + "a" * 24})
    with pytest.raises(ValueError, match="ID field values|opaque ID values"):
        policy.sanitize({"rowID": 1})
    assert policy.sanitize({"rowID": "1"})["row_id"].startswith("opaque:row_id:")


def test_shareable_privacy_fails_closed_on_unknown_pii_secrets_and_small_cells():
    policy = _privacy()
    with pytest.raises(ValueError, match="unknown shareable field"):
        policy.sanitize({"mysteryValue": None})
    with pytest.raises(ValueError, match="unknown shareable field"):
        policy.sanitize({"name": "must not bypass the allowlist"})
    with pytest.raises(ValueError, match="PII-bearing|credential-like"):
        policy.sanitize({"emailAddress": "person@example.org"})
    with pytest.raises(ValueError, match="secret|credential"):
        policy.sanitize({"apiKey": "secret-value"})
    with pytest.raises(ValueError, match="PII or markup|credential-like"):
        policy.sanitize({"victim@example.org": 1})
    with pytest.raises(ValueError, match="control"):
        policy.sanitize({"safe\u202ekey": 1})
    with pytest.raises(ValueError, match="format|control"):
        policy.sanitize({"promptText": "safe\u202eunsafe"})
    with pytest.raises(ValueError, match="direct PII literal|credential-like"):
        policy.sanitize({"promptText": "person@example.org"})
    with pytest.raises(ValueError, match="direct PII literal|credential-like"):
        policy.sanitize({"promptText": "+1 (212) 555-0199"})
    with pytest.raises(ValueError, match="numeric/bool/null"):
        policy.sanitize({"score": "private string"})
    with pytest.raises(ValueError, match="below minimum_cell_count"):
        policy.sanitize({"nRows": 4})
    assert policy.sanitize({"nRows": None}) == {"n_rows": None}
    assert policy.sanitize({"nRows": 5}) == {"n_rows": 5}
    cyclic = {}
    cyclic["nested"] = cyclic
    with pytest.raises(ValueError, match="cycle"):
        policy.sanitize(cyclic)


def test_shareable_recursive_schema_requires_typed_containers_and_leaves():
    policy = PrivacyPolicy.shareable(
        allow_fields=("score",),
        object_fields=("metrics",),
        list_fields=("items",),
        id_fields=(),
    )
    restored = PrivacyPolicy.from_manifest(policy.to_manifest())
    assert restored.object_fields == ("metrics",)
    assert restored.list_fields == ("items",)
    assert policy.sanitize({"metrics": {"score": 1.0}}) == {"metrics": {"score": 1.0}}
    assert policy.sanitize({"items": [{"score": 1.0}]}) == {"items": [{"score": 1.0}]}
    assert policy.sanitize({"metrics": None, "items": None}) == {
        "metrics": None,
        "items": None,
    }
    policy.validate_sanitized({"metrics": None, "items": None})
    with pytest.raises(ValueError, match="unknown shareable field"):
        policy.sanitize({"nested": {"score": 1.0}})
    with pytest.raises(ValueError, match="must contain typed objects"):
        policy.sanitize({"items": [1.0]})
    with pytest.raises(ValueError, match="not declared"):
        policy.sanitize({"score": {"score": 1.0}})
    with pytest.raises(ValueError, match="unknown shareable field"):
        policy.sanitize({"metrics": {"unknown": 1.0}})


def test_missing_numeric_values_become_null_but_infinity_is_rejected():
    policy = _privacy()
    clean = policy.sanitize({"score": np.nan, "featureID": np.int64(4)})
    assert clean == {"score": None, "feature_id": 4}
    with pytest.raises(ValueError, match="finite"):
        policy.sanitize({"score": np.inf})
    with pytest.raises(ValueError, match="browser-safe"):
        policy.sanitize({"score": 2**53})
    with pytest.raises(ValueError, match="missing numbers must be null"):
        policy.validate_sanitized({"score": float("nan")})


def test_local_full_text_is_explicit_and_unknown_local_fields_are_allowed():
    policy = PrivacyPolicy.profile(
        "local", text="full", text_fields=("prompt_text",), id_fields=("row_id",)
    )
    clean = policy.sanitize(
        {"rowID": "local-row", "promptText": "<b>full</b>", "customMetric": 1}
    )
    assert clean["row_id"] == "local-row"
    assert clean["prompt_text"] == "&lt;b&gt;full&lt;/b&gt;"
    assert clean["custom_metric"] == 1
    with pytest.raises(ValueError, match="direct PII literal|credential-like"):
        policy.sanitize({"custom": "person@example.org"})
    with pytest.raises(ValueError, match="direct PII literal|credential-like"):
        policy.sanitize({"custom": "212-555-0199"})


def test_canonical_json_table_round_trip_applies_contract_and_privacy():
    contract = TableContract(
        schema_name="scores",
        schema_version=1,
        required_columns=("feature_id", "score", "n_rows"),
        dtypes={"feature_id": "integer", "score": "float", "n_rows": "integer"},
        unique_key=("feature_id",),
        orientation="as_declared",
        units={"score": "unitless", "n_rows": "rows"},
    )
    table = pd.DataFrame(
        {
            "feature_id": np.array([4, 8], dtype=np.int64),
            "score": np.array([0.5, np.nan], dtype=float),
            "n_rows": np.array([5, 6], dtype=np.int64),
        }
    )
    policy = _privacy()
    wire = table_to_json_table(table, contract, policy)
    assert wire["format"] == JSON_TABLE_FORMAT
    assert wire["records"][1]["score"] is None
    policy.validate_sanitized(wire)
    restored, restored_contract = parse_json_table(
        json.loads(json.dumps(wire, allow_nan=False)),
        expected_contract=contract,
        privacy=policy,
    )
    assert restored_contract == contract
    assert restored["feature_id"].tolist() == [4, 8]
    assert np.isnan(restored.loc[1, "score"])

    renamed_wire = json.loads(json.dumps(wire))
    renamed_wire["schema"]["required_columns"] = ["featureID", "Score", "nRows"]
    renamed_wire["schema"]["dtypes"] = {
        "featureID": "integer",
        "Score": "float",
        "nRows": "integer",
    }
    renamed_wire["schema"]["unique_key"] = ["featureID"]
    renamed_wire["schema"]["units"] = {"Score": "unitless", "nRows": "rows"}
    renamed_wire["records"] = [
        {"featureID": row["feature_id"], "Score": row["score"], "nRows": row["n_rows"]}
        for row in renamed_wire["records"]
    ]
    with pytest.raises(ValueError, match="already use normalized"):
        parse_json_table(renamed_wire, expected_contract=contract, privacy=policy)
    collision_wire = json.loads(json.dumps(wire))
    collision_wire["schema"]["required_columns"] = [
        "feature_id",
        "score",
        "Score",
        "n_rows",
    ]
    with pytest.raises(ValueError, match="collides after normalization"):
        parse_json_table(collision_wire, privacy=policy)
    record_name_wire = json.loads(json.dumps(wire))
    record_name_wire["records"][0]["Score"] = record_name_wire["records"][0].pop(
        "score"
    )
    with pytest.raises(ValueError, match="already be normalized"):
        parse_json_table(record_name_wire, privacy=policy)
    record_collision_wire = json.loads(json.dumps(wire))
    record_collision_wire["records"][0]["Score"] = 0.5
    with pytest.raises(ValueError, match="collide after normalization"):
        parse_json_table(record_collision_wire, privacy=policy)

    all_null = table.copy()
    all_null["score"] = np.array([np.nan, np.nan], dtype=float)
    null_wire = table_to_json_table(all_null, contract, policy)
    null_table, _ = parse_json_table(null_wire, privacy=policy)
    assert str(null_table["score"].dtype) == "float64"
    assert null_table["score"].isna().all()

    empty_wire = table_to_json_table(table.iloc[:0], contract, policy)
    empty_table, _ = parse_json_table(empty_wire, privacy=policy)
    assert list(empty_table.columns) == list(contract.required_columns)
    assert str(empty_table["score"].dtype) == "float64"

    invalid_wire = json.loads(json.dumps(wire))
    invalid_wire["records"][0]["feature_id"] = "4"
    with pytest.raises(ValueError, match="invalid values|numeric/bool/null"):
        parse_json_table(invalid_wire, privacy=policy)

    bad = dict(wire)
    bad["surprise"] = True
    with pytest.raises(ValueError, match="fields must be exactly"):
        parse_json_table(bad)
    small = table.copy()
    small.loc[0, "n_rows"] = 2
    with pytest.raises(ValueError, match="suppress the cell"):
        table_to_json_table(small, contract, policy)
    redacting = _privacy(allow_fields=("score",), redact_fields=("feature_id",))
    with pytest.raises(ValueError, match="unique|invalid"):
        table_to_json_table(table, contract, redacting)

    extra_contract = replace(contract, allow_extra_columns=True)
    extra_table = table.assign(extra=np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="do not permit extra"):
        table_to_json_table(extra_table, extra_contract, policy)

    malformed_schema = contract.to_manifest()
    malformed_schema["required_columns"] = "x"
    malformed_wire = dict(wire)
    malformed_wire["schema"] = malformed_schema
    with pytest.raises(ValueError, match="must be an array|must be arrays"):
        parse_json_table(malformed_wire)

    capability_wire = ReportCapabilities(mode=ReportMode.CORPUS, n_rows=2).to_dict()
    capability_wire["feature_views"] = "x"
    with pytest.raises(ValueError, match="must be an array|must be arrays"):
        ReportCapabilities.from_dict(capability_wire)


def test_reporting_contract_modules_are_torch_free():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import prefscope.reporting.contracts; "
            "import prefscope.reporting.privacy; assert 'torch' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
