from __future__ import annotations

import subprocess
import sys

import prefscope
import prefscope.api as public_api
import prefscope.api.analysis as analysis_api
from prefscope.api import analysis_io
import prefscope.reporting as reporting
from prefscope.reporting import contracts, source


def test_analysis_io_public_facades_are_identical():
    assert prefscope.AnalysisDatasetReference is analysis_api.AnalysisDatasetReference
    assert prefscope.LoadedAnalysisResult is analysis_api.LoadedAnalysisResult
    assert prefscope.load_analysis_result is analysis_api.load_analysis_result
    assert prefscope.save_analysis_result is analysis_api.save_analysis_result
    assert public_api.AnalysisDatasetReference is analysis_api.AnalysisDatasetReference
    assert public_api.LoadedAnalysisResult is analysis_api.LoadedAnalysisResult
    assert public_api.load_analysis_result is analysis_api.load_analysis_result
    assert public_api.save_analysis_result is analysis_api.save_analysis_result
    assert analysis_io.LoadedAnalysisResult is analysis_api.LoadedAnalysisResult


def test_reporting_exports_canonical_foundation_only():
    assert reporting.FeatureSource is source.FeatureSource
    assert reporting.FeatureChunk is source.FeatureChunk
    assert reporting.FeatureBundleReader is source.FeatureBundleReader
    assert reporting.ReportManifest is contracts.ReportManifest
    assert reporting.ReportArtifact is contracts.ReportArtifact
    assert reporting.ReportLineage is contracts.ReportLineage
    assert reporting.DatasetLineage is contracts.DatasetLineage
    assert reporting.SourceArtifactReference is contracts.SourceArtifactReference
    assert reporting.CompilerProvenance is contracts.CompilerProvenance
    assert reporting.SamplingProvenance is contracts.SamplingProvenance
    assert reporting.PathPayload.__module__ == "prefscope.reporting.io"
    assert reporting.SectionOrientation is contracts.SectionOrientation
    assert reporting.artifact_sha256.__name__ == "artifact_sha256"
    for premature_alias in (
        "BundleArtifact", "Orientation", "SectionStatusReason",
        "load_bundle", "write_bundle",
    ):
        assert not hasattr(reporting, premature_alias)


def test_public_foundation_imports_are_torch_free():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import prefscope; import prefscope.api; "
            "import prefscope.api.analysis; import prefscope.observability; "
            "import prefscope.reporting; "
            "assert not any(n == 'torch' or n.startswith('torch.') for n in sys.modules)",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
