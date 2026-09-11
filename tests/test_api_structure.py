from __future__ import annotations

import subprocess
import sys

import prefscope
from prefscope.api.loaded_lens import pairs_to_battles


def test_public_api_exports_only_available_objects():
    import prefscope.api as api

    assert all(hasattr(api, name) for name in api.__all__)
    assert prefscope.analysis.__all__ == [
        "activation_summary",
        "coactivation_counts",
        "coactivation_pairs",
        "cross_coactivation_counts",
        "top_activating_rows",
    ]
    assert api.RepresentationSource is prefscope.RepresentationSource
    assert pairs_to_battles.__module__ == "prefscope.api.loaded_lens"


def test_recipe_framework_is_not_reexported_and_core_import_is_torch_free():
    removed = {
        "AnalysisArtifact", "AnalysisComponent", "AnalysisDataset", "AnalysisPlan",
        "AnalyzeConfig", "OutcomeSpec", "PrivacyPolicy", "ReportCompiler",
        "PresenceMatrix", "analyze_dataset", "compile_report", "concept_presence",
        "feature_thresholds", "load_plugins", "registry", "run_analysis",
        "semantic_presence",
    }
    assert removed.isdisjoint(prefscope.__all__)
    assert {
        "encode", "encode_one", "encode_pairs", "encode_items",
        "project", "project_representations", "presence", "top_concepts",
        "concept_activations",
    }.isdisjoint(vars(prefscope.Lens))
    code = (
        "import sys, prefscope, prefscope.analysis, prefscope.reporting\n"
        "assert 'torch' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
