from __future__ import annotations

import importlib
import pickle
import subprocess
import sys

import prefscope
import prefscope.api.analysis as analysis
import prefscope.api.analysis_components as components
import prefscope.api.analysis_contracts as contracts
import prefscope.api.analysis_execution as execution
from prefscope.api.loaded_lens import Lens, LoadedLens, pairs_to_battles
from prefscope.pipeline.analyze import AnalyzeConfig, LensSource
from prefscope.core import registry


def test_canonical_api_all_names_are_available():
    import prefscope.api as api

    assert all(hasattr(api, name) for name in api.__all__)
    assert api.TableContract is prefscope.TableContract
    assert api.RepresentationSource is prefscope.RepresentationSource


def test_analysis_facade_reexports_defining_module_objects():
    assert analysis.OutcomeSpec is contracts.OutcomeSpec
    assert analysis.AnalysisArtifact is contracts.AnalysisArtifact
    assert analysis.OutcomeAssociations is components.OutcomeAssociations
    assert analysis.AnalysisPlan is execution.AnalysisPlan
    assert analysis.analyze_dataset is execution.analyze_dataset
    assert prefscope.OutcomeSpec is contracts.OutcomeSpec
    assert LoadedLens is Lens


def test_public_objects_keep_historical_module_identity_and_pickle_path():
    assert analysis.OutcomeSpec.__module__ == "prefscope.api.analysis"
    assert analysis.OutcomeAssociations.__module__ == "prefscope.api.analysis"
    assert analysis.AnalysisPlan.__module__ == "prefscope.api.analysis"
    assert AnalyzeConfig.__module__ == "prefscope.pipeline.analyze"
    assert LensSource.__module__ == "prefscope.pipeline.analyze"
    assert pairs_to_battles.__module__ == "prefscope.api.loaded_lens"

    spec = analysis.OutcomeSpec([0.0], row_ids=("row",), kind="continuous")
    restored = pickle.loads(pickle.dumps(spec))
    assert type(restored) is analysis.OutcomeSpec


def test_analysis_component_registration_is_stable_across_facade_reload():
    expected = {
        "feature-artifact-diagnostics",
        "outcome-associations",
        "paired-concept-shift",
        "paired-outcome-shifts",
        "preference-length-confounds",
        "prompt-conditioned-outcome-shifts",
    }
    assert set(registry.available("analysis_component")) == expected
    importlib.reload(analysis)
    assert set(registry.available("analysis_component")) == expected


def test_public_import_orders_remain_torch_free():
    orders = (
        ("prefscope", "prefscope.api.analysis", "prefscope.api.loaded_lens",
         "prefscope.pipeline.analyze"),
        ("prefscope.pipeline.analyze", "prefscope.api.loaded_lens",
         "prefscope.api.analysis", "prefscope"),
        ("prefscope.pipeline.run", "prefscope", "prefscope.api.analysis"),
    )
    for modules in orders:
        code = "import importlib, sys\n" + "\n".join(
            f"importlib.import_module({module!r})" for module in modules
        ) + "\nassert 'torch' not in sys.modules\n"
        subprocess.run([sys.executable, "-c", code], check=True)


def test_analyze_config_does_not_initialize_viewer_export_package():
    code = (
        "import sys\n"
        "import prefscope.pipeline.analyze_config\n"
        "assert 'prefscope.viewer_export.cli' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
