from __future__ import annotations

import runpy

import numpy as np


def test_custom_analysis_api_example_runs_end_to_end():
    namespace = runpy.run_path("examples/advanced/custom_analysis_api.py")
    result = namespace["run_example"]()
    assert set(result.artifacts) == {
        "feature_artifact_diagnostics",
        "outcome_associations",
        "paired_outcome_shifts",
        "prompt_conditioned_outcome_shifts",
        "feature_magnitude",
    }
    heterogeneity = result.artifact("prompt_conditioned_outcome_shifts").table.iloc[0]
    assert np.isclose(heterogeneity["heterogeneity_present_minus_absent"], 0.5)
