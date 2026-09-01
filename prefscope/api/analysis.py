"""Compatibility facade for the stable task-centered analysis API.

Implementation lives in focused contract, component, and execution modules. Importing
from :mod:`prefscope.api.analysis` remains supported.
"""
from __future__ import annotations

from prefscope.api.analysis_components import (
    FeatureArtifactDiagnostics,
    OutcomeAssociations,
    PairedConceptShift,
    PairedOutcomeShifts,
    PreferenceLengthConfounds,
    PromptConditionedOutcomeShifts,
)
from prefscope.api.analysis_contracts import (
    AnalysisArtifact,
    AnalysisComponent,
    AnalysisDataset,
    OutcomeSpec,
    PairedOutcomeSpec,
)
from prefscope.api.analysis_execution import (
    AnalysisPlan,
    DatasetAnalysisResult,
    analyze_dataset,
)

# Keep the historical public module identity for repr, pickle, and introspection even
# though implementation now lives in focused modules.
for _public_object in (
    OutcomeSpec,
    PairedOutcomeSpec,
    AnalysisDataset,
    AnalysisArtifact,
    AnalysisComponent,
    OutcomeAssociations,
    FeatureArtifactDiagnostics,
    PreferenceLengthConfounds,
    PairedOutcomeShifts,
    PromptConditionedOutcomeShifts,
    PairedConceptShift,
    AnalysisPlan,
    DatasetAnalysisResult,
    analyze_dataset,
):
    _public_object.__module__ = __name__
del _public_object


__all__ = [
    "OutcomeSpec", "PairedOutcomeSpec", "AnalysisDataset", "AnalysisArtifact",
    "AnalysisComponent", "OutcomeAssociations", "FeatureArtifactDiagnostics",
    "PreferenceLengthConfounds", "PairedOutcomeShifts",
    "PromptConditionedOutcomeShifts", "PairedConceptShift", "AnalysisPlan",
    "DatasetAnalysisResult", "analyze_dataset",
]
