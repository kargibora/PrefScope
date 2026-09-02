from prefscope.api.analysis import (
    AnalysisArtifact,
    AnalysisComponent,
    AnalysisDataset,
    AnalysisPlan,
    AnalysisDatasetReference,
    LoadedAnalysisResult,
    DatasetAnalysisResult,
    FeatureArtifactDiagnostics,
    OutcomeAssociations,
    PairedConceptShift,
    PreferenceLengthConfounds,
    PairedOutcomeShifts,
    PairedOutcomeSpec,
    PromptConditionedOutcomeShifts,
    OutcomeSpec,
    analyze_dataset,
    load_analysis_result,
    save_analysis_result,
)
from prefscope.api.config import SAEConfig, TrainConfig
from prefscope.api.encoded import load_feature_batch, save_feature_batch
from prefscope.api.feature_activations import feature_activation_table
from prefscope.api.feature_catalog import FeatureCatalog
from prefscope.api.loaded_lens import Lens, LoadedLens, pairs_to_battles
from prefscope.api.preference import preference_relevance
from prefscope.api.representation import (
    EmbeddingRepresentationSource,
    PrecomputedRepresentationSource,
)
from prefscope.core.dataset import Dataset
from prefscope.core.features import FeatureBatch, FeatureMatrix
from prefscope.core.lens_backend import (
    LensBackend,
    LensCapabilities,
    pair_item_metadata,
)
from prefscope.core.representation import (
    CallableRepresentationSource,
    RepresentationBatch,
    RepresentationSource,
)
from prefscope.core.table_schema import TableContract
from prefscope.core.types import PairItem
from prefscope.integrations.saelens import SAELensProjector, SAELensTextBackend

__all__ = [
    "Lens",
    "LoadedLens",
    "pairs_to_battles",
    "PairItem",
    "Dataset",
    "SAEConfig",
    "TrainConfig",
    "LensBackend",
    "LensCapabilities",
    "pair_item_metadata",
    "SAELensProjector",
    "SAELensTextBackend",
    "RepresentationBatch",
    "RepresentationSource",
    "CallableRepresentationSource",
    "EmbeddingRepresentationSource",
    "PrecomputedRepresentationSource",
    "FeatureMatrix",
    "FeatureBatch",
    "FeatureCatalog",
    "feature_activation_table",
    "TableContract",
    "OutcomeSpec",
    "AnalysisDataset",
    "AnalysisArtifact",
    "AnalysisComponent",
    "FeatureArtifactDiagnostics",
    "OutcomeAssociations",
    "PreferenceLengthConfounds",
    "PairedOutcomeShifts",
    "PairedOutcomeSpec",
    "PromptConditionedOutcomeShifts",
    "PairedConceptShift",
    "AnalysisPlan",
    "DatasetAnalysisResult",
    "AnalysisDatasetReference",
    "LoadedAnalysisResult",
    "analyze_dataset",
    "load_analysis_result",
    "save_analysis_result",
    "preference_relevance",
    "load_feature_batch",
    "save_feature_batch",
]
