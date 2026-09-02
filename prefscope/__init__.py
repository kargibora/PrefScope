"""PrefScope: analyze post-training preference data by concept with sparse autoencoders."""

import logging

# Public API. These imports are all torch-free; the heavy
# Embedder/SAEProjector/build_lens imports stay lazy
# inside Lens methods so ``import prefscope`` never pulls in torch.
from prefscope.core.types import PairItem
from prefscope.core.dataset import Dataset
from prefscope.core.features import FeatureBatch, FeatureMatrix
from prefscope.core.lens_backend import (
    LensBackend,
    LensCapabilities,
    pair_item_metadata,
)
from prefscope.core.table_schema import TableContract
from prefscope.core.representation import (
    CallableRepresentationSource,
    RepresentationBatch,
    RepresentationSource,
)
from prefscope.data.datasets import HuggingFaceDataset, TableDataset
from prefscope.data.demo import create_demo, make_demo_corpus
from prefscope.data.tabular import ColumnMapping
from prefscope.pipeline.prepare_dataset import prepare_dataset
from prefscope.api.config import SAEConfig, TrainConfig
from prefscope.api.encoded import load_feature_batch, save_feature_batch
from prefscope.api.feature_activations import feature_activation_table
from prefscope.api.feature_catalog import FeatureCatalog
from prefscope.api.loaded_lens import Lens, LoadedLens
from prefscope.api.preference import preference_relevance
from prefscope.api.representation import (
    EmbeddingRepresentationSource,
    PrecomputedRepresentationSource,
)
from prefscope.api.analysis import (
    AnalysisArtifact,
    AnalysisComponent,
    AnalysisDataset,
    AnalysisDatasetReference,
    AnalysisPlan,
    DatasetAnalysisResult,
    LoadedAnalysisResult,
    FeatureArtifactDiagnostics,
    OutcomeAssociations,
    OutcomeSpec,
    PairedConceptShift,
    PreferenceLengthConfounds,
    PairedOutcomeShifts,
    PairedOutcomeSpec,
    PromptConditionedOutcomeShifts,
    analyze_dataset,
    load_analysis_result,
    save_analysis_result,
)
from prefscope.analysis import (
    diagnose,
    evaluate_preference,
    feature_preference_relevance,
)
from prefscope.analysis import (
    NormalizedOutcomes,
    OutcomeAssociationResult,
    associate_outcomes,
    associate_outcomes_by_group,
    concept_presence,
    normalize_outcomes,
    paired_concept_shift,
    paired_concept_shift_by_region,
    summarize_response_scope,
)
from prefscope.pipeline.compare import ResponseComparison, compare_encoded_responses
from prefscope.pipeline.confounds import screen_length_confound
from prefscope.pipeline.text_concepts import extract_text_concepts
from prefscope.core import registry
from prefscope.core.plugins import load_plugins

__version__ = "0.2.0"

# Library convention: emit logs under the ``prefscope`` namespace but stay silent
# unless the application attaches a handler / configures logging.
logging.getLogger(__name__).addHandler(logging.NullHandler())


def __getattr__(name):
    """Keep optional/heavy workflow internals out of a plain ``import prefscope``."""
    if name in {"AnalyzeConfig", "run_analysis"}:
        from prefscope.pipeline.analyze import AnalyzeConfig, run_analysis

        return {"AnalyzeConfig": AnalyzeConfig, "run_analysis": run_analysis}[name]
    if name in {"SAELensProjector", "SAELensTextBackend"}:
        from prefscope.integrations.saelens import SAELensProjector, SAELensTextBackend

        return {
            "SAELensProjector": SAELensProjector,
            "SAELensTextBackend": SAELensTextBackend,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def load_lens(
    path,
    *,
    device: str = "cpu",
    revision: str | None = None,
    cache_dir=None,
    token=None,
    local_files_only: bool = False,
    subfolder: str | None = None,
    annotations=None,
    embedding_cache=None,
    embed_backend: str = "hf",
    embed_batch_size: int | None = None,
):
    """Load a local lens or an ``hf://owner/repo[/subfolder]`` artifact.

    For an explicit Hub repo id without the ``hf://`` prefix, use
    :meth:`Lens.from_pretrained`.
    """
    if str(path).startswith("hf://"):
        from prefscope.api.hub import split_hf_source

        repo_id, source_subfolder = split_hf_source(str(path))
        if subfolder is not None and source_subfolder is not None:
            raise ValueError(
                "specify the Hub subfolder in either path or subfolder=, not both"
            )
        return Lens.from_pretrained(
            repo_id,
            device=device,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            subfolder=subfolder or source_subfolder,
            annotations=annotations,
            embedding_cache=embedding_cache,
            embed_backend=embed_backend,
            embed_batch_size=embed_batch_size,
        )
    if (
        any(x is not None for x in (revision, cache_dir, token, subfolder))
        or local_files_only
    ):
        raise ValueError(
            "revision/cache_dir/token/local_files_only/subfolder are Hub-only options; "
            "use hf://owner/repo or Lens.from_pretrained()"
        )
    kwargs = {"device": device}
    if embedding_cache is not None:
        kwargs["embedding_cache"] = embedding_cache
    if embed_backend != "hf":
        kwargs["embed_backend"] = embed_backend
    if embed_batch_size is not None:
        kwargs["embed_batch_size"] = embed_batch_size
    if annotations is not None:
        kwargs["annotations"] = annotations
    return Lens.load(path, **kwargs)


__all__ = [
    "Lens",
    "LoadedLens",
    "load_lens",
    "PairItem",
    "Dataset",
    "RepresentationBatch",
    "RepresentationSource",
    "CallableRepresentationSource",
    "EmbeddingRepresentationSource",
    "PrecomputedRepresentationSource",
    "LensBackend",
    "LensCapabilities",
    "pair_item_metadata",
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
    "TableDataset",
    "HuggingFaceDataset",
    "ColumnMapping",
    "prepare_dataset",
    "AnalyzeConfig",
    "run_analysis",
    "create_demo",
    "make_demo_corpus",
    "extract_text_concepts",
    "SAEConfig",
    "TrainConfig",
    "SAELensProjector",
    "SAELensTextBackend",
    "diagnose",
    "evaluate_preference",
    "feature_preference_relevance",
    "NormalizedOutcomes",
    "OutcomeAssociationResult",
    "normalize_outcomes",
    "associate_outcomes",
    "associate_outcomes_by_group",
    "concept_presence",
    "paired_concept_shift",
    "paired_concept_shift_by_region",
    "summarize_response_scope",
    "ResponseComparison",
    "compare_encoded_responses",
    "screen_length_confound",
    "registry",
    "load_plugins",
    "__version__",
]
