from prefscope.api.config import SAEConfig, TrainConfig
from prefscope.api.encoded import load_feature_batch, save_feature_batch
from prefscope.api.feature_activations import feature_activation_table
from prefscope.api.feature_catalog import FeatureCatalog
from prefscope.api.feature_catalog_io import (
    decode_feature_catalog,
    encode_feature_catalog,
    load_feature_catalog,
    save_feature_catalog,
)
from prefscope.api.loaded_lens import Lens, pairs_to_battles
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
from prefscope.core.types import PairItem
from prefscope.integrations.saelens import SAELensProjector, SAELensTextBackend

__all__ = [
    "Lens",
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
    "decode_feature_catalog",
    "encode_feature_catalog",
    "load_feature_catalog",
    "save_feature_catalog",
    "feature_activation_table",
    "load_feature_batch",
    "save_feature_batch",
]
