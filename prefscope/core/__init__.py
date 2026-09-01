from prefscope.core.dataset import Dataset
from prefscope.core.features import FeatureBatch, FeatureMatrix, validate_feature_ids
from prefscope.core.lens_backend import (
    LensBackend, LensCapabilities, pair_item_metadata,
)
from prefscope.core.plugins import load_plugins
from prefscope.core.table_schema import TableContract
from prefscope.core.representation import (
    CallableRepresentationSource,
    RepresentationBatch,
    RepresentationSource,
    validate_portable_mapping,
    validate_row_ids,
)
from prefscope.core.types import PairItem, SideVectors

__all__ = [
    "Dataset", "PairItem", "SideVectors", "RepresentationBatch",
    "RepresentationSource", "CallableRepresentationSource",
    "FeatureMatrix", "FeatureBatch", "LensBackend", "LensCapabilities",
    "pair_item_metadata",
    "TableContract", "load_plugins",
    "validate_feature_ids",
    "validate_row_ids", "validate_portable_mapping",
]
