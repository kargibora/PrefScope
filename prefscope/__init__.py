"""PrefScope: analyze post-training preference data by concept with sparse autoencoders."""
import logging

__version__ = "0.1.0"

# Library convention: emit logs under the ``prefscope`` namespace but stay silent
# unless the application attaches a handler / configures logging.
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Public API. Import order matters: pull in the analysis submodule before api so
# the absolute ``import prefscope.analysis`` inside api.loaded_lens resolves
# against a fully-initialized submodule (avoids a partial-init cycle). These are
# all torch-free; the heavy Embedder/SAEProjector/build_lens imports stay lazy
# inside Lens methods so ``import prefscope`` never pulls in torch.
from prefscope.core.types import PairItem
from prefscope.core.dataset import Dataset
from prefscope.api.config import SAEConfig, TrainConfig
from prefscope.api.loaded_lens import Lens, LoadedLens
from prefscope.analysis import diagnose, evaluate_preference, feature_preference_relevance
from prefscope.core import registry


def load_lens(path, *, device: str = "cpu"):
    """Load a trained lens directory into a :class:`Lens` (alias of ``Lens.load``)."""
    return Lens.load(path, device=device)


__all__ = [
    "Lens", "LoadedLens", "load_lens",
    "PairItem", "Dataset",
    "SAEConfig", "TrainConfig",
    "diagnose", "evaluate_preference", "feature_preference_relevance",
    "registry",
    "__version__",
]
