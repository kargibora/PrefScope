"""Backend-neutral contracts for turning dataset items into sparse feature codes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass

from prefscope.core.features import FeatureBatch
from prefscope.core.types import PairItem

CANONICAL_FEATURE_VIEWS = (
    "prompt",
    "response_a",
    "response_b",
    "response_difference",
)
VIEW_ARRAYS = {
    "prompt": "z_prompt",
    "response_a": "z_a",
    "response_b": "z_b",
    "response_difference": "z_diff",
}
ARRAY_VIEWS = {value: key for key, value in VIEW_ARRAYS.items()}


@dataclass(frozen=True)
class LensCapabilities:
    """Machine-readable behavior supported by one lens backend.

    ``views`` names item roles that can be encoded in one shared feature space.
    ``difference`` states how a paired contrast is formed; it is never inferred from
    an array name alone.
    """

    views: tuple[str, ...]
    shared_feature_space: bool = True
    difference: str = "none"
    input_kind: str = "pair_items"

    def __post_init__(self) -> None:
        views = tuple(self.views)
        if not views or len(set(views)) != len(views):
            raise ValueError("lens capability views must be unique and non-empty")
        unknown = set(views) - set(CANONICAL_FEATURE_VIEWS)
        if unknown:
            raise ValueError(f"unknown lens capability views: {sorted(unknown)}")
        if self.difference not in {
            "none",
            "a_minus_b_after_encoding",
            "direct_difference_projection",
        }:
            raise ValueError("unknown lens difference behavior")
        if not isinstance(self.shared_feature_space, bool):
            raise ValueError("shared_feature_space must be boolean")
        if not self.shared_feature_space:
            raise ValueError(
                "shared_feature_space=False is not supported by FeatureBatch")
        if self.difference != "none" and "response_difference" not in views:
            raise ValueError(
                "a declared difference behavior requires response_difference support"
            )
        if "response_difference" in views and self.difference == "none":
            raise ValueError(
                "response_difference support requires an explicit difference behavior"
            )
        if self.input_kind != "pair_items":
            raise ValueError("input_kind must be 'pair_items'")
        object.__setattr__(self, "views", views)

    def supports(self, *views: str) -> bool:
        return set(views).issubset(self.views)


def pair_item_metadata(items: Iterable[PairItem]) -> dict[str, tuple[object, ...]]:
    """Return canonical aligned metadata for normalized item rows."""
    rows = list(items)
    reserved = {
        "prompt", "response_a", "response_b", "pref", "model_a", "model_b",
        "response_length_a", "response_length_b", "response_length_difference",
    }
    custom = set()
    for item in rows:
        if not isinstance(item, PairItem):
            raise ValueError("item metadata needs PairItem rows")
        if not isinstance(item.meta, dict):
            raise ValueError("PairItem.meta must be a mapping")
        collisions = reserved & set(item.meta)
        if collisions:
            raise ValueError(
                f"PairItem.meta collides with canonical fields: {sorted(collisions)}")
        custom.update(item.meta)
    lengths_a = tuple(len(str(item.y_a).split()) for item in rows)
    lengths_b = tuple(
        None if item.y_b is None else len(str(item.y_b).split()) for item in rows)
    metadata = {
        "prompt": tuple(str(item.x) for item in rows),
        "response_a": tuple(str(item.y_a) for item in rows),
        "response_b": tuple(item.y_b for item in rows),
        "pref": tuple(item.pref for item in rows),
        "model_a": tuple(item.model_a for item in rows),
        "model_b": tuple(item.model_b for item in rows),
        "response_length_a": lengths_a,
        "response_length_b": lengths_b,
        "response_length_difference": tuple(
            None if b is None else a - b for a, b in zip(lengths_a, lengths_b)
        ),
    }
    metadata.update({
        name: tuple(item.meta.get(name) for item in rows)
        for name in sorted(custom)
    })
    return metadata


class LensBackend(ABC):
    """Extension contract for direct ``PairItem -> FeatureBatch`` backends.

    Implementations may use text embeddings, internal model activations, hosted APIs,
    or precomputed feature services. Heavy optional dependencies must stay behind the
    implementation's ``featurize`` call.
    """

    input_rep: str = "individual"

    @property
    @abstractmethod
    def capabilities(self) -> LensCapabilities:
        """Declare supported item views and paired-difference semantics."""

    @property
    @abstractmethod
    def m_total(self) -> int:
        """Return the total number of feature coordinates."""

    @property
    def activation_polarity(self) -> str:
        return "unknown"

    @property
    def code_semantics(self) -> str:
        return "custom"

    @abstractmethod
    def featurize(
        self,
        items: Iterable[PairItem],
        *,
        views: tuple[str, ...] | None = None,
        feature_ids: tuple[int, ...] | None = None,
        batch_size: int | None = None,
    ) -> FeatureBatch:
        """Encode aligned items into the requested role-aware feature views."""


__all__ = [
    "ARRAY_VIEWS",
    "CANONICAL_FEATURE_VIEWS",
    "LensBackend",
    "LensCapabilities",
    "VIEW_ARRAYS",
    "pair_item_metadata",
]
