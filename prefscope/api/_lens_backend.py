"""Shared validation and legacy-source adapter for :class:`prefscope.Lens`."""

from __future__ import annotations

from collections.abc import Iterable

from prefscope.core.features import FeatureBatch, validate_feature_ids
from prefscope.core.lens_backend import (
    ARRAY_VIEWS,
    VIEW_ARRAYS,
    LensBackend,
    LensCapabilities,
)
from prefscope.core.representation import validate_row_ids
from prefscope.core.types import PairItem


def normalize_items(dataset) -> list[PairItem]:
    items = list(dataset)
    if not items:
        raise ValueError("lens featurization needs at least one PairItem")
    if not all(isinstance(item, PairItem) for item in items):
        raise ValueError("lens featurization accepts only PairItem objects")
    validate_row_ids(tuple(item.id for item in items))
    paired = tuple(item.y_b is not None for item in items)
    if any(paired) and not all(paired):
        raise ValueError(
            "one feature batch cannot mix paired and single-response items"
        )
    return items


def resolve_views(
    requested: Iterable[str] | str | None,
    capabilities: LensCapabilities,
    *,
    paired: bool,
) -> tuple[str, ...]:
    if requested is None:
        views = tuple(
            view
            for view in capabilities.views
            if paired or view not in {"response_b", "response_difference"}
        )
    else:
        raw = (requested,) if isinstance(requested, str) else tuple(requested)
        views = tuple(ARRAY_VIEWS.get(str(view), str(view)) for view in raw)
        if not views or len(set(views)) != len(views):
            raise ValueError("views must contain unique supported view names")
    if not views:
        raise ValueError(
            "this lens has no supported view for single-response data; provide paired "
            "items or choose an individual/prompt lens")
    unknown = set(views) - set(capabilities.views)
    if unknown:
        raise ValueError(
            f"lens does not support views {sorted(unknown)}; available: "
            f"{list(capabilities.views)}"
        )
    paired_only = {"response_b", "response_difference"} & set(views)
    if paired_only and not paired:
        raise ValueError(
            f"single-response data cannot produce views {sorted(paired_only)}"
        )
    return views


def select_feature_batch(
    batch: FeatureBatch,
    *,
    views: tuple[str, ...],
    feature_ids: tuple[int, ...] | None,
    allow_extra: bool = False,
) -> FeatureBatch:
    names = tuple(VIEW_ARRAYS[view] for view in views)
    missing = set(names) - set(batch.arrays)
    if missing:
        raise ValueError(
            f"lens backend did not produce requested arrays {sorted(missing)}"
        )
    extra = set(batch.arrays) - set(names)
    if extra and not allow_extra:
        raise ValueError(
            f"lens backend produced unrequested arrays {sorted(extra)}")
    expected_orientations = {
        "prompt": "none",
        "response_a": "absolute_a",
        "response_b": "absolute_b",
        "response_difference": "a_minus_b",
    }
    for view, name in zip(views, names):
        if batch.roles[name] != view:
            raise ValueError(
                f"lens backend array {name!r} role must be {view!r}, got "
                f"{batch.roles[name]!r}"
            )
        expected = expected_orientations[view]
        if batch.orientations[name] != expected:
            raise ValueError(
                f"lens backend array {name!r} orientation must be {expected!r}, got "
                f"{batch.orientations[name]!r}"
            )
    if feature_ids is None:
        selected_ids = batch.feature_ids
        positions = tuple(range(len(selected_ids)))
    else:
        selected_ids = validate_feature_ids(feature_ids)
        lookup = {
            feature_id: index for index, feature_id in enumerate(batch.feature_ids)
        }
        missing_ids = [value for value in selected_ids if value not in lookup]
        if missing_ids:
            raise ValueError(
                f"feature_ids are outside this lens output: {missing_ids[:10]}"
            )
        positions = tuple(lookup[value] for value in selected_ids)
    arrays = {name: batch.array(name)[:, positions] for name in names}
    return FeatureBatch(
        row_ids=batch.row_ids,
        arrays=arrays,
        roles={name: batch.roles[name] for name in names},
        orientations={name: batch.orientations[name] for name in names},
        feature_ids=selected_ids,
        metadata=batch.metadata,
        activation_polarity=batch.activation_polarity,
        code_semantics=batch.code_semantics,
        provenance=batch.provenance,
    )


class RepresentationLensBackend(LensBackend):
    """Adapt the historical RepresentationSource + projector path."""

    def __init__(self, lens) -> None:
        self.lens = lens
        if lens.input_rep == "prompt":
            self._capabilities = LensCapabilities(("prompt",))
        elif lens.input_rep == "individual":
            self._capabilities = LensCapabilities(
                ("response_a", "response_b", "response_difference"),
                difference="a_minus_b_after_encoding",
            )
        else:
            self._capabilities = LensCapabilities(
                ("response_difference",),
                difference="direct_difference_projection",
            )
        self.input_rep = lens.input_rep

    @property
    def capabilities(self) -> LensCapabilities:
        return self._capabilities

    @property
    def m_total(self) -> int:
        return int(self.lens.projector.m_total)

    @property
    def activation_polarity(self) -> str:
        return self.lens.activation_polarity

    @property
    def code_semantics(self) -> str:
        return self.lens.code_semantics

    def featurize(
        self,
        items,
        *,
        views=None,
        feature_ids=None,
        batch_size=None,
    ) -> FeatureBatch:
        del batch_size  # batching is owned by the configured RepresentationSource
        if self.lens.representation_source is None:
            raise ValueError(
                "this lens has no item source; inject a RepresentationSource or use "
                "Lens.from_backend(...)"
            )
        rows = list(items)
        representations = self.lens.representation_source.encode(rows)
        expected = tuple(str(item.id) for item in rows)
        if representations.row_ids != expected:
            raise ValueError(
                "representation source row_ids must exactly match input item order"
            )
        batch = self.lens.project_representations(representations)
        return select_feature_batch(
            batch,
            views=tuple(views),
            feature_ids=feature_ids,
            allow_extra=True,
        )


__all__ = [
    "RepresentationLensBackend",
    "normalize_items",
    "resolve_views",
    "select_feature_batch",
]
