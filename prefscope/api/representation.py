"""Built-in representation sources for the public API."""
from __future__ import annotations

from typing import Iterable

from prefscope.core import registry
from prefscope.core.representation import RepresentationBatch, RepresentationSource
from prefscope.core.types import PairItem


@registry.register("representation_source", "precomputed")
class PrecomputedRepresentationSource(RepresentationSource):
    """Serve one immutable aligned batch of fixed-width precomputed vectors.

    Arrays may be static embeddings or already pooled residual activations. Their
    provenance must state which coordinate system produced them; this class does not
    reinterpret one representation family as another.
    """

    def __init__(
        self, batch: RepresentationBatch, *, source_name: str = "precomputed"
    ) -> None:
        if not isinstance(batch, RepresentationBatch):
            raise ValueError("batch must be a RepresentationBatch")
        if not isinstance(source_name, str) or not source_name:
            raise ValueError("source_name must be a non-empty string")
        self.batch = batch
        self.source_name = source_name

    def encode(self, items: Iterable[PairItem]) -> RepresentationBatch:
        rows = list(items)
        expected_ids = tuple(str(item.id) for item in rows)
        if expected_ids != self.batch.row_ids:
            raise ValueError(
                "precomputed batch row_ids must exactly match input item order")
        return RepresentationBatch(
            row_ids=self.batch.row_ids,
            arrays=self.batch.arrays,
            metadata=self.batch.metadata,
            granularity=self.batch.granularity,
            provenance={
                **dict(self.batch.provenance),
                "source_type": "precomputed",
                "source_name": self.source_name,
            },
        )


@registry.register("representation_source", "text-embedding")
class EmbeddingRepresentationSource(RepresentationSource):
    """Create prompt/response matrices with any duck-typed text embedder.

    The embedder needs ``encode_prompts(prompts)`` and
    ``encode(prompts, completions)``. The built-in Qwen ``Embedder`` satisfies
    this contract, but custom local or hosted embedding clients can be injected
    without subclassing PrefScope.
    """

    def __init__(
        self,
        embedder,
        *,
        include_prompt: bool = True,
        include_responses: bool = True,
    ) -> None:
        if not include_prompt and not include_responses:
            raise ValueError("at least one representation array must be requested")
        if include_prompt and not callable(getattr(embedder, "encode_prompts", None)):
            raise ValueError("embedder must provide encode_prompts(prompts)")
        if include_responses and not callable(getattr(embedder, "encode", None)):
            raise ValueError("embedder must provide encode(prompts, completions)")
        self.embedder = embedder
        self.include_prompt = bool(include_prompt)
        self.include_responses = bool(include_responses)

    def encode(self, items: Iterable[PairItem]) -> RepresentationBatch:
        rows = list(items)
        if not rows:
            raise ValueError("representation source needs at least one item")
        row_ids = tuple(str(item.id) for item in rows)
        if any(not row_id for row_id in row_ids) or len(set(row_ids)) != len(row_ids):
            raise ValueError("item ids must be unique non-empty strings")
        prompts = [str(item.x) for item in rows]
        reserved = {"prompt", "pref", "model_a", "model_b"}
        custom_names = set()
        for item in rows:
            if not isinstance(item.meta, dict):
                raise ValueError("PairItem.meta must be a mapping")
            collisions = reserved & set(item.meta)
            if collisions:
                raise ValueError(
                    f"PairItem.meta collides with canonical fields: {sorted(collisions)}")
            custom_names.update(item.meta)
        metadata = {
            "prompt": tuple(prompts),
            "pref": tuple(item.pref for item in rows),
            "model_a": tuple(item.model_a for item in rows),
            "model_b": tuple(item.model_b for item in rows),
            **{
                name: tuple(item.meta.get(name) for item in rows)
                for name in sorted(custom_names)
            },
        }
        arrays = {}
        if self.include_prompt:
            arrays["prompt"] = self.embedder.encode_prompts(prompts)
        if self.include_responses:
            arrays["response_a"] = self.embedder.encode(
                prompts, [str(item.y_a) for item in rows])
            paired = [item.y_b is not None for item in rows]
            if any(paired) and not all(paired):
                raise ValueError(
                    "a representation batch cannot mix paired and single-response items")
            if all(paired):
                arrays["response_b"] = self.embedder.encode(
                    prompts, [str(item.y_b) for item in rows])
        provenance = {
            "source_type": "text_embedding",
            "representation_family": "text_embedding",
        }
        contracts = {}
        provenance_fn = getattr(self.embedder, "provenance", None)
        if callable(provenance_fn):
            if self.include_prompt:
                descriptor = {
                    "representation_family": "text_embedding",
                    **dict(provenance_fn(prompt=True)),
                }
                provenance["prompt"] = descriptor
                contracts["prompt"] = descriptor
            if self.include_responses:
                descriptor = {
                    "representation_family": "text_embedding",
                    **dict(provenance_fn(prompt=False)),
                }
                provenance["response"] = descriptor
                contracts["response"] = descriptor
        else:
            model_id = getattr(self.embedder, "model_id", None)
            if model_id is not None:
                descriptor = {
                    "representation_family": "text_embedding",
                    "embed_model_id": str(model_id),
                }
                key = "prompt" if self.include_prompt and not self.include_responses else "response"
                contracts[key] = descriptor
                provenance[key] = descriptor
        if contracts:
            provenance["representation_contracts"] = contracts
        return RepresentationBatch(
            row_ids=row_ids, arrays=arrays, metadata=metadata,
            provenance=provenance)


__all__ = [
    "EmbeddingRepresentationSource", "PrecomputedRepresentationSource",
]
