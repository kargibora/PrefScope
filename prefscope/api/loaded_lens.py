"""The Lens: an SAE encoder + interpreted concept names + manifest, as one object.

A ``Lens`` turns a trained lens directory into a reusable inference artifact, or
trains a fresh one from preference data. Lifecycle: ``train -> save -> load ->
encode -> analyze``.

``load`` (alias ``from_dir``) builds the real (torch) projector + embedder; the
constructor takes them as objects so the orchestration is testable with fakes.
``encode_items(dataset)`` accepts homogeneous paired or single-response data;
``encode_pairs(dataset)`` (alias ``project``) embeds each PairItem's two
responses, forms the self-minus-other contrast the lens was trained on, and
projects it through the SAE to signed codes. ``encode`` projects single
(prompt, completion) responses (individual / prompt lenses only). The analysis
methods delegate to ``prefscope.analysis`` (the format-agnostic cores).

Convention: ``y_a`` is "self" (the model under study), ``y_b`` is "other"; codes
are self-minus-other and ``meta['pref']`` = P(self preferred), matching the
analysis contract.

``LoadedLens`` remains a back-compat alias for ``Lens``.
"""

from __future__ import annotations

import functools
import json
import os  # noqa: F401  - compatibility patch point for publication tests/callers
import re
import warnings
from contextvars import ContextVar
from pathlib import Path

import numpy as np
import pandas as pd

from prefscope.api._lens_annotations import _load_feature_table
from prefscope.api._lens_data import pairs_to_battles
from prefscope.api._lens_inspection import (
    concept_activations as inspect_concept_activations,
    concept_names as inspect_concept_names,
    diagnose as inspect_diagnose,
    evaluate_preference as inspect_evaluate_preference,
    feature_preference_relevance as inspect_feature_preference_relevance,
    feature_table as inspect_feature_table,
    fidelity_feature_ids as inspect_fidelity_feature_ids,
    presence as inspect_presence,
    top_concepts as inspect_top_concepts,
)
from prefscope.api._lens_projection import (
    encode,
    encode_items,
    encode_one,
    encode_pairs,
    expected_representation_contract,
    project_representations,
    representation_contract_fingerprint,
    validate_representation_contract,
)
from prefscope.api._lens_publication import (
    _publication_lock as _publication_lock,
    _recover_orphan_backup as _recover_orphan_backup,
    save_lens,
)
from prefscope.artifacts import MANIFEST, SAE_MODEL
from prefscope.observability.runtime import automatic_stage


_SAFE_INPUT_REPS = {"difference", "individual", "prompt"}
_SAFE_FEATURE_VIEWS = {"z_prompt", "z_a", "z_b", "z_diff"}
_FEATURE_OPERATION_ACTIVE: ContextVar[bool] = ContextVar(
    "prefscope_lens_feature_operation_active", default=False
)
_LOAD_LENS_ACTIVE: ContextVar[bool] = ContextVar(
    "prefscope_lens_load_operation_active", default=False
)
_FETCH_LENS_ACTIVE: ContextVar[bool] = ContextVar(
    "prefscope_lens_fetch_operation_active", default=False
)


def _operation_data(*, source_kind=None, input_rep=None, **booleans):
    """Return only the small, caller-approved metadata allowlist."""
    data = {}
    if source_kind is not None:
        data["source_kind"] = source_kind
    if input_rep in _SAFE_INPUT_REPS:
        data["input_rep"] = input_rep
    data.update({name: bool(value) for name, value in booleans.items()})
    return data


def _update_lens_span(span, lens) -> None:
    try:
        if not span.active:
            return
        data = {}
        input_rep = getattr(lens, "input_rep", None)
        if input_rep in _SAFE_INPUT_REPS:
            data["input_rep"] = input_rep
        width = getattr(getattr(lens, "backend", None), "m_total", None)
        if isinstance(width, int) and not isinstance(width, bool) and width > 0:
            data["n_features"] = int(width)
        span.update(**data)
    except BaseException:
        # Custom result properties must never turn success into failure.
        return


def _update_array_span(span, result) -> None:
    try:
        if not span.active:
            return
        value = result[0] if isinstance(result, tuple) else result
        shape = getattr(value, "shape", None)
        if shape is None:
            return
        dimensions = [int(size) for size in shape]
        data = {"shape": dimensions}
        if dimensions:
            data["n_rows"] = dimensions[0]
        if len(dimensions) > 1:
            data["n_features"] = dimensions[1]
        span.update(**data)
    except BaseException:
        # Structural telemetry is best effort and cannot alter return behavior.
        return


def _update_feature_batch_span(span, features) -> None:
    try:
        if not span.active:
            return
        arrays = getattr(features, "arrays", {})
        views = list(arrays)
        shapes = [[int(size) for size in arrays[name].shape] for name in views]
        data = {
            "n_rows": len(features.row_ids),
            "n_features": len(features.feature_ids),
            "n_views": len(views),
        }
        if all(
            name in _SAFE_FEATURE_VIEWS
            and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", name)
            for name in views
        ):
            data.update(views=views, shapes=dict(zip(views, shapes)))
        else:
            # Custom array names may contain private data. Preserve only dimensions.
            data["shapes"] = shapes
        span.update(**data)
    except BaseException:
        # Structural telemetry is best effort and cannot alter return behavior.
        return


def _observe_lens_result(stage, *, source_kind):
    """Instrument a constructor without inspecting its potentially private inputs."""

    def decorate(function):
        @functools.wraps(function)
        def observed(*args, **kwargs):
            if (
                stage == "load_lens"
                and _LOAD_LENS_ACTIVE.get()
                and not _FETCH_LENS_ACTIVE.get()
            ):
                return function(*args, **kwargs)
            context = (
                _LOAD_LENS_ACTIVE
                if stage == "load_lens"
                else _FETCH_LENS_ACTIVE
                if stage == "fetch_lens"
                else None
            )
            token = context.set(True) if context is not None else None
            try:
                with automatic_stage(stage, {"source_kind": source_kind}) as span:
                    result = function(*args, **kwargs)
                    _update_lens_span(span, result)
                    return result
            finally:
                if context is not None and token is not None:
                    context.reset(token)

        return observed

    return decorate


def _observe_feature_operation(stage, update_result):
    """Coalesce delegates while keeping each direct public operation observable."""

    def decorate(function):
        @functools.wraps(function)
        def observed(self, *args, **kwargs):
            if _FEATURE_OPERATION_ACTIVE.get():
                return function(self, *args, **kwargs)
            token = _FEATURE_OPERATION_ACTIVE.set(True)
            try:
                data = _operation_data(input_rep=getattr(self, "input_rep", None))
                with automatic_stage(stage, data) as span:
                    result = function(self, *args, **kwargs)
                    update_result(span, result)
                    return result
            finally:
                _FEATURE_OPERATION_ACTIVE.reset(token)

        return observed

    return decorate


def _observe_array_result(stage):
    """Instrument legacy ndarray operations, coalescing their internal delegates."""
    return _observe_feature_operation(stage, _update_array_span)


def _observe_feature_batch_result(stage):
    """Instrument FeatureBatch operations, coalescing their internal delegates."""
    return _observe_feature_operation(stage, _update_feature_batch_span)


class Lens:
    def __init__(
        self,
        projector,
        embedder=None,
        *,
        names=None,
        manifest=None,
        representation_source=None,
        backend=None,
    ) -> None:
        """Create a lens from injected numerical components.

        ``embedder`` preserves the historical text API. ``representation_source``
        is the general fixed-width source used by item encoding; custom in-memory
        residual sources can be injected without changing lens projection logic.
        """
        from prefscope.core.lens_backend import LensBackend
        from prefscope.core.representation import RepresentationSource

        if backend is not None and not isinstance(backend, LensBackend):
            raise ValueError("backend must implement LensBackend")
        if backend is not None:
            from numbers import Integral

            backend_width = backend.m_total
            if (
                isinstance(backend_width, bool)
                or not isinstance(backend_width, Integral)
                or int(backend_width) <= 0
            ):
                raise ValueError("backend m_total must be a positive integer")
            backend_width = int(backend_width)
        if representation_source is not None and backend is not None:
            raise ValueError("pass representation_source or backend, not both")
        if representation_source is not None and not isinstance(
            representation_source, RepresentationSource
        ):
            raise ValueError(
                "representation_source must implement RepresentationSource"
            )
        self.projector = projector
        self.embedder = embedder
        self.names = names
        self.manifest = dict(manifest or {})
        if self.manifest:
            # Parse through the versioned manifest so a real artifact's representation is
            # migrated/inferred — NEVER silently defaulted to "difference" (the old
            # `.get("input_rep", "difference")` corrupted every code when a lens omitted
            # it). from_dict raises rather than guess an undeterminable representation.
            from prefscope.core.manifest import LensManifest

            self.manifest_obj = LensManifest.from_dict(self.manifest)
            self.input_rep = self.manifest_obj.input_rep
            self.activation_polarity = self.manifest_obj.activation_polarity
            self.code_semantics = self.manifest_obj.code_semantics
            if backend is not None:
                comparisons = {
                    "input_rep": (self.input_rep, getattr(backend, "input_rep", None)),
                    "activation_polarity": (
                        self.activation_polarity,
                        getattr(backend, "activation_polarity", None),
                    ),
                    "code_semantics": (
                        self.code_semantics,
                        getattr(backend, "code_semantics", None),
                    ),
                }
                if self.manifest_obj.m_total is not None:
                    comparisons["m_total"] = (self.manifest_obj.m_total, backend_width)
                for field, (declared, actual) in comparisons.items():
                    if declared != actual:
                        raise ValueError(
                            f"backend manifest {field}={declared!r} conflicts with "
                            f"backend {field}={actual!r}"
                        )
        else:
            # in-memory Lens with no backing artifact — nothing to be wrong about
            self.manifest_obj = None
            semantic_source = backend if backend is not None else projector
            self.input_rep = getattr(semantic_source, "input_rep", "difference")
            if self.input_rep not in {"difference", "individual", "prompt"}:
                raise ValueError(
                    "in-memory projector input_rep must be difference, individual, "
                    "or prompt"
                )
            self.activation_polarity = getattr(
                semantic_source, "activation_polarity", "unknown"
            )
            self.code_semantics = getattr(semantic_source, "code_semantics", "custom")
        if representation_source is None and embedder is not None:
            from prefscope.api.representation import EmbeddingRepresentationSource

            representation_source = EmbeddingRepresentationSource(
                embedder,
                include_prompt=self.input_rep == "prompt",
                include_responses=self.input_rep != "prompt",
            )
        self.representation_source = representation_source
        self.source = representation_source
        if backend is None:
            from prefscope.api._lens_backend import RepresentationLensBackend

            backend = RepresentationLensBackend(self)
        self.backend = backend
        self.granularity = self.manifest.get("granularity", "response")
        self.lens_dir = None  # set by from_dir/load; None when constructed directly

    @classmethod
    @_observe_lens_result("load_lens", source_kind="config")
    def from_config(cls, config, *, device: str | None = None) -> "Lens":
        """Load a native, SAELens, or registered custom backend from YAML."""
        from prefscope.api.lens_config import load_lens_config

        return load_lens_config(config, device=device)

    @classmethod
    def from_backend(cls, backend, *, names=None, manifest=None) -> "Lens":
        """Create a lens from an extensible ``PairItem -> FeatureBatch`` backend."""
        from prefscope.core.lens_backend import LensBackend

        if not isinstance(backend, LensBackend):
            raise ValueError("backend must implement LensBackend")
        return cls(backend, names=names, manifest=manifest, backend=backend)

    @classmethod
    @_observe_lens_result("load_lens", source_kind="directory")
    def from_dir(
        cls,
        lens_dir,
        *,
        device: str = "cpu",
        annotations=None,
        embedding_cache=None,
        embed_backend: str = "hf",
        embed_batch_size: int | None = None,
        validate_arrays: bool = True,
    ) -> "Lens":
        """Load a trained lens directory.

        ``annotations`` may be an interpretation directory, one CSV, or an iterable
        of either.  Canonical names/fidelity/calibration/context/cluster tables are
        merged by ``feature_id`` and become available through ``feature_table``.
        """
        from prefscope.config import CONFIG
        from prefscope.encode.cache import NpyCache
        from prefscope.encode.embed import Embedder

        try:
            from prefscope.encode.sae import SAEProjector
        except ModuleNotFoundError as exc:
            if exc.name == "torch":
                raise ImportError(
                    "Lens inference needs PyTorch; install 'prefscope[torch]' (the "
                    "'cpu' alias is retained), or install a hardware-specific PyTorch "
                    "build before PrefScope"
                ) from exc
            raise
        from prefscope.core.manifest import LensManifest

        lens_dir = Path(lens_dir)
        manifest_path = lens_dir / MANIFEST
        model_path = lens_dir / SAE_MODEL
        missing = [str(p.name) for p in (manifest_path, model_path) if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                f"{lens_dir} is not a lens directory; missing {missing}"
            )
        manifest = json.loads(manifest_path.read_text())
        projector = SAEProjector(lens_dir, device=device)
        typed = LensManifest.from_dict(
            manifest, strict=int(manifest.get("schema_version") or 0) >= 2
        )
        typed.validate_projector(projector)
        if validate_arrays:
            typed.validate_arrays(lens_dir)
        input_rep = typed.input_rep
        names = _load_feature_table(
            lens_dir, input_rep, projector.m_total, annotations=annotations
        )
        mid = typed.embed_model_id
        if not mid:
            raise ValueError(
                "lens manifest does not record embed_model_id; it cannot be used for "
                "new-text inference safely"
            )
        unknown_preprocessing = [
            name
            for name in ("max_tokens", "embed_instruction", "pooling", "normalization")
            if getattr(typed, name) is None
        ]
        if unknown_preprocessing:
            warnings.warn(
                "lens has legacy/unknown embedding preprocessing fields "
                f"{unknown_preprocessing}; falling back to this PrefScope version's "
                "defaults, so new-text codes are not guaranteed reproducible. Rebuild "
                "the lens to publish exact provenance.",
                RuntimeWarning,
                stacklevel=3,  # account for the observability decorator
            )
        cache = NpyCache(embedding_cache or CONFIG.cache_dir)
        embedder = Embedder(
            cache,
            model_id=mid,
            model_revision=typed.embed_model_revision,
            device=device,
            max_tokens=typed.max_tokens or CONFIG.max_tokens,
            batch_size=embed_batch_size or CONFIG.embed_batch_size,
            backend=embed_backend,
            embed_instruction=(typed.embed_instruction or CONFIG.embed_instruction),
            prompt_embed_instruction=(
                typed.embed_instruction
                if input_rep == "prompt" and typed.embed_instruction
                else CONFIG.prompt_embed_instruction
            ),
            pooling=typed.pooling or "last-token",
            normalization=typed.normalization or "l2",
            dtype=typed.dtype,
        )
        lens = cls(projector, embedder, names=names, manifest=manifest)
        lens.lens_dir = lens_dir
        return lens

    # public name for from_dir; both work
    load = from_dir

    @classmethod
    @_observe_lens_result("fetch_lens", source_kind="hub")
    def from_pretrained(
        cls,
        repo_id: str,
        *,
        revision: str | None = None,
        cache_dir=None,
        token=None,
        local_files_only: bool = False,
        subfolder: str | None = None,
        device: str = "cpu",
        annotations=None,
        embedding_cache=None,
        embed_backend: str = "hf",
        embed_batch_size: int | None = None,
    ) -> "Lens":
        """Download a lens from the Hugging Face Hub and load it.

        A repository may contain one lens at its root or several lens directories,
        selected with ``subfolder``. Mutable or omitted revisions are resolved to an
        immutable commit before download. An explicit commit also works with
        ``local_files_only=True`` without a Hub metadata lookup.
        """
        from prefscope.api.hub import download_lens, resolve_hf_revision

        requested_revision = revision
        resolved_revision = resolve_hf_revision(
            repo_id,
            revision=requested_revision,
            repo_type="model",
            token=token,
            local_files_only=local_files_only,
        )
        lens_dir = download_lens(
            repo_id,
            revision=resolved_revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            subfolder=subfolder,
        )
        lens = cls.from_dir(
            lens_dir,
            device=device,
            annotations=annotations,
            embedding_cache=embedding_cache,
            embed_backend=embed_backend,
            embed_batch_size=embed_batch_size,
        )
        # Runtime source provenance belongs to the loaded object, not the published
        # artifact manifest. In particular, never add the access token to either.
        lens.pretrained_repo_id = repo_id
        lens.pretrained_revision = requested_revision
        lens.pretrained_subfolder = subfolder
        lens.pretrained_resolved_revision = resolved_revision
        lens.requested_revision = requested_revision
        lens.resolved_revision = resolved_revision
        return lens

    @classmethod
    @_observe_lens_result("load_lens", source_kind="saelens")
    def from_saelens(
        cls,
        release: str,
        sae_id: str,
        *,
        representation_source=None,
        input_rep: str = "individual",
        device: str = "cpu",
        dtype: str = "float32",
        force_download: bool = False,
        batch_size: int = 1024,
        text_batch_size: int = 8,
        max_output_bytes: int = 256 * 1024 * 1024,
        activation_polarity: str | None = None,
        long_text_policy: str = "truncate",
        include_bos: bool = False,
        reader_model_revision: str | None = None,
        item_projection_policy: str = "forbid",
        allow_unregistered_release: bool = False,
    ) -> "Lens":
        """Wrap a pretrained SAELens SAE without training a PrefScope SAE.

        By default, :meth:`featurize` accepts ``PairItem`` text and uses one lazy
        reader model for prompt and response views. :meth:`project_saelens_tokens`
        remains the advanced exact-activation escape hatch. ``representation_source``
        is usable only with the explicit
        ``item_projection_policy="single_token"`` contract. Direct difference
        projection is rejected because public pretrained SAEs were not trained on
        activation differences. See ``docs/how-to/use-saelens.md``.
        """
        from prefscope.integrations.saelens import SAELensProjector, SAELensTextBackend

        if (
            representation_source is not None
            and item_projection_policy != "single_token"
        ):
            raise ValueError(
                "representation_source requires item_projection_policy='single_token'; "
                "use project_saelens_tokens(...) for ordinary token activations"
            )
        projector = SAELensProjector.from_pretrained(
            release,
            sae_id,
            device=device,
            dtype=dtype,
            force_download=force_download,
            input_rep=input_rep,
            batch_size=batch_size,
            max_output_bytes=max_output_bytes,
            activation_polarity=activation_polarity,
            reader_model_revision=reader_model_revision,
            item_projection_policy=item_projection_policy,
            allow_unregistered_release=allow_unregistered_release,
        )
        if representation_source is None:
            backend = SAELensTextBackend(
                projector,
                device=device,
                text_batch_size=text_batch_size,
                long_text_policy=long_text_policy,
                include_bos=include_bos,
            )
            lens = cls(projector, backend=backend)
        else:
            lens = cls(projector, representation_source=representation_source)
        lens.pretrained_backend = "saelens"
        lens.saelens_release = release
        lens.saelens_id = sae_id
        return lens

    def project_saelens_tokens(
        self,
        *,
        row_ids,
        token_activations,
        token_row_ids,
        representation_contract,
        feature_ids=None,
        metadata=None,
        batch: int | None = None,
    ):
        """Encode exact-hook tokens, then max-pool features into a FeatureBatch."""
        from prefscope.integrations.saelens import project_saelens_tokens

        return project_saelens_tokens(
            self,
            row_ids=row_ids,
            token_activations=token_activations,
            token_row_ids=token_row_ids,
            representation_contract=representation_contract,
            feature_ids=feature_ids,
            metadata=metadata,
            batch=batch,
        )

    @classmethod
    def train(cls, data, config=None, *, out, columns=None) -> "Lens":
        """Train + save a fresh lens from preference data, then load it.

        ``data`` is anything ``pairs_to_battles`` accepts. ``config`` is a
        ``TrainConfig`` (defaults if omitted). Trains via ``build_lens`` and
        returns the loaded ``Lens``. Heavy imports (Embedder / build_lens) are
        lazy so ``import prefscope`` stays torch-free.
        """
        from prefscope.api.config import TrainConfig
        from prefscope.encode.embed import Embedder
        from prefscope.pipeline.build_lens import build_lens

        if config is None:
            config = TrainConfig()

        forbidden = {
            "m_total",
            "k",
            "matryoshka_prefix",
            "input_rep",
            "sae_type",
            "sparsity_coef",
            "bandwidth",
            "sparsity_warmup_steps",
            "val_frac",
            "device",
            "embed_model_id",
            "max_train_rows",
            "dump_embeddings",
        }
        overlap = forbidden & set(config.train_kwargs)
        if overlap:
            raise ValueError(
                f"train_kwargs may not override {sorted(overlap)}; set them via "
                f"SAEConfig/TrainConfig fields"
            )

        battles = pairs_to_battles(data, columns=columns)
        embedder = Embedder(
            None,
            device=config.device,
            model_revision=config.embed_model_revision,
            **({"model_id": config.embed_model_id} if config.embed_model_id else {}),
        )
        build_lens(
            battles,
            embedder,
            out,
            m_total=config.sae.m,
            k=config.sae.k,
            matryoshka_prefix=config.sae.matryoshka_prefix,
            input_rep=config.sae.input_rep,
            sae_type=config.sae.sae_type,
            sparsity_coef=config.sae.sparsity_coef,
            bandwidth=config.sae.bandwidth,
            sparsity_warmup_steps=config.sae.sparsity_warmup_steps,
            val_frac=config.val_frac,
            device=config.device,
            embed_model_id=config.embed_model_id,
            max_train_rows=config.max_train_rows,
            **config.train_kwargs,
        )
        return cls.load(out, device=config.device)

    @property
    def capabilities(self):
        """Return the backend's machine-readable supported feature views."""
        return self.backend.capabilities

    @_observe_feature_batch_result("featurize")
    def featurize(
        self,
        dataset,
        *,
        views=None,
        feature_ids=None,
        batch_size: int | None = None,
    ):
        """Encode ``PairItem`` rows into an aligned, role-aware ``FeatureBatch``.

        Existing ``encode``, ``encode_items``, and ``encode_pairs`` keep their
        historical ndarray return contracts.
        """
        from prefscope.api._lens_backend import (
            normalize_items,
            resolve_views,
            select_feature_batch,
        )
        from prefscope.core.features import FeatureBatch, validate_feature_ids

        items = normalize_items(dataset)
        resolved_views = resolve_views(
            views, self.capabilities, paired=items[0].y_b is not None
        )
        selected = None
        if feature_ids is not None:
            selected = validate_feature_ids(tuple(feature_ids))
            if not selected:
                raise ValueError("feature_ids must select at least one feature")
            if min(selected) < 0 or max(selected) >= int(self.backend.m_total):
                raise ValueError("feature_ids contain an index outside this lens")
        if batch_size is not None and (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer or None")
        features = self.backend.featurize(
            items, views=resolved_views, feature_ids=selected, batch_size=batch_size
        )
        if not isinstance(features, FeatureBatch):
            raise ValueError("lens backend featurize() must return a FeatureBatch")
        if features.row_ids != tuple(str(item.id) for item in items):
            raise ValueError(
                "lens backend FeatureBatch row_ids must exactly match item order"
            )
        if selected is not None and features.feature_ids != selected:
            raise ValueError(
                "lens backend feature_ids must exactly match the requested selection"
            )
        if selected is None and features.feature_ids != tuple(
            range(int(self.backend.m_total))
        ):
            raise ValueError(
                "unselected lens backend output must contain every feature ID in "
                "range(m_total)"
            )
        from prefscope.core.lens_backend import pair_item_metadata

        canonical_metadata = pair_item_metadata(items)
        for name in set(features.metadata) & set(canonical_metadata):
            if tuple(features.metadata[name]) != tuple(canonical_metadata[name]):
                raise ValueError(
                    f"lens backend metadata {name!r} contradicts PairItem rows"
                )
        features = FeatureBatch(
            row_ids=features.row_ids,
            arrays=features.arrays,
            roles=features.roles,
            orientations=features.orientations,
            feature_ids=features.feature_ids,
            metadata={**canonical_metadata, **dict(features.metadata)},
            activation_polarity=features.activation_polarity,
            code_semantics=features.code_semantics,
            provenance=features.provenance,
        )
        if (
            self.capabilities.difference == "a_minus_b_after_encoding"
            and {"z_a", "z_b", "z_diff"}.issubset(features.arrays)
            and not np.allclose(
                features.array("z_diff"),
                features.array("z_a") - features.array("z_b"),
                rtol=1e-5,
                atol=1e-6,
            )
        ):
            raise ValueError(
                "lens backend z_diff contradicts declared A-minus-B-after-encoding "
                "semantics"
            )
        return select_feature_batch(
            features, views=resolved_views, feature_ids=selected
        )

    @property
    def fidelity_feature_ids(self):
        return inspect_fidelity_feature_ids(self)

    @property
    def concept_names(self):
        """Series mapping feature IDs to names, or ``None`` when unnamed."""
        return inspect_concept_names(self)

    @property
    def feature_table(self) -> pd.DataFrame:
        """Return one row per feature with all bundled annotation columns."""
        return inspect_feature_table(self)

    @property
    def feature_catalog(self):
        """Return proposed display labels bound to this feature coordinate space."""
        from prefscope.api.feature_catalog import FeatureCatalog

        return FeatureCatalog.from_lens(self)

    @property
    def feature_space_identity(self) -> dict[str, str | None]:
        model_path = (
            Path(self.lens_dir) / SAE_MODEL if self.lens_dir is not None else None
        )
        if model_path is not None and model_path.is_file():
            stat = model_path.stat()
            cache_key = (
                str(model_path.resolve()),
                int(stat.st_size),
                int(stat.st_mtime_ns),
            )
        else:
            cache_key = None
        cached = getattr(self, "_feature_space_identity_cache", None)
        if cached is None or cached[0] != cache_key:
            from prefscope.api._feature_space import lens_feature_space_identity

            cached = (cache_key, lens_feature_space_identity(self))
            self._feature_space_identity_cache = cached
        return dict(cached[1])

    @property
    def feature_space_id(self) -> str | None:
        return self.feature_space_identity["feature_space_id"]

    @property
    def feature_space_status(self) -> str:
        return self.feature_space_identity["feature_space_status"]

    def presence(self, codes, *, feature_ids=None, policy: str = "calibrated"):
        """Resolve codes into concept presence under an explicit policy."""
        return inspect_presence(self, codes, feature_ids=feature_ids, policy=policy)

    @staticmethod
    def _representation_contract_fingerprint(contract) -> str:
        return representation_contract_fingerprint(contract)

    def _expected_representation_contract(self) -> dict | None:
        return expected_representation_contract(self)

    def _validate_representation_contract(
        self,
        batch,
        *,
        allow_mismatch: bool,
    ) -> dict:
        return validate_representation_contract(
            self, batch, allow_mismatch=allow_mismatch
        )

    @_observe_feature_batch_result("project_representations")
    def project_representations(
        self,
        batch,
        *,
        allow_representation_mismatch: bool = False,
    ):
        """Project an aligned representation batch through this lens."""
        return project_representations(
            self,
            batch,
            allow_representation_mismatch=allow_representation_mismatch,
        )

    @_observe_array_result("encode")
    def encode(self, prompts, completions=None) -> np.ndarray:
        """Encode prompt/response text with an individual or prompt lens."""
        return encode(self, prompts, completions)

    @_observe_array_result("encode")
    def encode_one(self, prompt, completion=None) -> np.ndarray:
        """Return concept codes for one response as a one-dimensional array."""
        return encode_one(self, prompt, completion)

    def top_concepts(self, codes, k: int = 5, *, matching_pole_only: bool = True):
        """Return each row's strongest active named concepts."""
        return inspect_top_concepts(
            self, codes, k=k, matching_pole_only=matching_pole_only
        )

    def concept_activations(
        self,
        codes,
        *,
        row_ids=None,
        active_only: bool = True,
        pole: str = "any",
        min_abs_activation: float = 0.0,
        top_k: int | None = None,
        fidelity_only: bool = False,
        semantic_presence_only: bool = False,
    ) -> pd.DataFrame:
        """Return sparse codes as a filterable long-form concept table."""
        return inspect_concept_activations(
            self,
            codes,
            row_ids=row_ids,
            active_only=active_only,
            pole=pole,
            min_abs_activation=min_abs_activation,
            top_k=top_k,
            fidelity_only=fidelity_only,
            semantic_presence_only=semantic_presence_only,
        )

    @_observe_array_result("encode_pairs")
    def encode_pairs(self, dataset, *, return_meta: bool = True):
        """Encode aligned response pairs and optionally return their metadata."""
        return encode_pairs(self, dataset, return_meta=return_meta)

    @_observe_array_result("encode_pairs")
    def encode_items(self, dataset, *, return_meta: bool = True):
        """Encode a homogeneous iterable of paired or single-response items."""
        return encode_items(self, dataset, return_meta=return_meta)

    # back-compat name; encode_pairs is the canonical method
    project = encode_pairs

    def save(
        self,
        dest,
        *,
        overwrite: bool = False,
        annotations=None,
        inference_only: bool = False,
    ):
        """Publish this lens as a transactional whole-directory replacement."""
        data = _operation_data(
            input_rep=self.input_rep,
            overwrite=overwrite,
            inference_only=inference_only,
            has_annotations=annotations is not None,
        )
        with automatic_stage("save_lens", data):
            return save_lens(
                self,
                dest,
                overwrite=overwrite,
                annotations=annotations,
                inference_only=inference_only,
            )

    def diagnose(self, codes, meta, *, fidelity_only: bool = False):
        """See ``prefscope.analysis.diagnose``."""
        return inspect_diagnose(self, codes, meta, fidelity_only=fidelity_only)

    def preference_relevance(
        self,
        features,
        *,
        preference_column: str = "pref",
        group_column: str | None = "group_id",
        feature_array: str = "z_diff",
    ) -> pd.DataFrame:
        """Analyze P(A preferred) against an aligned A-minus-B feature view."""
        from prefscope.api.preference import preference_relevance

        data = _operation_data(
            input_rep=self.input_rep,
            grouped=group_column is not None,
        )
        with automatic_stage("analyze_preference", data) as span:
            table = preference_relevance(
                features,
                preference_column=preference_column,
                group_column=group_column,
                feature_array=feature_array,
            )
            annotations = self.feature_table
            if "concept" in annotations:
                table = table.merge(
                    annotations[["feature_id", "concept"]].drop_duplicates(
                        "feature_id"
                    ),
                    on="feature_id",
                    how="left",
                )
            if span.active:
                try:
                    span.update(
                        output_rows=int(table.shape[0]),
                        output_features=int(table["feature_id"].nunique()),
                        shape=[int(size) for size in table.shape],
                    )
                except BaseException:
                    # Result telemetry cannot alter a successful analysis.
                    pass
            return table

    def feature_preference_relevance(self, codes, meta):
        """See ``prefscope.analysis.feature_preference_relevance``."""
        return inspect_feature_preference_relevance(self, codes, meta)

    def evaluate_preference(self, codes, meta, **kwargs):
        """See ``prefscope.analysis.evaluate_preference``."""
        return inspect_evaluate_preference(self, codes, meta, **kwargs)


# Preserve the historical public module identity after the implementation split.
pairs_to_battles.__module__ = __name__


# Back-compat alias: the class was formerly named LoadedLens.
LoadedLens = Lens
