"""The public Lens wrapper for backend-neutral feature extraction.

A Lens is loaded, trained, or constructed from a backend, then applied through
``Lens.featurize(...) -> FeatureBatch``. Historical ndarray extraction helpers remain
internal recipe support and are not methods on the public object.
"""

from __future__ import annotations

import hashlib
import json
import os  # noqa: F401  - compatibility patch point for publication tests/callers
import warnings
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from prefscope.api._lens_annotations import _load_feature_table
from prefscope.api._lens_data import pairs_to_battles
from prefscope.api._lens_inspection import (
    concept_names as inspect_concept_names,
    feature_table as inspect_feature_table,
)
from prefscope.api._lens_projection import (
    expected_representation_contract,
    representation_contract_fingerprint,
    validate_representation_contract,
)
from prefscope.api._lens_publication import (
    _publication_lock as _publication_lock,
    _recover_orphan_backup as _recover_orphan_backup,
    save_lens,
)
from prefscope.artifacts import MANIFEST, SAE_MODEL


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
                stacklevel=2,
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
        # The live projector keeps its loaded weights and whitener even if another
        # publisher later replaces this directory. Keep their coordinate identity too.
        lens._loaded_native_feature_space_identity = lens.feature_space_identity
        return lens

    # public name for from_dir; both work
    load = from_dir

    @classmethod
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
        expected_sae_weights_sha256: str | None = None,
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
            expected_sae_weights_sha256=expected_sae_weights_sha256,
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

    def featurize(
        self,
        dataset,
        *,
        views=None,
        feature_ids=None,
        batch_size: int | None = None,
    ):
        """Encode ``PairItem`` rows into an aligned, role-aware ``FeatureBatch``.

        ``batch_size`` is a per-call option for backends that support it, such as
        SAELens. Native representation lenses reject this option; set
        ``embed_batch_size`` when loading a native lens, or configure batching on
        the supplied ``RepresentationSource``.
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
        provenance = dict(features.provenance)
        raw_lens_provenance = provenance.get("lens", {})
        if not isinstance(raw_lens_provenance, Mapping):
            raise ValueError("lens backend provenance lens field must be a mapping")
        lens_provenance = dict(raw_lens_provenance)
        identity = self.feature_space_identity
        existing_id = lens_provenance.get("feature_space_id")
        if (
            existing_id is not None
            and identity["feature_space_id"] is not None
            and existing_id != identity["feature_space_id"]
        ):
            raise ValueError(
                "lens backend FeatureBatch contradicts its declared feature-space identity"
            )
        if identity["feature_space_id"] is not None or existing_id is None:
            lens_provenance.update(identity)
        provenance["lens"] = lens_provenance
        features = FeatureBatch(
            row_ids=features.row_ids,
            arrays=features.arrays,
            roles=features.roles,
            orientations=features.orientations,
            feature_ids=features.feature_ids,
            metadata={**canonical_metadata, **dict(features.metadata)},
            activation_polarity=features.activation_polarity,
            code_semantics=features.code_semantics,
            provenance=provenance,
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

        current = FeatureCatalog.from_lens(self)
        if self.lens_dir is None:
            return current
        from prefscope.artifacts import (
            FEATURE_CATALOG,
            FEATURE_NAMES,
            PROMPT_FEATURE_NAMES,
        )

        path = Path(self.lens_dir) / FEATURE_CATALOG
        if not path.is_file():
            return current
        from prefscope.api.feature_catalog_io import decode_feature_catalog

        bundled = decode_feature_catalog(path.read_bytes())
        if bundled.feature_ids != tuple(range(int(self.backend.m_total))):
            raise ValueError("bundled feature catalog must cover every native feature")
        names_filename = (
            PROMPT_FEATURE_NAMES if self.input_rep == "prompt" else FEATURE_NAMES
        )
        names_path = Path(self.lens_dir) / names_filename
        if not names_path.is_file():
            raise ValueError("bundled feature catalog names artifact is missing")
        names_sha256 = hashlib.sha256(names_path.read_bytes()).hexdigest()
        name_source = bundled.column_sources.get("name")
        if (
            bundled.provenance.get("names_artifact") != names_filename
            or bundled.provenance.get("names_sha256") != names_sha256
            or not isinstance(name_source, Mapping)
            or name_source.get("artifact") != names_filename
            or name_source.get("content_sha256") != names_sha256
        ):
            raise ValueError(
                "bundled feature catalog names provenance does not match its CSV"
            )
        if (
            bundled.feature_space_id != self.feature_space_id
            or bundled.feature_space_status != self.feature_space_status
        ):
            raise ValueError(
                "bundled feature catalog does not match the native feature space"
            )
        # Explicit runtime annotations may intentionally replace bundled proposed names.
        return bundled if dict(bundled.labels) == dict(current.labels) else current

    @property
    def feature_space_identity(self) -> dict[str, str | None]:
        loaded_identity = getattr(self, "_loaded_native_feature_space_identity", None)
        if loaded_identity is not None:
            return dict(loaded_identity)
        model_path = (
            Path(self.lens_dir) / SAE_MODEL if self.lens_dir is not None else None
        )
        if model_path is not None and model_path.is_file():
            model_stat = model_path.stat()
            whiten_path = Path(self.lens_dir) / "whiten.npz"
            if whiten_path.is_file():
                whiten_stat = whiten_path.stat()
                whiten_key = (
                    str(whiten_path.resolve()),
                    int(whiten_stat.st_dev),
                    int(whiten_stat.st_ino),
                    int(whiten_stat.st_size),
                    int(whiten_stat.st_mtime_ns),
                    int(whiten_stat.st_ctime_ns),
                )
            else:
                whiten_key = None
            cache_key = (
                str(model_path.resolve()),
                int(model_stat.st_dev),
                int(model_stat.st_ino),
                int(model_stat.st_size),
                int(model_stat.st_mtime_ns),
                int(model_stat.st_ctime_ns),
                whiten_key,
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
    def sae_weights_sha256(self) -> str | None:
        """Observed SAELens state digest for bootstrap pinning, when available."""
        provenance = dict(
            getattr(getattr(self, "projector", None), "projector_provenance", None)
            or {}
        )
        value = provenance.get("sae_weights_sha256")
        return str(value) if value is not None else None

    @property
    def sae_acquisition_status(self) -> str | None:
        """State whether the observed SAELens digest was expected and verified."""
        provenance = dict(
            getattr(getattr(self, "projector", None), "projector_provenance", None)
            or {}
        )
        value = provenance.get("sae_acquisition_status")
        return str(value) if value is not None else None

    @property
    def feature_space_id(self) -> str | None:
        return self.feature_space_identity["feature_space_id"]

    @property
    def feature_space_status(self) -> str:
        return self.feature_space_identity["feature_space_status"]

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

    def save(
        self,
        dest,
        *,
        overwrite: bool = False,
        annotations=None,
        inference_only: bool = False,
    ):
        """Publish this lens as a transactional whole-directory replacement."""
        return save_lens(
            self,
            dest,
            overwrite=overwrite,
            annotations=annotations,
            inference_only=inference_only,
        )



# Preserve the historical public module identity after the implementation split.
pairs_to_battles.__module__ = __name__
