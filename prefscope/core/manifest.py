"""Typed, versioned lens manifest.

A lens is a shared artifact: its ``manifest.json`` must say — unambiguously and
reproducibly — how the embeddings were produced and how the SAE was trained, so a
consumer can never load it with the *wrong* embedder/representation and silently produce
garbage. The previous manifest was an unversioned dict read with ``dict.get(key, default)``,
so a missing ``input_rep`` silently became ``"difference"`` and a missing embedder id
silently became ``None``. This module replaces that with:

* ``schema_version`` + migration for legacy (unversioned) manifests,
* **strict** required-field validation for freshly produced artifacts,
* a **safe** load path that infers ``input_rep`` from the saved arrays rather than
  guessing, and refuses to invent a representation it cannot determine,
* explicit provenance fields (recorded as ``null`` when genuinely unknown — never absent),
* array-shape validation against the on-disk ``.npy`` files.

Design: ``from_dict(strict=False)`` is the lenient LOAD path (migrate + infer, warn on
missing provenance, raise only when a representation can't be determined).
``from_dict(strict=True)`` / ``require_complete()`` is the PRODUCE path (every v2-required
field must be present) used when writing/validating a shareable artifact.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2

VALID_KINDS = ("difference", "individual", "prompt")

# fields that a COMPLETE (shareable) v2 artifact must carry
_REQUIRED = (
    "schema_version", "lens_kind", "input_rep", "m_total", "k", "input_dim",
    "matryoshka_prefix_lengths", "output_arrays", "embed_model_id", "sae_type",
    "activation_polarity", "code_semantics", "selection_rule",
)

# canonical (typed) fields; anything else in the source dict is preserved in ``extra``
_CANONICAL = _REQUIRED + (
    "embed_model_revision", "pooling", "normalization", "max_tokens",
    "embed_instruction", "dtype", "backend", "whiten", "seed",
    "best_val_norm_mse", "best_val_select_norm_mse", "n_epochs_trained",
    "best_val_explained_variance", "deployment_val_norm_mse",
    "deployment_val_explained_variance", "deployment_val_active",
    "deployment_dead_neurons", "deployment_rare_neurons", "target_l0",
    "calibration_l0", "threshold_calibration_rows", "optimizer", "weight_decay",
    "n_battles", "dataset_hash", "array_shapes",
)


def infer_sae_semantics(sae_type: str | None) -> tuple[str, str, str]:
    """Migrate pre-v2 artifacts without importing torch-backed SAE modules.

    Every historical checkpoint missing ``sae_type`` was loaded as ``batchtopk``,
    whose codes are signed.  That makes this a compatibility inference rather
    than a guess about the artifact's numerical behavior.
    """
    kind = sae_type or "batchtopk"
    if kind in ("batchtopk", "signed-batchtopk", "simple-topk"):
        return "signed", "axis", (
            "topk-absolute" if kind == "simple-topk" else "batchtopk-absolute")
    if kind == "batchtopk-relu":
        return "nonnegative", "presence", "batchtopk-relu"
    if kind == "jumprelu":
        return "nonnegative", "presence", "jumprelu"
    return "unknown", "custom", "custom"


def infer_lens_kind(input_rep, output_arrays) -> str | None:
    """Best-effort lens kind. Prefer an explicit ``input_rep``; else read the saved
    arrays: ``z_prompt`` → prompt, ``z_a``/``z_b`` → individual, ``z_diff`` only →
    difference. Returns None when nothing is determinable (caller must raise, not guess)."""
    if input_rep in VALID_KINDS:
        return input_rep
    arrays = set(output_arrays or [])
    if "z_prompt" in arrays:
        return "prompt"
    if {"z_a", "z_b"} & arrays:
        return "individual"
    if "z_diff" in arrays:
        return "difference"
    return None


@dataclass
class LensManifest:
    schema_version: int
    lens_kind: str
    input_rep: str
    m_total: int | None = None
    k: int | None = None
    input_dim: int | None = None
    matryoshka_prefix_lengths: list | None = None
    output_arrays: list | None = None
    embed_model_id: str | None = None
    # ── embedding provenance (explicit None = genuinely unknown, never silently absent) ──
    embed_model_revision: str | None = None
    pooling: str | None = None
    normalization: str | None = None
    max_tokens: int | None = None
    embed_instruction: str | None = None
    dtype: str | None = None
    backend: str | None = None
    # ── training / data provenance ──
    sae_type: str | None = None
    activation_polarity: str | None = None
    code_semantics: str | None = None
    selection_rule: str | None = None
    whiten: str | None = None
    seed: int | None = None
    best_val_norm_mse: float | None = None
    best_val_select_norm_mse: float | None = None
    best_val_explained_variance: float | None = None
    deployment_val_norm_mse: float | None = None
    deployment_val_explained_variance: float | None = None
    deployment_val_active: float | None = None
    deployment_dead_neurons: int | None = None
    deployment_rare_neurons: int | None = None
    target_l0: int | None = None
    calibration_l0: float | None = None
    threshold_calibration_rows: int | None = None
    optimizer: str | None = None
    weight_decay: float | None = None
    n_epochs_trained: int | None = None
    n_battles: int | None = None
    dataset_hash: str | None = None
    array_shapes: dict | None = None
    # forward-compat: any non-canonical keys are round-tripped here untouched
    extra: dict = field(default_factory=dict)

    # ── construction ──────────────────────────────────────────────────────────────
    @classmethod
    def from_dict(cls, d: dict, *, strict: bool = False) -> "LensManifest":
        """Parse a manifest dict. Migrates legacy (unversioned) manifests, infers
        ``input_rep`` from the saved arrays rather than defaulting it, and (when
        ``strict``) requires every field of a complete v2 artifact."""
        d = dict(d or {})
        legacy = "schema_version" not in d
        raw_version = d.get("schema_version")
        if not legacy and (
            type(raw_version) is not int or raw_version < 1
        ):
            raise ValueError("lens manifest schema_version must be a positive integer")
        source_version = 0 if legacy else raw_version
        if source_version > SCHEMA_VERSION:
            raise ValueError(
                f"lens manifest schema v{source_version} is newer than this "
                f"PrefScope build supports (v{SCHEMA_VERSION}); upgrade PrefScope")
        if legacy:
            logger.warning(
                "loading a legacy (unversioned) lens manifest; migrating to schema v%d — "
                "re-save the lens to persist provenance", SCHEMA_VERSION)

        kind = d.get("lens_kind") or infer_lens_kind(d.get("input_rep"),
                                                      d.get("output_arrays"))
        if kind is None:
            raise ValueError(
                "lens manifest has no input_rep/lens_kind and no recognizable output "
                "arrays; refusing to guess the representation (a wrong guess silently "
                "corrupts every code). Fields present: " + ", ".join(sorted(d)))
        if kind not in VALID_KINDS:
            raise ValueError(f"unknown lens_kind {kind!r}; expected one of {VALID_KINDS}")
        input_rep = d.get("input_rep") or kind
        if input_rep not in VALID_KINDS:
            raise ValueError(
                f"unknown input_rep {input_rep!r}; expected one of {VALID_KINDS}")
        if input_rep != kind:
            raise ValueError(
                f"contradictory lens manifest: lens_kind={kind!r} but "
                f"input_rep={input_rep!r}")

        sae_type = d.get("sae_type") or "batchtopk"
        polarity, semantics, selection = infer_sae_semantics(sae_type)
        if d.get("sae_type") is None:
            d["sae_type"] = sae_type
        if d.get("activation_polarity") is None:
            d["activation_polarity"] = polarity
        if d.get("code_semantics") is None:
            d["code_semantics"] = semantics
        if d.get("selection_rule") is None:
            d["selection_rule"] = selection
        if not legacy and source_version < SCHEMA_VERSION:
            logger.info("migrating lens manifest v%d to v%d", source_version,
                        SCHEMA_VERSION)

        for name in ("m_total", "k", "input_dim"):
            value = d.get(name)
            if value is not None and (
                type(value) is not int or value <= 0
            ):
                raise ValueError(
                    f"lens manifest {name} must be a positive integer")
        known = {k: d.get(k) for k in _CANONICAL if k in d}
        extra = {k: v for k, v in d.items() if k not in _CANONICAL}
        known.update(schema_version=SCHEMA_VERSION, lens_kind=kind, input_rep=input_rep)
        obj = cls(**{**{k: None for k in _CANONICAL if k not in ("schema_version",)},
                     **known, "extra": extra})
        if strict:
            obj.require_complete()
        return obj

    def require_complete(self) -> "LensManifest":
        """Raise if any field required for a shareable v2 artifact is missing."""
        # An empty Matryoshka list is meaningful in v2: nested-width training was
        # explicitly disabled. Output arrays, by contrast, must remain non-empty.
        inference_only = self.extra.get("artifact_scope") == "inference"
        missing = [
            f for f in _REQUIRED
            if getattr(self, f, None) is None
            or (f == "output_arrays" and getattr(self, f, None) == []
                and not inference_only)
        ]
        if missing:
            raise ValueError(
                f"lens manifest is missing required v{SCHEMA_VERSION} fields: {missing}. "
                "A complete artifact must record its representation, dimensions and "
                "embedding model so consumers never load it with the wrong config.")
        for name in ("m_total", "k", "input_dim"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"lens manifest {name} must be a positive integer")
        if self.k > self.m_total:
            raise ValueError("lens manifest k cannot exceed m_total")
        if self.max_tokens is not None and (
            not isinstance(self.max_tokens, int) or self.max_tokens <= 0
        ):
            raise ValueError("lens manifest max_tokens must be a positive integer")
        if self.pooling not in (None, "last-token"):
            raise ValueError(f"unsupported lens pooling {self.pooling!r}")
        if self.normalization not in (None, "l2"):
            raise ValueError(f"unsupported lens normalization {self.normalization!r}")
        return self

    def validate_projector(self, projector) -> "LensManifest":
        """Cross-check artifact metadata against the loaded SAE checkpoint."""
        checks = {
            "m_total": getattr(projector, "m_total", None),
            "input_dim": getattr(projector, "input_dim", None),
            "sae_type": getattr(projector, "sae_type", None),
            "activation_polarity": getattr(projector, "activation_polarity", None),
            "code_semantics": getattr(projector, "code_semantics", None),
            "selection_rule": getattr(projector, "selection_rule", None),
        }
        for name, observed in checks.items():
            declared = getattr(self, name)
            if declared is not None and observed is not None and declared != observed:
                raise ValueError(
                    f"lens manifest {name}={declared!r} disagrees with checkpoint "
                    f"{name}={observed!r}")
        checkpoint_k = getattr(projector, "k", None)
        if self.k is not None and checkpoint_k is not None and self.k != checkpoint_k:
            raise ValueError(
                f"lens manifest k={self.k} disagrees with checkpoint k={checkpoint_k}")
        return self

    def to_dict(self) -> dict:
        """Canonical serializable form: typed fields first, then any passthrough extras
        (so downstream readers keep every legacy key they relied on)."""
        out = {f: getattr(self, f) for f in _CANONICAL}
        out.update(self.extra)                    # passthrough (never overrides canonical)
        for f in _CANONICAL:                      # canonical wins over a same-named extra
            out[f] = getattr(self, f)
        return out

    # ── integrity ─────────────────────────────────────────────────────────────────
    def validate_arrays(self, lens_dir) -> "LensManifest":
        """Validate declared code arrays and their row-aligned metadata.

        Every declared array must be a two-dimensional ``(N, m_total)`` matrix. All
        arrays must share ``N``, agree with recorded ``array_shapes`` when present, and
        match ``n_battles`` and ``battles.parquet`` when those row-count records exist.
        Successful validation records the observed shapes in ``array_shapes``.
        """
        import numpy as np

        lens_dir = Path(lens_dir)
        names = list(self.output_arrays or [])
        if self.m_total is None and names:
            raise ValueError(
                "lens manifest has declared output arrays but no m_total to validate them")

        recorded_shapes = self.array_shapes
        if recorded_shapes is not None:
            if not isinstance(recorded_shapes, dict):
                raise ValueError("lens manifest array_shapes must be a mapping")
            if set(recorded_shapes) != set(names):
                raise ValueError(
                    "lens manifest array_shapes keys disagree with output_arrays: "
                    f"{sorted(recorded_shapes)} vs {sorted(names)}")

        shapes: dict[str, list[int]] = {}
        n_rows: int | None = None
        for name in names:
            path = lens_dir / f"{name}.npy"
            if not path.exists():
                raise FileNotFoundError(
                    f"manifest declares output array {name!r} but {path} is missing")
            array = np.load(path, mmap_mode="r")
            shape = list(array.shape)
            if array.ndim != 2:
                raise ValueError(
                    f"array {name} must be 2-D with shape (N, m_total), got {shape}")
            if array.shape[1] != self.m_total:
                raise ValueError(
                    f"array {name} has feature dim {array.shape[1]} but manifest m_total="
                    f"{self.m_total}; the lens and its codes disagree")
            if n_rows is None:
                n_rows = int(array.shape[0])
            elif array.shape[0] != n_rows:
                raise ValueError(
                    f"array {name} has {array.shape[0]} rows but the other declared "
                    f"arrays have {n_rows}; lens code arrays must stay row-aligned")
            if recorded_shapes is not None:
                try:
                    recorded = list(recorded_shapes[name])
                except TypeError:
                    message = (
                        f"lens manifest array_shapes[{name!r}] must be a shape sequence")
                    raise ValueError(message) from None
                if recorded != shape:
                    raise ValueError(
                        f"array {name} has shape {shape} but manifest array_shapes records "
                        f"{recorded}")
            shapes[name] = shape

        if n_rows is not None and self.n_battles is not None:
            if not isinstance(self.n_battles, int) or isinstance(self.n_battles, bool):
                raise ValueError("lens manifest n_battles must be an integer")
            if n_rows != self.n_battles:
                raise ValueError(
                    f"declared code arrays have {n_rows} rows but manifest n_battles="
                    f"{self.n_battles}")

        battles_path = lens_dir / "battles.parquet"
        if n_rows is not None and battles_path.is_file():
            import pandas as pd

            n_metadata_rows = len(pd.read_parquet(battles_path))
            if n_rows != n_metadata_rows:
                raise ValueError(
                    f"declared code arrays have {n_rows} rows but battles.parquet has "
                    f"{n_metadata_rows}; lens codes and metadata must stay row-aligned")

        self.array_shapes = shapes
        return self
