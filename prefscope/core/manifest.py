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
``from_dict(strict=True)`` / ``require_complete()`` is the PRODUCE path (every v1-required
field must be present) used when writing/validating a shareable artifact.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

VALID_KINDS = ("difference", "individual", "prompt")

# fields that a COMPLETE (shareable) v1 artifact must carry
_REQUIRED = (
    "schema_version", "lens_kind", "input_rep", "m_total", "k", "input_dim",
    "matryoshka_prefix_lengths", "output_arrays", "embed_model_id",
)

# canonical (typed) fields; anything else in the source dict is preserved in ``extra``
_CANONICAL = _REQUIRED + (
    "embed_model_revision", "pooling", "normalization", "max_tokens",
    "embed_instruction", "dtype", "backend", "sae_type", "whiten", "seed",
    "best_val_norm_mse", "best_val_select_norm_mse", "n_epochs_trained",
    "n_battles", "dataset_hash", "array_shapes",
)


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
    whiten: str | None = None
    seed: int | None = None
    best_val_norm_mse: float | None = None
    best_val_select_norm_mse: float | None = None
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
        ``strict``) requires every field of a complete v1 artifact."""
        d = dict(d or {})
        legacy = "schema_version" not in d
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

        known = {k: d.get(k) for k in _CANONICAL if k in d}
        extra = {k: v for k, v in d.items() if k not in _CANONICAL}
        known.update(schema_version=SCHEMA_VERSION, lens_kind=kind, input_rep=input_rep)
        obj = cls(**{**{k: None for k in _CANONICAL if k not in ("schema_version",)},
                     **known, "extra": extra})
        if strict:
            obj.require_complete()
        return obj

    def require_complete(self) -> "LensManifest":
        """Raise if any field required for a shareable v1 artifact is missing."""
        missing = [f for f in _REQUIRED if getattr(self, f, None) in (None, [])]
        if missing:
            raise ValueError(
                f"lens manifest is missing required v{SCHEMA_VERSION} fields: {missing}. "
                "A complete artifact must record its representation, dimensions and "
                "embedding model so consumers never load it with the wrong config.")
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
        """Check each declared output array exists and its feature dimension == m_total.
        Records the observed shapes into ``array_shapes``. Catches an m_total/embedding
        mismatch that a bare dict manifest would load right past."""
        import numpy as np
        lens_dir = Path(lens_dir)
        shapes: dict = {}
        for name in (self.output_arrays or []):
            p = lens_dir / f"{name}.npy"
            if not p.exists():
                raise FileNotFoundError(
                    f"manifest declares output array {name!r} but {p} is missing")
            arr = np.load(p, mmap_mode="r")
            shapes[name] = list(arr.shape)
            if self.m_total is not None and arr.ndim == 2 and arr.shape[1] != self.m_total:
                raise ValueError(
                    f"array {name} has feature dim {arr.shape[1]} but manifest m_total="
                    f"{self.m_total}; the lens and its codes disagree")
        self.array_shapes = shapes
        return self
