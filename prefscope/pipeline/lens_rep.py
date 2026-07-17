"""Lens contrast strategies — the single home for the ``input_rep`` branch logic.

A ``LensRep`` owns how a lens turns paired completion embeddings into (a) SAE training
rows, (b) inference-time contrast codes, and (c) the saved code arrays. The pipeline
resolves one by name (the manifest's ``input_rep``) via the registry, so adding a
representation is one class — not edits across build_lens / oriented_bank / diagnose /
loaded_lens.

This is the live pipeline's analogue of the item-level
``core.representation.Representation`` (used by the not-yet-implemented ``Lens`` facade),
which only assembles training rows and cannot express the inference-time contrast
(``project(e_a-e_b)`` vs ``project(e_a)-project(e_b)``) that the four pipeline sites
duplicated. The ``projector`` is duck-typed (anything with ``project(ndarray)->ndarray``),
so this module imports nothing from ``encode`` and is testable with a fake projector.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from prefscope.core import registry


def _f32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


class LensRep(ABC):
    """Train/project/save lifecycle for one lens input representation."""

    # contrastive reps build an A/B contrast lens (build_lens / bank / diagnose); a
    # non-contrastive rep (e.g. prompt) has no A/B pairing and is rejected up front.
    contrastive: bool = True
    # per_side reps have an encoder that codes a single response on its own, so they can
    # run absolute (no-pair) encoding; contrast-only reps cannot and are rejected up front.
    per_side: bool = False

    @abstractmethod
    def training_matrix(self, e_a, e_b) -> np.ndarray:
        """SAE training rows from paired (already train/val-masked) embeddings."""

    @abstractmethod
    def contrast_codes(self, projector, e_a, e_b) -> np.ndarray:
        """The A-vs-B contrast code (one orientation)."""

    @abstractmethod
    def oriented_codes(self, projector, e_a, e_b):
        """Both orientations ``(z_a_self, z_b_self)`` — A-as-self and B-as-self.

        Returned as a pair so the implementation can share forward passes; a nonlinear
        SAE means ``project(e_b-e_a) != -project(e_a-e_b)``, so this is NOT a sign flip.
        """

    @abstractmethod
    def output_arrays(self, projector, e_a, e_b) -> dict:
        """Named code arrays to persist at build time (full, unmasked)."""

    def single_output_arrays(self, projector, e) -> dict:
        """Named code arrays for a SINGLE response, with no A/B pair.

        Only a representation with a per-response encoder can code a lone response; the
        default refuses. Used by ``encode-dataset`` in absolute mode."""
        raise ValueError(
            "this lens has no single-response code — a contrast lens can only code an "
            "A/B pair. Provide a second response (battle mode) or use an 'individual' lens.")


@registry.register("lens_rep", "difference")
class DifferenceLensRep(LensRep):
    """WIMHF contrast: train and project on ``e_a - e_b``; the code IS ``project(diff)``."""

    def training_matrix(self, e_a, e_b) -> np.ndarray:
        return _f32(e_a) - _f32(e_b)

    def contrast_codes(self, projector, e_a, e_b) -> np.ndarray:
        return _f32(projector.project(_f32(e_a) - _f32(e_b)))

    def oriented_codes(self, projector, e_a, e_b):
        # two GENUINE forward passes — project(b-a) is not -project(a-b) for a nonlinear SAE
        return (self.contrast_codes(projector, e_a, e_b),
                self.contrast_codes(projector, e_b, e_a))

    def output_arrays(self, projector, e_a, e_b) -> dict:
        return {"z_diff": self.contrast_codes(projector, e_a, e_b)}


@registry.register("lens_rep", "individual")
class IndividualLensRep(LensRep):
    """Pool A and B as separate rows; the code is ``project(e_a) - project(e_b)``."""

    per_side = True   # the encoder codes a single response, so absolute encoding works

    def training_matrix(self, e_a, e_b) -> np.ndarray:
        return np.vstack([_f32(e_a), _f32(e_b)])

    def contrast_codes(self, projector, e_a, e_b) -> np.ndarray:
        return (_f32(projector.project(e_a)) - _f32(projector.project(e_b))).astype(np.float32)

    def oriented_codes(self, projector, e_a, e_b):
        # project each side ONCE and reuse for both orientations (no redundant passes)
        z_a = _f32(projector.project(e_a))
        z_b = _f32(projector.project(e_b))
        return ((z_a - z_b).astype(np.float32), (z_b - z_a).astype(np.float32))

    def output_arrays(self, projector, e_a, e_b) -> dict:
        z_a = _f32(projector.project(e_a))
        z_b = _f32(projector.project(e_b))
        return {"z_a": z_a, "z_b": z_b, "z_diff": (z_a - z_b).astype(np.float32)}

    def single_output_arrays(self, projector, e) -> dict:
        # a lone response codes directly through the per-response encoder
        return {"z_a": _f32(projector.project(e))}


@registry.register("lens_rep", "prompt")
class PromptLensRep(LensRep):
    """Prompt lens: a plain SAE over single prompt vectors — no A/B contrast.

    ``build_prompt_lens`` trains and saves these directly (it never routes through a
    ``LensRep``). These methods exist only so that pointing a contrast op (bank /
    diagnose / project) at a prompt lens raises a clear message instead of an opaque
    ``KeyError`` from the registry."""

    contrastive = False

    def _unsupported(self):
        raise ValueError(
            "prompt lens has no A/B contrast; analyze its z_prompt codes directly "
            "(contrast ops need a 'difference' or 'individual' lens).")

    def training_matrix(self, e_a, e_b):
        self._unsupported()

    def contrast_codes(self, projector, e_a, e_b):
        self._unsupported()

    def oriented_codes(self, projector, e_a, e_b):
        self._unsupported()

    def output_arrays(self, projector, e_a, e_b):
        self._unsupported()


def get_lens_rep(input_rep: str) -> LensRep:
    """Resolve the ``LensRep`` for a manifest's ``input_rep``. Raises ``ValueError``
    (conventional for a bad value) listing the available reps if unknown."""
    try:
        return registry.get("lens_rep", input_rep)()
    except KeyError:
        avail = ", ".join(registry.available("lens_rep"))
        raise ValueError(f"unknown input_rep {input_rep!r}; available: {avail}") from None
