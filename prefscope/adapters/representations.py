"""Pure-numpy Representations: combine a SideVectors pair into SAE rows."""
from __future__ import annotations

import numpy as np

from prefscope.core import registry
from prefscope.core.representation import Representation
from prefscope.core.types import SideVectors


@registry.register("representation", "identity")
class IdentityRepresentation(Representation):
    """Use the A-side vectors verbatim (works for response and token sources)."""
    compatible = frozenset({"response", "token"})

    def combine(self, sv: SideVectors) -> np.ndarray:
        return np.asarray(sv.a)


@registry.register("representation", "diff")
class DiffRepresentation(Representation):
    """e_a - e_b. Response-only: token-wise diff is ill-defined across lengths."""
    compatible = frozenset({"response"})

    def combine(self, sv: SideVectors) -> np.ndarray:
        if sv.b is None:
            raise ValueError("DiffRepresentation requires a paired item (y_b is None)")
        return np.asarray(sv.a) - np.asarray(sv.b)


@registry.register("representation", "concat")
class ConcatRepresentation(Representation):
    """[e_a | e_b] on the feature axis. Response-only (needs aligned rows)."""
    compatible = frozenset({"response"})

    def combine(self, sv: SideVectors) -> np.ndarray:
        if sv.b is None:
            raise ValueError("ConcatRepresentation requires a paired item (y_b is None)")
        return np.concatenate([np.asarray(sv.a), np.asarray(sv.b)], axis=1)


@registry.register("representation", "both")
class BothSidesRepresentation(Representation):
    """Stack A and B vectors as separate SAE rows (works for response/token)."""
    compatible = frozenset({"response", "token"})

    def combine(self, sv: SideVectors) -> np.ndarray:
        a = np.asarray(sv.a)
        if sv.b is None:
            return a
        return np.vstack([a, np.asarray(sv.b)])
