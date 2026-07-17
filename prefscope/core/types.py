"""Pure data contracts for the PrefScope framework (numpy/typing only)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class PairItem:
    """One post-training comparison. Single-response is y_b=None (degenerate)."""
    id: str
    x: str                       # prompt
    y_a: str
    y_b: str | None = None
    pref: float | None = None    # P(A preferred); 0.0 = B wins, 0.5 = tie
    model_a: str | None = None   # which model produced y_a (needed by diagnose)
    model_b: str | None = None   # which model produced y_b
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.pref is not None and not (0.0 <= float(self.pref) <= 1.0):
            raise ValueError(
                f"pref must be in [0, 1] as P(A preferred); got {self.pref!r}")

    @property
    def is_single(self) -> bool:
        return self.y_b is None


@dataclass
class SideVectors:
    """Raw per-side vectors from an ActivationSource.

    a/b are (n, d): n=1 for embeddings, n=#tokens for token activations.
    """
    a: np.ndarray
    b: np.ndarray | None
    item_id: str
    meta: dict = field(default_factory=dict)
