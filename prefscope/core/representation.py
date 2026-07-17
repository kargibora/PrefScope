from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from prefscope.core.types import SideVectors


class Representation(ABC):
    """Combines a SideVectors pair into the matrix rows fed to the SAE."""

    compatible: frozenset[str]   # subset of {"response", "token"}

    @abstractmethod
    def combine(self, sv: SideVectors) -> np.ndarray:
        ...
