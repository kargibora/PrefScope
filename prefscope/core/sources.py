from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Iterator, Literal

from prefscope.core.types import PairItem, SideVectors


class ActivationSource(ABC):
    """Turns PairItems into raw per-side vectors fed to a Representation."""

    dim: int
    granularity: Literal["response", "token"]

    @abstractmethod
    def encode(self, items: Iterable[PairItem]) -> Iterator[SideVectors]:
        ...
