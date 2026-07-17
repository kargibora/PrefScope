from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from prefscope.core.types import PairItem


class Dataset(ABC):
    """Yields normalized PairItems. Bring your own data by subclassing this."""

    @abstractmethod
    def __iter__(self) -> Iterator[PairItem]:
        ...

    def __len__(self) -> int:
        # A Dataset is defined by __iter__ and has no general length. Subclasses with a
        # known size override this; otherwise this must raise TypeError — the standard
        # "unsized object" signal. TypeError (not None, which violates the len() contract,
        # nor NotImplementedError, which would break list()/length-hint) is what len()
        # raises for unsized objects and what list()'s length-hint swallows (#10).
        raise TypeError(f"object of type {type(self).__name__!r} has no len()")
