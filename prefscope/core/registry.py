"""Internal constructors used by native lens-building and interpretation tools.

This module is not exported as an analysis API. Custom lens backends and representation
sources can be passed as ordinary objects.
"""
from __future__ import annotations

from typing import Callable, TypeVar

_REGISTRY: dict[str, dict[str, type]] = {}

T = TypeVar("T")


def register(kind: str, name: str) -> Callable[[type[T]], type[T]]:
    def deco(cls: type[T]) -> type[T]:
        bucket = _REGISTRY.setdefault(kind, {})
        if name in bucket:
            raise ValueError(f"{kind!r} already has a component named {name!r}")
        bucket[name] = cls
        return cls
    return deco


def get(kind: str, name: str) -> type:
    bucket = _REGISTRY.get(kind, {})
    if name not in bucket:
        opts = ", ".join(sorted(bucket)) or "(none registered)"
        raise KeyError(f"no {kind!r} named {name!r}; available: {opts}")
    return bucket[name]


def available(kind: str) -> list[str]:
    return sorted(_REGISTRY.get(kind, {}))


def make(kind: str, name: str, **kwargs):
    """Resolve one internal implementation or report the available names."""
    try:
        cls = get(kind, name)
    except KeyError:
        opts = ", ".join(available(kind)) or "(none registered)"
        raise ValueError(f"no {kind} named {name!r}; available: {opts}") from None
    return cls(**kwargs)
