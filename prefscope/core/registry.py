"""String-name -> class registry, keyed by component kind.

Adapters self-register on import via @register(kind, name). The Lens facade and
config resolution look classes up by name; power users bypass this and pass
instances directly.
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
    """Resolve and instantiate a registered component (the config-driven entry point).

    Raises ``ValueError`` listing the available names if ``name`` is unknown — the
    conventional error for a bad config/CLI value (vs. ``get``'s ``KeyError``). This is
    the single resolver every pipeline stage (interpreter, verifier, clusterer, …) and
    the YAML config runner go through, so a typo names its alternatives once, here."""
    try:
        cls = get(kind, name)
    except KeyError:
        opts = ", ".join(available(kind)) or "(none registered)"
        raise ValueError(f"no {kind} named {name!r}; available: {opts}") from None
    return cls(**kwargs)
