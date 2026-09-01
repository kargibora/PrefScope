"""Explicit loading for trusted third-party PrefScope plug-ins."""
from __future__ import annotations

import importlib
from collections.abc import Iterable
import re

_MODULE_NAME = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$")


def normalize_plugin_modules(modules: Iterable[str] | None) -> tuple[str, ...]:
    """Validate and deduplicate importable module names while preserving order."""
    if modules is None:
        return ()
    if isinstance(modules, str):
        raise ValueError("plugins must be a list of Python module names, not one string")
    try:
        values = tuple(modules)
    except TypeError as exc:
        raise ValueError("plugins must be a list of Python module names") from exc
    normalized = []
    for index, value in enumerate(values):
        if not isinstance(value, str) or not _MODULE_NAME.fullmatch(value):
            raise ValueError(
                f"plugins[{index}] must be a dotted Python module name; got {value!r}")
        if value not in normalized:
            normalized.append(value)
    return tuple(normalized)


def load_plugins(modules: Iterable[str] | None) -> tuple[str, ...]:
    """Import trusted plug-in modules and return their normalized module names.

    Importing a plug-in executes its Python module. Configuration files must therefore
    name only trusted installed packages. Registration remains explicit and deterministic;
    PrefScope does not scan the environment or import packages automatically.
    """
    normalized = normalize_plugin_modules(modules)
    for module_name in normalized:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            raise ValueError(
                f"failed to import PrefScope plugin {module_name!r}: "
                f"{type(exc).__name__}: {exc}") from exc
    return normalized


__all__ = ["load_plugins", "normalize_plugin_modules"]
