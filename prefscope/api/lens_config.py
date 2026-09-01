"""Strict, small YAML loader for backend-neutral lenses."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import yaml

_COMMON = {"version", "backend", "device"}
_PREFSCOPE = {
    "source",
    "revision",
    "cache_dir",
    "local_files_only",
    "subfolder",
    "annotations",
    "embedding_cache",
    "embed_backend",
    "embed_batch_size",
}
_SAELENS = {
    "release",
    "sae_id",
    "dtype",
    "force_download",
    "sae_batch_size",
    "text_batch_size",
    "max_output_bytes",
    "activation_polarity",
    "reader_model_revision",
    "long_text_policy",
    "include_bos",
    "allow_unregistered_release",
}
_CUSTOM = {"options"}


def _mapping(config):
    if isinstance(config, Mapping):
        return dict(config), Path.cwd()
    path = Path(config).expanduser().resolve()
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, Mapping):
        raise ValueError("lens config root must be a mapping")
    return dict(raw), path.parent


def _boolean(raw: dict, key: str, default: bool = False) -> bool:
    value = raw.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"lens config {key} must be a boolean")
    return value


def _local(value, base: Path):
    if value is None or str(value).startswith("hf://"):
        return value
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else (base / path).resolve())


def load_lens_config(config, *, device: str | None = None):
    """Load one ``Lens`` from a strict mapping or YAML file.

    Built-ins are ``prefscope`` for a published lens directory and ``saelens`` for a
    registered pretrained SAE. Other backend names resolve through the explicit
    ``lens_backend`` registry extension point.
    """
    from prefscope.api.loaded_lens import Lens

    raw, base = _mapping(config)
    version = raw.get("version", 1)
    if type(version) is not int or version != 1:
        raise ValueError(f"unsupported lens config version {version!r}; expected 1")
    backend = raw.get("backend")
    if not isinstance(backend, str) or not backend:
        raise ValueError("lens config needs a non-empty backend")
    backend = backend.casefold()
    allowed = _COMMON | (
        _PREFSCOPE
        if backend == "prefscope"
        else _SAELENS
        if backend == "saelens"
        else _CUSTOM
    )
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(
            f"unknown {backend} lens config key(s): {', '.join(sorted(unknown))}"
        )
    resolved_device = str(device if device is not None else raw.get("device", "cpu"))

    if backend == "prefscope":
        source = raw.get("source")
        if not isinstance(source, str) or not source:
            raise ValueError("prefscope lens config needs source")
        from prefscope import load_lens

        return load_lens(
            _local(source, base),
            device=resolved_device,
            revision=raw.get("revision"),
            cache_dir=_local(raw.get("cache_dir"), base),
            local_files_only=_boolean(raw, "local_files_only"),
            subfolder=raw.get("subfolder"),
            annotations=_local(raw.get("annotations"), base),
            embedding_cache=_local(raw.get("embedding_cache"), base),
            embed_backend=str(raw.get("embed_backend", "hf")),
            embed_batch_size=raw.get("embed_batch_size"),
        )

    if backend == "saelens":
        release = raw.get("release")
        sae_id = raw.get("sae_id")
        if not isinstance(release, str) or not release:
            raise ValueError("saelens lens config needs release")
        if not isinstance(sae_id, str) or not sae_id:
            raise ValueError("saelens lens config needs sae_id")
        return Lens.from_saelens(
            release,
            sae_id,
            device=resolved_device,
            dtype=str(raw.get("dtype", "float32")),
            force_download=_boolean(raw, "force_download"),
            batch_size=raw.get("sae_batch_size", 1024),
            text_batch_size=raw.get("text_batch_size", 8),
            max_output_bytes=raw.get("max_output_bytes", 256 * 1024 * 1024),
            activation_polarity=raw.get("activation_polarity"),
            reader_model_revision=raw.get("reader_model_revision"),
            long_text_policy=str(raw.get("long_text_policy", "truncate")),
            include_bos=_boolean(raw, "include_bos"),
            allow_unregistered_release=_boolean(raw, "allow_unregistered_release"),
        )

    options = raw.get("options", {})
    if not isinstance(options, Mapping):
        raise ValueError("custom lens config options must be a mapping")
    from prefscope.core import registry

    custom_options = dict(options)
    if "device" in custom_options:
        raise ValueError(
            "custom lens device must be set at the config top level, not in options")
    explicit_device = device is not None or "device" in raw
    if explicit_device:
        custom_options["device"] = resolved_device
    custom = registry.make("lens_backend", backend, **custom_options)
    return Lens.from_backend(custom)


__all__ = ["load_lens_config"]
