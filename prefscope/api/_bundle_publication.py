"""Transactional assembly of already-packaged lens children."""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
import shutil
import uuid

from prefscope.api._feature_space import _sha256_file
from prefscope.api._lens_annotations import _PROMPT_ANNOTATIONS, _RESPONSE_ANNOTATIONS
from prefscope.api._lens_publication import _publication_lock, _recover_orphan_backup
from prefscope.api.lens_bundle import (
    BUNDLE_ARTIFACT_TYPE,
    BUNDLE_SCHEMA_VERSION,
    LensBundle,
    _validate_member_path,
)
from prefscope.artifacts import DERIVED_FEATURE_CATALOG, FEATURE_CATALOG, MANIFEST, SAE_MODEL

# save_lens(inference_only=True) writes these files; package-lens may add README.md.
_PUBLIC_CHILD_FILES = {
    MANIFEST, SAE_MODEL, "whiten.npz", "README.md", "LICENSE",
    FEATURE_CATALOG, DERIVED_FEATURE_CATALOG,
    *_PROMPT_ANNOTATIONS, *_RESPONSE_ANNOTATIONS,
}


def _copy_public_child(source: Path, destination: Path) -> None:
    destination.mkdir()
    for item in source.iterdir():
        if item.is_symlink() or not item.is_file() or item.name not in _PUBLIC_CHILD_FILES:
            raise ValueError(f"bundle member contains an unexpected file: {item}")
        shutil.copy2(item, destination / item.name)


def package_bundle(members: Mapping[str, str | Path], dest, *, readme=None,
                   overwrite: bool = False, require_inference: bool = True) -> Path:
    """Atomically assemble validated inference lens directories into one bundle."""
    if not isinstance(members, Mapping) or not members:
        raise ValueError("package_bundle needs a non-empty members mapping")
    dest = Path(dest)
    sources = {}
    for name, path in members.items():
        _validate_member_path(name, name)
        source = Path(path)
        if source.is_symlink() or not source.is_dir():
            raise ValueError(f"bundle member {name!r} is not a real directory")
        sources[name] = source.resolve()
    if dest.is_symlink() or (dest.exists() and not dest.is_dir()):
        raise ValueError("bundle destination must be a directory path")
    dest_resolved = dest.resolve(strict=False)
    for name, source in sources.items():
        if source == dest_resolved or source in dest_resolved.parents or dest_resolved in source.parents:
            raise ValueError("bundle source and destination must not overlap")
        child_manifest = source / MANIFEST
        if not (child_manifest.is_file() and (source / SAE_MODEL).is_file()):
            raise ValueError(f"bundle member {name!r} is missing {MANIFEST} or {SAE_MODEL}")
        if require_inference:
            manifest = json.loads(child_manifest.read_text())
            if manifest.get("artifact_scope") != "inference" or manifest.get("output_arrays") not in (None, []):
                raise ValueError(f"bundle member {name!r} is not an inference-only lens")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with _publication_lock(dest):
        _recover_orphan_backup(dest)
        if dest.exists() and any(dest.iterdir()) and not overwrite:
            raise FileExistsError(f"{dest} exists and is not empty; pass overwrite=True")
        staging = dest.parent / f".{dest.name}.tmp-{uuid.uuid4().hex}"
        backup = dest.parent / f".{dest.name}.bak-{uuid.uuid4().hex}"
        try:
            staging.mkdir()
            descriptors = {}
            for name, source in sources.items():
                child_path = name
                _copy_public_child(source, staging / child_path)
                child = staging / child_path
                descriptors[name] = {
                    "path": child_path,
                    "manifest_sha256": _sha256_file(child / MANIFEST),
                    "weights_sha256": _sha256_file(child / SAE_MODEL),
                }
                if (child / "whiten.npz").is_file():
                    descriptors[name]["whiten_sha256"] = _sha256_file(child / "whiten.npz")
            descriptor = {"bundle_schema_version": BUNDLE_SCHEMA_VERSION, "members": descriptors}
            bundle_id = "sha256:" + hashlib.sha256(
                json.dumps(descriptor, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            ).hexdigest()
            root_manifest = {
                "artifact_type": BUNDLE_ARTIFACT_TYPE,
                **descriptor,
                "bundle_id": bundle_id,
            }
            (staging / MANIFEST).write_text(
                json.dumps(root_manifest, sort_keys=True, indent=2, allow_nan=False) + "\n"
            )
            if readme is not None:
                readme = Path(readme)
                if not readme.is_file():
                    raise FileNotFoundError(readme)
                shutil.copy2(readme, staging / "README.md")
            LensBundle.from_dir(staging)
            if dest.exists():
                os.replace(dest, backup)
            try:
                os.replace(staging, dest)
            except BaseException:
                if backup.exists() and not dest.exists():
                    os.replace(backup, dest)
                raise
            if backup.exists():
                shutil.rmtree(backup)
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
    return dest


__all__ = ["package_bundle"]
