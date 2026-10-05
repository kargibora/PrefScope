"""Validated containers for repositories containing multiple independent lenses."""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator, Mapping
from pathlib import Path, PurePosixPath

from prefscope.api._feature_space import _sha256_file
from prefscope.artifacts import MANIFEST, SAE_MODEL

BUNDLE_ARTIFACT_TYPE = "lens_bundle"
BUNDLE_SCHEMA_VERSION = 1
_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")


def _validate_member_path(name: str, value: str) -> PurePosixPath:
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        raise ValueError(f"invalid lens bundle member name: {name!r}")
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError(f"lens bundle member {name!r} needs a safe relative path")
    if any(ord(char) < 32 for char in value):
        raise ValueError(f"lens bundle member {name!r} has control characters")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"lens bundle member {name!r} has an unsafe path: {value!r}")
    return path


def _member_descriptors(manifest: Mapping) -> dict[str, dict]:
    if not isinstance(manifest, Mapping):
        raise ValueError("lens bundle manifest must be an object")
    if manifest.get("artifact_type") != BUNDLE_ARTIFACT_TYPE:
        raise ValueError("manifest is not a lens bundle")
    if manifest.get("bundle_schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ValueError("unsupported lens bundle schema version")
    members = manifest.get("members")
    if not isinstance(members, Mapping) or not members:
        raise ValueError("lens bundle manifest needs a non-empty members mapping")
    result: dict[str, dict] = {}
    seen_paths: set[str] = set()
    parsed_paths: list[tuple[str, PurePosixPath]] = []
    for name, descriptor in members.items():
        if not isinstance(descriptor, Mapping):
            raise ValueError(f"lens bundle member {name!r} must be an object")
        path = _validate_member_path(name, descriptor.get("path"))
        normalized = path.as_posix()
        if normalized in seen_paths:
            raise ValueError(f"lens bundle member paths must be unique: {normalized}")
        seen_paths.add(normalized)
        parsed_paths.append((str(name), path))
        result[str(name)] = dict(descriptor)
        result[str(name)]["path"] = normalized
    for name, path in parsed_paths:
        if any(path != other and (path in other.parents or other in path.parents)
               for other_name, other in parsed_paths if other_name != name):
            raise ValueError(f"lens bundle member paths overlap: {name!r}")
    return result


def validate_bundle_manifest(manifest: Mapping, root: str | Path) -> dict[str, str]:
    """Validate a bundle manifest and return its safe member paths."""
    descriptors = _member_descriptors(manifest)
    if "bundle_id" in manifest:
        if any(not isinstance(item.get(field), str)
               for item in descriptors.values()
               for field in ("manifest_sha256", "weights_sha256")):
            raise ValueError("identified lens bundle members need manifest and weights hashes")
        descriptor = {"bundle_schema_version": BUNDLE_SCHEMA_VERSION, "members": descriptors}
        expected_id = "sha256:" + hashlib.sha256(
            json.dumps(descriptor, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        if manifest["bundle_id"] != expected_id:
            raise ValueError("lens bundle ID does not match member descriptors")
    root = Path(root).resolve()
    result: dict[str, str] = {}
    folded_names: set[str] = set()
    for name, descriptor in descriptors.items():
        folded = name.casefold()
        if folded in folded_names:
            raise ValueError(f"lens bundle member names collide by case: {name!r}")
        folded_names.add(folded)
        path = descriptor["path"]
        child = root.joinpath(*PurePosixPath(path).parts)
        if child.is_symlink() or not child.is_dir():
            raise ValueError(f"lens bundle member is not a real directory: {name!r}")
        if root not in child.resolve().parents:
            raise ValueError(f"lens bundle member escapes its root: {name!r}")
        child_manifest_path = child / MANIFEST
        child_model_path = child / SAE_MODEL
        whitener = child / "whiten.npz"
        if any(path.is_symlink() for path in (child_manifest_path, child_model_path, whitener)):
            raise ValueError(f"bundle member contains a symlink: {name!r}")
        if not child_manifest_path.is_file() or not child_model_path.is_file():
            raise ValueError(
                f"lens bundle member {name!r} is missing {MANIFEST} or {SAE_MODEL}"
            )
        try:
            child_manifest = json.loads(child_manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid manifest for bundle member {name!r}") from exc
        if child_manifest.get("artifact_type") == BUNDLE_ARTIFACT_TYPE:
            raise ValueError(f"nested lens bundles are not supported: {name!r}")
        expected_hash = descriptor.get("manifest_sha256")
        if expected_hash is not None:
            actual_hash = _sha256_file(child_manifest_path)
            if expected_hash != actual_hash:
                raise ValueError(f"bundle member manifest hash mismatch: {name!r}")
        if "bundle_id" in manifest:
            if descriptor["weights_sha256"] != _sha256_file(child_model_path):
                raise ValueError(f"bundle member checkpoint hash mismatch: {name!r}")
            if whitener.is_file():
                if descriptor.get("whiten_sha256") != _sha256_file(whitener):
                    raise ValueError(f"bundle member whitener hash mismatch: {name!r}")
            elif "whiten_sha256" in descriptor:
                raise ValueError(f"bundle member whitener is missing: {name!r}")
        result[name] = path
    return result


class LensBundle(Mapping[str, object]):
    """A lazy mapping of names to independent :class:`Lens` objects."""

    def __init__(self, root, members: Mapping[str, str], *, manifest: Mapping | None = None,
                 load_kwargs: Mapping | None = None,
                 runtime: Mapping | None = None):
        self.root = Path(root).resolve()
        self._members = dict(members)
        self.manifest = dict(manifest or {})
        self._load_kwargs = dict(load_kwargs or {})
        self._runtime = dict(runtime or {})
        self._cache = {}

    @classmethod
    def from_dir(cls, path, *, device: str = "cpu", annotations=None,
                 derived_catalog=None, validate_arrays: bool = True):
        root = Path(path)
        manifest_path = root / MANIFEST
        if not manifest_path.is_file():
            raise FileNotFoundError(f"lens bundle is missing {MANIFEST}: {root}")
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid lens bundle manifest: {manifest_path}") from exc
        members = validate_bundle_manifest(manifest, root)
        return cls(
            root, members, manifest=manifest,
            load_kwargs={
                "device": device, "annotations": annotations,
                "derived_catalog": derived_catalog,
                "validate_arrays": validate_arrays,
            },
        )

    @classmethod
    def from_pretrained(cls, repo_id: str, *, revision: str | None = None,
                        cache_dir=None, token=None, local_files_only: bool = False,
                        subfolder: str | None = None, device: str = "cpu",
                        annotations=None, derived_catalog=None,
                        validate_arrays: bool = True):
        from prefscope.api.hub import download_bundle, resolve_hf_revision

        resolved_revision = resolve_hf_revision(
            repo_id, revision=revision, repo_type="model", token=token,
            local_files_only=local_files_only,
        )
        root = download_bundle(
            repo_id, revision=resolved_revision, cache_dir=cache_dir, token=token,
            local_files_only=local_files_only, subfolder=subfolder,
            _resolved_revision=resolved_revision,
        )
        bundle = cls.from_dir(
            root, device=device, annotations=annotations,
            derived_catalog=derived_catalog, validate_arrays=validate_arrays,
        )
        bundle._runtime.update({
            "pretrained_repo_id": repo_id,
            "pretrained_revision": resolved_revision,
            "pretrained_requested_revision": revision,
            "pretrained_subfolder": subfolder,
        })
        return bundle

    def __getitem__(self, name: str):
        if name not in self._members:
            raise KeyError(f"unknown lens {name!r}; available: {list(self._members)}")
        if name not in self._cache:
            from prefscope.api.loaded_lens import Lens

            kwargs = dict(self._load_kwargs)
            annotations = kwargs.get("annotations")
            if isinstance(annotations, Mapping):
                kwargs["annotations"] = annotations.get(name)
            derived_catalog = kwargs.get("derived_catalog")
            if isinstance(derived_catalog, Mapping):
                kwargs["derived_catalog"] = derived_catalog.get(name)
            lens = Lens.from_dir(self.root / self._members[name], **kwargs)
            if self._runtime:
                lens.pretrained_repo_id = self._runtime.get("pretrained_repo_id")
                lens.pretrained_revision = self._runtime.get("pretrained_revision")
                lens.pretrained_requested_revision = self._runtime.get(
                    "pretrained_requested_revision")
                lens.pretrained_subfolder = self._members[name]
                lens.bundle_member = name
                lens.bundle_root = str(self.root)
                lens.bundle_id = self.manifest.get("bundle_id")
            self._cache[name] = lens
        return self._cache[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._members)

    def __len__(self) -> int:
        return len(self._members)

    @property
    def prompt(self):
        return self["prompt"]

    @property
    def completion(self):
        return self["completion"]

    def __repr__(self) -> str:
        return f"LensBundle(root={str(self.root)!r}, lenses={list(self._members)!r})"


__all__ = ["BUNDLE_ARTIFACT_TYPE", "BUNDLE_SCHEMA_VERSION", "LensBundle", "validate_bundle_manifest"]
