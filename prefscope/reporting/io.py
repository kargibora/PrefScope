"""Transactional I/O for strict PrefScope report bundle v3 directories."""
from __future__ import annotations

from dataclasses import dataclass, field
from contextvars import ContextVar
import ctypes
import errno
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tempfile
from typing import BinaryIO, Mapping
import unicodedata
import uuid

from prefscope.api._lens_publication import (
    _publication_lock,
    _recover_orphan_backup,
)
from prefscope.observability.runtime import automatic_stage
from prefscope.reporting.contracts import (
    ArtifactPrivacy,
    ArtifactStatus,
    ReportArtifact,
    ReportManifest,
    parse_json_table,
)
from prefscope.reporting.privacy import (
    PrivacyPolicy,
    PrivacyProfile,
    sanitize_json,
    validate_html_neutral_snippet,
)

MANIFEST_FILENAME = "bundle_manifest.json"
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_JSON_DEPTH = 64
MAX_ARTIFACT_COUNT = 10_000
MAX_PATH_COMPONENTS = 32
MAX_BUNDLE_DIRECTORIES = 50_000
_COPY_CHUNK_BYTES = 1024 * 1024
_JSON_MEDIA = frozenset({"application/json", "application/vnd.prefscope+json"})
_WINDOWS_RESERVED = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{number}" for number in range(1, 10)}
    | {f"lpt{number}" for number in range(1, 10)}
)
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_SUPPRESS_LOAD_OBSERVATION: ContextVar[bool] = ContextVar(
    "prefscope_report_bundle_suppress_load_observation", default=False
)


@dataclass(frozen=True)
class PathPayload:
    """An explicit, path-backed payload for a LOCAL non-JSON artifact."""

    path: Path

    def __init__(self, path: str | os.PathLike[str]) -> None:
        object.__setattr__(self, "path", Path(path))


@dataclass(frozen=True)
class _PreparedPayload:
    artifact: ReportArtifact
    data: bytes | None = None
    source: Path | None = None


class _StrictJsonError(ValueError):
    pass


def _reject_constant(value: str) -> object:
    raise _StrictJsonError(f"JSON contains non-portable numeric constant {value}")


def _parse_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise _StrictJsonError("JSON contains a non-finite number")
    return number


def _no_duplicate_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _StrictJsonError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _check_json_depth(value: object) -> None:
    pending = [(value, 1)]
    while pending:
        item, depth = pending.pop()
        if depth > MAX_JSON_DEPTH:
            raise ValueError(
                f"JSON exceeds the maximum nesting depth of {MAX_JSON_DEPTH}")
        if isinstance(item, Mapping):
            pending.extend((nested, depth + 1) for nested in item.values())
        elif isinstance(item, (list, tuple)):
            pending.extend((nested, depth + 1) for nested in item)


def _load_json_bytes(value: bytes, *, where: str, require_canonical: bool = True):
    if not isinstance(value, bytes):
        raise ValueError(f"{where} must be bytes")
    if len(value) > MAX_JSON_BYTES:
        raise ValueError(f"{where} exceeds the {MAX_JSON_BYTES}-byte JSON limit")
    try:
        text = value.decode("utf-8")
        decoded = json.loads(
            text,
            parse_constant=_reject_constant,
            parse_float=_parse_float,
            object_pairs_hook=_no_duplicate_object,
        )
        _check_json_depth(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, _StrictJsonError) as exc:
        raise ValueError(f"{where} is not strict portable UTF-8 JSON") from exc
    if require_canonical and canonical_json_bytes(decoded) != value:
        raise ValueError(f"{where} is not canonically encoded JSON")
    return decoded


def canonical_json_bytes(
    value: object, *, privacy_policy: PrivacyPolicy | None = None
) -> bytes:
    """Encode deterministic UTF-8 JSON, sanitizing missing values when requested."""
    if privacy_policy is not None:
        if not isinstance(privacy_policy, PrivacyPolicy):
            raise ValueError("privacy_policy must be a PrivacyPolicy or null")
        value = sanitize_json(value, privacy_policy)
    _check_json_depth(value)
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        encoded = (text + "\n").encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError, RecursionError) as exc:
        raise ValueError("report JSON must contain portable finite JSON values") from exc
    if len(encoded) > MAX_JSON_BYTES:
        raise ValueError(f"report JSON exceeds the {MAX_JSON_BYTES}-byte limit")
    return encoded


def json_payload(value: object, *, privacy_policy: PrivacyPolicy) -> bytes:
    """Sanitize and canonically encode one JSON artifact payload."""
    if not isinstance(privacy_policy, PrivacyPolicy):
        raise ValueError("privacy_policy must be a PrivacyPolicy")
    return canonical_json_bytes(value, privacy_policy=privacy_policy)


def _open_regular_path(path: Path) -> int:
    try:
        descriptor = os.open(path, os.O_RDONLY | _NOFOLLOW)
    except OSError as exc:
        raise ValueError(f"payload path is not a readable regular file: {path}") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"payload path is not a regular file: {path}")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _hash_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    while True:
        block = os.read(descriptor, _COPY_CHUNK_BYTES)
        if not block:
            break
        digest.update(block)
    return digest.hexdigest()


def artifact_sha256(value: bytes | bytearray | memoryview | PathPayload) -> str:
    """Return a SHA-256 digest without loading a path-backed payload into memory."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return hashlib.sha256(bytes(value)).hexdigest()
    if isinstance(value, PathPayload):
        descriptor = _open_regular_path(value.path)
        try:
            return _hash_descriptor(descriptor)
        finally:
            os.close(descriptor)
    raise ValueError("artifact payload must be bytes or a PathPayload")


def _is_json_media(media_type: str | None) -> bool:
    return bool(
        media_type
        and (media_type in _JSON_MEDIA or media_type.endswith("+json"))
    )


def _safe_relative_path(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or len(value) > 4096 or "\\" in value:
        raise ValueError("artifact path must be a non-empty portable POSIX path")
    if unicodedata.normalize("NFC", value) != value:
        raise ValueError("artifact path has ambiguous Unicode normalization")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("artifact path components must contain ASCII only") from exc
    if value.startswith("/"):
        raise ValueError(f"unsafe report artifact path {value!r}")
    raw_parts = value.split("/")
    if len(raw_parts) > MAX_PATH_COMPONENTS:
        raise ValueError(
            f"artifact path exceeds the {MAX_PATH_COMPONENTS}-component limit")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise ValueError(f"unsafe report artifact path {value!r}")
    for part in raw_parts:
        if len(part) > 255 or part[-1] in {".", " "}:
            raise ValueError(f"non-portable report artifact path {value!r}")
        if part[0] == "." or any(
            ord(char) < 0x20 or ord(char) == 0x7F or char == ":" for char in part
        ):
            raise ValueError(f"non-portable report artifact path {value!r}")
        if not all(char.isalnum() or char in "._-" for char in part):
            raise ValueError(f"non-portable report artifact path {value!r}")
        if part.split(".", 1)[0].casefold() in _WINDOWS_RESERVED:
            raise ValueError(f"Windows-reserved report artifact path {value!r}")
        if part.casefold() == MANIFEST_FILENAME.casefold():
            raise ValueError("artifact path collides with the report manifest")
    return PurePosixPath(*raw_parts)


def _artifact_paths(manifest: ReportManifest) -> dict[str, PurePosixPath]:
    ready = [item for item in manifest.artifacts if item.status is ArtifactStatus.READY]
    if len(manifest.artifacts) > MAX_ARTIFACT_COUNT:
        raise ValueError(
            f"report manifest exceeds the {MAX_ARTIFACT_COUNT}-artifact limit")
    folded_paths: dict[str, PurePosixPath] = {}
    for artifact in ready:
        assert artifact.path is not None
        path = _safe_relative_path(artifact.path)
        folded = path.as_posix().casefold()
        if folded in folded_paths:
            raise ValueError("report artifact paths must be case-insensitively unique")
        folded_paths[folded] = path
    # Each path has a bounded number of components. Set lookups avoid the previous
    # quadratic all-pairs prefix scan while catching file/directory collisions.
    folded_set = set(folded_paths)
    for folded, path in folded_paths.items():
        parts = folded.split("/")
        if any("/".join(parts[:index]) in folded_set for index in range(1, len(parts))):
            raise ValueError("report artifact paths contain a file/directory collision")
    return {path.as_posix(): path for path in folded_paths.values()}


def _effective_policy(manifest: ReportManifest) -> PrivacyPolicy:
    return PrivacyPolicy.from_manifest(manifest.privacy)


def _validate_artifact_privacy(
    artifact: ReportArtifact, decoded: object, policy: PrivacyPolicy
) -> None:
    """Apply the artifact tier as a further restriction on the report policy."""
    no_ids = artifact.privacy in {
        ArtifactPrivacy.PUBLIC,
        ArtifactPrivacy.AGGREGATE,
    }
    no_text = artifact.privacy in {
        ArtifactPrivacy.PUBLIC,
        ArtifactPrivacy.AGGREGATE,
        ArtifactPrivacy.OPAQUE_ROWS,
    }
    snippets = artifact.privacy is ArtifactPrivacy.TEXT_SNIPPETS

    def visit(item: object, field_name: str | None = None) -> None:
        if item is None:
            return
        if field_name in policy.id_fields and no_ids:
            raise ValueError(
                f"artifact privacy {artifact.privacy.value!r} does not permit row IDs")
        if field_name in policy.text_fields:
            if no_text:
                raise ValueError(
                    f"artifact privacy {artifact.privacy.value!r} does not permit text")
            if snippets:
                validate_html_neutral_snippet(
                    item,
                    where="text-snippet artifact field",
                    max_chars=policy.snippet_chars,
                )
        if isinstance(item, Mapping):
            for key, nested in item.items():
                visit(nested, key)
        elif isinstance(item, list):
            for nested in item:
                visit(nested, field_name)

    visit(decoded)


def _validate_json_artifact(
    artifact: ReportArtifact, decoded: object, policy: PrivacyPolicy
) -> None:
    policy.validate_sanitized(decoded)
    if artifact.table_contract is not None:
        parse_json_table(
            decoded,
            expected_contract=artifact.table_contract,
            privacy=policy,
        )
        # Table schema metadata can legitimately contain a column whose name is an ID
        # or text field. Apply the artifact tier to records, not to schema declarations.
        assert isinstance(decoded, Mapping)
        _validate_artifact_privacy(artifact, decoded["records"], policy)
    else:
        _validate_artifact_privacy(artifact, decoded, policy)


def _normalize_payloads(
    manifest: ReportManifest, values: Mapping[str, object]
) -> dict[str, _PreparedPayload]:
    if not isinstance(values, Mapping):
        raise ValueError("artifacts must be a mapping")
    ready = [item for item in manifest.artifacts if item.status is ArtifactStatus.READY]
    for artifact in ready:
        if artifact.table_contract is not None and not _is_json_media(artifact.media_type):
            raise ValueError("table_contract is supported only for ready JSON artifacts")
    by_id = {item.artifact_id: item for item in ready}
    by_path = {item.path: item for item in ready}
    supplied = dict(values)
    keys = set(supplied)
    id_keys = set(by_id)
    path_keys = set(by_path)
    if keys == id_keys == path_keys:
        if any(item.artifact_id != item.path for item in ready):
            raise ValueError("artifact payload keys are ambiguous between IDs and paths")
        keyed = {item.artifact_id: supplied[item.artifact_id] for item in ready}
    elif keys == id_keys:
        keyed = {item.artifact_id: supplied[item.artifact_id] for item in ready}
    elif keys == path_keys:
        keyed = {item.artifact_id: supplied[item.path] for item in ready}
    else:
        raise ValueError(
            "artifact payload keys must exactly match ready artifact IDs or paths")

    policy = _effective_policy(manifest)
    prepared: dict[str, _PreparedPayload] = {}
    for artifact in ready:
        assert artifact.path is not None
        assert artifact.media_type is not None
        assert artifact.sha256 is not None
        value = keyed[artifact.artifact_id]
        if _is_json_media(artifact.media_type):
            if isinstance(value, PathPayload):
                raise ValueError("JSON artifacts must be supplied as values or canonical bytes")
            if isinstance(value, (bytes, bytearray, memoryview)):
                payload = bytes(value)
                decoded = _load_json_bytes(
                    payload, where=f"artifact {artifact.artifact_id!r}")
            else:
                # Object payloads are already-sanitized values (for example, the
                # output of table_to_json_table). Re-sanitizing can double-hash opaque
                # IDs and double-escape HTML entities. Raw values must go through
                # json_payload() before they are supplied to this writer.
                payload = canonical_json_bytes(value)
                decoded = _load_json_bytes(
                    payload, where=f"artifact {artifact.artifact_id!r}")
            _validate_json_artifact(artifact, decoded, policy)
            if artifact_sha256(payload) != artifact.sha256:
                raise ValueError(
                    f"artifact {artifact.artifact_id!r} payload does not match sha256")
            prepared[artifact.path] = _PreparedPayload(artifact, data=payload)
            continue
        if policy.profile_name is PrivacyProfile.SHAREABLE:
            raise ValueError("Phase 1 SHAREABLE bundles may contain only canonical JSON")
        if artifact.table_contract is not None:
            raise ValueError("table_contract is not supported for non-JSON media")
        if isinstance(value, (bytes, bytearray, memoryview)):
            payload = bytes(value)
            if artifact_sha256(payload) != artifact.sha256:
                raise ValueError(
                    f"artifact {artifact.artifact_id!r} payload does not match sha256")
            prepared[artifact.path] = _PreparedPayload(artifact, data=payload)
        elif isinstance(value, PathPayload):
            prepared[artifact.path] = _PreparedPayload(artifact, source=value.path)
        else:
            raise ValueError(
                "LOCAL non-JSON payloads must be bytes or an explicit PathPayload")
    return prepared


def _write_all(handle: BinaryIO, value: bytes) -> None:
    offset = 0
    while offset < len(value):
        written = handle.write(value[offset:])
        if written is None or written <= 0:
            raise OSError("short write while staging report artifact")
        offset += written


def _write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        _write_all(handle, value)
        handle.flush()
        os.fsync(handle.fileno())


def _copy_path_payload(path: Path, source: Path, expected_sha256: str) -> None:
    descriptor = _open_regular_path(source)
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    try:
        with path.open("xb") as target:
            while True:
                block = os.read(descriptor, _COPY_CHUNK_BYTES)
                if not block:
                    break
                digest.update(block)
                _write_all(target, block)
            target.flush()
            os.fsync(target.fileno())
    finally:
        os.close(descriptor)
    if digest.hexdigest() != expected_sha256:
        raise ValueError("path-backed artifact payload does not match sha256")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | _DIRECTORY | _NOFOLLOW)
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise ValueError(f"expected a directory while syncing {path}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree_directories(root: Path) -> None:
    directories = [path for path in root.rglob("*") if path.is_dir()]
    for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        _fsync_directory(path)
    _fsync_directory(root)


def _open_bundle_root(root: Path) -> int:
    try:
        before = root.lstat()
    except OSError as exc:
        raise ValueError("report bundle path must be an existing directory") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
        raise ValueError("report bundle path must be a non-symlink directory")
    try:
        descriptor = os.open(root, os.O_RDONLY | _DIRECTORY | _NOFOLLOW)
    except OSError as exc:
        raise ValueError("report bundle directory could not be opened safely") from exc
    opened = os.fstat(descriptor)
    if not stat.S_ISDIR(opened.st_mode) or (before.st_dev, before.st_ino) != (
        opened.st_dev,
        opened.st_ino,
    ):
        os.close(descriptor)
        raise ValueError("report bundle root changed while it was opened")
    return descriptor


def _descriptor_identity(descriptor: int) -> tuple[int, int]:
    info = os.fstat(descriptor)
    return info.st_dev, info.st_ino


def _verify_root_path_identity(root: Path, expected: tuple[int, int]) -> None:
    try:
        info = root.lstat()
    except OSError as exc:
        raise ValueError("report bundle root disappeared after validation") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode) or (
        info.st_dev,
        info.st_ino,
    ) != expected:
        raise ValueError("report bundle root changed after validation")


def _open_relative_regular(root_descriptor: int, relative: PurePosixPath) -> int:
    current = os.dup(root_descriptor)
    try:
        for part in relative.parts[:-1]:
            next_descriptor = os.open(
                part,
                os.O_RDONLY | _DIRECTORY | _NOFOLLOW,
                dir_fd=current,
            )
            if not stat.S_ISDIR(os.fstat(next_descriptor).st_mode):
                os.close(next_descriptor)
                raise ValueError("artifact path crosses a non-directory entry")
            os.close(current)
            current = next_descriptor
        descriptor = os.open(
            relative.parts[-1], os.O_RDONLY | _NOFOLLOW, dir_fd=current)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"report member {relative.as_posix()!r} cannot be opened safely") from exc
    finally:
        os.close(current)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ValueError(f"report member {relative.as_posix()!r} is not regular")
    return descriptor


def _read_descriptor_limited(descriptor: int, *, limit: int, where: str) -> bytes:
    chunks = []
    total = 0
    while True:
        block = os.read(descriptor, min(_COPY_CHUNK_BYTES, limit + 1 - total))
        if not block:
            break
        chunks.append(block)
        total += len(block)
        if total > limit:
            raise ValueError(f"{where} exceeds the {limit}-byte limit")
    return b"".join(chunks)


def _read_relative_json(root_descriptor: int, relative: PurePosixPath, *, where: str):
    descriptor = _open_relative_regular(root_descriptor, relative)
    try:
        value = _read_descriptor_limited(
            descriptor, limit=MAX_JSON_BYTES, where=where)
    finally:
        os.close(descriptor)
    return value, _load_json_bytes(value, where=where)


def _hash_relative(root_descriptor: int, relative: PurePosixPath) -> str:
    descriptor = _open_relative_regular(root_descriptor, relative)
    try:
        return _hash_descriptor(descriptor)
    finally:
        os.close(descriptor)


def _expected_directories(paths: Mapping[str, PurePosixPath]) -> set[str]:
    result = set()
    for path in paths.values():
        parent = path.parent
        while parent != PurePosixPath("."):
            result.add(parent.as_posix())
            if len(result) > MAX_BUNDLE_DIRECTORIES:
                raise ValueError(
                    f"report bundle exceeds the {MAX_BUNDLE_DIRECTORIES}-directory limit")
            parent = parent.parent
    return result


def _directory_entries(
    root_descriptor: int, *, expected_directories: set[str]
) -> set[str]:
    """Inventory from the verified root fd without following path-based replacements."""
    files: set[str] = set()
    directory_count = 0

    def inventory_entry(directory_descriptor, prefix, entry) -> None:
        nonlocal directory_count
        relative = "/".join((*prefix, entry.name))
        try:
            before = entry.stat(follow_symlinks=False)
        except OSError as exc:
            raise ValueError(
                f"report bundle member changed during inventory: {relative!r}") from exc
        if stat.S_ISLNK(before.st_mode):
            raise ValueError(
                f"report bundle must not contain symlinks: {relative!r}")
        if stat.S_ISDIR(before.st_mode):
            directory_count += 1
            if directory_count > MAX_BUNDLE_DIRECTORIES:
                raise ValueError("report bundle exceeds the aggregate directory limit")
            if relative not in expected_directories:
                raise ValueError(f"report bundle contains stale directory {relative!r}")
            try:
                child = os.open(
                    entry.name,
                    os.O_RDONLY | _DIRECTORY | _NOFOLLOW,
                    dir_fd=directory_descriptor,
                )
            except OSError as exc:
                raise ValueError(
                    f"report directory changed during inventory: {relative!r}") from exc
            try:
                opened = os.fstat(child)
                if not stat.S_ISDIR(opened.st_mode) or (
                    before.st_dev,
                    before.st_ino,
                ) != (opened.st_dev, opened.st_ino):
                    raise ValueError(
                        f"report directory changed during inventory: {relative!r}")
                visit(child, (*prefix, entry.name))
            finally:
                os.close(child)
        elif stat.S_ISREG(before.st_mode):
            files.add(relative)
        else:
            raise ValueError(
                f"report bundle contains a non-regular member: {relative!r}")

    def visit(directory_descriptor: int, prefix: tuple[str, ...]) -> None:
        try:
            entries = os.scandir(directory_descriptor)
        except OSError as exc:
            raise ValueError("report bundle directory could not be inventoried") from exc
        with entries:
            for entry in entries:
                inventory_entry(directory_descriptor, prefix, entry)

    visit(root_descriptor, ())
    return files


@dataclass(frozen=True)
class ReportBundle:
    """A validated on-disk report bundle and its typed manifest."""

    root: Path
    manifest: ReportManifest
    _root_identity: tuple[int, int] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "root", Path(os.path.abspath(os.fspath(self.root))))
        if not isinstance(self.manifest, ReportManifest):
            raise ValueError("bundle manifest must be a ReportManifest")
        if (
            not isinstance(self._root_identity, tuple)
            or len(self._root_identity) != 2
            or not all(isinstance(item, int) for item in self._root_identity)
        ):
            raise ValueError("bundle root identity must contain device and inode")
        descriptor = _open_bundle_root(self.root)
        try:
            if _descriptor_identity(descriptor) != self._root_identity:
                raise ValueError("report bundle root changed before it was returned")
        finally:
            os.close(descriptor)

    def _open_root(self) -> int:
        descriptor = _open_bundle_root(self.root)
        if _descriptor_identity(descriptor) != self._root_identity:
            os.close(descriptor)
            raise ValueError("report bundle root was replaced after loading")
        return descriptor

    def artifact(self, artifact_id: str) -> ReportArtifact:
        return self.manifest.artifact(artifact_id)

    def read_bytes(self, artifact_id: str) -> bytes:
        artifact = self.artifact(artifact_id)
        if artifact.status is not ArtifactStatus.READY or artifact.path is None:
            raise ValueError(f"report artifact {artifact_id!r} has no ready payload")
        relative = _safe_relative_path(artifact.path)
        root_descriptor = self._open_root()
        try:
            descriptor = _open_relative_regular(root_descriptor, relative)
            try:
                chunks = []
                digest = hashlib.sha256()
                while True:
                    block = os.read(descriptor, _COPY_CHUNK_BYTES)
                    if not block:
                        break
                    chunks.append(block)
                    digest.update(block)
            finally:
                os.close(descriptor)
            _verify_root_path_identity(self.root, self._root_identity)
        finally:
            os.close(root_descriptor)
        if digest.hexdigest() != artifact.sha256:
            raise ValueError(f"report artifact {artifact_id!r} is corrupt")
        return b"".join(chunks)

    def read_json(self, artifact_id: str):
        artifact = self.artifact(artifact_id)
        if artifact.status is not ArtifactStatus.READY or artifact.path is None:
            raise ValueError(f"report artifact {artifact_id!r} has no ready payload")
        if not _is_json_media(artifact.media_type):
            raise ValueError(f"report artifact {artifact_id!r} is not JSON media")
        root_descriptor = self._open_root()
        try:
            value, decoded = _read_relative_json(
                root_descriptor,
                _safe_relative_path(artifact.path),
                where=f"artifact {artifact_id!r}",
            )
            _verify_root_path_identity(self.root, self._root_identity)
        finally:
            os.close(root_descriptor)
        if artifact_sha256(value) != artifact.sha256:
            raise ValueError(f"report artifact {artifact_id!r} is corrupt")
        policy = _effective_policy(self.manifest)
        _validate_json_artifact(artifact, decoded, policy)
        return decoded


def _load_report_bundle(
    directory: str | os.PathLike[str],
    *,
    verify_hashes: bool = True,
    reject_stale: bool = True,
) -> ReportBundle:
    """Load and strictly validate one complete report bundle v3 directory."""
    if not isinstance(verify_hashes, bool) or not isinstance(reject_stale, bool):
        raise ValueError("verify_hashes and reject_stale must be booleans")
    # abspath, unlike resolve(), preserves the supplied final component so lstat can
    # still reject a symlink while making the returned bundle independent of cwd.
    root = Path(os.path.abspath(os.fspath(directory)))
    root_descriptor = _open_bundle_root(root)
    root_identity = _descriptor_identity(root_descriptor)
    try:
        manifest_bytes, raw = _read_relative_json(
            root_descriptor,
            PurePosixPath(MANIFEST_FILENAME),
            where="report manifest",
        )
        del manifest_bytes
        if not isinstance(raw, Mapping):
            raise ValueError("report manifest must be a JSON object")
        artifacts_raw = raw.get("artifacts")
        if not isinstance(artifacts_raw, list):
            raise ValueError("report manifest artifacts must be an array")
        if len(artifacts_raw) > MAX_ARTIFACT_COUNT:
            raise ValueError(
                f"report manifest exceeds the {MAX_ARTIFACT_COUNT}-artifact limit")
        try:
            manifest = ReportManifest.from_dict(raw)
        except (TypeError, KeyError, ValueError) as exc:
            raise ValueError("report manifest does not satisfy the v3 schema") from exc
        paths = _artifact_paths(manifest)
        policy = _effective_policy(manifest)
        for artifact in manifest.artifacts:
            if (
                artifact.status is ArtifactStatus.READY
                and artifact.table_contract is not None
                and not _is_json_media(artifact.media_type)
            ):
                raise ValueError(
                    "table_contract is supported only for ready JSON artifacts")
        expected = {MANIFEST_FILENAME, *paths}
        expected_directories = _expected_directories(paths)
        if reject_stale:
            observed = _directory_entries(
                root_descriptor,
                expected_directories=expected_directories,
            )
            if observed != expected:
                raise ValueError(
                    "report bundle file set mismatch; "
                    f"missing={sorted(expected - observed)}, "
                    f"stale={sorted(observed - expected)}"
                )
        for artifact in manifest.artifacts:
            if artifact.status is not ArtifactStatus.READY:
                continue
            assert artifact.path is not None
            assert artifact.sha256 is not None
            assert artifact.media_type is not None
            relative = paths[artifact.path]
            if _is_json_media(artifact.media_type):
                payload, decoded = _read_relative_json(
                    root_descriptor,
                    relative,
                    where=f"artifact {artifact.artifact_id!r}",
                )
                if verify_hashes and artifact_sha256(payload) != artifact.sha256:
                    raise ValueError(
                        f"report artifact {artifact.artifact_id!r} is corrupt")
                _validate_json_artifact(artifact, decoded, policy)
            else:
                if policy.profile_name is PrivacyProfile.SHAREABLE:
                    raise ValueError(
                        "Phase 1 SHAREABLE bundles may contain only canonical JSON")
                if verify_hashes:
                    actual = _hash_relative(root_descriptor, relative)
                    if actual != artifact.sha256:
                        raise ValueError(
                            f"report artifact {artifact.artifact_id!r} is corrupt")
                else:
                    descriptor = _open_relative_regular(root_descriptor, relative)
                    os.close(descriptor)
        if _descriptor_identity(root_descriptor) != root_identity:
            raise ValueError("report bundle root descriptor changed during validation")
        _verify_root_path_identity(root, root_identity)
    finally:
        os.close(root_descriptor)
    return ReportBundle(
        root=root, manifest=manifest, _root_identity=root_identity)


def _rename_no_replace(source: Path, destination: Path) -> None:
    """Rename a staged directory only if the destination is still absent."""
    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform == "darwin":
        rename = getattr(libc, "renamex_np", None)
        if rename is None:
            error = errno.ENOSYS
            result = -1
        else:
            rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
            rename.restype = ctypes.c_int
            result = rename(source_bytes, destination_bytes, 0x00000004)  # RENAME_EXCL
            error = ctypes.get_errno()
    elif sys.platform.startswith("linux"):
        rename = getattr(libc, "renameat2", None)
        if rename is None:
            error = errno.ENOSYS
            result = -1
        else:
            rename.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            ]
            rename.restype = ctypes.c_int
            result = rename(
                -100, source_bytes, -100, destination_bytes, 0x1)  # RENAME_NOREPLACE
            error = ctypes.get_errno()
    else:
        error = errno.ENOSYS
        result = -1
    if result == 0:
        return
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error,
            "report bundle destination appeared during staged installation",
            destination,
        )
    raise OSError(error, os.strerror(error), destination)


def _remove_staging(path: Path) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(info.st_mode):
        path.unlink()
    elif stat.S_ISDIR(info.st_mode):
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink()


def _quarantine_unexpected(destination: Path) -> Path | None:
    if not (destination.exists() or destination.is_symlink()):
        return None
    quarantine = destination.parent / (
        f".{destination.name}.quarantine-{uuid.uuid4().hex}")
    os.replace(destination, quarantine)
    _fsync_directory(destination.parent)
    return quarantine


def _publish_locked(
    destination: Path,
    manifest: ReportManifest,
    payloads: Mapping[str, _PreparedPayload],
    *,
    overwrite: bool,
) -> ReportBundle:
    parent = destination.parent
    exists_initially = destination.exists() or destination.is_symlink()
    if exists_initially:
        if destination.is_symlink() or not destination.is_dir():
            raise ValueError("existing report destination must be a non-symlink directory")
        if not overwrite:
            raise FileExistsError(
                f"report bundle destination already exists: {destination}")
        # Never replace an arbitrary directory. This also checks every old digest.
        _load_report_bundle_for_validation(destination)
        old_info = destination.lstat()
        old_identity = (old_info.st_dev, old_info.st_ino)
    else:
        old_identity = None

    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=parent))
    backup = parent / f".{destination.name}.bak-{uuid.uuid4().hex}"
    moved_old = False
    installed_new = False
    completed = False
    try:
        for relative, prepared in payloads.items():
            target = staging.joinpath(*_safe_relative_path(relative).parts)
            if prepared.data is not None:
                _write_bytes(target, prepared.data)
            else:
                assert prepared.source is not None
                assert prepared.artifact.sha256 is not None
                _copy_path_payload(
                    target, prepared.source, prepared.artifact.sha256)
        # Completion marker: it is intentionally the final staged file.
        _write_bytes(
            staging / MANIFEST_FILENAME,
            canonical_json_bytes(manifest.to_dict()),
        )
        _fsync_tree_directories(staging)
        _load_report_bundle_for_validation(staging)

        # An uncooperative writer can ignore our lock. Never move a destination that
        # appeared during staging in no-overwrite mode.
        if destination.exists() or destination.is_symlink():
            if not overwrite:
                raise FileExistsError(
                    f"report bundle destination appeared during staging: {destination}")
            if not exists_initially:
                raise FileExistsError(
                    f"unexpected report destination appeared during staging: {destination}")
            # Validate again immediately before the destructive rename. The lock is
            # advisory; an actor that ignored it may have replaced or mutated the old
            # bundle while the new one was staged.
            _load_report_bundle_for_validation(destination)
            current_info = destination.lstat()
            if (current_info.st_dev, current_info.st_ino) != old_identity:
                raise RuntimeError(
                    "existing report destination changed during staging")
            os.replace(destination, backup)
            moved_old = True
            _fsync_directory(parent)
        elif exists_initially:
            raise RuntimeError("existing report destination disappeared during publication")

        # Overwrite has a visible old->backup gap and is not linearizable. A writer
        # that ignores the owner lock can fill that gap. No-replace installation must
        # leave such an occupant untouched and retain the recoverable old backup.
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                "unexpected destination appeared after backup was created")
        _rename_no_replace(staging, destination)
        installed_new = True
        _fsync_directory(parent)
        result = _load_report_bundle_for_validation(destination)
        completed = True
        if moved_old:
            shutil.rmtree(backup)
            _fsync_directory(parent)
        return result
    except BaseException as exc:
        destination_collision = isinstance(exc, FileExistsError)
        if not completed and not destination_collision:
            # For failures other than a no-replace collision, quarantine a partial new
            # tree and put the prior valid bundle back when possible.
            if installed_new or (moved_old and (destination.exists() or destination.is_symlink())):
                try:
                    _quarantine_unexpected(destination)
                except BaseException:
                    pass
            if moved_old and backup.exists() and not (
                destination.exists() or destination.is_symlink()
            ):
                try:
                    os.replace(backup, destination)
                    _fsync_directory(parent)
                    moved_old = False
                except BaseException:
                    # The .bak-* path is the explicit recovery record. Never remove it.
                    pass
        # A no-replace collision is different: never move or delete the late occupant.
        # If old->backup already happened, retain that backup for explicit recovery.
        raise
    finally:
        _remove_staging(staging)
        # Deliberately do not remove backup here. On every exceptional path it is
        # either restored or remains as a recoverable .bak-* directory.


def _write_report_bundle(
    directory: str | os.PathLike[str],
    manifest: ReportManifest,
    artifacts: Mapping[str, object],
    *,
    overwrite: bool = False,
) -> ReportBundle:
    """Stage, validate, and transactionally install a complete report bundle v3.

    Final installation is no-replace. Overwrite uses an old-to-backup gap and is not
    linearizable against writers that ignore the owner lock.
    """
    if not isinstance(manifest, ReportManifest):
        raise ValueError("manifest must be a ReportManifest")
    if not isinstance(overwrite, bool):
        raise ValueError("overwrite must be a boolean")
    _artifact_paths(manifest)
    payloads = _normalize_payloads(manifest, artifacts)
    raw_destination = Path(directory)
    if raw_destination.name in {"", ".", ".."}:
        raise ValueError("report bundle destination must name a directory")
    destination = Path(os.path.abspath(os.fspath(raw_destination)))
    parent = destination.parent
    if parent.exists():
        info = parent.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ValueError("report bundle parent must be a non-symlink directory")
    else:
        missing = []
        cursor = parent
        while not cursor.exists():
            missing.append(cursor)
            cursor = cursor.parent
        cursor_info = cursor.lstat()
        if stat.S_ISLNK(cursor_info.st_mode) or not stat.S_ISDIR(cursor_info.st_mode):
            raise ValueError("report bundle parent ancestry must be a directory")
        parent.mkdir(parents=True)
        # Persist every new directory entry, including the entry for the immediate
        # report parent in its containing directory.
        for created in reversed(missing):
            _fsync_directory(created)
            _fsync_directory(created.parent)
    with _publication_lock(destination):
        _recover_orphan_backup(destination)
        return _publish_locked(
            destination,
            manifest,
            payloads,
            overwrite=overwrite,
        )


def _report_bundle_observation(manifest: ReportManifest) -> dict[str, object]:
    capabilities = manifest.capabilities
    policy = _effective_policy(manifest)
    return {
        "n_rows": capabilities.n_rows,
        "n_groups": capabilities.n_groups,
        "artifact_count": len(manifest.artifacts),
        "status": manifest.status.value,
        "profile": policy.profile_name.value,
        "feature_view_count": len(capabilities.feature_views),
        "evidence_layer_count": len(capabilities.evidence_layers),
    }


def load_report_bundle(
    directory: str | os.PathLike[str],
    *,
    verify_hashes: bool = True,
    reject_stale: bool = True,
) -> ReportBundle:
    """Load and strictly validate one complete report bundle v3 directory."""
    if _SUPPRESS_LOAD_OBSERVATION.get():
        return _load_report_bundle(
            directory,
            verify_hashes=verify_hashes,
            reject_stale=reject_stale,
        )
    with automatic_stage("report_bundle.load") as operation:
        result = _load_report_bundle(
            directory,
            verify_hashes=verify_hashes,
            reject_stale=reject_stale,
        )
        if operation.active:
            try:
                operation.update(**_report_bundle_observation(result.manifest))
            except BaseException:
                pass
        return result


def _load_report_bundle_for_validation(directory) -> ReportBundle:
    token = _SUPPRESS_LOAD_OBSERVATION.set(True)
    try:
        return load_report_bundle(directory)
    finally:
        _SUPPRESS_LOAD_OBSERVATION.reset(token)


def write_report_bundle(
    directory: str | os.PathLike[str],
    manifest: ReportManifest,
    artifacts: Mapping[str, object],
    *,
    overwrite: bool = False,
) -> ReportBundle:
    """Stage, validate, and transactionally install a complete report bundle v3."""
    with automatic_stage(
        "report_bundle.write", ({"overwrite": overwrite} if isinstance(overwrite, bool) else {})
    ) as operation:
        result = _write_report_bundle(
            directory, manifest, artifacts, overwrite=overwrite
        )
        if operation.active:
            try:
                operation.update(**_report_bundle_observation(result.manifest))
            except BaseException:
                pass
        return result


__all__ = [
    "MANIFEST_FILENAME",
    "PathPayload",
    "ReportBundle",
    "artifact_sha256",
    "canonical_json_bytes",
    "json_payload",
    "load_report_bundle",
    "write_report_bundle",
]
