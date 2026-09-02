"""Internal transactional-publication primitives for the Lens facade."""
from __future__ import annotations

from contextlib import contextmanager
import errno
import json
import os
from pathlib import Path
import shutil
import socket
import stat
import uuid

from prefscope.api._lens_annotations import _annotation_paths
from prefscope.artifacts import MANIFEST, SAE_MODEL


def _locking_module():
    try:
        import fcntl
    except ImportError as exc:
        raise RuntimeError(
            "transactional lens publication locking is unavailable on this platform"
        ) from exc
    return fcntl


def _validate_lock_parent(parent: Path) -> None:
    if parent.is_symlink() or not parent.is_dir():
        raise RuntimeError(f"publication lock parent must be a real directory: {parent}")
    opened = parent.stat(follow_symlinks=False)
    mode = stat.S_IMODE(opened.st_mode)
    sticky = bool(opened.st_mode & stat.S_ISVTX)
    if hasattr(os, "geteuid"):
        owner_matches = opened.st_uid == os.geteuid()
        if not owner_matches and not sticky:
            raise RuntimeError(
                f"publication lock parent is not owned by this user: {parent}"
            )
        if mode & 0o022 and not sticky:
            raise RuntimeError(
                f"publication lock parent is writable by other users: {parent}"
            )


def _secure_lock_descriptor(lock_path: Path) -> int:
    _validate_lock_parent(lock_path.parent)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    cloexec = getattr(os, "O_CLOEXEC", None)
    if nofollow is None or cloexec is None:
        raise RuntimeError("secure publication locking requires O_NOFOLLOW and O_CLOEXEC")
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_APPEND
        | getattr(os, "O_NONBLOCK", 0)
        | nofollow
        | cloexec
    )
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise RuntimeError(
                f"publication lock must be a regular file: {lock_path}")
        if opened.st_nlink != 1:
            raise RuntimeError(
                f"publication lock must have exactly one hard link: {lock_path}"
            )
        if hasattr(os, "geteuid") and opened.st_uid != os.geteuid():
            raise RuntimeError(
                f"publication lock must be owned by this user: {lock_path}"
            )
        os.fchmod(descriptor, 0o600)
        if stat.S_IMODE(os.fstat(descriptor).st_mode) != 0o600:
            raise RuntimeError(
                f"publication lock permissions must be 0600: {lock_path}")
        entry = lock_path.stat(follow_symlinks=False)
        if (entry.st_dev, entry.st_ino) != (opened.st_dev, opened.st_ino):
            raise RuntimeError(
                f"publication lock changed while opening: {lock_path}")
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _write_lock_metadata(descriptor: int, owner_id: str) -> None:
    payload = json.dumps(
        {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "owner_id": owner_id,
        },
        sort_keys=True,
    ).encode("utf-8")
    os.ftruncate(descriptor, 0)
    os.lseek(descriptor, 0, os.SEEK_SET)
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError("short write while updating publication lock metadata")
        offset += written
    os.fsync(descriptor)


@contextmanager
def _publication_lock(destination: Path):
    """Hold a stable, never-unlinked advisory lock for one destination."""
    fcntl = _locking_module()
    lock_path = destination.parent / f".{destination.name}.lock"
    try:
        descriptor = _secure_lock_descriptor(lock_path)
    except OSError as exc:
        raise RuntimeError(
            f"cannot securely open publication lock {lock_path}") from exc

    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                raise
            raise RuntimeError(
                f"cannot publish {destination}: another active publisher holds "
                f"{lock_path}"
            ) from exc

        # Metadata helps operators identify the last/acquiring process, but it
        # never decides ownership. Only the kernel advisory lock does that.
        owner_id = uuid.uuid4().hex
        _write_lock_metadata(descriptor, owner_id)
        opened = os.fstat(descriptor)
        entry = lock_path.stat(follow_symlinks=False)
        if opened.st_nlink != 1:
            raise RuntimeError(
                f"publication lock acquired an unsafe hard link: {lock_path}"
            )
        if (entry.st_dev, entry.st_ino) != (opened.st_dev, opened.st_ino):
            raise RuntimeError(
                f"publication lock directory entry changed: {lock_path}")
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _recover_orphan_backup(destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        return
    backups = sorted(destination.parent.glob(f".{destination.name}.bak-*"))
    if not backups:
        return
    if len(backups) != 1:
        raise RuntimeError(
            f"cannot recover {destination}: found multiple orphan backups "
            f"{[path.name for path in backups]}")
    backup = backups[0]
    if backup.is_symlink() or not backup.is_dir():
        raise RuntimeError(
            f"cannot recover {destination}: orphan backup {backup} is not a directory")
    os.replace(backup, destination)


def save_lens(
    lens, dest, *, overwrite: bool = False, annotations=None,
    inference_only: bool = False,
):
    """Copy the backing lens dir to ``dest`` as a staged whole-directory replacement.

    The old behaviour merged into ``dest`` (``copytree(dirs_exist_ok=True)``), which
    could leave a *hybrid* lens: new ``sae_model.pt`` beside a stale ``feature_names``
    or ``manifest`` from a previous artifact. Instead we stage into a temp sibling and
    swap it in with filesystem renames, so files from two artifacts are never merged.
    Existing destinations are restored if the final rename fails. A non-empty ``dest``
    is refused unless ``overwrite=True``.
    ``annotations`` may point to interpretation CSVs/directories to bundle into the
    staged artifact before it is published or uploaded. ``inference_only=True``
    omits corpus-aligned ``z_*.npy``/battles/training files and rewrites the copied
    manifest as an inference artifact; this is the compact form intended for the
    Hugging Face Hub.
    """
    if dest is None:
        raise ValueError("save() requires a destination path")
    if lens.lens_dir is None:
        raise ValueError("this Lens has no backing directory to save")
    src = Path(lens.lens_dir)
    dest = Path(dest)
    if not src.is_dir():
        raise ValueError(f"backing lens path must be a directory: {src}")
    src_resolved = src.resolve()
    dest_resolved = dest.resolve(strict=False)
    if src_resolved == dest_resolved:
        if inference_only or annotations is not None:
            raise ValueError(
                "cannot bundle annotations or create an inference-only artifact "
                "in place; choose a different destination")
        return dest
    if src_resolved in dest_resolved.parents or dest_resolved in src_resolved.parents:
        raise ValueError(
            "lens source and destination must not be ancestors or descendants "
            "of each other")
    if dest.is_symlink() or (dest.exists() and not dest.is_dir()):
        raise ValueError("save destination must be a directory path, not a file/symlink")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with _publication_lock(dest):
        _recover_orphan_backup(dest)
        if dest.exists() and any(dest.iterdir()) and not overwrite:
            raise FileExistsError(
                f"{dest} exists and is not empty; pass overwrite=True to replace it "
                "(save() never merges — that would risk a hybrid lens)")
        # UUID names make stale paths attributable to one publication attempt and
        # prevent PID reuse or concurrent destinations from colliding.
        staging = dest.parent / f".{dest.name}.tmp-{uuid.uuid4().hex}"
        backup = dest.parent / f".{dest.name}.bak-{uuid.uuid4().hex}"
        try:
            if inference_only:
                staging.mkdir()
                for name in (SAE_MODEL, "whiten.npz", "README.md", "LICENSE"):
                    path = src / name
                    if path.is_file():
                        shutil.copy2(path, staging / name)
                manifest = json.loads((src / MANIFEST).read_text())
                source_arrays = list(
                    manifest.get("source_output_arrays")
                    or manifest.get("output_arrays")
                    or [])
                # Packaging is also the migration boundary: a shared Hub artifact
                # should never retain a stale schema or declare corpus-aligned arrays
                # that were deliberately omitted. Preserve the original arrays only as
                # provenance and persist inferred SAE semantics explicitly.
                from prefscope.core.manifest import LensManifest

                typed = LensManifest.from_dict(manifest)
                projector = getattr(lens, "projector", None)
                if projector is not None:
                    typed.m_total = typed.m_total or int(projector.m_total)
                    typed.input_dim = typed.input_dim or int(projector.input_dim)
                    typed.k = typed.k or int(getattr(projector, "k", 1))
                    typed.matryoshka_prefix_lengths = (
                        typed.matryoshka_prefix_lengths
                        if typed.matryoshka_prefix_lengths is not None
                        else list(getattr(
                            getattr(projector, "_model", None),
                            "matryoshka_prefix_lengths", [])))
                typed.output_arrays = []
                typed.array_shapes = {}
                typed.extra["artifact_scope"] = "inference"
                typed.extra["source_output_arrays"] = source_arrays
                (staging / MANIFEST).write_text(
                    json.dumps(typed.to_dict(), indent=2))
            else:
                shutil.copytree(src, staging)  # fresh dir: never merge artifacts
            if inference_only or annotations is not None:
                for path in _annotation_paths(src, lens.input_rep, annotations):
                    # src files were already copied; external annotations replace the
                    # canonical file with the same name inside the staged artifact.
                    if inference_only or path.parent != src.resolve():
                        shutil.copy2(path, staging / path.name)
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
            if staging.is_symlink():
                staging.unlink()
            elif staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
        return dest
