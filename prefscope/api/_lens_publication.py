"""Internal transactional-publication primitives for the Lens facade."""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import socket
import stat
import uuid

from prefscope.api._lens_annotations import _annotation_paths
from prefscope.artifacts import MANIFEST, SAE_MODEL

def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _lock_owner(path: Path) -> tuple[dict | None, str]:
    try:
        before = path.lstat()
        raw = path.read_text()
        after = path.lstat()
    except FileNotFoundError:
        return None, "lock disappeared"
    if not stat.S_ISREG(before.st_mode) or (before.st_dev, before.st_ino) != (
        after.st_dev, after.st_ino
    ):
        return None, "lock is not a stable regular file"
    try:
        owner = json.loads(raw)
        pid = int(owner["pid"])
        hostname = str(owner["hostname"])
        owner_id = str(owner["owner_id"])
        uuid.UUID(hex=owner_id)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None, "lock metadata is invalid"
    if pid <= 0 or hostname != socket.gethostname():
        return None, "lock owner cannot be safely checked on this host"
    return {"pid": pid, "hostname": hostname, "owner_id": owner_id}, raw


@contextmanager
def _publication_lock(destination: Path):
    lock_path = destination.parent / f".{destination.name}.lock"
    owner_id = uuid.uuid4().hex
    payload = json.dumps({
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "owner_id": owner_id,
    }, sort_keys=True)
    for _ in range(3):
        try:
            descriptor = os.open(
                lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            owner, detail = _lock_owner(lock_path)
            if owner is None:
                if detail == "lock disappeared":
                    continue
                raise RuntimeError(
                    f"cannot publish {destination}: publication lock {lock_path} "
                    f"is present but {detail}; refusing to remove it")
            if _pid_is_alive(owner["pid"]):
                raise RuntimeError(
                    f"cannot publish {destination}: another active publisher "
                    f"holds {lock_path} (pid {owner['pid']})")
            current, current_raw = _lock_owner(lock_path)
            if current != owner or current_raw != detail:
                continue
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
            continue
        else:
            lock_stat = os.fstat(descriptor)
            try:
                encoded = payload.encode("utf-8")
                offset = 0
                while offset < len(encoded):
                    written = os.write(descriptor, encoded[offset:])
                    if written <= 0:
                        raise OSError("short write while creating publication lock")
                    offset += written
                os.fsync(descriptor)
            except BaseException:
                os.close(descriptor)
                try:
                    current_stat = lock_path.stat(follow_symlinks=False)
                    if (
                        current_stat.st_dev == lock_stat.st_dev
                        and current_stat.st_ino == lock_stat.st_ino
                    ):
                        lock_path.unlink()
                except FileNotFoundError:
                    pass
                raise
            else:
                os.close(descriptor)
            break
    else:
        raise RuntimeError(f"could not acquire publication lock for {destination}")
    try:
        yield
    finally:
        try:
            owner, _ = _lock_owner(lock_path)
            if owner is not None and owner["owner_id"] == owner_id:
                lock_path.unlink()
        except FileNotFoundError:
            pass


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
