"""Resolve shareable PrefScope lens artifacts from the Hugging Face Hub.

A Hub lens is deliberately just a normal lens directory.  Downloading changes
where the files live, not their format or the inference path: after
``snapshot_download`` the regular :class:`prefscope.api.loaded_lens.Lens` loader
validates and opens the cached directory.
"""
from __future__ import annotations

from pathlib import Path
import re

from prefscope.artifacts import MANIFEST, SAE_MODEL

_COMMIT_SHA = re.compile(r"^[0-9a-fA-F]{40}$")


def _safe_subfolder(subfolder: str | None) -> Path:
    """Return a relative artifact subfolder and reject path traversal."""
    if not subfolder:
        return Path()
    path = Path(subfolder)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("subfolder must be a relative path inside the Hub repository")
    return path


def split_hf_source(source: str) -> tuple[str, str | None]:
    """Parse ``hf://owner/repo[/subfolder]`` into a Hub repo id and subfolder."""
    if not str(source).startswith("hf://"):
        raise ValueError("Hugging Face lens sources must start with 'hf://'")
    parts = str(source)[5:].strip("/").split("/")
    if len(parts) < 2 or not all(parts[:2]):
        raise ValueError(
            "Hugging Face lens source must look like hf://owner/repository"
        )
    repo_id = "/".join(parts[:2])
    subfolder = "/".join(parts[2:]) or None
    return repo_id, subfolder


def resolve_hf_revision(
    repo_id: str,
    *,
    revision: str | None = None,
    repo_type: str = "model",
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> str:
    """Resolve a Hub ref to an immutable 40-hex repository commit.

    Exact commit revisions pass through without a Hub request, so pinned artifacts
    remain loadable offline. Mutable refs require the Hub API; offline callers must
    supply a commit rather than silently trusting a cached branch name.
    """
    requested = None if revision is None else str(revision).strip()
    if requested and _COMMIT_SHA.fullmatch(requested):
        return requested.lower()
    if local_files_only:
        raise ValueError(
            f"local_files_only=True cannot resolve mutable Hub revision {revision!r}; "
            "pass an exact 40-character commit SHA")
    try:
        from huggingface_hub import HfApi

        info = HfApi().repo_info(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            token=token,
        )
    except Exception as exc:
        raise ValueError(
            f"could not resolve Hugging Face {repo_type} repository {repo_id!r} "
            f"revision {revision!r} to an immutable commit SHA") from exc
    resolved = str(getattr(info, "sha", "") or "").strip()
    if not _COMMIT_SHA.fullmatch(resolved):
        raise ValueError(
            f"Hugging Face returned no immutable 40-character commit SHA for "
            f"{repo_type} repository {repo_id!r} revision {revision!r}")
    return resolved.lower()


def download_lens(
    repo_id: str,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    subfolder: str | None = None,
) -> Path:
    """Download a Hub model repository and return its local lens directory.

    ``revision`` may be a branch, tag, or commit hash. Mutable refs are resolved
    first, and the snapshot is always requested by its immutable commit SHA.
    ``subfolder`` supports a repository containing multiple artifacts (for example
    ``completion/`` and ``prompt/``). Required lens files are checked before the
    caller constructs PyTorch objects, producing a direct artifact error instead
    of an obscure checkpoint failure.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - declared dependency; useful for vendoring
        raise ImportError(
            "Lens.from_pretrained() needs huggingface-hub; install or upgrade prefscope"
        ) from exc

    relative = _safe_subfolder(subfolder)
    resolved_revision = resolve_hf_revision(
        repo_id,
        revision=revision,
        repo_type="model",
        token=token,
        local_files_only=local_files_only,
    )
    download_kwargs = {}
    if relative != Path():
        download_kwargs["allow_patterns"] = [f"{relative.as_posix()}/**"]
    root = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            revision=resolved_revision,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            token=token,
            local_files_only=local_files_only,
            library_name="prefscope",
            **download_kwargs,
        )
    )
    lens_dir = root / relative
    missing = [name for name in (MANIFEST, SAE_MODEL) if not (lens_dir / name).is_file()]
    if missing:
        where = f" in subfolder {subfolder!r}" if subfolder else ""
        raise FileNotFoundError(
            f"{repo_id!r}{where} is not a PrefScope lens: missing {missing}"
        )
    return lens_dir
