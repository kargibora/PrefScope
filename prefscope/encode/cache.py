"""Content-addressed cache of float32 arrays keyed by a text hash.

Vectors are grouped into block files:

    <root>/<key>.npy                     legacy single-vector files (read-only)
    <root>/blocks/blk_<worker>_<n>.npz   {vecs: (n, d) float32, keys: (n,)}

Each process writes its own ``<worker>`` blocks, so concurrent writers sharing one
cache directory need no locking.
"""
from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path

import numpy as np

BLOCK_SIZE = 256
_BLOCK_DIR = "blocks"
_BLOCK_LRU = 4


def text_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class NpyCache:
    def __init__(self, root: str | Path, *, block_size: int = BLOCK_SIZE) -> None:
        if int(block_size) < 1:
            raise ValueError("block_size must be positive")
        self.root = Path(root)
        self.block_size = int(block_size)
        self.root.mkdir(parents=True, exist_ok=True)
        self._blocks = self.root / _BLOCK_DIR
        self._worker = f"{os.getpid():d}{uuid.uuid4().hex[:6]}"
        self._seq = 0
        self._buf_keys: list[str] = []
        self._buf_vecs: list[np.ndarray] = []
        self._buf_pos: dict[str, int] = {}
        self._index: dict[str, tuple[Path, int]] | None = None
        self._loaded: dict[Path, np.ndarray] = {}

    def put(self, key: str, arr: np.ndarray) -> None:
        vec = np.asarray(arr, dtype=np.float32)
        if self._buf_vecs and vec.shape != self._buf_vecs[0].shape:
            self.flush()
        self._buf_pos[key] = len(self._buf_keys)
        self._buf_keys.append(key)
        self._buf_vecs.append(vec)
        if len(self._buf_keys) >= self.block_size:
            self.flush()

    def flush(self) -> None:
        """Write buffered vectors as one block. Idempotent.

        Callers must flush when finished: buffered vectors are otherwise lost.
        """
        if not self._buf_keys:
            return
        self._blocks.mkdir(parents=True, exist_ok=True)
        path = self._blocks / f"blk_{self._worker}_{self._seq:06d}.npz"
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez(
            tmp,
            vecs=np.stack(self._buf_vecs).astype(np.float32),
            keys=np.asarray(self._buf_keys, dtype=object),
        )
        os.replace(tmp, path)
        if self._index is not None:
            for row, key in enumerate(self._buf_keys):
                self._index[key] = (path, row)
        self._seq += 1
        self._buf_keys, self._buf_vecs, self._buf_pos = [], [], {}

    def _legacy_path(self, key: str) -> Path:
        return self.root / f"{key}.npy"

    def _block_index(self) -> dict[str, tuple[Path, int]]:
        if self._index is None:
            index: dict[str, tuple[Path, int]] = {}
            if self._blocks.is_dir():
                for path in sorted(self._blocks.glob("blk_*.npz")):
                    try:
                        with np.load(path, allow_pickle=True) as handle:
                            keys = handle["keys"]
                    except (OSError, ValueError, KeyError):
                        continue      # incomplete block from an interrupted run
                    for row, key in enumerate(keys):
                        index[str(key)] = (path, row)
            self._index = index
        return self._index

    def _block_vecs(self, path: Path) -> np.ndarray:
        if path not in self._loaded:
            with np.load(path, allow_pickle=True) as handle:
                self._loaded[path] = handle["vecs"]
            if len(self._loaded) > _BLOCK_LRU:
                self._loaded.pop(next(iter(self._loaded)))
        return self._loaded[path]

    def has(self, key: str) -> bool:
        return (key in self._buf_pos
                or self._legacy_path(key).exists()
                or key in self._block_index())

    def get(self, key: str) -> np.ndarray:
        if key in self._buf_pos:
            return self._buf_vecs[self._buf_pos[key]]
        legacy = self._legacy_path(key)
        if legacy.exists():
            return np.load(legacy)
        entry = self._block_index().get(key)
        if entry is None:
            raise KeyError(f"no cached vector for {key!r}")
        path, row = entry
        return np.asarray(self._block_vecs(path)[row], dtype=np.float32)

    def get_many(self, keys, *, workers: int = 8) -> dict[str, np.ndarray]:
        """Return ``{key: vector}``, reading each block file once.

        Preferred over repeated :meth:`get` for bulk reads: callers request keys in
        corpus order while blocks are written per worker, so key-at-a-time access
        reloads the same blocks repeatedly.
        """
        out: dict[str, np.ndarray] = {}
        legacy: list[str] = []
        by_block: dict[Path, list[tuple[str, int]]] = {}
        index = self._block_index()
        for key in keys:
            if key in self._buf_pos:
                out[key] = self._buf_vecs[self._buf_pos[key]]
            elif self._legacy_path(key).exists():
                legacy.append(key)
            elif key in index:
                path, row = index[key]
                by_block.setdefault(path, []).append((key, row))
            else:
                raise KeyError(f"no cached vector for {key!r}")
        for path, items in by_block.items():
            with np.load(path, allow_pickle=True) as handle:
                vecs = handle["vecs"]
                for key, row in items:
                    out[key] = np.asarray(vecs[row], dtype=np.float32)
        if legacy:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
                for key, vec in zip(legacy, pool.map(
                        lambda k: np.load(self._legacy_path(k)), legacy)):
                    out[key] = vec
        return out

    def existing_keys(self) -> set[str]:
        """All cached keys: buffered, legacy files, and block members."""
        out = set(self._buf_pos)
        with os.scandir(self.root) as it:
            for entry in it:
                if entry.name.endswith(".npy"):
                    out.add(entry.name[:-4])
        out.update(self._block_index())
        return out
