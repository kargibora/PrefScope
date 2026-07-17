"""Content-addressed cache of float32 arrays keyed by a text hash."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np


def text_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class NpyCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.npy"

    def has(self, key: str) -> bool:
        return self._path(key).exists()

    def existing_keys(self) -> set[str]:
        """All cached keys via a single directory scan.

        One `scandir` pass is far cheaper than N per-key `exists()` stat calls on
        a parallel filesystem, where the cache may hold hundreds of thousands of
        small files.
        """
        out: set[str] = set()
        with os.scandir(self.root) as it:
            for entry in it:
                if entry.name.endswith(".npy"):
                    out.add(entry.name[:-4])
        return out

    def get(self, key: str) -> np.ndarray:
        return np.load(self._path(key))

    def put(self, key: str, arr: np.ndarray) -> None:
        np.save(self._path(key), np.asarray(arr, dtype=np.float32))
