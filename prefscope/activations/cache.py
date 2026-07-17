"""Memmap-backed cache of token-level activations + a (battle_id, span, token_idx) index.

Activations are appended row-by-row to a flat float16 binary file; the index is
accumulated in memory and written to parquet at finalize(). No torch — this is
pure NumPy/pandas so it is testable without a GPU stack.

Write a new cache via ActivationCache(root, hidden_dim) + append() + finalize();
read an existing one via ActivationCache.open(root).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

_ACTS = "acts.f16"
_INDEX = "index.parquet"
_MANIFEST = "manifest.json"


class ActivationCache:
    """Memmap-backed cache of token-level activations + a (battle_id, span, token_idx) index.

    Write a new cache via ActivationCache(root, hidden_dim) + append() + finalize();
    read an existing one via ActivationCache.open(root).
    """

    def __init__(self, root, hidden_dim: int, dtype: str = "float16") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.hidden_dim = int(hidden_dim)
        self.dtype = dtype
        self._rows: list[dict] = []
        self._n = 0
        self._fh = open(self.root / _ACTS, "wb")
        self._finalized = False

    def append(self, vectors: np.ndarray, rows: list[dict]) -> None:
        if getattr(self, "_finalized", True):
            raise RuntimeError("cache is read-only or already finalized; "
                               "create a new ActivationCache(...) to write")
        v = np.ascontiguousarray(vectors, dtype=self.dtype)
        if v.ndim != 2 or v.shape[1] != self.hidden_dim:
            raise ValueError(f"expected (n, {self.hidden_dim}), got {v.shape}")
        if v.shape[0] != len(rows):
            raise ValueError(f"{v.shape[0]} vectors vs {len(rows)} index rows")
        self._fh.write(v.tobytes())
        self._rows.extend(rows)
        self._n += v.shape[0]

    def finalize(self, extra_manifest: dict | None = None) -> None:
        if getattr(self, "_finalized", True):
            raise RuntimeError("cache is read-only or already finalized; "
                               "create a new ActivationCache(...) to write")
        self._fh.flush()
        self._fh.close()
        pd.DataFrame(self._rows).to_parquet(self.root / _INDEX)
        manifest = {"n_tokens": self._n, "hidden_dim": self.hidden_dim,
                    "dtype": self.dtype}
        manifest.update(extra_manifest or {})
        (self.root / _MANIFEST).write_text(json.dumps(manifest, indent=2))
        self._finalized = True

    # ── read side ──

    @classmethod
    def open(cls, root) -> "ActivationCache":
        root = Path(root)
        mpath = root / _MANIFEST
        if not mpath.exists():
            raise FileNotFoundError(f"no {_MANIFEST} in {root}")
        obj = cls.__new__(cls)
        obj.root = root
        obj.manifest = json.loads(mpath.read_text())
        obj.hidden_dim = int(obj.manifest["hidden_dim"])
        obj.dtype = obj.manifest["dtype"]
        obj.n_tokens = int(obj.manifest["n_tokens"])
        if obj.n_tokens == 0:
            obj.acts = np.empty((0, obj.hidden_dim), dtype=obj.dtype)
        else:
            obj.acts = np.memmap(root / _ACTS, dtype=obj.dtype, mode="r",
                                 shape=(obj.n_tokens, obj.hidden_dim))
        obj.index = pd.read_parquet(root / _INDEX)
        obj._finalized = True
        return obj


def train_val_row_indices(n_tokens: int, val_frac: float,
                          max_train_tokens: int | None, seed: int = 0
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic disjoint (train, val) row-index arrays over [0, n_tokens).

    Validation is a fixed fraction and is never capped. The remaining rows are
    the training pool; if ``max_train_tokens`` is set, the pool is randomly
    subsampled to that many rows (a reservoir cap that bounds GPU/disk traffic
    and is the scale-up knob for the full corpus).
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_tokens)
    n_val = int(round(val_frac * n_tokens))
    val = np.sort(perm[:n_val])
    train = perm[n_val:]
    if max_train_tokens is not None and len(train) > max_train_tokens:
        train = train[:max_train_tokens]
    return np.sort(train), val
