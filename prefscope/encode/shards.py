# prefscope/encode/shards.py
"""Order-based sharded vector storage: fixed-size blocks of rows, named by global
start row, written atomically. Resumable + splittable across processes. Pure I/O,
independent of how vectors are produced."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

SHARD_DEFAULT = 1024


def block_starts(n_rows: int, shard: int = SHARD_DEFAULT) -> list[int]:
    return list(range(0, n_rows, shard))


def shard_path(root, side: str, start: int) -> Path:
    return Path(root) / side / f"shard_{start:09d}.npy"


def assigned_starts(n_rows: int, shard_idx: int, num_shards: int,
                    shard: int = SHARD_DEFAULT) -> list[int]:
    """Blocks owned by task `shard_idx` of `num_shards` (round-robin by block)."""
    starts = block_starts(n_rows, shard)
    return [s for j, s in enumerate(starts) if j % num_shards == shard_idx]


def write_block(root, side: str, start: int, vecs) -> None:
    p = shard_path(root, side, start)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")                 # shard_XXXXXXXXX.npy.tmp
    with open(tmp, "wb") as f:                          # file object => np.save won't append ".npy"
        np.save(f, np.asarray(vecs, dtype=np.float32))
    os.replace(tmp, p)                                 # atomic on POSIX


def missing_starts(root, side: str, starts) -> list[int]:
    return [s for s in starts if not shard_path(root, side, s).exists()]


def assemble(root, side: str, n_rows: int, shard: int = SHARD_DEFAULT) -> np.ndarray:
    parts = []
    for s in block_starts(n_rows, shard):
        p = shard_path(root, side, s)
        if not p.exists():
            raise FileNotFoundError(f"missing shard {p}")
        parts.append(np.load(p))
    return np.concatenate(parts, axis=0).astype(np.float32)
