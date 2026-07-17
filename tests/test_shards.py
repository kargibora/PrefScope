# tests/test_shards.py
import numpy as np
import pytest

from prefscope.encode import shards


def test_block_starts():
    assert shards.block_starts(0, shard=4) == []
    assert shards.block_starts(4, shard=4) == [0]
    assert shards.block_starts(10, shard=4) == [0, 4, 8]


def test_assigned_starts_is_disjoint_cover():
    n, shard, k = 4000, 1024, 3                      # 4 blocks (0,1024,2048,3072)
    allb = set(shards.block_starts(n, shard))
    owned = [set(shards.assigned_starts(n, i, k, shard)) for i in range(k)]
    assert set().union(*owned) == allb               # covers every block
    for i in range(k):                               # disjoint
        for j in range(i + 1, k):
            assert owned[i].isdisjoint(owned[j])


def test_write_assemble_roundtrip(tmp_path):
    n, shard = 10, 4
    x = np.arange(n * 3, dtype=np.float32).reshape(n, 3)
    for s in shards.block_starts(n, shard):
        shards.write_block(tmp_path, "a", s, x[s:s + shard])
    out = shards.assemble(tmp_path, "a", n, shard=shard)
    np.testing.assert_array_equal(out, x)
    assert not list(tmp_path.glob("a/*.tmp"))         # atomic: no temp left


def test_missing_starts(tmp_path):
    n, shard = 10, 4
    starts = shards.block_starts(n, shard)            # [0,4,8]
    shards.write_block(tmp_path, "a", 0, np.zeros((4, 2), np.float32))
    assert shards.missing_starts(tmp_path, "a", starts) == [4, 8]


def test_assemble_raises_on_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        shards.assemble(tmp_path, "a", 8, shard=4)    # nothing written
