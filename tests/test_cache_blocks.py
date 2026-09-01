import numpy as np
import pytest

from prefscope.encode.cache import NpyCache, text_key


def _vec(i: int, dim: int = 8) -> np.ndarray:
    return np.full(dim, float(i), dtype=np.float32)


def test_blocks_collapse_many_texts_into_few_files(tmp_path):
    cache = NpyCache(tmp_path, block_size=256)
    for i in range(1000):
        cache.put(text_key(f"t{i}"), _vec(i))
    cache.flush()
    files = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert len(files) <= 8


def test_roundtrip_across_block_boundaries(tmp_path):
    cache = NpyCache(tmp_path, block_size=4)
    keys = [text_key(f"t{i}") for i in range(10)]
    for i, k in enumerate(keys):
        cache.put(k, _vec(i))
    cache.flush()
    reader = NpyCache(tmp_path, block_size=4)
    for i, k in enumerate(keys):
        np.testing.assert_array_equal(reader.get(k), _vec(i))
    assert reader.existing_keys() == set(keys)


def test_unflushed_values_are_readable(tmp_path):
    cache = NpyCache(tmp_path, block_size=256)
    k = text_key("pending")
    cache.put(k, _vec(7))
    assert cache.has(k)
    np.testing.assert_array_equal(cache.get(k), _vec(7))
    assert k in cache.existing_keys()


def test_flush_is_idempotent_and_writes_no_empty_block(tmp_path):
    cache = NpyCache(tmp_path, block_size=4)
    cache.flush()
    cache.flush()
    assert [p for p in tmp_path.rglob("*") if p.is_file()] == []


def test_concurrent_writers_do_not_collide(tmp_path):
    a, b = NpyCache(tmp_path, block_size=2), NpyCache(tmp_path, block_size=2)
    for i in range(6):
        a.put(text_key(f"a{i}"), _vec(i))
        b.put(text_key(f"b{i}"), _vec(100 + i))
    a.flush()
    b.flush()
    reader = NpyCache(tmp_path)
    for i in range(6):
        np.testing.assert_array_equal(reader.get(text_key(f"a{i}")), _vec(i))
        np.testing.assert_array_equal(reader.get(text_key(f"b{i}")), _vec(100 + i))


def test_legacy_per_text_cache_is_still_readable(tmp_path):
    legacy_key = text_key("legacy")
    np.save(tmp_path / f"{legacy_key}.npy", _vec(42))
    cache = NpyCache(tmp_path, block_size=4)
    assert cache.has(legacy_key)
    np.testing.assert_array_equal(cache.get(legacy_key), _vec(42))
    assert legacy_key in cache.existing_keys()
    cache.put(text_key("fresh"), _vec(1))
    cache.flush()
    assert {legacy_key, text_key("fresh")} <= cache.existing_keys()


def test_missing_key_raises(tmp_path):
    with pytest.raises(KeyError):
        NpyCache(tmp_path).get(text_key("absent"))


def test_mixed_dimensions_do_not_corrupt_a_block(tmp_path):
    cache = NpyCache(tmp_path, block_size=8)
    cache.put("k4", np.ones(4, dtype=np.float32))
    cache.put("k8", np.ones(8, dtype=np.float32))
    cache.flush()
    reader = NpyCache(tmp_path)
    assert reader.get("k4").shape == (4,)
    assert reader.get("k8").shape == (8,)


def test_get_many_reads_each_block_once(tmp_path, monkeypatch):
    cache = NpyCache(tmp_path, block_size=4)
    keys = [text_key(f"t{i}") for i in range(16)]
    for i, k in enumerate(keys):
        cache.put(k, _vec(i))
    cache.flush()

    reader = NpyCache(tmp_path, block_size=4)
    reader._block_index()
    loads = []
    real_load = np.load
    monkeypatch.setattr(np, "load", lambda p, **kw: (loads.append(p), real_load(p, **kw))[1])
    # interleaved order, as round-robin shards produce
    order = [keys[i] for i in [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]]
    got = reader.get_many(order)
    assert len(loads) == 4
    for i, k in enumerate(keys):
        np.testing.assert_array_equal(got[k], _vec(i))


def test_get_many_handles_legacy_and_missing(tmp_path):
    legacy_key = text_key("legacy")
    np.save(tmp_path / f"{legacy_key}.npy", _vec(9))
    cache = NpyCache(tmp_path, block_size=4)
    cache.put(text_key("blocked"), _vec(1))
    cache.flush()
    got = cache.get_many([legacy_key, text_key("blocked")])
    np.testing.assert_array_equal(got[legacy_key], _vec(9))
    np.testing.assert_array_equal(got[text_key("blocked")], _vec(1))
    with pytest.raises(KeyError):
        cache.get_many([text_key("absent")])
