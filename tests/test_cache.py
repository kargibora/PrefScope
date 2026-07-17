import numpy as np

from prefscope.encode.cache import NpyCache, text_key


def test_text_key_is_stable_and_distinct():
    assert text_key("hello") == text_key("hello")
    assert text_key("hello") != text_key("world")


def test_cache_roundtrip(tmp_path):
    cache = NpyCache(tmp_path)
    key = text_key("abc")
    assert not cache.has(key)
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    cache.put(key, arr)
    assert cache.has(key)
    np.testing.assert_array_equal(cache.get(key), arr)


def test_existing_keys_bulk_scan(tmp_path):
    import numpy as np
    from prefscope.encode.cache import NpyCache
    c = NpyCache(tmp_path)
    c.put("aaa", np.zeros(3, dtype="float32"))
    c.put("bbb", np.ones(3, dtype="float32"))
    keys = c.existing_keys()
    assert keys == {"aaa", "bbb"}
    assert c.existing_keys() == {k for k in keys if c.has(k)}
