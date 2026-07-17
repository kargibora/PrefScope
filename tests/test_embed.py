import numpy as np

from prefscope.encode.cache import NpyCache
from prefscope.encode.embed import Embedder


def test_embedder_caches_and_batches(tmp_path, monkeypatch):
    cache = NpyCache(tmp_path)
    emb = Embedder(cache=cache)

    calls = {"n": 0}

    def fake_uncached(texts, keys=None):
        calls["n"] += 1
        # deterministic: vector keyed on length of the formatted text
        vecs = np.array([[float(len(t)), 1.0, 2.0, 3.0] for t in texts],
                        dtype=np.float32)
        # mirror the real method: cache per-key so the next call hits cache
        if keys is not None and emb.cache is not None:
            for k, v in zip(keys, vecs):
                emb.cache.put(k, v)
        return vecs

    monkeypatch.setattr(emb, "_encode_uncached", fake_uncached)

    prompts = ["p", "p"]
    comps = ["alpha", "beta"]
    out1 = emb.encode(prompts, comps)
    assert out1.shape == (2, 4)
    assert calls["n"] == 1  # one batch call for two uncached items

    # second call: fully cached -> no new uncached calls
    out2 = emb.encode(prompts, comps)
    assert calls["n"] == 1
    np.testing.assert_array_equal(out1, out2)


def test_encode_uncached_preserves_input_order_despite_length_sorting(monkeypatch):
    import torch
    emb = Embedder(cache=None)
    emb.batch_size = 2          # force >1 batch so cross-batch reordering is exercised

    class FakeTok:
        def __call__(self, batch, padding, truncation, max_length, return_tensors):
            class Enc(dict):
                def to(self, device):
                    return self
            ids = torch.tensor([[ord(t[0])] for t in batch])
            return Enc(input_ids=ids, attention_mask=torch.ones_like(ids))

    class FakeOut:
        pass

    class FakeModel:
        def __call__(self, **enc):
            ids = enc["input_ids"]
            b = ids.shape[0]
            h = torch.zeros(b, 1, 4)
            for r in range(b):
                h[r, 0, int(ids[r, 0]) % 4] = 1.0     # one-hot identity = ord%4
            out = FakeOut(); out.last_hidden_state = h
            return out

    emb._tok = FakeTok(); emb._model = FakeModel()
    monkeypatch.setattr(emb, "_ensure_model", lambda: None)

    # input NOT in length order: lengths 4,1,3,2  -> sorted internally, must come back in order
    out = emb._encode_uncached(["dddd", "a", "ccc", "bb"])
    expected = np.array([[1, 0, 0, 0],   # 'd'=100 %4=0
                         [0, 1, 0, 0],   # 'a'=97  %4=1
                         [0, 0, 0, 1],   # 'c'=99  %4=3
                         [0, 0, 1, 0]],  # 'b'=98  %4=2
                        dtype=np.float32)
    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_unload_clears_model():
    emb = Embedder(cache=None)
    emb._model = object()
    emb._tok = object()
    emb.unload()
    assert emb._model is None and emb._tok is None


def test_format_query_wimhf_template():
    emb = Embedder(cache=None)
    q = emb.format_query("PROMPT", "RESPONSE")
    assert q.startswith("Represent this user-assistant exchange")
    assert "User: PROMPT" in q and "Assistant: RESPONSE" in q


def test_cache_key_namespaced_by_model():
    a = Embedder(cache=None, model_id="m-0.6B")
    b = Embedder(cache=None, model_id="m-8B")
    q = a.format_query("p", "c")
    # same text, different model -> different cache key (no cross-model collision)
    assert a._cache_key(q) != b._cache_key(q)
    assert a._cache_key(q) == a._cache_key(q)


def test_vllm_backend_embeds_and_normalizes(monkeypatch):
    class _Out:
        def __init__(self, e): self.outputs = type("O", (), {"embedding": e})()

    class _FakeLLM:
        def __init__(self, **kw): self.kw = kw
        def embed(self, texts):
            return [_Out([3.0, 4.0]) for _ in texts]   # norm 5 -> normalizes to .6/.8

    emb = Embedder(cache=None, backend="vllm", model_id="Qwen/Qwen3-Embedding-8B",
                   tensor_parallel_size=2)
    emb._model = _FakeLLM()                       # skip real vLLM init
    monkeypatch.setattr(emb, "_ensure_model", lambda: None)
    out = emb._encode_uncached(["t1", "t2"])
    assert out.shape == (2, 2)
    np.testing.assert_allclose(out[0], [0.6, 0.8], atol=1e-6)   # L2-normalized


def test_embedder_rejects_bad_backend():
    import pytest
    with pytest.raises(ValueError):
        Embedder(cache=None, backend="nope")


def test_encode_dedups_identical_pairs_within_call(monkeypatch):
    emb = Embedder(cache=None)
    calls = {"texts": []}

    def fake_uncached(texts, keys=None):
        calls["texts"].append(list(texts))
        return np.array([[float(len(t)), 0.0, 0.0, 0.0] for t in texts],
                        dtype=np.float32)

    monkeypatch.setattr(emb, "_encode_uncached", fake_uncached)
    out = emb.encode(["p", "p"], ["same", "same"])
    assert out.shape == (2, 4)
    # only ONE unique text should have been sent to the model
    assert len(calls["texts"]) == 1
    assert len(calls["texts"][0]) == 1
    np.testing.assert_array_equal(out[0], out[1])


def test_encode_uncached_pools_last_real_token_when_left_padded(monkeypatch):
    import torch
    emb = Embedder(cache=None)

    class FakeTok:
        def __call__(self, batch, padding, truncation, max_length, return_tensors):
            # two sequences left-padded to width 3 (lengths 2 and 3)
            class Enc(dict):
                def to(self, device):
                    return self
            return Enc(
                input_ids=torch.tensor([[0, 11, 12], [21, 22, 23]]),
                attention_mask=torch.tensor([[0, 1, 1], [1, 1, 1]]),
            )

    class FakeOut:
        pass

    class FakeModel:
        def __call__(self, **enc):
            ids = enc["input_ids"]
            b, length = ids.shape
            # hidden[:, t, :] is the one-hot vector e_t (dim 4); pooling position p -> e_p
            h = torch.zeros(b, length, 4)
            for t in range(length):
                h[:, t, t] = 1.0
            out = FakeOut()
            out.last_hidden_state = h
            return out

    emb._tok = FakeTok()
    emb._model = FakeModel()
    monkeypatch.setattr(emb, "_ensure_model", lambda: None)

    out = emb._encode_uncached(["x", "y"])
    # last real token is column index 2 for both rows -> one-hot e_2 = [0,0,1,0]
    # (the buggy formula would pool column 1 for the first row -> [0,1,0,0])
    assert out.shape == (2, 4)
    np.testing.assert_allclose(out, [[0, 0, 1, 0], [0, 0, 1, 0]], atol=1e-6)
