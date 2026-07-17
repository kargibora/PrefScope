"""Qwen3-Embedding wrapper with per-completion caching.

Cache key = hash of the formatted (prompt, completion) query, so identical
completions paired with identical prompts are embedded once. The transformer is
loaded lazily on first uncached batch.
"""
from __future__ import annotations

import logging

import numpy as np

from prefscope.config import CONFIG
from prefscope.encode.cache import NpyCache, text_key

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self, cache: NpyCache | None, *,
                 model_id: str = CONFIG.embed_model_id, device: str = "cpu",
                 max_tokens: int = CONFIG.max_tokens,
                 batch_size: int = CONFIG.embed_batch_size, dtype=None,
                 cache_workers: int = 32, backend: str = "hf",
                 tensor_parallel_size: int = 1,
                 api_base: str | None = None,
                 api_key_env: str = "OPENAI_API_KEY") -> None:
        if backend not in ("hf", "vllm", "vllm-server"):
            raise ValueError(
                f"backend must be 'hf', 'vllm', or 'vllm-server', got {backend!r}")
        self.cache = cache
        self.model_id = model_id
        self.device = device
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.dtype = dtype     # None -> bf16 on cuda, fp32 elsewhere
        self.cache_workers = max(1, cache_workers)   # parallel cache reads
        self.backend = backend
        self.tensor_parallel_size = tensor_parallel_size
        self.api_base = api_base          # vllm-server: OpenAI-compatible /v1 URL
        self.api_key_env = api_key_env
        self._model = None
        self._tok = None
        self._client = None

    def format_query(self, prompt: str, completion: str) -> str:
        # WIMHF instruction-aware format (verbatim instruction in config)
        return (f"{CONFIG.embed_instruction}\n\n"
                f"User: {prompt}\n\nAssistant: {completion}")

    def format_prompt_query(self, prompt: str) -> str:
        # prompt-only format for the prompt-concept lens (no Assistant turn)
        return f"{CONFIG.prompt_embed_instruction}\n\nUser: {prompt}"

    def _cache_key(self, query: str) -> str:
        # namespace by model_id so different embedders never collide in one cache
        return text_key(f"{self.model_id}\x1f{query}")

    @staticmethod
    def _engine_arg_fields() -> set:
        """Field names this vLLM's EngineArgs accepts (for version-tolerant init)."""
        try:
            from vllm import EngineArgs
        except Exception:
            try:
                from vllm.engine.arg_utils import EngineArgs
            except Exception:
                return set()
        try:
            import dataclasses
            return {f.name for f in dataclasses.fields(EngineArgs)}
        except Exception:
            try:
                import inspect
                return set(inspect.signature(EngineArgs).parameters)
            except Exception:
                return set()

    def _ensure_client(self):
        """OpenAI-compatible client for the vLLM embedding server (no GPU here)."""
        if self._client is not None:
            return
        import os

        from openai import OpenAI
        if not self.api_base:
            raise ValueError("vllm-server backend needs api_base (the server /v1 URL)")
        key = os.environ.get(self.api_key_env) or "EMPTY"   # vLLM ignores the value
        self._client = OpenAI(base_url=self.api_base, api_key=key)

    def _ensure_model(self):
        if self.backend == "vllm-server":
            self._ensure_client()
            return
        if self._model is not None:
            return
        if self.backend == "vllm":
            from vllm import LLM
            kwargs = dict(model=self.model_id,
                          tensor_parallel_size=self.tensor_parallel_size,
                          max_model_len=self.max_tokens)
            # vLLM renamed the embedding selector across versions:
            #   old (<=0.8): task="embed"   new: runner="pooling"
            # Inspect what THIS EngineArgs accepts so we don't pass a dead kwarg
            # (recent versions raise TypeError on the wrong one; an embedding
            #  model like Qwen3-Embedding is also auto-detected, so neither is
            #  strictly required as a last resort).
            fields = self._engine_arg_fields()
            if "runner" in fields:
                kwargs["runner"] = "pooling"
            elif "task" in fields:
                kwargs["task"] = "embed"
            self._model = LLM(**kwargs)
            return
        import torch
        from transformers import AutoModel, AutoTokenizer
        dtype = self.dtype
        if dtype is None:
            dtype = torch.bfloat16 if str(self.device).startswith("cuda") else torch.float32
        self._tok = AutoTokenizer.from_pretrained(self.model_id, padding_side="left")
        self._model = AutoModel.from_pretrained(
            self.model_id, torch_dtype=dtype).to(self.device)
        self._model.eval()

    def unload(self) -> None:
        """Release the transformer + free GPU memory (call before downstream GPU work)."""
        self._model = None
        self._tok = None
        self._client = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _encode_uncached(self, texts: list[str],
                         keys: list[str] | None = None) -> np.ndarray:
        self._ensure_model()
        if self.backend == "vllm-server":
            return self._encode_vllm_server(texts, keys)
        if self.backend == "vllm":
            return self._encode_vllm(texts, keys)
        return self._encode_hf(texts, keys)

    def _server_tokenizer(self):
        """HF tokenizer (CPU only) used to right-truncate before POSTing."""
        if self._tok is not None:
            return self._tok
        from transformers import AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        return self._tok

    def _truncate_for_server(self, batch, tok) -> list[str]:
        """Right-truncate to max_tokens, KEEPING the front (WIMHF instruction).

        vLLM's truncate_prompt_tokens drops the *front* (left truncation), which
        would strip the instruction. We instead tokenize and cut the tail, then
        decode — matching the HF backend's truncation side. Only items actually
        at the cap are re-decoded; shorter texts pass through unchanged.
        """
        margin = 8                              # headroom for server-added specials
        maxlen = max(8, self.max_tokens - margin)
        ids_list = tok(batch, truncation=True, max_length=maxlen,
                       add_special_tokens=False)["input_ids"]
        return [batch[i] if len(ids) < maxlen
                else tok.decode(ids, skip_special_tokens=True)
                for i, ids in enumerate(ids_list)]

    def _encode_vllm_server(self, texts, keys=None) -> np.ndarray:
        """Embed via a vLLM OpenAI-compatible /v1/embeddings server (HTTP).

        The heavy model lives in the server (e.g. a Singularity vLLM container);
        this side only right-truncates, POSTs the text, and caches the result. No
        GPU is touched here, so the host needs no working CUDA torch for embedding.
        """
        import time

        from tqdm.auto import tqdm
        tok = self._server_tokenizer()
        n = len(texts)
        chunk = max(self.batch_size, 256)      # server batches internally
        out: list[np.ndarray | None] = [None] * n
        for s in tqdm(range(0, n, chunk), desc="embedding (vllm-server)",
                      unit="chunk", total=len(range(0, n, chunk))):
            batch = self._truncate_for_server(texts[s:s + chunk], tok)
            resp = None
            for attempt in range(5):
                try:
                    # truncate_prompt_tokens is only a safety net for tokenizer
                    # drift; the host-side right-truncation above does the real work.
                    resp = self._client.embeddings.create(
                        model=self.model_id, input=batch,
                        extra_body={"truncate_prompt_tokens": self.max_tokens})
                    break
                except Exception:                # transient server hiccup / warmup
                    if attempt == 4:
                        raise
                    time.sleep(2 * (attempt + 1))
            data = sorted(resp.data, key=lambda d: d.index)
            vecs = np.asarray([d.embedding for d in data], dtype=np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.clip(norms, 1e-8, None)
            for local in range(len(batch)):
                gi = s + local
                out[gi] = vecs[local]
                if self.cache is not None and keys is not None:
                    self.cache.put(keys[gi], vecs[local])
        return np.vstack(out).astype(np.float32)

    def _encode_vllm(self, texts, keys=None) -> np.ndarray:
        """Embed with vLLM (instruction-aware pooling model), L2-normalized.

        Chunked so a killed run resumes from the last cached item.
        """
        from tqdm.auto import tqdm
        n = len(texts)
        chunk = max(self.batch_size, 1024)   # vLLM batches internally; large chunks
        out: list[np.ndarray | None] = [None] * n
        for s in tqdm(range(0, n, chunk), desc="embedding (vllm)", unit="chunk",
                      total=len(range(0, n, chunk))):
            batch = texts[s:s + chunk]
            results = self._model.embed(batch)
            vecs = np.asarray([r.outputs.embedding for r in results], dtype=np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.clip(norms, 1e-8, None)
            for local in range(len(batch)):
                gi = s + local
                out[gi] = vecs[local]
                if self.cache is not None and keys is not None:
                    self.cache.put(keys[gi], vecs[local])
        return np.vstack(out).astype(np.float32)

    def _encode_hf(self, texts: list[str],
                   keys: list[str] | None = None) -> np.ndarray:
        import torch
        from tqdm.auto import tqdm
        n = len(texts)
        # Process similar-length texts together so a single long completion doesn't
        # pad an entire batch up to max_tokens (attention is quadratic in length).
        order = sorted(range(n), key=lambda i: len(texts[i]))
        out: list[np.ndarray | None] = [None] * n
        for s in tqdm(range(0, n, self.batch_size), desc="embedding", unit="batch",
                      total=len(range(0, n, self.batch_size))):
            idx = order[s:s + self.batch_size]
            batch = [texts[i] for i in idx]
            enc = self._tok(batch, padding=True, truncation=True,
                            max_length=self.max_tokens, return_tensors="pt").to(self.device)
            with torch.no_grad():
                hidden = self._model(**enc).last_hidden_state
            mask = enc["attention_mask"]
            if int(mask[:, -1].sum()) == hidden.size(0):
                # left-padded (padding_side="left"): last real token is the final column
                pooled = hidden[:, -1]
            else:
                seq_len = mask.sum(dim=1) - 1
                pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), seq_len]
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vecs = pooled.float().cpu().numpy()
            # scatter back to original order; cache per-item so a killed run resumes
            for local, gi in enumerate(idx):
                out[gi] = vecs[local]
                if self.cache is not None and keys is not None:
                    self.cache.put(keys[gi], vecs[local])
        return np.vstack(out).astype(np.float32)

    def encode(self, prompts: list[str], completions: list[str]) -> np.ndarray:
        return self.encode_queries(
            [self.format_query(p, c) for p, c in zip(prompts, completions)])

    def encode_prompts(self, prompts: list[str]) -> np.ndarray:
        """Embed prompts ALONE (prompt-only query) for the prompt-concept lens."""
        return self.encode_queries([self.format_prompt_query(p) for p in prompts])

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        from tqdm.auto import tqdm
        keys = [self._cache_key(q) for q in queries]
        results: list[np.ndarray | None] = [None] * len(queries)

        # One bulk directory scan instead of N per-key exists() stat calls.
        existing = self.cache.existing_keys() if self.cache is not None else set()

        # First pass: partition into cached vs to-embed (deduped)
        seen: dict[str, int] = {}   # cache-key -> index in unique_texts
        unique_texts: list[str] = []
        unique_keys: list[str] = []
        pending: list[tuple[int, int]] = []   # (result_idx, unique_idx)
        cached: list[tuple[int, str]] = []     # (result_idx, cache_key)
        for i, (q, k) in enumerate(zip(queries, keys)):
            if k in existing:
                cached.append((i, k))
            elif k in seen:
                pending.append((i, seen[k]))   # duplicate within this call
            else:
                uid = len(unique_texts)
                seen[k] = uid
                unique_texts.append(q)
                unique_keys.append(k)
                pending.append((i, uid))

        logger.info("  %d cached, %d to embed (%d total)",
                    len(cached), len(unique_texts), len(queries))
        # Parallel reads: np.load is I/O-bound and releases the GIL, and a parallel
        # filesystem (Lustre) serves many small reads far faster than serial ones.
        if cached:
            from concurrent.futures import ThreadPoolExecutor

            def _get(item):
                i, k = item
                return i, self.cache.get(k)

            with ThreadPoolExecutor(max_workers=self.cache_workers) as ex:
                for i, vec in tqdm(ex.map(_get, cached), total=len(cached),
                                   desc="cache load", unit="vec"):
                    results[i] = vec

        if unique_texts:
            # _encode_uncached caches each vector per-item (keyed by unique_keys)
            vecs = self._encode_uncached(unique_texts, unique_keys)
            for i, uid in pending:
                if results[i] is None:
                    results[i] = vecs[uid]

        return np.vstack([r.reshape(-1) for r in results]).astype(np.float32)
