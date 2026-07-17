"""Pure-numpy example selection for feature interpretation and verification."""
from __future__ import annotations

import hashlib

import numpy as np

from prefscope.core import registry


def name_verify_split(instruction_ids, verify_frac: float = 0.2):
    """Deterministic disjoint split: hash instruction_id, low buckets -> verify.
    Returns (name_mask, verify_mask), both (N,) bool, covering all rows."""
    thresh = int(verify_frac * 1000)
    verify = np.zeros(len(instruction_ids), dtype=bool)
    for i, iid in enumerate(instruction_ids):
        bucket = int(hashlib.sha1(str(iid).encode()).hexdigest(), 16) % 1000
        verify[i] = bucket < thresh
    return ~verify, verify


def top_pairs(z_col: np.ndarray, pool: np.ndarray, n_active: int, n_zero: int,
              rng: np.random.Generator, *, active_pool_factor: int = 1) -> dict:
    """Within `pool`: up to `n_active` highest-z battles from the POSITIVE pole
    (z > 0) + `n_zero` random silent battles (z == 0). WIMHF sample_top_zero.
    Global indices. Requiring z > 0 for actives matters for a signed SAE: without
    it, a feature with few positive examples pulls in negative-pole (z < 0) or even
    silent (z == 0) rows as "actives", so the same row could land in both buckets."""
    pool = np.asarray(pool)
    z_pool = z_col[pool]
    order = pool[np.argsort(-z_pool)]
    positive = order[z_col[order] > 0]
    factor = max(1, int(active_pool_factor))
    candidates = positive[:max(n_active, n_active * factor)]
    if factor > 1 and len(candidates) > n_active:
        # Multi-candidate naming needs genuinely different evidence views. Sample
        # within the strongly activating pool; factor=1 preserves the exact-top
        # historical/default protocol.
        active = rng.choice(candidates, size=n_active, replace=False)
        active = active[np.argsort(-z_col[active])]
    else:
        active = candidates[:n_active]
    zero_pool = pool[z_pool == 0]
    if len(zero_pool) > n_zero:
        zero = rng.choice(zero_pool, size=n_zero, replace=False)
    else:
        zero = zero_pool
    return {"active": active, "zero": zero}


def close_silent_order(active_codes: np.ndarray, silent_codes: np.ndarray, f: int) -> np.ndarray:
    """Order silent candidates by code-space similarity to the activators, feature f removed.

    'Hard negatives' for NAMING: silent-on-f responses that share the model's OTHER concepts
    with the activators but don't fire f. Zeroing column f (and using the lens's own codes as
    the similarity space) holds the confounding surface traits ~constant, so the naming
    contrast isolates feature f instead of a generic property many responses share (e.g. the
    LLM can't call it "uses bullet lists" when the similar controls use bullet lists too).
    Returns an argsort order over ``silent_codes`` rows (most-similar first)."""
    A = np.asarray(active_codes, dtype=np.float64).copy()
    S = np.asarray(silent_codes, dtype=np.float64).copy()
    A[:, f] = 0.0
    S[:, f] = 0.0
    cen = A.mean(axis=0)
    cen /= (np.linalg.norm(cen) + 1e-12)
    S /= (np.linalg.norm(S, axis=1, keepdims=True) + 1e-12)
    return np.argsort(-(S @ cen))


def holdout_buckets(z_col: np.ndarray, pool: np.ndarray, n_per_bucket: int,
                    rng: np.random.Generator, *, sampling: str = "extremes") -> dict:
    """Held-out pos/neg/tie buckets from ``pool``.

    ``extremes`` keeps the original top-|z| case-control protocol.
    ``stratified-random`` samples uniformly within each sign bucket, covering the
    activation range while retaining enough positive/negative/control cases.
    """
    if sampling not in ("extremes", "stratified-random"):
        raise ValueError(
            "sampling must be 'extremes' or 'stratified-random', "
            f"got {sampling!r}")
    pool = np.asarray(pool)
    z_pool = z_col[pool]
    pos_pool = pool[z_pool > 0]
    neg_pool = pool[z_pool < 0]
    tie_pool = pool[z_pool == 0]

    def choose(idx_pool):
        if len(idx_pool) == 0:
            return np.array([], dtype=int)
        if sampling == "stratified-random":
            if len(idx_pool) > n_per_bucket:
                return rng.choice(idx_pool, size=n_per_bucket, replace=False)
            return idx_pool
        return idx_pool[np.argsort(-np.abs(z_col[idx_pool]))][:n_per_bucket]

    pos = choose(pos_pool)
    neg = choose(neg_pool)
    if len(tie_pool) > n_per_bucket:
        tie = rng.choice(tie_pool, size=n_per_bucket, replace=False)
    else:
        tie = tie_pool
    return {"pos": pos, "neg": neg, "tie": tie}


@registry.register("negative_sampler", "random")
def _random_negatives(z_col, pool, n, *, active_idx=None, embeddings=None, rng=None):
    """Random silent examples — identical to top_pairs' zero bucket."""
    pool = np.asarray(pool)
    silent = pool[z_col[pool] == 0]
    if rng is None:
        rng = np.random.default_rng(0)
    if len(silent) > n:
        return rng.choice(silent, size=n, replace=False)
    return silent


# "close" is the pipeline/CLI name for hard negatives (naming implements it via
# close_silent_order; verify resolves it here). Register the same cosine sampler under BOTH
# "similar" and "close" so `--negatives close` doesn't crash verification (was: KeyError,
# only random/similar registered — the blocking bug).
@registry.register("negative_sampler", "close")
@registry.register("negative_sampler", "similar")
def _similar_negatives(z_col, pool, n, *, active_idx=None, embeddings=None, rng=None):
    """Silent examples most cosine-similar to the top-activating exemplars.

    Ranks silent candidates by mean cosine similarity to the ``active_idx`` rows'
    embeddings (= cosine to the normalized active centroid). Deterministic stable
    sort; returns min(n, n_silent).
    """
    if embeddings is None or active_idx is None or len(active_idx) == 0:
        raise ValueError("'similar' negatives need embeddings and a non-empty active_idx")
    pool = np.asarray(pool)
    silent = pool[z_col[pool] == 0]
    if len(silent) == 0:
        return silent
    e = np.asarray(embeddings, dtype=np.float64)
    e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)
    centroid = e[np.asarray(active_idx, dtype=int)].mean(axis=0)
    sims = e[silent] @ centroid                          # mean cosine to active set
    order = np.argsort(-sims, kind="stable")
    return silent[order[:n]]


def select_negatives(z_col, pool, n, *, strategy="random", active_idx=None,
                     embeddings=None, rng=None) -> np.ndarray:
    """Pick ``n`` silent (z==0) negative examples for falsifying a feature's
    explanation, by named strategy (registry kind ``negative_sampler``)."""
    sampler = registry.get("negative_sampler", strategy)
    return sampler(z_col, pool, n, active_idx=active_idx, embeddings=embeddings, rng=rng)
