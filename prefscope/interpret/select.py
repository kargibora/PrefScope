"""Pure-numpy example selection for feature interpretation and verification."""
from __future__ import annotations

import hashlib

import numpy as np

from prefscope.core import registry


# Hard-negative mining is only a candidate-ranking step: scanning every 4,096-d
# embedding for every feature is both unnecessary and catastrophic at corpus scale.
# Four thousand candidates gives a broad deterministic search pool while keeping one
# feature's temporary matrix around 66 MiB in float32 for Qwen3-Embedding-8B.
_SIMILAR_CANDIDATE_CAP = 4_000

_SAMPLING_ALIASES = {
    # Historical name: this sampled uniformly inside each active/sign bucket; it
    # did not stratify by activation magnitude. Keep accepting it for checkpoints
    # and scripts, but expose the honest name in new CLI output.
    "stratified-random": "random-active",
}


def normalize_verification_sampling(sampling: str) -> str:
    """Return the canonical verification sampling name."""
    value = _SAMPLING_ALIASES.get(str(sampling), str(sampling))
    valid = {"extremes", "random-active", "quantile-stratified"}
    if value not in valid:
        raise ValueError(
            "sampling must be 'extremes', 'random-active', or "
            f"'quantile-stratified', got {sampling!r}")
    return value


def quantile_stratified_sample(indices: np.ndarray, values: np.ndarray, n: int,
                               rng: np.random.Generator,
                               *, max_strata: int = 5) -> np.ndarray:
    """Sample approximately equally across activation-magnitude quantiles.

    ``indices`` are global row indices and ``values`` is the aligned full feature
    column. Each non-empty stratum contributes before any stratum contributes a
    second row, so weak, medium, and strong activations are all represented.
    """
    indices = np.asarray(indices, dtype=int)
    if n <= 0 or len(indices) == 0:
        return np.array([], dtype=int)
    if len(indices) <= n:
        return indices.copy()
    order = indices[np.argsort(np.abs(values[indices]), kind="stable")]
    strata = [np.asarray(x, dtype=int) for x in np.array_split(
        order, min(max_strata, n, len(order))) if len(x)]
    strata = [rng.permutation(x).tolist() for x in strata]
    out: list[int] = []
    while len(out) < n and any(strata):
        for stratum in strata:
            if stratum and len(out) < n:
                out.append(int(stratum.pop()))
    return np.asarray(out, dtype=int)


def name_verify_split(instruction_ids, verify_frac: float = 0.2):
    """Deterministic disjoint split: hash instruction_id, low buckets -> verify.
    Returns (name_mask, verify_mask), both (N,) bool, covering all rows."""
    thresh = int(verify_frac * 1000)
    verify = np.zeros(len(instruction_ids), dtype=bool)
    for i, iid in enumerate(instruction_ids):
        bucket = int(hashlib.sha1(str(iid).encode()).hexdigest(), 16) % 1000
        verify[i] = bucket < thresh
    return ~verify, verify


def split_group_ids(battles) -> list[str]:
    """Prompt-level groups used to keep repeated prompts in the same split."""
    if "group_id" in battles.columns:
        return battles["group_id"].astype(str).tolist()
    if "prompt" in battles.columns:
        return [hashlib.sha1(str(value).encode()).hexdigest()[:16]
                for value in battles["prompt"]]
    return battles["instruction_id"].astype(str).tolist()


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
    ``random-active`` samples uniformly inside each sign bucket.
    ``quantile-stratified`` covers weak through strong activation magnitudes.
    The legacy name ``stratified-random`` is an alias for ``random-active``.
    """
    sampling = normalize_verification_sampling(sampling)
    pool = np.asarray(pool)
    z_pool = z_col[pool]
    pos_pool = pool[z_pool > 0]
    neg_pool = pool[z_pool < 0]
    tie_pool = pool[z_pool == 0]

    def choose(idx_pool):
        if len(idx_pool) == 0:
            return np.array([], dtype=int)
        if sampling == "random-active":
            if len(idx_pool) > n_per_bucket:
                return rng.choice(idx_pool, size=n_per_bucket, replace=False)
            return idx_pool
        if sampling == "quantile-stratified":
            return quantile_stratified_sample(idx_pool, z_col, n_per_bucket, rng)
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

    Ranks a bounded sample of silent candidates by mean cosine similarity to the
    ``active_idx`` rows' embeddings (= cosine to the normalized active centroid).
    Only the indexed rows are materialized, in float32: a corpus-scale embedding
    memmap must never be copied wholesale merely to choose a handful of controls.
    """
    if embeddings is None or active_idx is None or len(active_idx) == 0:
        raise ValueError("'similar' negatives need embeddings and a non-empty active_idx")
    pool = np.asarray(pool)
    silent = pool[z_col[pool] == 0]
    if len(silent) == 0:
        return silent
    if rng is None:
        rng = np.random.default_rng(0)
    candidates = silent
    if len(candidates) > _SIMILAR_CANDIDATE_CAP:
        candidates = rng.choice(
            candidates, size=_SIMILAR_CANDIDATE_CAP, replace=False)

    active_idx = np.asarray(active_idx, dtype=int)
    candidates = np.asarray(candidates, dtype=int)
    # np.memmap advanced indexing reads only these rows. Keep the fallback for small
    # Python-list inputs used by third-party callers; real lens embeddings are ndarrays.
    try:
        active_e = np.asarray(embeddings[active_idx], dtype=np.float32)
        candidate_e = np.asarray(embeddings[candidates], dtype=np.float32)
    except (TypeError, IndexError):
        e = np.asarray(embeddings, dtype=np.float32)
        active_e, candidate_e = e[active_idx], e[candidates]
    active_e = active_e / (np.linalg.norm(active_e, axis=1, keepdims=True) + 1e-12)
    candidate_e = candidate_e / (
        np.linalg.norm(candidate_e, axis=1, keepdims=True) + 1e-12)
    centroid = active_e.mean(axis=0)
    sims = candidate_e @ centroid                        # mean cosine to active set
    order = np.argsort(-sims, kind="stable")
    return candidates[order[:n]]


def select_negatives(z_col, pool, n, *, strategy="random", active_idx=None,
                     embeddings=None, rng=None) -> np.ndarray:
    """Pick ``n`` silent (z==0) negative examples for falsifying a feature's
    explanation, by named strategy (registry kind ``negative_sampler``)."""
    sampler = registry.get("negative_sampler", strategy)
    return sampler(z_col, pool, n, active_idx=active_idx, embeddings=embeddings, rng=rng)
