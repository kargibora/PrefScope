"""2D map exports: battle-level (z_diff), prompt-space (z_prompt), and
single-response (z_a, plus z_b for paired data) UMAP scatters."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from prefscope.artifacts import (
    BATTLES, FEATURE_NAMES, Z_A, Z_B, Z_DIFF, Z_PROMPT, battle_id_col,
    lens_battle_ids,
)
from prefscope.data.pair_schema import orient_by_label

from .sanitize import _concept_or_none, _read_csv


def export_map(lens: Path, corpus_path: str, features: pd.DataFrame, *,
               sample: int = 2500, seed: int = 0, metric: str = "euclidean",
               mode: str = "hybrid") -> dict | None:
    """UMAP the per-battle SAE codes (z_diff) to 2D for the scatter map.

    Each point is one battle, positioned by its difference-activation pattern and
    colored by its dominant verified feature. Needs umap-learn.

    ``mode`` chooses which battles to show:
      - ``random``: uniform sample (faithful to the dataset, but many points have
        no verified feature firing — noise);
      - ``top-activating``: the strongest-activating battles per verified feature
        (clean clusters, over-represents strong signal);
      - ``hybrid`` (default): half top-activating + half random background.
    """
    try:
        import umap  # noqa: F401
    except Exception:
        print("  (skipping map.json — `uv pip install umap-learn` to enable)",
              file=sys.stderr)
        return None

    z = np.load(lens / Z_DIFF)
    battles = None
    if corpus_path:
        from prefscope.interpret.io import load_lens_battles
        battles, z, _ = load_lens_battles(lens, corpus=corpus_path)   # row-aligned
    else:
        bp = lens / BATTLES
        battles = pd.read_parquet(bp) if bp.exists() else None

    n = z.shape[0]
    rng = np.random.default_rng(seed)
    verified = (features.loc[features.get("fidelity_pass", False) == True, "feature_id"]
                .astype(int).tolist() if "fidelity_pass" in features
                else features["feature_id"].astype(int).tolist())
    vset = verified if verified else list(range(z.shape[1]))

    def _top_activating(budget: int) -> np.ndarray:
        per = max(1, budget // max(1, len(vset)))
        keep: set[int] = set()
        for f in vset:
            col = np.abs(z[:, f])
            for i in np.argsort(-col)[: per * 2]:
                if col[i] > 0:
                    keep.add(int(i))
                if len(keep) and len([j for j in keep]) >= budget:
                    break
        return np.array(sorted(keep), dtype=int)[:budget]

    if mode == "random" or not verified:
        idx = rng.choice(n, size=min(sample, n), replace=False)
    elif mode == "top-activating":
        idx = _top_activating(sample)
    else:  # hybrid
        top = _top_activating(sample // 2)
        rest = sample - len(top)
        pool = np.setdiff1d(np.arange(n), top, assume_unique=False)
        rnd = rng.choice(pool, size=min(rest, len(pool)), replace=False) if rest > 0 else np.array([], int)
        idx = np.concatenate([top, rnd])
    idx = np.sort(idx)
    zs = z[idx].astype(np.float32)

    import umap
    nn = max(2, min(15, len(idx) - 1))
    xy = umap.UMAP(n_components=2, n_neighbors=nn, min_dist=0.1,
                   metric=metric, random_state=seed).fit_transform(zs)
    concept_by = features.set_index("feature_id")["concept"].to_dict() \
        if "concept" in features else {}
    vcols = zs[:, verified] if verified else zs
    top_local = np.argmax(np.abs(vcols), axis=1)
    top_fid = [verified[t] if verified else int(t) for t in top_local]
    # dominant verified-feature magnitude per point (0 = nothing fired = noise)
    mags = np.abs(vcols).max(axis=1)
    any_fire = (np.abs(vcols).sum(axis=1) > 0) if verified else np.ones(len(idx), bool)

    def _clip(s, n):
        s = str(s)
        return s if len(s) <= n else s[:n] + " …[truncated]"

    pts = []
    for k, i in enumerate(idx):
        b = battles.iloc[int(i)] if battles is not None else None
        pts.append({
            "x": round(float(xy[k, 0]), 3), "y": round(float(xy[k, 1]), 3),
            "f": int(top_fid[k]) if any_fire[k] else -1,
            "m": round(float(mags[k]), 3),       # dominant activation magnitude
            "ma": str(b.get("model_a", "")) if b is not None else "",
            "mb": str(b.get("model_b", "")) if b is not None else "",
            "p": _clip(b.get("prompt", ""), 1000) if b is not None else "",
            # full(ish) text so a click can show prompt + both responses
            "ca": _clip(b.get("completion_a", ""), 1800) if b is not None else "",
            "cb": _clip(b.get("completion_b", ""), 1800) if b is not None else "",
        })
    return {"n_total": int(n), "n_sampled": int(len(idx)), "metric": metric, "mode": mode,
            "features": verified,
            # _concept_or_none guards against a NaN name leaking a non-JSON `NaN` token
            "concepts": [_concept_or_none(concept_by, f) or f"feature {f}" for f in verified],
            "points": pts}


def _battle_ids_of(lens_dir: Path) -> np.ndarray:
    return lens_battle_ids(lens_dir)


def _concept_map(p: Path, key="feature_id", val="concept") -> dict:
    d = _read_csv(p)
    return dict(zip(d[key].astype(int), d[val])) if d is not None and key in d and val in d else {}


def _clip_text(s, n: int) -> str:
    s = str(s)
    return s if len(s) <= n else s[:n] + " …[truncated]"


def _project2d(z, seed: int = 0):
    """UMAP to 2D; deterministic top-2 SVD fallback when umap-learn isn't installed."""
    z = np.asarray(z, dtype=np.float32)
    try:
        import umap
        nn = max(2, min(15, len(z) - 1))
        return umap.UMAP(n_components=2, n_neighbors=nn, min_dist=0.1,
                         random_state=seed).fit_transform(z)
    except Exception:
        x = z - z.mean(0, keepdims=True)
        u, s, _ = np.linalg.svd(x, full_matrices=False)
        comp = u[:, :2] * s[:2]
        if comp.shape[1] < 2:
            comp = np.pad(comp, ((0, 0), (0, 2 - comp.shape[1])))
        return comp


def export_prompt_map(prompt_lens, completion_lens, delta_csv, prompt_interpret_dir, *,
                      sample: int = 2000, mode: str = "hybrid", seed: int = 0,
                      topk: int = 6, corpus_path: str = "") -> dict | None:
    """Prompt-space map: one point per (decisive) battle, positioned by z_prompt and
    colored by dominant prompt behavior. Each point carries the prompt features firing
    (z_prompt) and the winner's completion features (oriented z_diff), with the cells
    that are significant in the aggregate Δ relation badged. See the design spec."""
    prompt_lens, completion_lens = Path(prompt_lens), Path(completion_lens)
    pint = Path(prompt_interpret_dir)
    zp = np.load(prompt_lens / Z_PROMPT)
    zd = np.load(completion_lens / Z_DIFF)
    pb, cb = _battle_ids_of(prompt_lens), _battle_ids_of(completion_lens)

    # row-align the two lenses by battle id (same pattern as pipeline.prompt_delta)
    if len(pb) == len(cb) and bool((pb == cb).all()):
        bids = cb
    else:
        common = pd.Index(cb).intersection(pd.Index(pb))
        cpos = {b: i for i, b in enumerate(cb)}
        ppos = {b: i for i, b in enumerate(pb)}
        ic = np.array([cpos[b] for b in common])
        ip = np.array([ppos[b] for b in common])
        zd, zp, bids = zd[ic], zp[ip], common.to_numpy()

    # battle metadata (prompt / winner / models): pull each column from whichever
    # source has it — the completion-lens battles or the corpus — keyed by battle id.
    # (The difference lens's battles.parquet may not store prompt/model text.)
    def _indexed(df):
        return df.set_index(df[battle_id_col(df)].astype(str))
    cbat = _indexed(pd.read_parquet(completion_lens / BATTLES))
    corp = _indexed(pd.read_parquet(corpus_path)) if corpus_path else None

    def meta_col(name, default=""):
        """values aligned to the current `bids`, from lens battles else corpus."""
        ids = pd.Index(bids).astype(str)
        if name in cbat.columns:
            return cbat[name].reindex(ids)
        if corp is not None and name in corp.columns:
            return corp[name].reindex(ids)
        return pd.Series(default, index=ids)

    hp = meta_col("human_pref", np.nan)
    if bool(hp.isna().all()):
        print("  (skipping prompt_map.json — no human_pref in lens battles or corpus; "
              "rebuild corpus with --keep-labels and pass --corpus)", file=sys.stderr)
        return None
    y = hp.to_numpy(dtype=float)

    # orient z_diff toward the human-preferred response; drop ties (no winner, no Δ row)
    zd, keep = orient_by_label(y, zd)
    zp, bids, y = zp[keep], bids[keep], y[keep]
    n = len(bids)
    if n == 0:
        return None

    pnames = _concept_map(pint / "prompt_feature_names.csv")
    cnames = _concept_map(completion_lens / FEATURE_NAMES)
    pfid = _read_csv(pint / "prompt_feature_fidelity.csv")
    verified = (pfid.loc[pfid.get("fidelity_pass", False) == True, "feature_id"]
                .astype(int).tolist() if pfid is not None and "fidelity_pass" in pfid else [])
    cand = [c for c in (verified or sorted(pnames) or list(range(zp.shape[1]))) if c < zp.shape[1]]
    candA = np.array(cand)
    # dominant prompt feature per battle, POSITIVE pole only (max > 0) — abs-argmax could
    # pick the negative pole and label the point with a concept name verified only for the
    # positive pole. -1 = no positive prompt concept present.
    zc = zp[:, cand]
    dom = np.where(zc.max(axis=1) > 0, candA[zc.argmax(axis=1)], -1)

    # prompt feature -> cluster (optional)
    pclu = _read_csv(pint / "prompt_feature_clusters.csv")
    f2c, behaviors = {}, {}
    if pclu is not None and "cluster_id" in pclu.columns:
        f2c = dict(zip(pclu["feature_id"].astype(int), pclu["cluster_id"].astype(int)))
        if "behavior" in pclu.columns:
            behaviors = {int(c): str(b) for c, b in pclu.dropna(subset=["behavior"])
                         .groupby("cluster_id")["behavior"].first().items()}
    clu_key = np.array([f2c.get(int(d), -1) for d in dom]) if f2c else None

    # Δ lookup; derive the point's key in the SAME keyspace the delta CSV used
    delta = _read_csv(Path(delta_csv)) if delta_csv else None
    dlook, dkeys = {}, set()
    if delta is not None and {"prompt_concept", "completion_feature"} <= set(delta.columns):
        sig = (delta["stable"].astype(bool) & (delta["p_bonferroni"].astype(float) < 0.05)
               if "stable" in delta and "p_bonferroni" in delta
               else pd.Series(False, index=delta.index))
        for pc_, cf_, dv, sg in zip(delta["prompt_concept"].astype(int),
                                    delta["completion_feature"].astype(int),
                                    delta["delta"].astype(float), sig):
            dlook[(int(pc_), int(cf_))] = (float(dv), bool(sg))
        dkeys = set(int(k) for k in delta["prompt_concept"].astype(int))
    use_cluster = bool(f2c) and (clu_key is not None) and (
        len(dkeys & set(int(c) for c in clu_key)) >= len(dkeys & set(int(d) for d in dom)))
    point_concept = clu_key if use_cluster else dom

    # sample (ties already dropped)
    rng = np.random.default_rng(seed)
    if n <= sample:
        idx = np.arange(n)
    elif mode == "random" or not verified:
        idx = rng.choice(n, size=sample, replace=False)
    else:  # hybrid: top-activating per candidate prompt feature + random background
        top = set()
        per = max(1, (sample // 2) // max(1, len(cand)))
        for c in cand:
            col = np.abs(zp[:, c])
            for i in np.argsort(-col)[:per]:
                if col[i] > 0:
                    top.add(int(i))
        top = np.array(sorted(top), int)[: sample // 2]
        pool = np.setdiff1d(np.arange(n), top)
        rest = min(sample - len(top), len(pool))
        rnd = rng.choice(pool, size=rest, replace=False) if rest > 0 else np.array([], int)
        idx = np.concatenate([top, rnd])
    idx = np.sort(idx.astype(int))

    xy = _project2d(zp[idx], seed)
    prompts = meta_col("prompt").astype(str).to_numpy()
    ma = meta_col("model_a").astype(str).to_numpy()
    mb = meta_col("model_b").astype(str).to_numpy()
    ca = meta_col("completion_a").astype(str).to_numpy()
    cb = meta_col("completion_b").astype(str).to_numpy()

    pts = []
    for k, i in enumerate(idx):
        zpi, zdi = zp[i], zd[i]
        # prompt features: top 5 that actually FIRE (positive activation only)
        pf = []
        for j in np.argsort(-zpi[cand]):
            z = float(zpi[candA[j]])
            if z <= 1e-9:
                break
            pf.append({"id": int(candA[j]), "concept": str(pnames.get(int(candA[j]), candA[j])),
                       "z": round(z, 3)})
            if len(pf) >= 5:
                break
        # completion features: top |Δ| in BOTH directions (+ = winner-more, − = loser-more)
        pc = int(point_concept[i])
        cf = []
        for j in np.argsort(-np.abs(zdi)):
            z = float(zdi[j])
            if abs(z) < 1e-9:
                break
            dv, sg = dlook.get((pc, int(j)), (None, False))
            cf.append({"id": int(j), "concept": str(cnames.get(int(j), j)), "z": round(z, 3),
                       "delta": (round(dv, 3) if dv is not None else None), "sig": bool(sg)})
            if len(cf) >= topk:
                break
        di = int(dom[i])                                   # -1 = no positive prompt concept
        pts.append({"x": round(float(xy[k, 0]), 3), "y": round(float(xy[k, 1]), 3),
                    "f": di, "m": round(float(zpi[di]), 3) if di >= 0 else 0.0, "pc": pc,
                    "ma": ma[i], "mb": mb[i], "win": "A" if y[i] > 0.5 else "B",
                    "p": _clip_text(prompts[i], 1000),
                    "ca": _clip_text(ca[i], 1800), "cb": _clip_text(cb[i], 1800),
                    "pf": pf, "cf": cf})

    feats = [int(c) for c in cand]
    out = {"n_total": int(n), "n_sampled": int(len(idx)), "mode": mode,
           "features": feats, "concepts": [str(pnames.get(c, c)) for c in feats],
           "points": pts}
    if f2c:
        out["clusters"] = [int(f2c.get(c, -1)) for c in feats]
        if behaviors:
            out["behaviors"] = {str(c): b for c, b in behaviors.items()}
    return out


def export_response_map(lens, corpus_path, features, *, sample: int = 2500,
                        seed: int = 0, mode: str = "hybrid") -> dict | None:
    """Feature map at the RESPONSE level (individual lens): one point per single
    response (A and, for paired data, B), positioned by its individual SAE code and colored by
    its dominant verified feature. Clicking a point shows that ONE response + its
    activation — unlike the battle map (z_diff) where a point is an A/B pair."""
    lens = Path(lens)
    if not (lens / Z_A).exists():
        return None
    za = np.load(lens / Z_A)
    paired = (lens / Z_B).exists()
    zb = np.load(lens / Z_B) if paired else None
    Z = (np.vstack([za, zb]) if paired else za).astype(np.float32)
    lb = pd.read_parquet(lens / BATTLES)
    ids = lens_battle_ids(lb)
    N = len(ids)
    corp = pd.read_parquet(corpus_path) if corpus_path else None
    if corp is not None:
        corp = corp.set_index(corp[battle_id_col(corp)].astype(str))

    def col(name, default=""):
        if corp is not None and name in corp.columns:
            return corp[name].reindex(pd.Index(ids)).fillna(default).astype(str).to_numpy()
        if name in lb.columns:
            return lb[name].fillna(default).astype(str).to_numpy()
        return np.array([default] * N, dtype=object)
    prompts, ca, cb = col("prompt"), col("completion_a"), col("completion_b")
    ma, mb = col("model_a"), col("model_b")

    verified = (features.loc[features.get("fidelity_pass", False) == True, "feature_id"]
                .astype(int).tolist() if "fidelity_pass" in features
                else features["feature_id"].astype(int).tolist())
    vset = verified if verified else list(range(Z.shape[1]))
    n2 = Z.shape[0]
    rng = np.random.default_rng(seed)

    def _top(budget):
        per = max(1, budget // max(1, len(vset)))
        keep: set[int] = set()
        for f in vset:
            # Feature names describe the verified positive pole.  A large negative
            # code is evidence for the opposite pole, not a strong example of the
            # named behaviour.
            col_ = Z[:, f]
            for i in np.argsort(-col_)[:per]:
                if col_[i] > 0:
                    keep.add(int(i))
        return np.array(sorted(keep), dtype=int)[:budget]
    if mode == "random" or not verified:
        idx = rng.choice(n2, size=min(sample, n2), replace=False)
    elif mode == "top-activating":
        idx = _top(sample)
    else:
        top = _top(sample // 2)
        pool = np.setdiff1d(np.arange(n2), top)
        rest = sample - len(top)
        rnd = rng.choice(pool, size=min(rest, len(pool)), replace=False) if rest > 0 else np.array([], int)
        idx = np.concatenate([top, rnd])
    idx = np.sort(idx.astype(int))

    xy = _project2d(Z[idx], seed)
    vcols = Z[idx][:, vset]
    top_local = np.argmax(vcols, axis=1)
    top_fid = [vset[t] for t in top_local]
    mags = vcols.max(axis=1)
    anyf = mags > 0
    concept_by = features.set_index("feature_id")["concept"].to_dict() if "concept" in features else {}

    pts = []
    for k, j in enumerate(idx):
        b = int(j % N); is_a = (not paired) or j < N
        pts.append({"x": round(float(xy[k, 0]), 3), "y": round(float(xy[k, 1]), 3),
                    "f": int(top_fid[k]) if anyf[k] else -1,
                    "m": round(float(mags[k]), 3) if anyf[k] else 0.0,
                    "side": "A" if is_a else "B", "model": (ma[b] if is_a else mb[b]),
                    "p": _clip_text(prompts[b], 600),
                    "r": _clip_text(ca[b] if is_a else cb[b], 1800)})
    return {"n_total": int(n2), "n_sampled": int(len(idx)), "mode": mode,
            "features": [int(f) for f in vset],
            "concepts": [str(concept_by.get(f, f)) for f in vset], "points": pts}
