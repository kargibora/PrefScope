"""Per-model diagnosis exports: report-card rows from the oriented bank and the
paired head-to-head feature contrast."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _absolute_fire_rate(lens: Path, models, feats):
    """Per-model ABSOLUTE prevalence ``P(z_self > 0)``: the fraction of a model's OWN
    responses that express each feature's positive pole.

    Read from the lens's **per-side absolute** codes ``z_a``/``z_b`` (``z_a = f(e_a)``,
    ``z_b = f(e_b)``) — the honest "did this model's answer express f" signal. The oriented
    bank stores CONTRAST codes (self − other), so a rate off the bank measures how often the
    model DIFFERS from its opponent, not how often it does the thing — which is what the
    "Does a lot" panel implies. ``> 0`` (not ``!= 0``) because the concept name describes the
    positive pole; the opposite pole is a different concept.

    Returns ``{model: np.ndarray(len(feats))}`` or ``None`` when the lens has no per-side
    codes (a difference lens) or the dump is misaligned — caller then falls back to the
    contrast rate and labels it as such."""
    za_p, zb_p, battles_p = lens / "z_a.npy", lens / "z_b.npy", lens / "battles.parquet"
    if not (za_p.exists() and zb_p.exists() and battles_p.exists()):
        return None
    z_a = np.load(za_p, mmap_mode="r")
    z_b = np.load(zb_p, mmap_mode="r")
    battles = pd.read_parquet(battles_p)
    if not ({"model_a", "model_b"} <= set(battles.columns)):
        return None
    if len(battles) != len(z_a) or len(battles) != len(z_b):
        return None
    feats = [int(f) for f in feats]
    pa = np.asarray(z_a[:, feats]) > 0                   # (N, F) A's answer expresses f (+pole)
    pb = np.asarray(z_b[:, feats]) > 0
    ma = battles["model_a"].to_numpy()
    mb = battles["model_b"].to_numpy()
    out = {}
    for m in models:
        rows = []
        if (ma == m).any():
            rows.append(pa[ma == m])                      # m's answers where it was model_a
        if (mb == m).any():
            rows.append(pb[mb == m])                      # m's answers where it was model_b
        out[m] = np.vstack(rows).mean(axis=0) if rows else np.full(len(feats), np.nan)
    return out


def export_diagnosis(lens: Path, features: pd.DataFrame, min_battles=20, *,
                     prompt_lens=None, prompt_names=None) -> dict | None:
    """Per-model net_direction + delta_vs_pool for verified features, from the bank.

    Also emits, per model, the report-card extras the viewer's Report card tab needs:
    ``fire_rate`` (per-feature activation rate for the model) and — when a prompt lens
    is supplied — ``prompt_types`` (per-prompt-concept win rates, via
    ``prompt_concept_winrates``). Both degrade gracefully (``prompt_types`` → [] when
    no prompt lens / no battle-id column)."""
    bank = lens / "bank"
    if not (bank / "bank_codes.npy").exists():
        return None
    from prefscope.pipeline.oriented_bank import load_bank
    Z, meta, _ = load_bank(bank)

    feats = features.loc[features.get("fidelity_pass", False) == True, "feature_id"] \
        if "fidelity_pass" in features else features["feature_id"]
    feats = feats.astype(int).tolist()
    cols = Z[:, feats]                                  # (2N, F)
    sm = meta["self_model"].to_numpy()
    win = meta["win"].to_numpy(dtype=float)

    # battle-id column in the bank meta (build_oriented_codes carries instruction_id);
    # prompt_concept_winrates needs it to align a model's battles to the prompt lens.
    bid_col = next((c for c in ("battle_id", "id", "instruction_id")
                    if c in meta.columns), None)
    prompt_wr_fn = rel_fn = None
    if prompt_lens is not None and bid_col is not None:
        from prefscope.pipeline.report import (prompt_concept_winrates,
                                               prompt_to_response_winrates)
        prompt_wr_fn = prompt_concept_winrates
        rel_fn = prompt_to_response_winrates

    # response-feature id -> concept name, for the per-model prompt→response edges
    if "concept" in features.columns:
        cmap = features.dropna(subset=["concept"]).drop_duplicates("feature_id")
        resp_names = dict(zip(cmap["feature_id"].astype(int), cmap["concept"]))
    else:
        resp_names = {}

    pos = (cols > 0).astype(np.int64)
    neg = (cols < 0).astype(np.int64)
    g = pd.DataFrame(pos, columns=[f"p{f}" for f in feats])
    g["self_model"] = sm
    gp = g.groupby("self_model").sum()
    gn = pd.DataFrame(neg, columns=[f"p{f}" for f in feats]); gn["self_model"] = sm
    gn = gn.groupby("self_model").sum()
    cnt = pd.Series(sm).value_counts()

    keep_models = cnt[cnt >= min_battles].index.tolist()
    # ABSOLUTE per-model prevalence P(z_self>0) for the "Does a lot" panel, from per-side
    # codes. None on a difference lens -> fall back to the bank's contrast disagreement rate
    # and label it honestly via fire_rate_kind so the viewer doesn't imply prevalence (#1).
    abs_fire = _absolute_fire_rate(lens, keep_models, feats)
    fire_rate_kind = "absolute" if abs_fire is not None else "contrast"
    tot = len(sm)
    tot_pos = pos.sum(axis=0).astype(float)             # (F,)
    tot_neg = neg.sum(axis=0).astype(float)

    win_rate = pd.Series(win).groupby(sm).mean()
    rows = {}
    for m in keep_models:
        n = int(cnt[m])
        pm = gp.loc[m].to_numpy(dtype=float)
        nm = gn.loc[m].to_numpy(dtype=float)
        nd_model = pm / n - nm / n
        # pool = everyone else (inside-vs-outside, exact)
        pool_n = tot - n
        nd_pool = (tot_pos - pm) / pool_n - (tot_neg - nm) / pool_n
        # "Does a lot" rate: ABSOLUTE prevalence P(z_self>0) when per-side codes exist,
        # else the bank's contrast disagreement rate (fires positive + negative)/n (#1).
        fire_rate = abs_fire[m] if abs_fire is not None else (pm + nm) / n
        # per-prompt-concept win rates + per-model prompt→response edges
        # (both graceful: [] without a prompt lens / battle-id column)
        prompt_types: list[dict] = []
        relations: list[dict] = []
        mask = (sm == m)
        if prompt_wr_fn is not None:
            try:
                pw = prompt_wr_fn(prompt_lens, meta.loc[mask, bid_col].tolist(),
                                  win[mask], prompt_names=prompt_names,
                                  min_battles=min_battles)
                prompt_types = [{"concept": str(r["prompt_concept"]),
                                 "win_rate": round(float(r["win_rate"]), 4),
                                 "n": int(r["n"])} for _, r in pw.iterrows()]
            except Exception as e:  # never crash the whole export over one model
                print(f"  (prompt_types skipped for {m}: {e})", file=sys.stderr)
        if rel_fn is not None:
            try:
                rl = rel_fn(prompt_lens, meta.loc[mask, bid_col].tolist(),
                            cols[mask], feats, win[mask], prompt_names=prompt_names,
                            response_names=resp_names, min_support=min_battles)
                relations = [{"prompt_concept": str(r["prompt_concept"]),
                              "response_concept": str(r["response_concept"]),
                              "delta_win": round(float(r["delta_win"]), 4),
                              "n": int(r["n"])} for _, r in rl.iterrows()]
            except Exception as e:
                print(f"  (relations skipped for {m}: {e})", file=sys.stderr)
        rows[m] = {
            "win_rate": float(win_rate.get(m, np.nan)),
            "n_battles": n,
            "net_direction": [round(float(v), 5) for v in nd_model],
            "delta_vs_pool": [round(float(a - b), 5) for a, b in zip(nd_model, nd_pool)],
            "fire_rate": [round(float(v), 5) for v in fire_rate],
            # raw per-feature counts (fires-positive / fires-negative in this model's
            # battles) — with the pool totals below, the viewer can compute a proper
            # model-vs-pool z-test + BH instead of showing delta_vs_pool as bare effect.
            "fire_pos": [int(v) for v in pm],
            "fire_neg": [int(v) for v in nm],
            "prompt_types": prompt_types,
            "relations": relations,
        }
    concepts = (features.set_index("feature_id").reindex(feats)["concept"].tolist()
                if "concept" in features else [str(f) for f in feats])
    return {"features": feats, "concepts": concepts, "models": keep_models, "rows": rows,
            # "absolute" = fire_rate is P(z_self>0) prevalence; "contrast" = it's the bank
            # disagreement rate (difference lens) and the viewer must label it distinctly (#1).
            "fire_rate_kind": fire_rate_kind,
            # pool totals over ALL battles (incl. each model's own — client subtracts):
            # pool_pos_f = tot_pos[f] - fire_pos[f], pool_n = n_total - n_battles.
            "tot_pos": [int(v) for v in tot_pos], "tot_neg": [int(v) for v in tot_neg],
            "n_total": int(tot)}


def export_head_to_head(lens: Path, features: pd.DataFrame, diag: dict | None,
                        *, min_shared: int = 30) -> dict | None:
    """Paired, prompt-matched head-to-head feature contrast between model pairs.

    For each unordered model pair (A, B) with at least ``min_shared`` shared battles, and
    each diagnosis feature f, count the *discordant* battles::

        bpos_f = # shared battles where f fires in A's answer but NOT in B's
        cpos_f = # shared battles where f fires in B's answer but NOT in A's

    "fires" == the per-response code is nonzero, read from the lens's **per-side absolute**
    codes ``z_a``/``z_b`` (``z_a = f(e_a)``, ``z_b = f(e_b)``) — NOT the oriented bank. The
    bank stores contrast codes (``z_a-z_b`` and ``z_b-z_a``), which are sign flips with the
    *same* nonzero mask on both sides, so a discordant count off the bank is always zero.
    ``z_a``/``z_b`` are the genuine "did this model's answer express f" signal. Because both
    answers respond to the *same* prompt, this is prompt-matched. The viewer forms the paired
    estimate ``(bpos - cpos)/n`` AND a McNemar test from ``(bpos, cpos)``; ``bpos + cpos``
    (the discordant count) is the effective sample size, so the viewer can gate on power and
    FDR-correct rather than trust ``n``. Concordant battles (both/neither fire) contribute
    nothing, by design.

    Returns ``{models, features, concepts, min_shared, pairs:[{a, b, n, bpos, cpos}]}`` where
    ``a < b`` index into ``models`` and ``bpos``/``cpos`` are per-feature int arrays parallel
    to ``features``. ``None`` if the lens has no per-side codes (a difference lens: only
    ``z_diff``) or there is no diagnosis."""
    za_p, zb_p = lens / "z_a.npy", lens / "z_b.npy"
    battles_p = lens / "battles.parquet"
    if not (za_p.exists() and zb_p.exists() and battles_p.exists()):
        return None                                     # needs a per-side (individual) lens
    if diag is None or not diag.get("models"):
        return None

    # use the exact feature subset + order the report card uses, so viewer lookups align
    feats = [int(f) for f in diag["features"]]
    if not feats:
        return None
    z_a = np.load(za_p, mmap_mode="r")
    z_b = np.load(zb_p, mmap_mode="r")
    battles = pd.read_parquet(battles_p)
    if not ({"model_a", "model_b"} <= set(battles.columns)):
        return None
    if len(battles) != len(z_a) or len(battles) != len(z_b):
        return None                                     # misaligned dump — refuse silently

    # > 0 (not != 0): a signed BatchTopK feature's negative pole is the OPPOSITE concept,
    # so counting it as "expressed f" would let head-to-head tally "opposite of X" as X (#2).
    fa = np.asarray(z_a[:, feats]) > 0                  # (N, F) A's answer expresses f (+pole)
    fb = np.asarray(z_b[:, feats]) > 0                  # (N, F) B's answer expresses f (+pole)
    ma = battles["model_a"].to_numpy()
    mb = battles["model_b"].to_numpy()

    models = diag["models"]
    midx = {m: k for k, m in enumerate(models)}
    M = len(models)
    mi = np.array([midx.get(m, -1) for m in ma])        # model_a index (the "A" side, = fa)
    mj = np.array([midx.get(m, -1) for m in mb])
    valid = (mi >= 0) & (mj >= 0) & (mi != mj)          # both in the report-card universe
    if not valid.any():
        return None
    mi, mj = mi[valid], mj[valid]
    faV, fbV = fa[valid], fb[valid]                     # (Nv, F)
    a_is_lo = mi < mj                                    # orient so pair key is (lo, hi)
    lo = np.minimum(mi, mj)
    hi = np.maximum(mi, mj)

    discA = faV & ~fbV                                   # model_a fires, model_b doesn't
    discB = ~faV & fbV
    a_not_b = np.where(a_is_lo[:, None], discA, discB)   # lo-model fires, hi doesn't
    b_not_a = np.where(a_is_lo[:, None], discB, discA)

    key = lo.astype(np.int64) * M + hi
    uk, inv = np.unique(key, return_inverse=True)
    P = len(uk)
    F = len(feats)
    bpos = np.zeros((P, F), dtype=np.int64)
    cpos = np.zeros((P, F), dtype=np.int64)
    npair = np.zeros(P, dtype=np.int64)
    np.add.at(bpos, inv, a_not_b.astype(np.int64))
    np.add.at(cpos, inv, b_not_a.astype(np.int64))
    np.add.at(npair, inv, 1)

    concepts = diag.get("concepts") or [str(f) for f in feats]
    pairs = []
    for p in range(P):
        if npair[p] < min_shared:
            continue
        pairs.append({"a": int(uk[p] // M), "b": int(uk[p] % M), "n": int(npair[p]),
                      "bpos": [int(x) for x in bpos[p]],
                      "cpos": [int(x) for x in cpos[p]]})
    return {"models": models, "features": feats,
            "concepts": [None if pd.isna(c) else str(c) for c in concepts],
            "min_shared": int(min_shared), "pairs": pairs}
