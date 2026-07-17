"""Relationship-table exports: delta / conditional heatmaps, elicitation edges,
the bias screen, and the prompt-feature table."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .sanitize import _concept_or_none, _read_csv, _round


def export_delta(delta_csv, features: pd.DataFrame, bias_csv=None) -> dict | None:
    """Prompt-conditioned delta matrix Δ_{k,f} for the relationship heatmap.

    Rows = prompt concepts, cols = completion features (with behavior, win_assoc,
    fidelity, confound flag attached). All tested cells are emitted; the viewer
    masks non-stable / non-significant cells (it must never imply an untested null).
    """
    d = _read_csv(Path(delta_csv)) if delta_csv else None
    if d is None or not len(d):
        return None
    fcols = features.copy()
    bias = _read_csv(Path(bias_csv)) if bias_csv else None
    if bias is not None and "confound_entangled" in bias.columns:
        fcols = fcols.merge(bias[["feature_id", "confound_entangled"]],
                            on="feature_id", how="left")
    fmap = fcols.set_index("feature_id")

    def _col(fid: int) -> dict:
        r = fmap.loc[fid].to_dict() if fid in fmap.index else {}
        g = lambda k: (None if k not in r or pd.isna(r[k]) else r[k])  # noqa: E731
        return {"id": int(fid), "concept": g("concept"), "behavior": g("behavior"),
                "cluster_id": (int(r["cluster_id"]) if pd.notna(r.get("cluster_id")) else None),
                "win_assoc": g("win_assoc"),
                "fidelity_pass": (bool(r["fidelity_pass"]) if pd.notna(r.get("fidelity_pass")) else None),
                "confound_entangled": (bool(r["confound_entangled"]) if pd.notna(r.get("confound_entangled")) else None)}

    pc_name = (dict(zip(d["prompt_concept"], d["prompt_concept_name"]))
               if "prompt_concept_name" in d.columns else {})
    pcs = sorted(d["prompt_concept"].unique().tolist())
    feat_ids = sorted(d["completion_feature"].unique().tolist())
    cells = [{"pc": int(r.prompt_concept), "cf": int(r.completion_feature),
              "delta": round(float(r.delta), 4),
              "p": (round(float(r.p_bonferroni), 4) if pd.notna(r.p_bonferroni) else None),
              "stable": bool(r.stable)} for r in d.itertuples()]
    n_sig = sum(1 for c in cells if c["stable"] and c["p"] is not None and c["p"] < 0.05)
    return {"prompt_concepts": [{"id": int(k), "name": _concept_or_none(pc_name, k)} for k in pcs],
            "completion_features": [_col(f) for f in feat_ids],
            "cells": cells, "n_cells": len(cells), "n_significant": n_sig}


def export_conditional(cond_csv, features: pd.DataFrame, delta_csv=None) -> dict | None:
    """Conditional δ_{f,k}: length-controlled Δwin-rate of behavior f *within* prompt
    type k. This is the framework's thesis as a statistic — a behavior can win for one
    prompt type and lose for another. Emits the (k, f) cells + labels; the viewer masks
    non-significant cells and ranks the sign-flips (rewarded here, penalised there)."""
    d = _read_csv(Path(cond_csv)) if cond_csv else None
    if d is None or not len(d):
        return None
    need = {"prompt_concept", "feature_id", "delta_win_rate"}
    if not need <= set(d.columns):
        return None
    cmap = (features.set_index("feature_id")["concept"].to_dict()
            if "concept" in features else {})
    # prompt-concept names. The conditional table is keyed by prompt CLUSTERS, and the
    # conditional CSV itself carries the cluster name (`prompt_concept_name`, attached
    # by prompt_delta from the clusters' `behavior`). Prefer that — it's the right
    # keyspace. Fall back to the delta CSV's column only if the conditional CSV lacks
    # it (delta is keyed by RAW concepts, so it can mislabel clusters — last resort).
    pc_name = {}
    if "prompt_concept_name" in d.columns:
        nm = d.dropna(subset=["prompt_concept_name"])
        pc_name = dict(zip(nm["prompt_concept"].astype(int), nm["prompt_concept_name"]))
    if not pc_name:
        dd = _read_csv(Path(delta_csv)) if delta_csv else None
        if dd is not None and "prompt_concept_name" in dd.columns:
            pc_name = dict(zip(dd["prompt_concept"].astype(int), dd["prompt_concept_name"]))

    def g(r, k):
        return None if k not in r or pd.isna(r[k]) else r[k]

    cells = []
    for r in d.itertuples():
        rd = r._asdict()
        cells.append({
            "pc": int(r.prompt_concept), "f": int(r.feature_id),
            "delta": round(float(r.delta_win_rate), 4),
            "p": (round(float(rd["cond_p_bonferroni"]), 4)
                  if "cond_p_bonferroni" in rd and pd.notna(rd["cond_p_bonferroni"]) else None),
            "sig": bool(g(rd, "cond_significant")) if "cond_significant" in rd else False,
            "n": (int(rd["n_battles"]) if "n_battles" in rd and pd.notna(rd["n_battles"]) else None),
            # effective support: battles of this type where the feature FIRES (the honest
            # per-cell n — n_battles alone overstates a rarely-firing feature's support).
            "nf": (int(rd["n_fire"]) if "n_fire" in rd and pd.notna(rd["n_fire"]) else None),
        })
    pcs = sorted({c["pc"] for c in cells})
    fids = sorted({c["f"] for c in cells})
    n_sig = sum(1 for c in cells if c["sig"])
    return {"prompt_concepts": [{"id": k, "name": _concept_or_none(pc_name, k)} for k in pcs],
            "features": [{"id": f, "concept": _concept_or_none(cmap, f)} for f in fids],
            "cells": cells, "n_cells": len(cells), "n_significant": n_sig}


def export_elicitation(elic_csv, *, max_edges: int = 24000,
                       per_feature: int = 15, per_prompt: int = 30) -> dict | None:
    """Prompt-concept → response-concept elicitation edges (co-activation lift).

    Descriptive, preference-independent: when prompt concept X is present, which
    response concepts Y appear above base rate. Emits a directed edge list (lift,
    P(Y|X), support, significance) that BOTH hubs read — the Feature panel reads it
    cy→px ("activated by") and the Prompt panel reads it px→cy ("elicits").

    Coverage-first cap: a plain global top-N by |lift| starved individual features (a
    content-bound feature's *activating* edge could rank below other features' stronger
    edges and get dropped). Instead keep, per response feature, its top ``per_feature``
    edges by lift AND, per prompt concept, its top ``per_prompt`` — plus every significant
    edge — so every concept has its strongest edges on both sides. Union, then a generous
    ``max_edges`` ceiling as a payload backstop."""
    d = _read_csv(Path(elic_csv)) if elic_csv else None
    if d is None or not len(d):
        return None
    need = {"prompt_feature", "completion_feature", "lift"}
    if not need <= set(d.columns):
        return None
    d = d.copy()
    if "log2_lift" not in d.columns:
        d["log2_lift"] = np.log2(d["lift"].clip(lower=1e-6))
    # NB: helper columns must NOT start with "_" — itertuples() renames such columns to
    # positional names, which silently broke reading `sig` back per row.
    d["absl2"] = d["log2_lift"].abs()
    d["sigf"] = (d["significant"].astype(bool) if "significant" in d.columns
                 else pd.Series(False, index=d.index))
    d = d.drop_duplicates(["prompt_feature", "completion_feature"])
    # TRUE totals over the full tested set (the CSV holds every reported tested cell) —
    # the headline "N of M significant" must NOT be computed over the capped subset.
    n_total = int(len(d))
    n_sig_total = int(d["sigf"].sum())
    # per-concept coverage: each feature's / prompt's strongest edges by |log2 lift| —
    # SYMMETRIC, so suppression edges (lift<1, l2<0) are kept alongside activation edges
    # (sorting by raw lift desc would systematically drop the suppression side) — plus
    # every significant edge, so no concept is starved.
    by_l2 = d.sort_values("absl2", ascending=False)
    kept = pd.concat([
        by_l2.groupby("completion_feature", sort=False).head(per_feature),
        by_l2.groupby("prompt_feature", sort=False).head(per_prompt),
        d[d["sigf"]],
    ]).drop_duplicates(["prompt_feature", "completion_feature"])
    if len(kept) > max_edges:  # payload backstop: keep significant-first then strongest
        kept = kept.sort_values(["sigf", "absl2"], ascending=[False, False]).head(max_edges)

    pn = (dict(zip(d["prompt_feature"], d["prompt_feature_name"]))
          if "prompt_feature_name" in d.columns else {})
    cn = (dict(zip(d["completion_feature"], d["completion_feature_name"]))
          if "completion_feature_name" in d.columns else {})

    def g(r, k, default=None):
        return default if k not in r or pd.isna(r[k]) else r[k]

    edges = []
    for r in kept.itertuples():
        rd = r._asdict()
        edges.append({
            "px": int(r.prompt_feature), "cy": int(r.completion_feature),
            "lift": round(float(r.lift), 3),
            "l2": round(float(g(rd, "log2_lift", 0.0)), 3),
            "pyx": round(float(g(rd, "p_y_given_x", float("nan"))), 4),
            "py": round(float(g(rd, "p_y", float("nan"))), 4),
            "nx": int(g(rd, "n_x", 0)), "nco": int(g(rd, "n_cooccur", 0)),
            "p": (round(float(rd["p_bonferroni"]), 5)
                  if "p_bonferroni" in rd and pd.notna(rd["p_bonferroni"]) else None),
            "sig": bool(g(rd, "sigf", False)),
        })
    pxs = sorted({e["px"] for e in edges})
    cys = sorted({e["cy"] for e in edges})
    return {"prompt_concepts": [{"id": k, "concept": _concept_or_none(pn, k)} for k in pxs],
            "response_concepts": [{"id": k, "concept": _concept_or_none(cn, k)} for k in cys],
            "edges": edges,
            "n_edges": n_total,            # full tested-and-reported count
            "n_significant": n_sig_total,  # significant among ALL tested, not just shown
            "n_shown": len(edges)}         # how many made it past the payload cap


def export_bias_screen(bias_csv) -> list | None:
    b = _read_csv(Path(bias_csv)) if bias_csv else None
    return _round(b) if b is not None and len(b) else None


def export_prompt_features(interpret_dir) -> dict | None:
    """Prompt-lens concepts (names + optional fidelity + cluster) for the prompt tab."""
    if not interpret_dir:
        return None
    idir = Path(interpret_dir)
    names = _read_csv(idir / "prompt_feature_names.csv")
    fid = _read_csv(idir / "prompt_feature_fidelity.csv")
    clusters = _read_csv(idir / "prompt_feature_clusters.csv")
    base = fid if fid is not None else names
    if base is None:
        return None
    df = base.copy()
    if names is not None and "concept" not in df.columns:
        df = df.merge(names[["feature_id", "concept"]], on="feature_id", how="left")
    if clusters is not None and "cluster_id" in clusters.columns:
        cc = ["feature_id", "cluster_id"] + (["behavior"] if "behavior" in clusters.columns else [])
        df = df.merge(clusters[cc], on="feature_id", how="left")
    return {"features": _round(df)}
