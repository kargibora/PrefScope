"""Lens-level exports: bundle meta, the per-feature table, and generality signals."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from prefscope.artifacts import (
    FEATURE_FIDELITY, FEATURE_NAMES, MANIFEST, WIN_RELEVANCE,
)

from .sanitize import _read_csv


def export_meta(lens: Path, validation, features) -> dict:
    manifest = json.loads((lens / MANIFEST).read_text())
    ev = None
    log = lens / "sae_training_log.csv"
    if log.exists():
        ldf = pd.read_csv(log)
        if "val_ev" in ldf.columns and len(ldf):
            ev = float(ldf["val_ev"].iloc[-1])
    # Predictor fit quality. Two honesty rules: (a) report TRUE R² (1 − SS_res/SS_tot on
    # linearly rescaled predictions), not squared Pearson r — a scale/offset-miscalibrated
    # predictor must not score perfectly; (b) never silently pass an in-sample fit off as
    # held-out: `is_loo` says which one this is, and `loo_r2` is null unless it IS LOO.
    r2 = None
    is_loo = False
    if validation is not None:
        is_loo = "predicted_score_loo" in validation.columns
        xc = "predicted_score_loo" if is_loo else "predicted_score"
        if {xc, "actual_win_rate"} <= set(validation.columns) and len(validation) >= 3:
            yv = validation["actual_win_rate"].to_numpy(dtype=float)
            xv = validation[xc].to_numpy(dtype=float)
            # predictions are scores, not win rates — put them on the win-rate scale via
            # least squares before R², so R² measures explained variance, not correlation.
            A = np.column_stack([xv, np.ones_like(xv)])
            coef, *_ = np.linalg.lstsq(A, yv, rcond=None)
            resid = yv - A @ coef
            ss_tot = float(((yv - yv.mean()) ** 2).sum())
            r2 = float(1 - resid @ resid / ss_tot) if ss_tot > 0 else None
    loo_r2 = r2 if is_loo else None
    n_verified = int(features["fidelity_pass"].sum()) if features is not None \
        and "fidelity_pass" in features else None
    # n_named = features actually surfaced (named). The verified fraction should read against
    # this, NOT m_total (the SAE width) — "45/2048" wrongly looks like 98% is broken.
    n_named = int(len(features)) if features is not None else None
    # has_preference: does this dataset carry usable preference labels? win-relevance is
    # only computed (and merged into features) when the corpus had a preference column, so
    # its columns are a reliable proxy. When false, the viewer hides every preference-derived
    # surface (Bias screen, Validation, reward columns, "what wins" panels).
    has_preference = features is not None and any(
        c in features.columns for c in ("delta_win_rate", "win_assoc"))
    return {
        "lens": lens.name,
        "input_rep": manifest.get("input_rep"),   # let the viewer describe the RIGHT lens
        "embed_model_id": manifest.get("embed_model_id"),
        "m_total": manifest.get("m_total"),
        "k": manifest.get("k"),
        "input_dim": manifest.get("input_dim"),
        "n_battles": manifest.get("n_battles"),
        "ev": ev,
        "n_verified": n_verified,
        "n_named": n_named,
        "r2": r2,               # fit quality of whatever predictions exist
        "is_loo": is_loo,       # True only when predictions are leave-one-model-out
        "loo_r2": loo_r2,       # back-compat: null unless genuinely LOO
        "n_models": int(len(validation)) if validation is not None else None,
        "has_preference": bool(has_preference),
    }


def export_features(lens: Path) -> pd.DataFrame:
    names = _read_csv(lens / FEATURE_NAMES)
    fid = _read_csv(lens / FEATURE_FIDELITY)
    wr = _read_csv(lens / WIN_RELEVANCE)
    base = names if names is not None else fid
    df = base[["feature_id"]].copy()
    if names is not None and "concept" in names:
        df = df.merge(names[["feature_id", "concept"]], on="feature_id", how="left")
    # optional coarse type (capability/format/style/topic/safety) from label_feature_types
    types = _read_csv(lens / "feature_types.csv")
    if types is not None and "type" in types:
        df = df.merge(types[["feature_id", "type"]], on="feature_id", how="left")
    # optional length-confound flag (bias screen): a "does more" that's really "does longer"
    bias = _read_csv(lens / "bias_screen.csv")
    if bias is not None and "corr_confound_len" in bias:
        keep = [c for c in ["feature_id", "corr_confound_len", "confound_entangled"] if c in bias]
        df = df.merge(bias[keep], on="feature_id", how="left")
    if fid is not None:
        # keep the FULL fidelity verdict — n / precision / recall / f1 / fp_rate let the
        # viewer show "verified on n=14" vs "n=200" instead of one opaque pass/fail.
        keep = [c for c in ["feature_id", "correlation", "sign", "p_bonferroni",
                            "fidelity_pass", "n", "precision", "recall", "f1",
                            "fp_rate", "agreement"] if c in fid.columns]
        df = df.merge(fid[keep].rename(columns={"n": "fidelity_n"}),
                      on="feature_id", how="left")
    if wr is not None:
        # win_assoc is the RAW gap; delta_win_rate is the length-controlled AME
        # (WIMHF App. A.2) — the honest quantity. Carry both + their n + significance.
        keep = [c for c in ["feature_id", "win_assoc", "fire_rate", "significant",
                            "n_fire", "win_rate_a_more", "win_rate_a_less",
                            "delta_win_rate", "delta_win_significant"]
                if c in wr.columns]
        df = df.merge(wr[keep].rename(columns={"significant": "win_significant"}),
                      on="feature_id", how="left")
    return df


def feature_fire_rate(lens: Path, *, chunk: int = 20000) -> dict[int, float]:
    """Per completion feature: **pervasiveness** = the fraction of responses it fires in.

    This is our ``generality`` signal. A behaviour that appears in a large fraction of
    responses is general ('refuses', 'produces a list'); one firing in a tiny fraction is
    niche / content-bound ('American football'). Topic-based measures can't isolate niche
    content when the prompt lens has no matching concept, but fire rate doesn't care — 0.5%
    of responses is niche regardless. Computed from the individual lens's per-side codes
    (a feature expresses its concept when the top-k code is > 0 — the positive pole; a
    negative code is the opposite pole, not presence), over both responses of every battle.

    Returns ``{feature_id: rate}`` over all axes, or ``{}`` for a difference lens (no
    per-side codes — a lone response's activation can't be defined there)."""
    za_p, zb_p = lens / "z_a.npy", lens / "z_b.npy"
    if not (za_p.exists() and zb_p.exists()):
        return {}
    counts = None
    n = 0
    for p in (za_p, zb_p):
        arr = np.load(p, mmap_mode="r")
        if counts is None:
            counts = np.zeros(arr.shape[1], dtype=np.int64)
        for s in range(0, arr.shape[0], chunk):
            block = np.asarray(arr[s:s + chunk])
            counts += (block > 0).sum(axis=0)   # +pole = concept present (not != 0)
            n += block.shape[0]
    if not n:
        return {}
    rate = counts / n
    return {int(f): round(float(r), 4) for f, r in enumerate(rate)}


def feature_prompt_types(elic_csv) -> dict[int, int]:
    """Per completion feature: how many prompt concepts *significantly* elicit it — a
    secondary context signal shown next to ``generality`` (a topic-gated feature has few
    concepts driving it). From the elicitation co-occurrence table; ``{}`` if unavailable."""
    d = _read_csv(Path(elic_csv)) if elic_csv else None
    if d is None or not len(d):
        return {}
    if not {"prompt_feature", "completion_feature"} <= set(d.columns):
        return {}
    has_sig = "significant" in d.columns
    has_lift = "lift" in d.columns
    out: dict[int, int] = {}
    for cy, g in d.groupby("completion_feature"):
        sig = g
        if has_sig:
            sig = sig[sig["significant"].astype(bool)]
        if has_lift:
            sig = sig[sig["lift"] > 1.0]
        out[int(cy)] = int(sig["prompt_feature"].nunique())
    return out
