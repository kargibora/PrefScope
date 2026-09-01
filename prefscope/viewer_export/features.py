"""Lens-level exports: bundle meta, the per-feature table, and generality signals."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from prefscope.analysis.presence import annotation_flag
from prefscope.artifacts import (
    FEATURE_CALIBRATION, FEATURE_CONTEXT, FEATURE_FIDELITY, FEATURE_NAMES, FEATURE_ROLES,
    MANIFEST, WIN_RELEVANCE,
)
from prefscope.core.manifest import LensManifest

from .sanitize import _read_csv


def export_meta(lens: Path, validation, features) -> dict:
    manifest = json.loads((lens / MANIFEST).read_text())
    input_rep = LensManifest.from_dict(manifest).input_rep
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
    n_verified = int(features["fidelity_pass"].map(annotation_flag).sum()) \
        if features is not None and "fidelity_pass" in features else None
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
        "input_rep": input_rep,   # let the viewer describe the RIGHT lens
        "dataset_mode": manifest.get("dataset_mode", "paired"),
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


def export_features(lens: Path, analysis_dir: Path | str | None = None) -> pd.DataFrame:
    """Assemble the viewer's feature catalog from a lens and its analysis outputs.

    Arrays and the manifest always come from ``lens``.  Interpretation tables normally
    live in a separate run directory, so ``analysis_dir`` is authoritative when given;
    refusing to fall back to the lens prevents stale tables from two runs being mixed.
    """
    lens = Path(lens)
    analysis = Path(analysis_dir) if analysis_dir is not None else lens
    names = _read_csv(analysis / FEATURE_NAMES)
    fid = _read_csv(analysis / FEATURE_FIDELITY)
    roles = _read_csv(analysis / FEATURE_ROLES)
    wr = _read_csv(analysis / WIN_RELEVANCE)
    # The feature catalog inventories the SAE, not merely the subset interpreted in
    # one run. This keeps unnamed and failed axes visible in the feature atlas and makes
    # partially completed, resumable interpretation runs explicit.
    manifest = LensManifest.from_dict(json.loads((lens / MANIFEST).read_text()))
    width = int(manifest.m_total or 0)
    if width <= 0:
        candidates = [
            int(pd.to_numeric(table["feature_id"], errors="coerce").max()) + 1
            for table in (names, fid)
            if table is not None and "feature_id" in table and len(table)
        ]
        width = max(candidates, default=0)
    df = pd.DataFrame({"feature_id": np.arange(width, dtype=int)})
    if names is not None and "concept" in names:
        df = df.merge(names[["feature_id", "concept"]], on="feature_id", how="left")
    # Backward compatibility for early bundles that carried a coarse feature_types.csv.
    types = _read_csv(analysis / "feature_types.csv")
    if types is not None and "type" in types:
        df = df.merge(types[["feature_id", "type"]], on="feature_id", how="left")
    # optional length-confound flag (bias screen): a "does more" that's really "does longer"
    bias = _read_csv(analysis / "bias_screen.csv")
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
    if roles is not None:
        keep = [c for c in [
            "feature_id", "classification_status", "semantic_role", "semantic_family",
            "role_confidence", "role_agreement", "prompt_relation",
            "relation_agreement", "requested_share", "elicited_share",
            "prompt_driven_share", "independent_share", "prompt_scope",
            "behavior_scope", "feature_summary", "n_examples", "n_labelled",
            "n_present", "label_coverage", "concept_present_rate",
        ] if c in roles.columns]
        df = df.merge(roles[keep].drop_duplicates("feature_id"),
                      on="feature_id", how="left")
        if "behavior_scope" in df.columns:
            role_category = df["behavior_scope"].map({
                "candidate_cross_prompt_behavior": "general",
                "context_conditional_behavior": "context_specific",
                "prompt_content": "prompt_content",
            })
            df["behavior_category"] = role_category.fillna("unclassified")
    calibration = _read_csv(analysis / FEATURE_CALIBRATION)
    if calibration is not None:
        keep = [c for c in [
            "feature_id", "calibration_status", "semantic_threshold",
            "threshold_quantile", "precision_lcb", "semantic_coverage",
            "silent_concept_rate", "semantic_role", "requested_share",
            "presence_pass",
        ] if c in calibration.columns]
        keep = [c for c in keep if c == "feature_id" or c not in df.columns]
        df = df.merge(calibration[keep].drop_duplicates("feature_id"),
                      on="feature_id", how="left")
    context = _read_csv(analysis / FEATURE_CONTEXT)
    if context is not None:
        keep = [c for c in [
            "feature_id", "semantic_presence_rate", "prompt_dependence_nmi",
            "prompt_context_js", "effective_prompt_contexts",
            "max_prompt_context_share", "n_supported_prompt_contexts",
            "paired_choice_ratio", "behavior_category", "top_prompt_contexts_json",
        ] if c in context.columns]
        context_table = context[keep].drop_duplicates("feature_id")
        if "behavior_category" in context_table.columns and "behavior_category" in df.columns:
            context_table = context_table.rename(
                columns={"behavior_category": "_context_behavior_category"})
        df = df.merge(context_table, on="feature_id", how="left")
        if "_context_behavior_category" in df.columns:
            df["behavior_category"] = df["_context_behavior_category"].combine_first(
                df["behavior_category"])
            df = df.drop(columns="_context_behavior_category")
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


def feature_fire_rate(lens: Path, features: pd.DataFrame | None = None, *,
                      chunk: int = 20000) -> dict[int, float]:
    """Per completion feature: **pervasiveness** = the fraction of responses it fires in.

    This is our ``generality`` signal. A behaviour that appears in a large fraction of
    responses is general ('refuses', 'produces a list'); one firing in a tiny fraction is
    niche / content-bound ('American football'). Topic-based measures can't isolate niche
    content when the prompt lens has no matching concept, but fire rate doesn't care — 0.5%
    of responses is niche regardless. Computed from the individual lens's per-side codes
    (a feature expresses its concept when the top-k code is > 0 — the positive pole; a
    negative code is the opposite pole, not presence), over both responses of every battle.

    Returns ``{feature_id: rate}`` over all axes, or ``{}`` for a difference lens (no
    per-side codes — a lone response's activation can't be defined there). Single-response
    lenses are rated over their one side."""
    za_p, zb_p = lens / "z_a.npy", lens / "z_b.npy"
    if not za_p.exists():
        return {}
    sides = [za_p] + ([zb_p] if zb_p.exists() else [])
    from .presence import feature_thresholds

    counts = None
    n = 0
    for p in sides:
        arr = np.load(p, mmap_mode="r")
        if counts is None:
            counts = np.zeros(arr.shape[1], dtype=np.int64)
            feats = list(range(arr.shape[1]))
            thresholds, calibrated = feature_thresholds(
                features if features is not None else pd.DataFrame({"feature_id": feats}),
                feats)
        for s in range(0, arr.shape[0], chunk):
            block = np.asarray(arr[s:s + chunk])
            present = block > 0
            if calibrated.any():
                present[:, calibrated] = block[:, calibrated] >= thresholds[calibrated]
            counts += present.sum(axis=0)
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
            sig = sig[sig["significant"].map(annotation_flag)]
        if has_lift:
            sig = sig[sig["lift"] > 1.0]
        out[int(cy)] = int(sig["prompt_feature"].nunique())
    return out
