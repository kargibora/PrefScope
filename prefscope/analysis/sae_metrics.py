"""Cheap SAE health metrics for tracking a lens (e.g. across an M sweep).

These are REDUNDANCY + FIT signals, NOT a feature-absorption score. Absorption is
a behavioral probe+ablation test (SAEBench / Chanin et al. 2024) and can occur
between near-orthogonal decoder atoms, so decoder cosine cannot detect it — do not
read these as absorption. Rough reading: decoder max-cosine <0.3 healthy / >0.7
duplicated features; FVU 0.1-0.2 healthy, higher is a red flag.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

from prefscope.pipeline.cluster import feature_mi


def _decoder_weight(lens_dir: Path) -> np.ndarray:
    """Decoder weight (input_dim, M) from the saved checkpoint; columns are the
    per-feature directions (unit-norm in training, re-normalized here defensively)."""
    import torch
    ckpt = torch.load(lens_dir / "sae_model.pt", map_location="cpu", weights_only=True)
    return ckpt["state_dict"]["decoder.weight"].float().cpu().numpy()


def decoder_cos_mean_max(W: np.ndarray) -> float:
    """Mean over features of each feature's max off-diagonal |cosine| with another
    decoder column (the standard cheap redundancy metric). float32 to stay memory-safe
    at large M (M=8192 -> ~0.27 GB gram)."""
    W = np.asarray(W, dtype=np.float32)
    norms = np.linalg.norm(W, axis=0, keepdims=True)
    Wn = np.divide(W, norms, out=np.zeros_like(W), where=norms > 1e-8)
    gram = Wn.T @ Wn                       # (M, M) cosine similarity
    np.fill_diagonal(gram, 0.0)
    return float(np.mean(np.max(np.abs(gram), axis=1)))


def _offdiag_mean(mat: np.ndarray) -> float:
    m = mat.shape[0]
    if m < 2:
        return float("nan")
    return float(mat[~np.eye(m, dtype=bool)].mean())


def lens_metrics(lens_dir) -> dict:
    """One row of redundancy/fit metrics for a trained lens directory."""
    lens_dir = Path(lens_dir)
    man = json.loads((lens_dir / "manifest.json").read_text())
    fvu = man.get("best_val_norm_mse")
    out: dict = {
        "lens": lens_dir.name,
        "m_total": man.get("m_total"), "k": man.get("k"),
        "input_dim": man.get("input_dim"),
        "fvu": fvu,
        "explained_variance": (1.0 - fvu) if fvu is not None else None,
    }

    out["decoder_cos_mean_max"] = decoder_cos_mean_max(_decoder_weight(lens_dir))

    # activation-based metrics from whichever code matrix the lens saved
    zname = next((n for n in ("z_diff.npy", "z_prompt.npy", "z_a.npy")
                  if (lens_dir / n).exists()), None)
    if zname is not None:
        z = np.load(lens_dir / zname)
        with warnings.catch_warnings():           # feature_mi/BLAS can emit benign warnings
            warnings.simplefilter("ignore")
            out["redundancy_mi_mean"] = _offdiag_mean(feature_mi(z))
            zc = z - z.mean(axis=0, keepdims=True)
            denom = np.sqrt((zc ** 2).sum(axis=0))
            corr = np.divide(zc.T @ zc, np.outer(denom, denom),
                             out=np.zeros((z.shape[1], z.shape[1])),
                             where=np.outer(denom, denom) > 1e-12)
        out["redundancy_abscorr_mean"] = _offdiag_mean(np.abs(corr))
        out["dead_frac"] = float((np.abs(z).sum(axis=0) == 0).mean())
        out["l0_mean"] = float((z != 0).sum(axis=1).mean())
        out["code_array"] = zname
    return out
