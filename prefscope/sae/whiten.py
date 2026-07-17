"""Input whitening for SAE training.

Sentence embeddings are anisotropic — a few directions dominate the variance — which
makes a sparse dictionary spend latents on rotations of those dominant axes (the
duplicate "directness" latents). Whitening the input first de-emphasizes that and
improves feature disentanglement ("Data Whitening Improves SAE Learning",
arXiv:2511.13981), at a small reconstruction cost.

Two modes:
- ``standardize`` — per-dimension (mean 0, unit variance); diagonal whitening, tiny.
- ``pca`` — full PCA whitening (decorrelate + unit variance); the paper's method.

The transform is fit on the SAE's training rows, saved next to the lens
(``whiten.npz``), and applied by ``SAEProjector`` so projection/reconstruction stay
in the space the SAE was trained on. ``none`` disables it (no file written).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

FNAME = "whiten.npz"


class Whitener:
    def __init__(self, method: str, mean, std=None, components=None, scale=None):
        self.method = method
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = None if std is None else np.asarray(std, dtype=np.float32)            # standardize
        self.components = None if components is None else np.asarray(components, np.float32)  # pca: V (D,D)
        self.scale = None if scale is None else np.asarray(scale, dtype=np.float32)      # pca: 1/sqrt(λ+ε)

    @classmethod
    def fit(cls, X: np.ndarray, method: str = "standardize", eps: float = 1e-5) -> "Whitener":
        X = np.asarray(X, dtype=np.float64)
        mean = X.mean(axis=0)
        Xc = X - mean
        if method == "standardize":
            std = Xc.std(axis=0)
            std = np.where(std > eps, std, 1.0)              # leave ~constant dims unscaled
            return cls("standardize", mean, std=std)
        if method == "pca":
            cov = (Xc.T @ Xc) / max(1, Xc.shape[0])
            evals, V = np.linalg.eigh(cov)                   # ascending; V columns orthonormal
            scale = 1.0 / np.sqrt(np.clip(evals, 0.0, None) + eps)
            return cls("pca", mean, components=V, scale=scale)
        raise ValueError(f"whiten method must be 'standardize' or 'pca', got {method!r}")

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xc = np.asarray(X, dtype=np.float32) - self.mean
        if self.method == "standardize":
            return (Xc / self.std).astype(np.float32)
        return ((Xc @ self.components) * self.scale).astype(np.float32)   # rotate then scale

    def inverse_transform(self, Xw: np.ndarray) -> np.ndarray:
        Xw = np.asarray(Xw, dtype=np.float32)
        if self.method == "standardize":
            return (Xw * self.std + self.mean).astype(np.float32)
        return (((Xw / self.scale) @ self.components.T) + self.mean).astype(np.float32)

    def save(self, out_dir) -> Path:
        p = Path(out_dir) / FNAME
        arrs = {"method": np.array(self.method), "mean": self.mean}
        if self.method == "standardize":
            arrs["std"] = self.std
        else:
            arrs["components"] = self.components
            arrs["scale"] = self.scale
        np.savez(p, **arrs)
        return p

    @classmethod
    def load(cls, lens_dir) -> "Whitener | None":
        p = Path(lens_dir) / FNAME
        if not p.exists():
            return None
        d = np.load(p, allow_pickle=False)
        method = str(d["method"])
        if method == "standardize":
            return cls("standardize", d["mean"], std=d["std"])
        return cls("pca", d["mean"], components=d["components"], scale=d["scale"])
