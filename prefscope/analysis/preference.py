"""Predict which response a human/judge prefers from the interpretable codes.

A cross-validated logistic-regression readout from the sparse SAE codes onto the
preference label — the "build on top" test harness. It quantifies how much of the
human choice the interpretable concept features explain, and which features drive
it. Univariate per-feature relevance lives in ``run.feature_preference_relevance``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import (StratifiedGroupKFold, StratifiedKFold,
                                      cross_val_predict)


def evaluate_preference(codes, meta, *, n_splits: int = 5, seed: int = 0,
                        names=None, group_col: str | None = None) -> dict:
    """Cross-validated preference prediction from codes.

    Drops ties (pref == 0.5); label y = 1 if pref > 0.5 (A/self preferred) else 0.
    Returns a dict: ``n`` (non-tie examples), ``accuracy``, ``auc``,
    ``baseline_accuracy`` (majority class), ``n_features``, and ``top_features``
    (a DataFrame of feature_id + coefficient, by |coefficient| descending).

    ``group_col``: a column in ``meta`` (e.g. prompt / battle / user id) that groups
    non-independent rows. When given, folds are split by GROUP (``StratifiedGroupKFold``)
    so duplicated prompts / per-user structure can't leak across folds and inflate the
    score. Without it, splitting is per-row and the AUC is optimistic on any corpus with
    repeated prompts (Arena has them) — pass a group key for an honest estimate (#6).
    """
    codes = np.asarray(codes, dtype=np.float32)
    if "pref" not in meta.columns:
        raise ValueError("meta must have a 'pref' column (P(A preferred) per row)")
    pref = np.asarray(meta["pref"], dtype=float)
    keep = pref != 0.5
    X, y = codes[keep], (pref[keep] > 0.5).astype(int)
    if len(np.unique(y)) < 2:
        raise ValueError(
            "need both preferred and non-preferred examples after dropping ties")

    groups = None
    if group_col is not None:
        if group_col not in meta.columns:
            raise ValueError(f"group_col {group_col!r} not in meta columns")
        groups = np.asarray(meta[group_col])[keep]

    n_splits = int(min(n_splits, np.bincount(y).min()))
    if groups is not None:
        n_splits = int(min(n_splits, len(np.unique(groups))))
    if n_splits < 2:
        raise ValueError("too few examples per class (or groups) for cross-validation")

    clf = LogisticRegression(max_iter=1000)
    if groups is not None:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        proba = cross_val_predict(clf, X, y, cv=cv, groups=groups,
                                  method="predict_proba")[:, 1]
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    pred = (proba >= 0.5).astype(int)

    clf.fit(X, y)
    coef = clf.coef_[0]
    order = np.argsort(np.abs(coef))[::-1]
    top = pd.DataFrame({"feature_id": order.astype(int), "coefficient": coef[order]})
    if names is not None and "feature_id" in getattr(names, "columns", []):
        keepc = [c for c in ("feature_id", "concept") if c in names.columns]
        top = top.merge(names[keepc], on="feature_id", how="left")

    return {
        "n": int(keep.sum()),
        "accuracy": float(accuracy_score(y, pred)),
        "auc": float(roc_auc_score(y, proba)),
        "baseline_accuracy": float(max(y.mean(), 1.0 - y.mean())),
        "n_features": int(codes.shape[1]),
        "top_features": top,
    }
