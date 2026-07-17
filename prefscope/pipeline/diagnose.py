"""Diagnose what a target model lacks (and excels at) relative to a pool.

Orient a corpus around a target model M (M = "self", every opponent = "other"),
project each battle's target-minus-opponent contrast through a frozen SAE lens,
then aggregate the signed codes into per-feature tendencies:

- ``net_direction`` > 0  -> M over-expresses the concept relative to its peers;
  < 0 -> M under-expresses it (a gap).
- ``outcome_assoc`` > 0  -> over-expressing the concept goes with M winning;
  combined with ``net_direction`` this separates strengths from weaknesses:
  over-express + helps = strength; under-express + helps = a gap worth closing.

The contrast is formed the same way the lens was trained. For a ``difference``
lens the SAE was fit on ``e_a - e_b``, so we project the contrast vector itself
(``project(e_self - e_other)``); naively subtracting two individual projections
would be off-distribution because the threshold is non-linear.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from prefscope.data.orient import orient_to_model

_OUTCOME_NUM = {"win": 1.0, "tie": 0.5, "loss": 0.0}


def _welch_contrast(inside: np.ndarray, outside: np.ndarray) -> dict:
    """Inside-vs-outside contrast on the signed presence s = sign(z).

    ``mean(sign(z)) = P(z>0) - P(z<0) = net_direction``, so a Welch two-sample
    test on ``sign(z)`` directly tests whether the target's net_direction for a
    feature differs from the pool's. ``delta_vs_pool`` = net_direction(target) -
    net_direction(pool); Cohen's d uses the pooled sd. Delegates the statistics to
    the shared ``inside_outside_contrast`` primitive.
    """
    from prefscope.analysis.stats import inside_outside_contrast

    c = inside_outside_contrast(np.sign(inside), np.sign(outside))
    return {"net_direction_pool": c["mean_outside"], "delta_vs_pool": c["delta"],
            "welch_t": c["welch_t"], "welch_p": c["welch_p"],
            "cohens_d": c["cohens_d"]}


def diagnose_features(z_diff: np.ndarray, win: np.ndarray, *,
                      features=None, z_outside: np.ndarray | None = None,
                      length: np.ndarray | None = None) -> pd.DataFrame:
    """Aggregate target-minus-opponent contrast codes into per-feature stats.

    z_diff: (N, M) signed SAE codes; z>0 means the target over-expresses
            feature f relative to its opponent in that battle.
    win:    (N,) target outcome in {0.0, 0.5, 1.0} (loss / tie / win).
    z_outside: optional (K, M) pool baseline codes (every other model's oriented
            codes). When given, adds the inside-vs-outside Welch contrast columns
            ``net_direction_pool``, ``delta_vs_pool``, ``welch_t``, ``welch_p``,
            ``welch_p_bonferroni``, ``cohens_d`` — i.e. how DISTINCTIVELY the
            target over/under-expresses each concept relative to the pool.
            CAVEAT: each battle contributes two dependent orientations (self/other),
            so a battle can land in both the target and the pool group — the Welch
            test's independence assumption is violated and ``welch_p`` is
            anti-conservative. Treat ``delta_vs_pool`` as a descriptive effect size,
            NOT the p-value as inference; use the prompt-matched paired head-to-head
            (``export_head_to_head`` / viewer "vs model") for trustworthy significance.
            A battle/prompt-clustered test is future work (deferred #3).
    length: optional (N,) per-battle word-count gap (self − other) for the target's
            battles. When given, adds ``outcome_assoc_lc`` (a length-controlled
            helps-win signal: the per-feature logistic average marginal effect) and
            ``length_confound`` (corr of the feature's sign with the length gap, so a
            verbosity-proxy feature is visible). Arena's strong length bias means the
            raw ``outcome_assoc`` partly reflects verbosity; the ``_lc`` column nets
            it out. The raw ``outcome_assoc`` column is kept for back-compat.
    """
    z_diff = np.asarray(z_diff, dtype=np.float32)
    win = np.asarray(win, dtype=np.float64)
    n = z_diff.shape[0]
    feats = range(z_diff.shape[1]) if features is None else list(features)
    feats = list(feats)
    win_rate = float(win.mean()) if n else float("nan")
    has_pool = z_outside is not None
    if has_pool:
        z_outside = np.asarray(z_outside, dtype=np.float32)

    rows = []
    for f in feats:
        col = z_diff[:, f]
        more = col > 0
        less = col < 0
        row = {
            "feature_id": int(f),
            "n": int(n),
            "fire_rate": float((col != 0).mean()) if n else float("nan"),
            "self_more_rate": float(more.mean()) if n else float("nan"),
            "self_less_rate": float(less.mean()) if n else float("nan"),
            "net_direction": float(more.mean() - less.mean()) if n else float("nan"),
            "mean_abs_z": float(np.abs(col).mean()) if n else float("nan"),
            "win_rate": win_rate,
            "win_rate_self_more": float(win[more].mean()) if more.any() else float("nan"),
            "win_rate_self_less": float(win[less].mean()) if less.any() else float("nan"),
        }
        if has_pool:
            row.update(_welch_contrast(col, z_outside[:, int(f)]))
        rows.append(row)
    df = pd.DataFrame(rows)
    df["outcome_assoc"] = df["win_rate_self_more"] - df["win_rate_self_less"]
    if has_pool:
        df["welch_p_bonferroni"] = (df["welch_p"] * len(df)).clip(upper=1.0)
    if length is not None and n:
        df = _attach_length_controlled(df, z_diff, win, length, feats)
    return df


def _attach_length_controlled(df, z_diff, win, length, feats) -> pd.DataFrame:
    """Add ``outcome_assoc_lc`` (length-controlled helps-win AME) and
    ``length_confound`` (each feature's sign-vs-length correlation)."""
    from prefscope.pipeline.winrelevance import win_relevance_logistic

    length = np.asarray(length, dtype=np.float64)
    cols = z_diff[:, feats]
    lc = win_relevance_logistic(cols, win, length, features=list(range(len(feats))))
    lc = lc.set_index("feature_id")
    df["outcome_assoc_lc"] = [
        float(lc.loc[i, "delta_win_rate"]) if i in lc.index else float("nan")
        for i in range(len(feats))
    ]

    try:
        from prefscope.analysis.dataset import feature_confound_correlation
        conf = feature_confound_correlation(cols, length).set_index("feature_id")["corr"]
        df["length_confound"] = [float(conf.get(i, float("nan"))) for i in range(len(feats))]
    except Exception:
        s = np.sign(cols.astype(np.float64))
        out = np.full(len(feats), np.nan)
        if np.std(length) > 0:
            for j in range(len(feats)):
                c = s[:, j]
                if np.std(c) > 0:
                    out[j] = float(np.corrcoef(c, length)[0, 1])
        df["length_confound"] = out
    return df


def _target_codes(oriented: pd.DataFrame, embedder, projector,
                  input_rep: str) -> np.ndarray:
    """Project each battle's target-minus-opponent contrast through the lens."""
    prompts = oriented["prompt"].tolist()
    e_self = np.asarray(embedder.encode(prompts, oriented["self_completion"].tolist()),
                        dtype=np.float32)
    e_other = np.asarray(embedder.encode(prompts, oriented["other_completion"].tolist()),
                         dtype=np.float32)
    from prefscope.pipeline.lens_rep import get_lens_rep
    return get_lens_rep(input_rep).contrast_codes(projector, e_self, e_other)


def battles_frame(oriented: pd.DataFrame, z: np.ndarray, win: np.ndarray,
                  features=None) -> pd.DataFrame:
    """Per-battle evidence: target/opponent text, outcome, and each axis code.

    One row per oriented battle, with the target-minus-opponent activation for
    every (selected) feature as a ``z{f}`` column — so the viewer can show the
    actual battles where the target most over/under-expresses a concept.
    """
    meta = ["instruction_id", "self_model", "other_model", "prompt",
            "self_completion", "other_completion", "outcome"]
    out = pd.DataFrame({c: oriented[c].values for c in meta if c in oriented.columns})
    out["win"] = win
    feat_list = range(z.shape[1]) if features is None else list(features)
    for f in feat_list:
        out[f"z{int(f)}"] = z[:, int(f)]
    return out.reset_index(drop=True)


def _attach_helps_win(df, win_relevance):
    """Merge the global length-controlled ``delta_win_rate`` as the headline
    ``helps_win`` signal (option ii). No-op if ``win_relevance`` is None or lacks it."""
    if win_relevance is None or "delta_win_rate" not in win_relevance.columns:
        return df
    wr = win_relevance[["feature_id", "delta_win_rate"]].copy()
    wr["feature_id"] = wr["feature_id"].astype(int)
    wr = wr.rename(columns={"delta_win_rate": "helps_win"})
    return df.merge(wr, on="feature_id", how="left")


def _attach_and_sort(df, names, sort_col):
    if names is not None:
        keep = [c for c in ("feature_id", "concept", "concept_abbrev") if c in names.columns]
        df = df.merge(names[keep], on="feature_id", how="left")
        front = [c for c in ("feature_id", "concept", "concept_abbrev") if c in df.columns]
        df = df[front + [c for c in df.columns if c not in front]]
    return df.sort_values(sort_col, ascending=False).reset_index(drop=True)


def _feat_filter(names, fidelity_only):
    if names is not None and fidelity_only and "fidelity_pass" in names.columns:
        return names.loc[names["fidelity_pass"].astype(bool), "feature_id"].astype(int).tolist()
    return None


def _oriented_length(oriented: pd.DataFrame) -> np.ndarray | None:
    """Per-battle word-count gap self − other (the length confound), if text is present."""
    if not {"self_completion", "other_completion"} <= set(oriented.columns):
        return None
    wc = lambda s: oriented[s].fillna("").str.split().str.len().to_numpy(dtype=float)  # noqa: E731
    return wc("self_completion") - wc("other_completion")


def run_diagnose(battles: pd.DataFrame, model: str, embedder, projector, *,
                 input_rep: str = "difference", names: pd.DataFrame | None = None,
                 fidelity_only: bool = True, return_battles: bool = False,
                 baseline_z: np.ndarray | None = None,
                 win_relevance: pd.DataFrame | None = None):
    """Diagnose ``model`` against the rest of the corpus.

    Returns ``(df, summary)``. When ``names`` is given, its ``concept`` columns
    are attached; if it has a ``fidelity_pass`` column and ``fidelity_only`` is
    True (default), only verified features are diagnosed.

    ``baseline_z`` (optional, (K, M)) is the pool's oriented codes — usually
    ``oriented_bank`` rows for every *other* model. When supplied, the result
    gains the inside-vs-outside Welch contrast columns and is sorted by
    ``delta_vs_pool`` (distinctiveness) instead of raw ``net_direction``.

    ``win_relevance`` (optional) is a global win-relevance frame carrying the
    length-controlled ``delta_win_rate``; when given it is merged as ``helps_win``
    (the headline helps-win signal). The within-model ``outcome_assoc_lc`` is the
    secondary signal.
    """
    oriented = orient_to_model(battles, model)
    if oriented.empty:
        raise ValueError(f"model {model!r} does not appear in the corpus")

    feats = _feat_filter(names, fidelity_only)
    z = _target_codes(oriented, embedder, projector, input_rep)
    win = oriented["outcome"].map(_OUTCOME_NUM).to_numpy(dtype=float)
    length = _oriented_length(oriented)
    df = diagnose_features(z, win, features=feats, z_outside=baseline_z, length=length)
    df = _attach_helps_win(df, win_relevance)

    sort_col = "delta_vs_pool" if baseline_z is not None else "net_direction"
    df = _attach_and_sort(df, names, sort_col)
    summary = {
        "model": model,
        "n_battles": int(len(oriented)),
        "win_rate": float(win.mean()) if len(win) else float("nan"),
        "n_features": int(len(df)),
        "input_rep": input_rep,
        "has_baseline": baseline_z is not None,
    }
    if return_battles:
        return df, summary, battles_frame(oriented, z, win, feats)
    return df, summary


def diagnose_from_bank(bank_Z: np.ndarray, bank_meta: pd.DataFrame, model: str, *,
                       names: pd.DataFrame | None = None, fidelity_only: bool = True,
                       win_relevance: pd.DataFrame | None = None):
    """Diagnose a model already present in an oriented-code bank (no embedding).

    Inside = the bank's ``self_model == model`` rows; outside = every other row.
    Always computes the inside-vs-outside Welch contrast. When the bank carries a
    per-battle ``length`` column the within-model length-controlled
    ``outcome_assoc_lc`` is added; ``win_relevance`` (if given) merges the global
    ``delta_win_rate`` as ``helps_win``. Returns ``(df, summary)``.
    """
    inside = (bank_meta["self_model"] == model).to_numpy()
    if not inside.any():
        raise ValueError(f"model {model!r} not in bank")
    feats = _feat_filter(names, fidelity_only)
    z = bank_Z[inside]
    win = np.asarray(bank_meta.loc[inside, "win"], dtype=float)
    length = (np.asarray(bank_meta.loc[inside, "length"], dtype=float)
              if "length" in bank_meta.columns else None)
    df = diagnose_features(z, win, features=feats, z_outside=bank_Z[~inside], length=length)
    df = _attach_helps_win(df, win_relevance)
    df = _attach_and_sort(df, names, "delta_vs_pool")
    summary = {
        "model": model,
        "n_battles": int(inside.sum()),
        "win_rate": float(win.mean()) if inside.any() else float("nan"),
        "n_features": int(len(df)),
        "has_baseline": True,
        "source": "bank",
    }
    return df, summary
