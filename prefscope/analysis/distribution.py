"""Corpus-level concept statistics: how often concepts fire, and with what.

Both summaries are computed in bounded memory over row chunks and return payloads
whose size depends on the number of features, never on the number of rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from prefscope.core.features import validate_feature_ids

ACTIVATION_QUANTILES = (0.5, 0.75, 0.9, 0.99, 1.0)


def _chunks(n_rows: int, chunk: int):
    for start in range(0, n_rows, chunk):
        yield start, min(start + chunk, n_rows)


def _columns(codes, columns, feature_ids):
    total = codes.shape[1]
    cols = (
        np.arange(total, dtype=int)
        if columns is None
        else np.asarray(validate_feature_ids(columns), dtype=int)
    )
    if cols.ndim != 1 or ((cols < 0) | (cols >= total)).any():
        raise ValueError("columns must be valid feature-column indices")
    ids = (
        cols.copy()
        if feature_ids is None
        else np.asarray(validate_feature_ids(feature_ids, width=len(cols)), dtype=int)
    )
    return cols, ids


def _positive_block(codes, start, stop, columns):
    """Values on the named concept's positive pole, materialized one row chunk at a time."""
    raw = np.asarray(codes[start:stop][:, columns], dtype=np.float32)
    return np.maximum(raw, 0.0)


def _matrix_like(codes):
    """Preserve bounded-memory row-stack views while normalizing ordinary inputs."""
    if hasattr(codes, "shape") and hasattr(codes, "ndim") and hasattr(codes, "__getitem__"):
        return codes
    return np.asarray(codes)


def concept_distribution(codes, *, columns=None, feature_ids=None, groups=None,
                         chunk_rows: int = 50_000) -> dict:
    """Per-feature prevalence and activation spread, plus per-row concept counts.

    ``groups`` is an optional per-row label (language, source, model) giving the
    per-group fire rate needed to compare subsets of one corpus.
    """
    codes = _matrix_like(codes)
    if codes.ndim != 2:
        raise ValueError("codes must be a 2-D (rows, features) array")
    n_rows, _ = codes.shape
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    columns, ids = _columns(codes, columns, feature_ids)
    n_features = len(columns)

    active_counts = np.zeros(n_features, dtype=np.int64)
    sums = np.zeros(n_features, dtype=np.float64)
    maxima = np.zeros(n_features, dtype=np.float32)
    per_row_counts = np.zeros(n_rows, dtype=np.int32)
    group_labels = None if groups is None else np.asarray(groups).astype(str)
    if group_labels is not None and len(group_labels) != n_rows:
        raise ValueError("groups must have one entry per row")
    uniq = None if group_labels is None else np.unique(group_labels)
    group_active = (None if uniq is None
                    else {g: np.zeros(n_features, dtype=np.int64) for g in uniq})
    group_totals = (None if uniq is None
                    else {g: int((group_labels == g).sum()) for g in uniq})

    for start, stop in _chunks(n_rows, chunk_rows):
        block = _positive_block(codes, start, stop, columns)
        active = block > 0
        active_counts += active.sum(axis=0)
        sums += block.sum(axis=0)
        if len(block):
            maxima = np.maximum(maxima, block.max(axis=0))
        per_row_counts[start:stop] = active.sum(axis=1)
        if uniq is not None:
            labels = group_labels[start:stop]
            for g in uniq:
                mask = labels == g
                if mask.any():
                    group_active[g] += active[mask].sum(axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_when_active = np.where(active_counts > 0, sums / np.maximum(active_counts, 1), 0.0)
    fire_rate = active_counts / float(n_rows) if n_rows else np.zeros(n_features)

    features = [
        {
            "feature_id": int(ids[j]),
            "n_active": int(active_counts[j]),
            "fire_rate": float(fire_rate[j]),
            "mean_activation": float(mean_when_active[j]),
            "max_activation": float(maxima[j]),
        }
        for j in range(n_features)
    ]
    if uniq is not None:
        for j, row in enumerate(features):
            row["group_fire_rate"] = {
                str(g): (float(group_active[g][j] / group_totals[g])
                         if group_totals[g] else 0.0)
                for g in uniq
            }

    counts = np.bincount(per_row_counts, minlength=1)
    return {
        "n_rows": int(n_rows),
        "n_features": int(n_features),
        "rows_with_any_concept": int((per_row_counts > 0).sum()),
        "coverage": float((per_row_counts > 0).mean()) if n_rows else 0.0,
        "concepts_per_row": {
            "mean": float(per_row_counts.mean()) if n_rows else 0.0,
            "quantiles": {
                str(q): float(np.quantile(per_row_counts, q)) for q in ACTIVATION_QUANTILES
            } if n_rows else {},
            "histogram": [int(c) for c in counts],
        },
        "dead_features": [int(ids[j]) for j in np.flatnonzero(active_counts == 0)],
        "groups": [str(g) for g in uniq] if uniq is not None else [],
        "group_totals": ({str(g): group_totals[g] for g in uniq}
                         if uniq is not None else {}),
        "features": features,
    }


def concept_coactivation(codes, *, columns=None, feature_ids=None, top_k: int = 20,
                         min_pair_count: int = 5, max_pairs: int = 20_000,
                         n_examples: int = 6, example_pairs: int = 500,
                         chunk_rows: int = 50_000) -> dict:
    """Which concepts co-fire, as each feature's strongest partners by lift.

    Ranking per feature rather than globally keeps every feature represented instead
    of letting a few dense ones fill a global top-N. Lift is
    ``P(i and j) / (P(i) * P(j))``; a value above 1 means the pair co-fires more than
    independence predicts. Example row indices are collected for the retained pairs so
    the viewer can show evidence without a second pass over the codes.
    """
    codes = _matrix_like(codes)
    if codes.ndim != 2:
        raise ValueError("codes must be a 2-D (rows, features) array")
    if top_k < 1 or n_examples < 0 or max_pairs < 1 or example_pairs < 0:
        raise ValueError("top_k and max_pairs must be positive; counts non-negative")
    n_rows, _ = codes.shape
    columns, ids = _columns(codes, columns, feature_ids)
    n_features = len(columns)

    # float32 so the co-occurrence counts go through BLAS; an integer matmul falls back
    # to a naive loop and dominates the runtime at realistic feature counts.
    joint_f = np.zeros((n_features, n_features), dtype=np.float64)
    sums = np.zeros(n_features, dtype=np.float64)
    for start, stop in _chunks(n_rows, chunk_rows):
        values = _positive_block(codes, start, stop, columns)
        active = (values > 0).astype(np.float32)
        joint_f += active.T @ active
        sums += values.sum(axis=0)
    joint = np.rint(joint_f).astype(np.int64)
    counts = np.diag(joint).astype(np.float64)

    with np.errstate(invalid="ignore", divide="ignore"):
        expected = np.outer(counts, counts) / float(n_rows) if n_rows else np.zeros_like(joint)
        lift = np.where(expected > 0, joint / expected, 0.0)
    np.fill_diagonal(lift, 0.0)
    eligible = joint >= min_pair_count
    lift = np.where(eligible, lift, 0.0)

    pairs = []
    for i in range(n_features):
        row = lift[i]
        if not np.any(row > 0):
            continue
        k = min(top_k, int((row > 0).sum()))
        partners = np.argpartition(row, -k)[-k:]
        for j in partners[np.argsort(row[partners])[::-1]]:
            if j == i:
                continue
            a, b = (i, int(j)) if i < j else (int(j), i)
            pairs.append((float(row[j]), int(joint[i, j]), a, b))

    seen, unique_pairs = set(), []
    for lift_value, count, a, b in sorted(pairs, key=lambda p: -p[0]):
        if (a, b) in seen:
            continue
        seen.add((a, b))
        unique_pairs.append((lift_value, count, a, b))
        if len(unique_pairs) >= max_pairs:
            break

    examples: dict[tuple[int, int], list[int]] = {}
    if n_examples and unique_pairs:
        # Only the strongest pairs carry examples: the scan is (rows x pairs), so an
        # unbounded pair set makes this the dominant cost of the whole export. Rank by
        # the weaker of the two mean-normalized activations. The previous first-match
        # scan often attached weak false-positive rows that visibly contradicted both
        # feature names even though strong joint activators existed later in the corpus.
        wanted = [(a, b) for _, _, a, b in unique_pairs[:example_pairs]]
        left = np.array([a for a, _ in wanted], dtype=int)
        right = np.array([b for _, b in wanted], dtype=int)
        scales = sums / np.maximum(counts, 1.0)
        ranked: dict[tuple[int, int], list[tuple[float, int]]] = {
            pair: [] for pair in wanted
        }
        for start, stop in _chunks(n_rows, chunk_rows):
            values = _positive_block(codes, start, stop, columns)
            both = (values[:, left] > 0) & (values[:, right] > 0)
            for pair_index, pair in enumerate(wanted):
                local_rows = np.flatnonzero(both[:, pair_index])
                if not len(local_rows):
                    continue
                score = np.minimum(
                    values[local_rows, left[pair_index]]
                    / max(scales[left[pair_index]], 1e-12),
                    values[local_rows, right[pair_index]]
                    / max(scales[right[pair_index]], 1e-12),
                )
                take = min(n_examples, len(local_rows))
                if take < len(local_rows):
                    if float(score.max()) == float(score.min()):
                        keep = np.arange(take)
                    else:
                        threshold = np.partition(score, -take)[-take]
                        candidates = np.flatnonzero(score >= threshold)
                        keep = candidates[np.lexsort((
                            local_rows[candidates], -score[candidates]
                        ))[:take]]
                    local_rows, score = local_rows[keep], score[keep]
                candidates = ranked[pair] + [
                    (float(s), int(start + row))
                    for s, row in zip(score, local_rows)
                ]
                ranked[pair] = sorted(candidates, key=lambda item: (-item[0], item[1]))[
                    :n_examples
                ]
        examples = {
            pair: [row for _, row in rows]
            for pair, rows in ranked.items()
        }

    return {
        "n_rows": int(n_rows),
        "min_pair_count": int(min_pair_count),
        "truncated": len(seen) >= max_pairs,
        "pairs": [
            {
                "a": int(ids[a]),
                "b": int(ids[b]),
                "count": int(count),
                "lift": round(float(lift_value), 4),
                "rows": examples.get((a, b), []),
            }
            for lift_value, count, a, b in unique_pairs
        ],
    }


def distribution_frame(distribution: dict) -> pd.DataFrame:
    """The per-feature part of :func:`concept_distribution` as a table."""
    return pd.DataFrame(distribution["features"])
