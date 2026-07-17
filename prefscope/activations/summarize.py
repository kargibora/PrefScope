"""Reduce token-level SAE codes to per-span X^max / X^freq summaries.

reduce_span_summaries is pure NumPy/pandas (testable without torch). The
summarize_spans driver (added in a later task) projects cached activations
through a trained SAE and calls it; that driver imports torch via SAEProjector
and can run on any supported accelerator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def reduce_span_summaries(codes: np.ndarray, index: pd.DataFrame) -> pd.DataFrame:
    """Long-format (battle_id, span, feature_id, x_max, x_freq), nonzero features only.

    x_max  = max signed activation over the span's tokens (per feature).
    x_freq = fraction of span tokens on which the feature is nonzero.
    A feature is emitted for a span only if it fires there (x_freq > 0).
    """
    codes = np.asarray(codes, dtype=np.float32)
    out = []
    grp = index.reset_index(drop=True).groupby(["battle_id", "span"], sort=False)
    for (bid, span), rows in grp.indices.items():
        sub = codes[np.asarray(rows)]
        fired = sub != 0
        freq = fired.mean(axis=0)
        x_max = sub.max(axis=0)
        feats = np.where(freq > 0)[0]
        for f in feats:
            out.append({"battle_id": bid, "span": span, "feature_id": int(f),
                        "x_max": float(x_max[f]), "x_freq": float(freq[f])})
    return pd.DataFrame(out, columns=["battle_id", "span", "feature_id", "x_max", "x_freq"])


def summarize_spans(cache, projector, *, batch: int = 8192):
    """Project cached token activations through the SAE and reduce to per-span
    X^max/X^freq summaries; also return per-span token counts.

    Streams (battle_id, span) groups in buffers of about ``batch`` rows so the
    full (n_tokens, M) code matrix is never materialized — at M = expansion x
    hidden (tens of thousands of features) that matrix would be petabyte-scale.
    Peak memory is bounded by ``batch`` rows of codes, not the cache size.

    Returns (summaries_df, span_meta_df). summaries_df is the concatenated output
    of reduce_span_summaries across all groups; span_meta_df has one row per
    (battle_id, span) with n_tokens.
    """
    idx = cache.index.reset_index(drop=True)
    groups = list(idx.groupby(["battle_id", "span"], sort=False).indices.items())

    parts: list = []
    span_meta_rows: list[dict] = []
    buf: list = []          # list of ((battle_id, span), row_array)
    buf_count = 0

    def _flush(buffer):
        if not buffer:
            return
        all_rows = np.concatenate([r for _, r in buffer])
        codes = projector.project(np.asarray(cache.acts[all_rows]))
        off = 0
        for (bid, span), r in buffer:
            n = len(r)
            sub = codes[off:off + n]
            off += n
            sub_index = pd.DataFrame({"battle_id": bid, "span": span,
                                      "token_idx": np.arange(n)})
            parts.append(reduce_span_summaries(sub, sub_index))

    for (bid, span), rows in groups:
        rows = np.asarray(rows)
        span_meta_rows.append({"battle_id": bid, "span": span, "n_tokens": int(len(rows))})
        buf.append(((bid, span), rows))
        buf_count += len(rows)
        if buf_count >= batch:
            _flush(buf)
            buf = []
            buf_count = 0
    _flush(buf)

    cols = ["battle_id", "span", "feature_id", "x_max", "x_freq"]
    summaries = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)
    span_meta = pd.DataFrame(span_meta_rows, columns=["battle_id", "span", "n_tokens"])
    return summaries, span_meta
