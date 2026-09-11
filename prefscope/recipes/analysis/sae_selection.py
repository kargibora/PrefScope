"""Pick an SAE width and sparsity from a sweep of lens metrics.

Reconstruction improves monotonically with capacity, so it cannot select a
configuration on its own. A configuration is admissible only when it also keeps its
dictionary alive, keeps decoder directions distinct, and has enough training rows per
feature to learn a direction rather than memorise examples; among admissible
configurations the best reconstruction wins.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Duplicate-direction ceiling. Feature splitting shows up as near-parallel decoder
# columns; see the reading guide in analysis/sae_metrics.py.
MAX_DECODER_COS = 0.5
# Above this share of never-firing features the width is wasted capacity.
MAX_DEAD_FRAC = 0.05
# Rows per feature. Document-embedding SAEs train on one vector per document, orders
# of magnitude fewer samples than token-level SAEs, so width is bounded by the corpus.
MIN_ROWS_PER_FEATURE = 20
# Realised sparsity should track the target; a large shortfall means unused capacity.
MIN_L0_RATIO = 0.75

REQUIRED_COLUMNS = ("m_total", "k", "fvu", "dead_frac", "l0_mean")


def _reasons(row, n_rows: int | None) -> list[str]:
    out = []
    if row["dead_frac"] > MAX_DEAD_FRAC:
        out.append(f"dead_frac {row['dead_frac']:.3f} > {MAX_DEAD_FRAC}")
    cos = row.get("decoder_cos_mean_max")
    if cos is not None and np.isfinite(cos) and cos > MAX_DECODER_COS:
        out.append(f"decoder_cos {cos:.3f} > {MAX_DECODER_COS} (duplicated directions)")
    if np.isfinite(row["l0_mean"]) and row["k"] and row["l0_mean"] < MIN_L0_RATIO * row["k"]:
        out.append(f"l0 {row['l0_mean']:.1f} far below k={int(row['k'])}")
    if n_rows is not None and row["m_total"] > n_rows / MIN_ROWS_PER_FEATURE:
        out.append(
            f"{n_rows / row['m_total']:.0f} rows/feature < {MIN_ROWS_PER_FEATURE} "
            "(memorisation risk)")
    return out


def evaluate_sweep(rows, *, n_rows: int | None = None) -> pd.DataFrame:
    """Annotate each swept configuration with admissibility and the reasons against it."""
    frame = pd.DataFrame(rows).copy()
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"sweep rows are missing {missing}")
    if frame.empty:
        raise ValueError("sweep is empty")
    for column in REQUIRED_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "decoder_cos_mean_max" in frame.columns:
        frame["decoder_cos_mean_max"] = pd.to_numeric(
            frame["decoder_cos_mean_max"], errors="coerce")
    notes = [_reasons(row, n_rows) for _, row in frame.iterrows()]
    frame["rejected_because"] = ["; ".join(n) for n in notes]
    frame["admissible"] = [not n for n in notes]
    return frame.sort_values(["admissible", "fvu"], ascending=[False, True])


def recommend_config(rows, *, n_rows: int | None = None) -> dict:
    """Return the best admissible configuration, or the least-bad one with a warning."""
    frame = evaluate_sweep(rows, n_rows=n_rows)
    admissible = frame[frame["admissible"]]
    chosen = (admissible if not admissible.empty else frame).iloc[0]
    return {
        "m_total": int(chosen["m_total"]),
        "k": int(chosen["k"]),
        "fvu": float(chosen["fvu"]),
        "admissible": bool(chosen["admissible"]),
        "rejected_because": str(chosen["rejected_because"]),
        "n_admissible": int(len(admissible)),
        "n_evaluated": int(len(frame)),
        "table": frame,
    }


def expansion_ratio(m_total: int, input_dim: int) -> float:
    """Dictionary width relative to the representation it decomposes."""
    if input_dim <= 0:
        raise ValueError("input_dim must be positive")
    return float(m_total) / float(input_dim)
