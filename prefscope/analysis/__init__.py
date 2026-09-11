"""Small numerical operations over feature matrices."""

from prefscope.analysis.basic import (
    activation_summary,
    coactivation_counts,
    coactivation_pairs,
    cross_coactivation_counts,
    top_activating_rows,
)

__all__ = [
    "activation_summary",
    "coactivation_counts",
    "coactivation_pairs",
    "cross_coactivation_counts",
    "top_activating_rows",
]
