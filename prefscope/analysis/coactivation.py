"""Prompt-concept co-activation — which prompt concepts fire together above chance.

The prompt lens fires top-k concepts per prompt (``z_prompt > 0``), so each prompt has
a concept SET, not a single argmax. This finds the pairs that co-occur more than their
marginals predict — a "compound prompt concept" vocabulary (e.g. coding ∧ english =
"english coding questions"). It reuses ``elicitation.prompt_response_association`` (the
same 2×2 co-occurrence-lift + Yates χ² + Bonferroni) with ``z_prompt`` as BOTH matrices,
dropping the diagonal and deduping symmetric ``(A,B)``/``(B,A)`` rows.

Lift is symmetric and co-occurrence, NOT causal (see the elicitation module): a flagged
pair means "these prompt concepts tend to appear together", not that one drives the other.
Prompt codes are one-per-battle (no A/B stacking), so ``n_clusters`` is left at the row
count — no design-effect correction (unlike the stacked prompt→response case).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def prompt_coactivation(z_prompt: np.ndarray, *, features=None,
                        min_support: int = 30, min_cooccur: int = 5) -> pd.DataFrame:
    """Co-firing prompt-concept pairs with lift + significance.

    ``z_prompt``: (N, M) prompt codes (fire ⇔ ``> 0``). ``features``: optional subset of
    concept ids to test (e.g. verified prompt features). Returns one row per unordered
    pair (``concept_a < concept_b``): ``lift``, ``log2_lift``, ``p_bonferroni``,
    ``significant`` (+ the support/count columns from the underlying association).
    """
    from prefscope.analysis.elicitation import prompt_response_association

    fire = np.asarray(z_prompt) > 0
    df = prompt_response_association(
        fire, fire, prompt_features=features, resp_features=features,
        min_support=min_support, min_cooccur=min_cooccur, n_clusters=None)
    df = df.rename(columns={"prompt_feature": "concept_a", "completion_feature": "concept_b"})
    df = df[df["concept_a"] != df["concept_b"]].copy()          # drop self-pairs (lift = 1/P)
    # dedupe symmetric pairs: keep the ordered (min, max) representative
    lo = np.minimum(df["concept_a"], df["concept_b"])
    hi = np.maximum(df["concept_a"], df["concept_b"])
    df["concept_a"], df["concept_b"] = lo, hi
    df = df.drop_duplicates(subset=["concept_a", "concept_b"]).reset_index(drop=True)
    return df
