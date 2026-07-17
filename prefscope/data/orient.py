"""Orient battles around a target model.

The target model M becomes "self"; its opponent becomes "other". When M was on
side B, `sign = -1` so that any downstream `sign * (vec_a - vec_b)` is always
M-minus-opponent. The judge outcome is expressed from M's perspective:
win / tie / loss.

Contract with build_codes: build_codes orients via the `self_completion` /
`other_completion` columns, so its `z_diff = z_self - z_other` is already
M-minus-opponent. The `sign` column is provided for callers that instead work
from raw side-A/side-B vectors and would compute `sign * (vec_a - vec_b)`.
"""
from __future__ import annotations

import pandas as pd

_OUTCOME = {1.0: "win", 0.0: "loss", 0.5: "tie"}


def model_counts(battles: pd.DataFrame) -> pd.Series:
    """Number of battles each model participated in (either side)."""
    return pd.concat([battles["model_a"], battles["model_b"]]).value_counts()


def orient_to_model(battles: pd.DataFrame, model: str) -> pd.DataFrame:
    mask = (battles["model_a"] == model) | (battles["model_b"] == model)
    sub = battles[mask].copy()
    if sub.empty:
        return sub.reset_index(drop=True)
    is_a = sub["model_a"] == model
    sub["sign"] = is_a.map({True: 1, False: -1})
    sub["self_model"] = model
    sub["other_model"] = sub["model_b"].where(is_a, sub["model_a"])
    sub["self_completion"] = sub["completion_a"].where(is_a, sub["completion_b"])
    sub["other_completion"] = sub["completion_b"].where(is_a, sub["completion_a"])
    if "len_a" in sub.columns and "len_b" in sub.columns:
        sub["self_len"] = sub["len_a"].where(is_a, sub["len_b"])
        sub["other_len"] = sub["len_b"].where(is_a, sub["len_a"])
    # probability that self (M) is preferred: y_judge if M=A, else 1 - y_judge
    self_pref = sub["y_judge"].where(is_a, 1.0 - sub["y_judge"])
    sub["outcome"] = self_pref.map(_OUTCOME)
    if sub["outcome"].isna().any():
        bad = sorted(set(sub.loc[sub["outcome"].isna(), "y_judge"]))
        raise ValueError(f"y_judge values outside {{0.0, 0.5, 1.0}}: {bad}")
    return sub.reset_index(drop=True)
