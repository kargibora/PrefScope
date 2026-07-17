"""Embed + SAE-project oriented battles into sparse codes."""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_codes(oriented: pd.DataFrame, embedder, projector) -> dict[str, np.ndarray]:
    """Return z_self, z_other, z_diff (= z_self - z_other), and residual_self.

    `oriented` must have columns: prompt, self_completion, other_completion.
    `embedder.encode(prompts, completions) -> (N, D)`.
    `projector.project((N, D)) -> (N, M)`.

    `residual_self` is the per-row SAE reconstruction residual of the self/M
    responses — a coverage signal indicating how well the SAE dictionary
    represents each input (large = off-dictionary behavior).

    Note: build_codes orients via the `self_completion`/`other_completion`
    columns, so `z_diff = z_self - z_other` is already M-minus-opponent. The
    `sign` column (from orient_to_model) is provided for callers that instead
    work from raw side-A/side-B vectors and would compute
    `sign * (vec_a - vec_b)`.
    """
    prompts = oriented["prompt"].tolist()
    e_self = embedder.encode(prompts, oriented["self_completion"].tolist())
    e_other = embedder.encode(prompts, oriented["other_completion"].tolist())
    z_self = projector.project(e_self)
    z_other = projector.project(e_other)
    z_diff = (z_self - z_other).astype(np.float32)
    residual_self = projector.residual_norm(e_self)
    return {"z_self": z_self, "z_other": z_other, "z_diff": z_diff,
            "residual_self": residual_self}
