"""Caller-computed example UMAP tables for the normal static Viewer exporter.

Requires the optional ``umap-learn`` package. This example does not fit a lens,
change feature values, choose concept labels, or build/export the Viewer.
"""
from __future__ import annotations

import hashlib
from importlib.metadata import version
import json

import numpy as np
import pandas as pd

from prefscope import FeatureBatch

_RESPONSE_ROLES = {"response", "response_a", "response_b"}
_DIFFERENCE_ROLES = {"difference", "response_difference"}


def _projection(
    batch: FeatureBatch, *, projection_id: str, space: str, seed: int
) -> tuple[pd.DataFrame, dict]:
    views = list(batch.arrays)
    if projection_id == "answers":
        views = [view for view in views if batch.roles[view] in _RESPONSE_ROLES]
    matrices = [batch.matrix(view) for view in views]
    semantics = {(matrix.activation_polarity, matrix.code_semantics) for matrix in matrices}
    if len(semantics) != 1:
        raise ValueError("Shared answer UMAP requires compatible per-view code semantics")
    if projection_id != "answers" and len(views) != 1:
        raise ValueError("Prompt and pair UMAPs require exactly one vector per original row")

    # One fitted observation per (original row, view), in that order. A/B are
    # separate points in ONE shared response geometry, never averaged.
    values = np.stack([matrix.values for matrix in matrices], axis=1)
    values = np.asarray(values.reshape(-1, len(batch.feature_ids)), dtype=np.float32)
    if len(values) < 4:
        raise ValueError(f"UMAP unavailable for {projection_id}: at least four points required")
    if np.all(values == values[0]):
        raise ValueError(f"UMAP unavailable for {projection_id}: all activation vectors identical")

    try:
        import umap
    except ImportError as exc:
        raise ImportError("Example UMAP export requires optional umap-learn") from exc

    parameters = {
        "n_components": 2,
        "n_neighbors": min(15, len(values) - 1),
        "min_dist": 0.1,
        "metric": "euclidean",
        "random_state": int(seed),
        "n_jobs": 1,
        "init": "spectral",
        "n_epochs": 500,
    }
    coordinates = umap.UMAP(**parameters).fit_transform(values)
    if coordinates.shape != (len(values), 2) or not np.isfinite(coordinates).all():
        raise ValueError(f"UMAP returned invalid coordinates for {projection_id}")
    zero = ~np.any(values, axis=1)
    descriptors = [
        {
            "view": view,
            "role": matrix.role,
            "orientation": matrix.orientation,
            "activation_polarity": matrix.activation_polarity,
            "code_semantics": matrix.code_semantics,
        }
        for view, matrix in zip(views, matrices)
    ]
    feature_space_id = batch.provenance.get("lens", {}).get("feature_space_id")
    identity = {
        "space": space,
        "feature_space_id": feature_space_id,
        "feature_ids": list(batch.feature_ids),
        "row_ids": list(batch.row_ids),
        "view_descriptors": descriptors,
        "row_order": "row_major_view_order",
        "shape": list(values.shape),
        "dtype": "float32",
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    digest.update(b"\0")
    digest.update(values.astype("<f4", copy=False).tobytes(order="C"))
    points = pd.DataFrame({
        "projection_id": projection_id,
        "space": space,
        "view": np.tile(views, len(batch.row_ids)),
        "row_id": np.repeat(batch.row_ids, len(views)),
        "x": coordinates[:, 0],
        "y": coordinates[:, 1],
        "zero_vector": zero,
    })
    metadata = {
        "projection_id": projection_id,
        "method": "umap",
        "basis": "full_feature_activations",
        "space": space,
        "views": views,
        "view_descriptors": descriptors,
        "feature_space_id": feature_space_id,
        "feature_ids": list(batch.feature_ids),
        "row_order": "row_major_view_order",
        "n_points": len(values),
        "n_rows": len(batch.row_ids),
        "n_features": len(batch.feature_ids),
        "n_zero_rows": int(zero.sum()),
        "parameters": parameters,
        "versions": {
            package: version(package)
            for package in ("umap-learn", "numpy", "scipy", "scikit-learn", "numba", "pynndescent")
        },
        "input_hash": digest.hexdigest(),
        "preprocessing": "none",
        "dtype": "float32",
    }
    return points, metadata


def example_umap_tables(
    features: FeatureBatch,
    prompt_features: FeatureBatch | None = None,
    *,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Fit separate prompt and answer/pair UMAPs over complete activation vectors.

    A/B observations in the root batch share ONE fitted answer map. If both
    response sides are present, derived difference views are excluded. In a
    difference-only batch, codes stay in their recorded orientation. A prompt-only
    root is supported.
    An optional separate prompt batch must have exactly aligned original rows.
    Every row and zero vector is retained; no labels, preference orientation,
    sampling, normalization, SVD fallback, or post-fit jitter are used.

    Identical or fewer than four fitted vectors fail explicitly as unavailable.
    Coordinate equality is reproducible in the recorded package environment,
    not guaranteed across package versions or platforms. Duplicate/zero inputs
    can separate during UMAP optimization; this is not evidence of a difference
    in their measured activations.
    """
    roles = set(features.roles.values())
    paired_answers = {"response_a", "response_b"} <= roles
    if roles <= _RESPONSE_ROLES or (
        paired_answers and roles <= _RESPONSE_ROLES | _DIFFERENCE_ROLES
    ):
        projection_id = "answers"
    elif roles <= _DIFFERENCE_ROLES:
        projection_id = "pair"
    elif roles == {"prompt"}:
        projection_id = "prompt"
    else:
        raise ValueError("Example UMAP requires response, pair-difference, or prompt views")

    batches = [(features, projection_id, "main")]
    if prompt_features is not None:
        if projection_id == "prompt":
            raise ValueError("Supply a prompt-only root or a separate prompt batch, not both")
        if prompt_features.row_ids != features.row_ids:
            raise ValueError("Prompt UMAP row IDs must exactly match root rows in order")
        if set(prompt_features.roles.values()) != {"prompt"}:
            raise ValueError("Separate prompt UMAP requires only prompt-role views")
        batches.append((prompt_features, "prompt", "prompt"))

    point_tables, metadata = [], []
    for batch, projection_id, space in batches:
        points, meta = _projection(batch, projection_id=projection_id, space=space, seed=seed)
        point_tables.append(points)
        metadata.append(meta)
    return {
        "example_umap_points": pd.concat(point_tables, ignore_index=True),
        "example_umap_meta": pd.DataFrame(metadata),
    }
