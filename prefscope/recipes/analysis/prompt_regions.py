"""Turn sparse prompt codes into overlapping concept/cluster membership."""
from __future__ import annotations

import numpy as np
import pandas as pd

from prefscope.core.features import validate_feature_ids


def regions_from_feature_presence(
    presence,
    feature_ids,
    *,
    clusters: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn feature-aligned boolean presence into overlapping prompt regions.

    Without ``clusters`` each feature is one region. With a ``feature_id, cluster_id``
    table, membership is the union of every present member feature. Empty regions are
    removed. This works with calibrated presence and therefore complements
    :func:`prompt_region_membership`, whose input is a raw activation matrix.
    """
    raw = np.asarray(presence)
    ids = np.asarray(validate_feature_ids(feature_ids), dtype=int)
    if raw.ndim != 2 or raw.shape[1] != len(ids):
        raise ValueError("presence must be 2-D with one column per feature id")
    if raw.dtype == bool:
        values = raw
    elif np.issubdtype(raw.dtype, np.number):
        numeric = np.asarray(raw, dtype=float)
        if not np.isfinite(numeric).all() or not np.isin(numeric, [0.0, 1.0]).all():
            raise ValueError(
                "presence must contain finite boolean or numeric 0/1 values")
        values = numeric.astype(bool)
    else:
        raise ValueError("presence must contain boolean or numeric 0/1 values")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("feature_ids must be unique")
    if clusters is None:
        active = values.any(axis=0)
        return ids[active], values[:, active]
    required = {"feature_id", "cluster_id"}
    if not required <= set(clusters.columns):
        raise ValueError(f"prompt clusters need columns {sorted(required)}")
    table = clusters.dropna(subset=list(required)).copy()
    table["feature_id"] = table["feature_id"].astype(int)
    by_feature = {int(feature_id): j for j, feature_id in enumerate(ids)}
    table = table[table["feature_id"].isin(by_feature)]
    region_ids = np.asarray(sorted(table["cluster_id"].unique()))
    membership = np.zeros((values.shape[0], len(region_ids)), dtype=bool)
    for j, region_id in enumerate(region_ids):
        members = table.loc[table["cluster_id"] == region_id, "feature_id"].unique()
        columns = [by_feature[int(feature_id)] for feature_id in members]
        if columns:
            membership[:, j] = values[:, columns].any(axis=1)
    active = membership.any(axis=0)
    return region_ids[active], membership[:, active]


def prompt_region_membership(
    z_prompt,
    *,
    feature_ids=None,
    clusters: pd.DataFrame | None = None,
    min_activation: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(region_ids, membership, strength)`` for every prompt concept.

    ``membership[n, k]`` is true when prompt row ``n`` positively activates region
    ``k`` above ``min_activation``. Without ``clusters``, regions are individual
    feature ids. With a ``feature_id, cluster_id`` table, a prompt belongs to every
    cluster for which any member feature fires; ``strength`` is the maximum member
    activation. Regions overlap—no dominant ``argmax`` label is imposed.
    """
    z = np.asarray(z_prompt)
    if z.ndim != 2 or not np.issubdtype(z.dtype, np.number):
        raise ValueError(f"z_prompt must be a 2-D numeric matrix, got shape {z.shape}")
    if not np.isfinite(z).all():
        raise ValueError("z_prompt must contain only finite values")
    if not np.isfinite(float(min_activation)) or float(min_activation) < 0:
        raise ValueError("min_activation must be finite and non-negative")
    if feature_ids is None:
        selected = np.arange(z.shape[1], dtype=int)
    else:
        selected = np.asarray(validate_feature_ids(feature_ids), dtype=int)
    if len(selected) and ((selected < 0).any() or (selected >= z.shape[1]).any()):
        raise ValueError(f"prompt feature ids must be inside [0, {z.shape[1]})")

    if clusters is None:
        strength = np.asarray(z[:, selected], dtype=np.float32)
        membership = strength > float(min_activation)
        active = membership.any(axis=0)
        return selected[active], membership[:, active], strength[:, active]

    required = {"feature_id", "cluster_id"}
    if not required <= set(clusters.columns):
        raise ValueError(f"prompt clusters need columns {sorted(required)}")
    table = clusters.dropna(subset=["feature_id", "cluster_id"]).copy()
    table["feature_id"] = table["feature_id"].astype(int)
    table["cluster_id"] = table["cluster_id"].astype(int)
    table = table[table["feature_id"].isin(set(selected))]
    table = table[
        (table["feature_id"] >= 0) & (table["feature_id"] < z.shape[1])]
    region_ids = np.asarray(sorted(table["cluster_id"].unique()), dtype=int)
    membership = np.zeros((z.shape[0], len(region_ids)), dtype=bool)
    strength = np.zeros((z.shape[0], len(region_ids)), dtype=np.float32)
    for j, cluster_id in enumerate(region_ids):
        members = table.loc[
            table["cluster_id"] == int(cluster_id), "feature_id"].unique()
        if not len(members):
            continue
        block = np.asarray(z[:, members], dtype=np.float32)
        strength[:, j] = block.max(axis=1)
        membership[:, j] = (block > float(min_activation)).any(axis=1)
    active = membership.any(axis=0)
    return region_ids[active], membership[:, active], strength[:, active]
