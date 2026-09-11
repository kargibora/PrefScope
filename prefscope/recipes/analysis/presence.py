"""Reusable rules for turning sparse codes into semantic concept presence.

The SAE selecting a positive feature is weaker evidence than the named property being
present in the text.  A calibrated feature therefore uses its learned semantic threshold;
callers must opt in explicitly when they want the historical ``z > 0`` fallback.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from prefscope.core.features import validate_feature_ids


PRESENCE_POLICIES = ("calibrated", "positive_nonzero", "mixed")


@dataclass(frozen=True, eq=False)
class PresenceMatrix:
    """Boolean concept presence with feature-aligned provenance.

    ``values`` has shape ``(n_items, n_features)`` and is parallel to ``feature_ids``.
    ``basis`` is either ``semantic_threshold`` or ``positive_nonzero`` for each retained
    feature.  Under the ``calibrated`` policy, uncalibrated features are omitted entirely.
    """

    values: np.ndarray
    feature_ids: np.ndarray
    basis: np.ndarray
    thresholds: np.ndarray
    calibrated: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        if values.ndim != 2 or values.dtype != bool:
            raise ValueError("presence values must be a 2-D boolean matrix")
        width = values.shape[1]
        feature_ids = np.asarray(self.feature_ids)
        basis = np.asarray(self.basis, dtype=str)
        thresholds = np.asarray(self.thresholds, dtype=float)
        calibrated = np.asarray(self.calibrated)
        for name, column in {
            "feature_ids": feature_ids,
            "basis": basis,
            "thresholds": thresholds,
            "calibrated": calibrated,
        }.items():
            if column.ndim != 1 or len(column) != width:
                raise ValueError(f"{name} must have one entry per presence column")
        feature_ids = validate_feature_ids(feature_ids, width=width)
        if not np.isfinite(thresholds).all():
            raise ValueError("presence thresholds must be finite")
        if calibrated.dtype != bool:
            raise ValueError("calibrated must be a boolean vector")
        allowed_basis = {"semantic_threshold", "positive_nonzero"}
        if any(value not in allowed_basis for value in basis):
            raise ValueError(f"presence basis must be one of {sorted(allowed_basis)}")
        if not np.array_equal(calibrated, basis == "semantic_threshold"):
            raise ValueError(
                "calibrated must exactly identify semantic_threshold features")
        detached = {
            "values": np.array(values, dtype=bool, copy=True),
            "feature_ids": np.array(feature_ids, dtype=int, copy=True),
            "basis": np.array(basis, dtype=str, copy=True),
            "thresholds": np.array(thresholds, dtype=float, copy=True),
            "calibrated": np.array(calibrated, dtype=bool, copy=True),
        }
        for name, column in detached.items():
            immutable = np.frombuffer(
                column.tobytes(order="C"), dtype=column.dtype
            ).reshape(column.shape)
            object.__setattr__(self, name, immutable)


def _feature_table(features: pd.DataFrame | None) -> pd.DataFrame:
    if features is None or features.empty:
        return pd.DataFrame(index=pd.Index([], name="feature_id"))
    if "feature_id" not in features.columns:
        raise ValueError("feature annotations need a feature_id column")
    table = features.copy()
    table["feature_id"] = pd.to_numeric(table["feature_id"], errors="raise").astype(int)
    # Annotation artifacts are often concatenated (names, fidelity, calibration).  Keep
    # the last non-null value per column instead of dropping the earlier partial rows.
    return table.groupby("feature_id", sort=True).last()


def annotation_flag(value, *, default: bool = False) -> bool:
    """Parse a persisted boolean without treating ``NaN``/``"False"`` as true."""
    if value is None or pd.isna(value):
        return bool(default)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n", ""}:
            return False
        return False
    return bool(value)


def feature_thresholds(
    features: pd.DataFrame | None,
    feature_ids,
) -> tuple[np.ndarray, np.ndarray]:
    """Return semantic thresholds and a calibrated mask parallel to ``feature_ids``."""
    ids = np.asarray(validate_feature_ids(feature_ids), dtype=int)
    thresholds = np.zeros(len(ids), dtype=np.float32)
    calibrated = np.zeros(len(ids), dtype=bool)
    table = _feature_table(features)
    if "semantic_threshold" not in table.columns:
        return thresholds, calibrated
    has_pass = "presence_pass" in table.columns
    for j, feature_id in enumerate(ids):
        if feature_id not in table.index:
            continue
        row = table.loc[feature_id]
        threshold = pd.to_numeric(row.get("semantic_threshold"), errors="coerce")
        passed = annotation_flag(row.get("presence_pass")) if has_pass else True
        if pd.notna(threshold) and np.isfinite(float(threshold)) and passed:
            thresholds[j] = float(threshold)
            calibrated[j] = True
    return thresholds, calibrated


def concept_presence(
    codes,
    features: pd.DataFrame | None = None,
    *,
    feature_ids=None,
    policy: str = "calibrated",
) -> PresenceMatrix:
    """Convert a code matrix into feature-aligned semantic presence.

    Parameters
    ----------
    policy:
        ``calibrated`` omits features without a passing semantic threshold;
        ``positive_nonzero`` uses ``z > 0`` for every selected feature; ``mixed`` uses
        semantic thresholds where available and the positive-nonzero fallback elsewhere.
    """
    values = np.asarray(codes)
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"codes must be a 2-D numeric matrix, got shape {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("codes must contain only finite values")
    if policy not in PRESENCE_POLICIES:
        raise ValueError(f"policy must be one of {list(PRESENCE_POLICIES)}")
    if feature_ids is None:
        if features is not None and not features.empty and "feature_id" in features.columns:
            id_values = pd.to_numeric(
                features["feature_id"], errors="raise").astype(int).tolist()
            ids = np.asarray(list(dict.fromkeys(id_values)), dtype=int)
        else:
            ids = np.arange(values.shape[1], dtype=int)
    else:
        ids = np.asarray(validate_feature_ids(feature_ids), dtype=int)
    if len(ids) and ((ids < 0).any() or (ids >= values.shape[1]).any()):
        raise ValueError(f"feature ids must be inside [0, {values.shape[1]})")

    thresholds, calibrated = feature_thresholds(features, ids)
    if policy == "calibrated":
        ids = ids[calibrated]
        thresholds = thresholds[calibrated]
        calibrated = calibrated[calibrated]
        present = np.asarray(values[:, ids] >= thresholds, dtype=bool)
        basis = np.full(len(ids), "semantic_threshold", dtype=object)
    elif policy == "positive_nonzero":
        present = np.asarray(values[:, ids] > 0, dtype=bool)
        basis = np.full(len(ids), "positive_nonzero", dtype=object)
        thresholds = np.zeros(len(ids), dtype=np.float32)
        calibrated = np.zeros(len(ids), dtype=bool)
    else:
        raw = np.asarray(values[:, ids])
        present = raw > 0
        if calibrated.any():
            present[:, calibrated] = raw[:, calibrated] >= thresholds[calibrated]
        basis = np.where(calibrated, "semantic_threshold", "positive_nonzero")

    return PresenceMatrix(
        values=np.asarray(present, dtype=bool),
        feature_ids=np.asarray(ids, dtype=int),
        basis=np.asarray(basis, dtype=object),
        thresholds=np.asarray(thresholds, dtype=np.float32),
        calibrated=np.asarray(calibrated, dtype=bool),
    )


def semantic_presence(codes, feature_ids, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility helper implementing the historical mixed-presence behavior."""
    result = concept_presence(
        codes, features, feature_ids=feature_ids, policy="mixed")
    return result.values, result.calibrated
