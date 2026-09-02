"""Feature-coordinate identity helpers shared by catalogs and projection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

from prefscope.artifacts import SAE_MODEL
from prefscope.core.features import FeatureMatrix


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity_payload(payload: Mapping[str, object], *, status: str) -> dict:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return {
        "feature_space_id": f"sha256:{hashlib.sha256(encoded).hexdigest()}",
        "feature_space_status": status,
    }


def projector_feature_space_identity(
    projector, *, input_rep: str, backend: str
) -> dict:
    provenance = dict(getattr(projector, "projector_provenance", None) or {})
    if not provenance:
        return {"feature_space_id": None, "feature_space_status": "unbound"}
    payload = {
        "backend": backend,
        "m_total": int(projector.m_total),
        "input_rep": str(input_rep),
        "projector": provenance,
    }
    pin_status = provenance.get("coordinate_pin_status")
    pinned = pin_status == "pinned"
    status = "declared_pinned_coordinate" if pinned else "declared_unpinned"
    try:
        return _identity_payload(payload, status=status)
    except (TypeError, ValueError):
        return {"feature_space_id": None, "feature_space_status": "unbound"}


def lens_feature_space_identity(lens) -> dict:
    """Return exact native or declared external feature-coordinate identity."""
    width = int(lens.backend.m_total)
    lens_dir = getattr(lens, "lens_dir", None)
    model_path = Path(lens_dir) / SAE_MODEL if lens_dir is not None else None
    if model_path is not None and model_path.is_file():
        return _identity_payload(
            {
                "weights_sha256": _sha256_file(model_path),
                "m_total": width,
                "input_rep": str(lens.input_rep),
            },
            status="exact_weights",
        )
    projector = getattr(lens, "projector", None)
    return projector_feature_space_identity(
        projector,
        input_rep=str(lens.input_rep),
        backend=str(getattr(lens, "pretrained_backend", type(projector).__name__)),
    )


def matrix_feature_space_identity(
    matrix: FeatureMatrix,
) -> tuple[str | None, str | None]:
    lens = matrix.provenance.get("lens", {})
    if not isinstance(lens, Mapping):
        return None, None
    feature_space_id = lens.get("feature_space_id")
    status = lens.get("feature_space_status")
    return (
        str(feature_space_id) if feature_space_id is not None else None,
        str(status) if status is not None else None,
    )


__all__ = [
    "lens_feature_space_identity",
    "matrix_feature_space_identity",
    "projector_feature_space_identity",
]
