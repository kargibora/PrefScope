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


_FEATURE_SPACE_STATUSES = {
    "exact_weights",
    "declared_pinned_coordinate",
    "declared_unpinned",
    "unbound",
}


def backend_feature_space_identity(backend) -> dict[str, str | None]:
    """Validate the optional backend-neutral coordinate identity declaration."""
    value = getattr(backend, "feature_space_identity", None)
    if value is None:
        return {"feature_space_id": None, "feature_space_status": "unbound"}
    if not isinstance(value, Mapping) or set(value) != {
        "feature_space_id", "feature_space_status"
    }:
        raise ValueError(
            "backend feature_space_identity must contain feature_space_id and "
            "feature_space_status"
        )
    feature_space_id = value["feature_space_id"]
    status = value["feature_space_status"]
    if status not in _FEATURE_SPACE_STATUSES:
        raise ValueError("backend feature_space_identity has an unknown status")
    if feature_space_id is None:
        if status != "unbound":
            raise ValueError("a bound backend feature-space status needs an ID")
    elif not isinstance(feature_space_id, str) or not feature_space_id.strip():
        raise ValueError("backend feature_space_id must be a non-empty string or None")
    elif status == "unbound":
        raise ValueError("an unbound backend feature space must not declare an ID")
    return {
        "feature_space_id": feature_space_id,
        "feature_space_status": status,
    }


def projector_feature_space_identity(
    projector, *, input_rep: str, backend: str
) -> dict:
    provenance = dict(getattr(projector, "projector_provenance", None) or {})
    if not provenance:
        return {"feature_space_id": None, "feature_space_status": "unbound"}
    weights_sha256 = provenance.get("sae_weights_sha256")
    coordinate_cfg_sha256 = provenance.get("sae_coordinate_config_fingerprint")
    if (
        provenance.get("backend") == "saelens"
        and isinstance(weights_sha256, str)
        and len(weights_sha256) == 64
        and isinstance(coordinate_cfg_sha256, str)
        and len(coordinate_cfg_sha256) == 64
    ):
        try:
            int(weights_sha256, 16)
            int(coordinate_cfg_sha256, 16)
            exact_payload = {
                "identity_schema": "saelens-operative-state-v1",
                "backend": "saelens",
                "architecture": provenance.get("architecture"),
                "d_in": int(provenance["d_in"]),
                "d_sae": int(provenance["d_sae"]),
                "sae_weights_sha256": weights_sha256,
                "sae_coordinate_config_fingerprint": coordinate_cfg_sha256,
            }
            return _identity_payload(exact_payload, status="exact_weights")
        except (KeyError, TypeError, ValueError):
            pass
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


def native_lens_feature_space_identity(
    model_path: str | Path,
    *,
    m_total: int,
    input_rep: str,
    input_dim: int | None = None,
    whiten_path: str | Path | None = None,
) -> dict:
    """Bind one native coordinate system to its checkpoint and optional whitener."""
    model = Path(model_path)
    if not model.is_file():
        return {"feature_space_id": None, "feature_space_status": "unbound"}
    whiten = Path(whiten_path) if whiten_path is not None else None
    return _identity_payload(
        {
            "identity_schema": "prefscope-native-feature-space-v2",
            "weights_sha256": _sha256_file(model),
            "whiten_sha256": (
                _sha256_file(whiten) if whiten is not None and whiten.is_file() else None
            ),
            "m_total": int(m_total),
            "input_dim": int(input_dim) if input_dim is not None else None,
            "input_rep": str(input_rep),
        },
        status="exact_weights",
    )


def lens_feature_space_identity(lens) -> dict:
    """Return exact native or declared external feature-coordinate identity."""
    width = int(lens.backend.m_total)
    backend_identity = backend_feature_space_identity(lens.backend)
    if backend_identity["feature_space_id"] is not None:
        return backend_identity
    lens_dir = getattr(lens, "lens_dir", None)
    model_path = Path(lens_dir) / SAE_MODEL if lens_dir is not None else None
    if model_path is not None and model_path.is_file():
        return native_lens_feature_space_identity(
            model_path,
            m_total=width,
            input_dim=(
                getattr(lens.backend, "input_dim", None)
                or getattr(getattr(lens, "projector", None), "input_dim", None)
            ),
            input_rep=str(lens.input_rep),
            whiten_path=Path(lens_dir) / "whiten.npz",
        )
    projector = getattr(lens, "projector", None)
    projector_provenance = dict(
        getattr(projector, "projector_provenance", None) or {}
    )
    backend_name = projector_provenance.get(
        "backend",
        getattr(lens, "pretrained_backend", type(projector).__name__),
    )
    return projector_feature_space_identity(
        projector,
        input_rep=str(lens.input_rep),
        backend=str(backend_name),
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
    "backend_feature_space_identity",
    "lens_feature_space_identity",
    "native_lens_feature_space_identity",
    "matrix_feature_space_identity",
    "projector_feature_space_identity",
]
