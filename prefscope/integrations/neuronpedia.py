"""Explicit Neuronpedia feature-description catalog provider."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from numbers import Real
from typing import Mapping
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd

from prefscope.api.feature_catalog import CATALOG_SCHEMA_VERSION, FeatureCatalog
from prefscope.core.features import validate_feature_ids

_MAX_RESPONSE_BYTES = 8 * 1024 * 1024


class NeuronpediaProvider:
    """Fetch proposed feature descriptions for one declared Neuronpedia SAE."""

    def __init__(
        self,
        neuronpedia_id: str,
        *,
        timeout: float = 10.0,
        user_agent: str = "PrefScope/0.2 feature-catalog",
        feature_space_identity: Mapping[str, str | None] | None = None,
    ) -> None:
        if (
            not isinstance(neuronpedia_id, str)
            or neuronpedia_id.count("/") != 1
            or any(not part.strip() for part in neuronpedia_id.split("/"))
        ):
            raise ValueError("neuronpedia_id must have the form 'model/layer'")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, Real)
            or not math.isfinite(float(timeout))
            or float(timeout) <= 0
        ):
            raise ValueError("timeout must be a positive number")
        if (
            not isinstance(user_agent, str)
            or not user_agent.strip()
            or any(character in user_agent for character in "\r\n")
        ):
            raise ValueError("user_agent must be a non-empty single-line string")
        identity = dict(feature_space_identity or {})
        if identity and set(identity) != {"feature_space_id", "feature_space_status"}:
            raise ValueError(
                "feature_space_identity needs feature_space_id and feature_space_status"
            )
        if identity:
            feature_space_id = identity["feature_space_id"]
            status = identity["feature_space_status"]
            if feature_space_id is not None and (
                not isinstance(feature_space_id, str) or not feature_space_id.strip()
            ):
                raise ValueError("feature_space_id must be a non-empty string or None")
            if status not in {
                "exact_weights",
                "declared_pinned_coordinate",
                "declared_unpinned",
                "unbound",
            }:
                raise ValueError("unknown feature_space_status")
            if (feature_space_id is None) != (status == "unbound"):
                raise ValueError("unbound feature-space identity is inconsistent")
        self.neuronpedia_id = neuronpedia_id
        self.timeout = float(timeout)
        self.user_agent = user_agent
        self.feature_space_identity = identity

    @classmethod
    def from_lens(cls, lens, **kwargs) -> "NeuronpediaProvider | None":
        """Use the Neuronpedia coordinate declared by a loaded SAELens checkpoint."""
        metadata = getattr(
            getattr(getattr(lens, "projector", None), "sae", None), "cfg", None
        )
        metadata = getattr(metadata, "metadata", None)
        neuronpedia_id = getattr(metadata, "neuronpedia_id", None)
        if not isinstance(neuronpedia_id, str) or "/" not in neuronpedia_id:
            return None
        return cls(
            neuronpedia_id,
            feature_space_identity=lens.feature_space_identity,
            **kwargs,
        )

    def _url(self, feature_id: int) -> str:
        model, layer = self.neuronpedia_id.split("/", 1)
        return (
            "https://www.neuronpedia.org/api/feature/"
            f"{quote(model, safe='')}/{quote(layer, safe='')}/{feature_id}"
        )

    @staticmethod
    def _description(payload: object) -> str | None:
        if not isinstance(payload, dict):
            return None
        explanations = payload.get("explanations", [])
        if not isinstance(explanations, list):
            return None

        def rank(item):
            if not isinstance(item, dict):
                return (False, float("-inf"))
            values = []
            for score in item.get("scores", []):
                if isinstance(score, dict) and score.get("value") is not None:
                    try:
                        values.append(float(score["value"]))
                    except (TypeError, ValueError):
                        pass
            return (bool(values), max(values, default=float("-inf")))

        candidates = [
            item
            for item in explanations
            if isinstance(item, dict) and str(item.get("description", "")).strip()
        ]
        if not candidates:
            return None
        return str(max(candidates, key=rank)["description"])

    def fetch(self, feature_ids, *, strict: bool = True) -> FeatureCatalog:
        """Fetch a catalog for explicit IDs without sending prompts or activations."""
        if not isinstance(strict, bool):
            raise ValueError("strict must be boolean")
        selected = validate_feature_ids(tuple(feature_ids))
        if any(feature_id < 0 for feature_id in selected):
            raise ValueError("Neuronpedia feature IDs must be non-negative")
        retrieved_at = datetime.now(timezone.utc).isoformat()
        rows = []
        for feature_id in selected:
            url = self._url(feature_id)
            description = None
            digest = None
            status = "unavailable"
            try:
                request = Request(url, headers={"User-Agent": self.user_agent})
                with urlopen(request, timeout=self.timeout) as response:
                    body = response.read(_MAX_RESPONSE_BYTES + 1)
                if len(body) > _MAX_RESPONSE_BYTES:
                    raise ValueError("Neuronpedia response exceeds the byte limit")
                payload = json.loads(body)
                description = self._description(payload)
                digest = hashlib.sha256(body).hexdigest()
                status = "available" if description else "unnamed"
            except Exception as error:
                if strict:
                    raise RuntimeError(
                        f"could not fetch Neuronpedia feature {feature_id}"
                    ) from error
            rows.append(
                {
                    "feature_id": feature_id,
                    "description": description,
                    "source": "neuronpedia",
                    "source_ref": url,
                    "evidence_layer": "proposed_label",
                    "retrieval_status": status,
                    "content_sha256": digest,
                }
            )
        source = {
            "kind": "neuronpedia",
            "neuronpedia_id": self.neuronpedia_id,
            "evidence_layer": "proposed_label",
            "retrieved_at": retrieved_at,
        }
        return FeatureCatalog(
            pd.DataFrame.from_records(
                rows,
                columns=[
                    "feature_id",
                    "description",
                    "source",
                    "source_ref",
                    "evidence_layer",
                    "retrieval_status",
                    "content_sha256",
                ],
            ),
            provenance={
                "schema_version": CATALOG_SCHEMA_VERSION,
                "source_kind": "neuronpedia",
                "neuronpedia_id": self.neuronpedia_id,
                "retrieved_at": retrieved_at,
                "n_features": len(selected),
                **self.feature_space_identity,
            },
            column_sources={"description": source},
        )


__all__ = ["NeuronpediaProvider"]
