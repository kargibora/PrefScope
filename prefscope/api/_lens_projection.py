"""Internal representation projection and encoding for the Lens facade."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json

import numpy as np


def _get_lens_rep(name):
    from prefscope.pipeline.lens_rep import get_lens_rep

    return get_lens_rep(name)


def representation_contract_fingerprint(contract) -> str:
    encoded = json.dumps(
        dict(contract),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def expected_representation_contract(lens) -> dict | None:
    declared = getattr(lens.projector, "representation_contract", None)
    if declared is not None:
        if not isinstance(declared, Mapping):
            raise ValueError("projector representation_contract must be a mapping")
        return {key: value for key, value in declared.items() if value is not None}
    typed = lens.manifest_obj
    if typed is None or not typed.embed_model_id:
        return None
    return {
        key: value
        for key, value in {
            "representation_family": "text_embedding",
            "embed_model_id": typed.embed_model_id,
            "embed_model_revision": typed.embed_model_revision,
            "max_tokens": typed.max_tokens,
            "embed_instruction": typed.embed_instruction,
            "pooling": typed.pooling,
            "normalization": typed.normalization,
            "dtype": typed.dtype,
            "backend": typed.backend,
        }.items()
        if value is not None
    }


def validate_representation_contract(
    lens,
    batch,
    *,
    allow_mismatch: bool,
) -> dict[str, object]:
    expected = lens._expected_representation_contract()
    if expected is None:
        return {
            "status": "lens_contract_not_declared",
            "unsafe_override": False,
        }
    provenance = dict(batch.provenance)
    contracts = provenance.get("representation_contracts")
    view = "prompt" if lens.input_rep == "prompt" else "response"
    actual = contracts.get(view) if isinstance(contracts, Mapping) else None
    if actual is None:
        actual = provenance.get("representation_contract")
    if actual is None:
        actual = provenance
    if not isinstance(actual, Mapping):
        actual = None
    expected_fingerprint = lens._representation_contract_fingerprint(expected)
    observed_fingerprint = provenance.get("representation_fingerprint")
    mismatch = []
    if actual is None:
        mismatch.append("representation contract is absent")
    else:
        for key, expected_value in expected.items():
            if key not in actual:
                mismatch.append(f"missing {key}")
            elif actual[key] != expected_value:
                mismatch.append(
                    f"{key}: expected {expected_value!r}, got {actual[key]!r}"
                )
        comparable_actual = {key: actual[key] for key in expected if key in actual}
        observed_fingerprint = lens._representation_contract_fingerprint(
            comparable_actual
        )
    if mismatch and not allow_mismatch:
        raise ValueError(
            "representation source is incompatible with this lens: "
            + "; ".join(mismatch)
            + ". Pass allow_representation_mismatch=True only for an explicitly "
            "audited unsafe projection."
        )
    return {
        "status": "unsafe_override" if mismatch else "matched",
        "unsafe_override": bool(mismatch),
        "expected_fingerprint": expected_fingerprint,
        "observed_fingerprint": observed_fingerprint,
        "mismatches": mismatch,
    }


def project_representations(
    lens,
    batch,
    *,
    allow_representation_mismatch: bool = False,
):
    """Project an interchangeable representation batch through this lens.

    This is the source-agnostic inference boundary. Text embeddings from
    :class:`EmbeddingRepresentationSource`, pooled residual activations, or a
    custom :class:`RepresentationSource` use the same validated contract.
    The source must publish canonical arrays: ``prompt`` for a prompt lens,
    and ``response_a``/``response_b`` for response lenses.
    """
    from prefscope.core.features import FeatureBatch
    from prefscope.core.representation import RepresentationBatch

    if not isinstance(batch, RepresentationBatch):
        raise ValueError("batch must be a RepresentationBatch")
    if getattr(lens.projector, "supports_item_projection", True) is False:
        raise ValueError(
            "this SAELens backend expects token activations; use "
            "lens.project_saelens_tokens(...) so SAE encoding happens before pooling"
        )
    if not isinstance(allow_representation_mismatch, bool):
        raise ValueError("allow_representation_mismatch must be boolean")
    compatibility = lens._validate_representation_contract(
        batch, allow_mismatch=allow_representation_mismatch
    )
    pin_status = getattr(lens.projector, "coordinate_pin_status", None)
    if compatibility.get("status") == "matched" and pin_status is not None:
        compatibility = {
            **compatibility,
            "status": "matched_declared_unpinned",
            "coordinate_pin_status": pin_status,
        }
    expected_dim = getattr(lens.manifest_obj, "input_dim", None)
    if expected_dim is None:
        expected_dim = getattr(lens.projector, "input_dim", None)

    def vector(name):
        value = np.asarray(batch.array(name), dtype=np.float32)
        if expected_dim is not None and value.shape[1] != int(expected_dim):
            raise ValueError(
                f"representation array {name!r} has width {value.shape[1]} "
                f"but lens input_dim is {expected_dim}"
            )
        return value

    if lens.input_rep == "prompt":
        arrays = {"z_prompt": lens.projector.project(vector("prompt"))}
    else:
        e_a = vector("response_a")
        rep = _get_lens_rep(lens.input_rep)
        if "response_b" in batch.arrays:
            arrays = rep.output_arrays(lens.projector, e_a, vector("response_b"))
        else:
            arrays = rep.single_output_arrays(lens.projector, e_a)
    roles = {
        "z_prompt": "prompt",
        "z_a": "response_a",
        "z_b": "response_b",
        "z_diff": "response_difference",
    }
    orientations = {
        "z_prompt": "none",
        "z_a": "absolute_a",
        "z_b": "absolute_b",
        "z_diff": "a_minus_b",
    }
    lens_provenance = {
        "schema_version": lens.manifest.get("schema_version"),
        "lens_kind": lens.manifest.get("lens_kind", lens.input_rep),
        "input_rep": lens.input_rep,
        "m_total": int(lens.projector.m_total),
        "input_dim": expected_dim,
        "dataset_hash": lens.manifest.get("dataset_hash"),
        "representation_compatibility": compatibility,
        **lens.feature_space_identity,
    }
    projector_provenance = getattr(lens.projector, "projector_provenance", None)
    if projector_provenance is not None:
        if not isinstance(projector_provenance, Mapping):
            raise ValueError("projector_provenance must be a mapping")
        lens_provenance["projector"] = dict(projector_provenance)
    return FeatureBatch(
        row_ids=batch.row_ids,
        arrays=arrays,
        roles={name: roles[name] for name in arrays},
        orientations={name: orientations[name] for name in arrays},
        feature_ids=tuple(range(int(lens.projector.m_total))),
        activation_polarity=lens.activation_polarity,
        code_semantics=lens.code_semantics,
        metadata=batch.metadata,
        provenance={
            "representation_source": dict(batch.provenance),
            "lens": lens_provenance,
            "views": {
                name: {
                    "role": roles[name],
                    "orientation": orientations[name],
                    "activation_polarity": (
                        "signed"
                        if name == "z_diff" and lens.input_rep == "individual"
                        else lens.activation_polarity
                    ),
                    "code_semantics": (
                        "activity_difference"
                        if name == "z_diff" and lens.input_rep == "individual"
                        else lens.code_semantics
                    ),
                    "derivation": (
                        "a_minus_b_after_encoding"
                        if name == "z_diff" and lens.input_rep == "individual"
                        else "direct_projection"
                    ),
                }
                for name in arrays
            },
        },
    )
