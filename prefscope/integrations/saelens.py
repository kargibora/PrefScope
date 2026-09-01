"""Lazy adapter for pretrained SAEs loaded through SAELens.

The adapter implements PrefScope's numerical projector protocol without importing
SAELens or PyTorch at module import time. A plain ``import prefscope`` therefore
remains Torch-free. The wrapped SAE still expects activations from its exact reader
model and hook point; it is not a text encoder and is not portable across activation
coordinate systems.
"""
from __future__ import annotations

import hashlib
from importlib import metadata as importlib_metadata
import json
from numbers import Integral
from typing import Mapping

import numpy as np

from prefscope.core.lens_backend import LensBackend, LensCapabilities
from prefscope.core.representation import validate_portable_mapping, validate_row_ids


def _value(obj, name, default=None):
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _json_value(value):
    """Convert small config values to stable JSON-compatible provenance."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_json_value(item) for item in value)
    return str(value)


def _architecture(cfg) -> str:
    value = getattr(cfg, "architecture", None)
    if callable(value):
        value = value()
    if value is None:
        value = type(cfg).__name__
    return str(value).casefold()


def _infer_activation_polarity(cfg, architecture: str) -> str:
    activation = str(
        _value(cfg, "activation_fn_str", _value(cfg, "activation_fn", ""))
    ).casefold()
    known_nonnegative = {
        "batchtopk", "gated", "jumprelu", "jumprelu_skip_transcoder",
        "jumprelu_transcoder", "skip_transcoder", "standard", "topk", "transcoder",
    }
    if architecture in known_nonnegative or any(
        name in activation for name in ("relu", "topk", "jumprelu")
    ):
        return "nonnegative"
    return "unknown"


def _torch_module():
    try:
        import torch
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise ImportError(
                "SAELens projection needs PyTorch; install 'prefscope[saelens]' "
                "or install a hardware-specific PyTorch build and 'sae-lens>=6.50,<7'"
            ) from exc
        raise
    return torch


class SAELensProjector:
    """Adapt one loaded SAELens SAE to PrefScope's frozen projector protocol.

    ``project`` accepts a two-dimensional matrix of activations from the exact model
    and hook declared by the SAE. It returns numerical SAE activity. These values are
    not semantic presence until a separate PrefScope calibration confirms a label and
    threshold on the target data.

    Token-level users should project every token and pool feature activations after
    projection. Pooling model activations before this nonlinear projector is generally
    not equivalent.
    """

    def __init__(
        self,
        sae,
        *,
        release: str | None = None,
        sae_id: str | None = None,
        input_rep: str = "individual",
        batch_size: int = 1024,
        max_output_bytes: int = 256 * 1024 * 1024,
        activation_polarity: str | None = None,
        reader_model_revision: str | None = None,
        item_projection_policy: str = "forbid",
        representation_contract: Mapping[str, object] | None = None,
    ) -> None:
        if not callable(getattr(sae, "encode", None)):
            raise ValueError("sae must provide encode(activations)")
        cfg = getattr(sae, "cfg", None)
        if cfg is None:
            raise ValueError("sae must expose its SAELens cfg")
        if input_rep not in {"individual", "prompt"}:
            raise ValueError(
                "a pretrained SAELens SAE supports individual or prompt activity; "
                "direct difference projection is out of distribution"
            )
        if item_projection_policy not in {"forbid", "single_token"}:
            raise ValueError(
                "item_projection_policy must be 'forbid' or 'single_token'"
            )
        if isinstance(batch_size, bool) or not isinstance(batch_size, Integral):
            raise ValueError("batch_size must be a positive integer")
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if isinstance(max_output_bytes, bool) or not isinstance(max_output_bytes, Integral):
            raise ValueError("max_output_bytes must be a positive integer")
        max_output_bytes = int(max_output_bytes)
        if max_output_bytes < 1:
            raise ValueError("max_output_bytes must be a positive integer")
        for name, value in (("release", release), ("sae_id", sae_id),
                            ("reader_model_revision", reader_model_revision)):
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be a non-empty string or None")

        d_in = _value(cfg, "d_in")
        d_sae = _value(cfg, "d_sae")
        if (
            isinstance(d_in, bool) or not isinstance(d_in, Integral) or int(d_in) < 1
            or isinstance(d_sae, bool) or not isinstance(d_sae, Integral)
            or int(d_sae) < 1
        ):
            raise ValueError("SAELens cfg must declare positive integer d_in and d_sae")
        reshape = _value(cfg, "reshape_activations", "none") or "none"
        if str(reshape) != "none":
            raise ValueError(
                "this adapter currently supports flat residual/MLP activations only; "
                f"SAELens reshape_activations={reshape!r} needs a structured hook adapter"
            )

        architecture = _architecture(cfg)
        supported_architectures = {
            "gated", "jumprelu", "jumprelu_skip_transcoder",
            "jumprelu_transcoder", "skip_transcoder", "standard", "topk",
            "transcoder",
        }
        if architecture not in supported_architectures:
            raise ValueError(
                f"SAELens architecture {architecture!r} is not supported by the flat "
                "adapter; temporal and other sequence-aware SAEs need a shape-aware "
                "integration"
            )
        polarity = (
            _infer_activation_polarity(cfg, architecture)
            if activation_polarity is None else activation_polarity
        )
        if polarity not in {"nonnegative", "signed", "unknown"}:
            raise ValueError(
                "activation_polarity must be nonnegative, signed, unknown, or None"
            )

        metadata = _value(cfg, "metadata", {}) or {}
        layout = "token" if item_projection_policy == "forbid" else "one_token_per_item"
        if representation_contract is None:
            model_name = _value(metadata, "model_name")
            hook_name = _value(metadata, "hook_name")
            if not model_name or not hook_name:
                raise ValueError(
                    "SAELens cfg metadata must declare model_name and hook_name, or "
                    "representation_contract must be supplied explicitly"
                )
            contract = {
                "representation_family": "internal_activation",
                "model_id": str(model_name),
                "hook_name": str(hook_name),
                "source_activation_preprocessing": "raw_hook_activation",
                "sae_input_normalization": str(
                    _value(cfg, "normalize_activations", "none") or "none"),
                "activation_reshape": str(reshape),
                "activation_layout": layout,
            }
            if reader_model_revision is not None:
                contract["model_revision"] = reader_model_revision
            if item_projection_policy == "single_token":
                contract["item_reduction"] = "single_token"
            optional = {
                "hook_layer": _value(metadata, "hook_layer"),
                "hook_head_index": _value(metadata, "hook_head_index"),
                "context_size": _value(metadata, "context_size"),
                "prepend_bos": _value(metadata, "prepend_bos"),
                "seqpos_slice": _value(metadata, "seqpos_slice"),
                "model_from_pretrained_kwargs": _value(
                    metadata, "model_from_pretrained_kwargs"),
                "exclude_special_tokens": (
                    False
                    if _value(metadata, "exclude_special_tokens", False) is None
                    else _value(metadata, "exclude_special_tokens", False)
                ),
            }
            contract.update({
                key: _json_value(value)
                for key, value in optional.items() if value is not None
            })
        else:
            contract = {str(key): _json_value(value)
                        for key, value in dict(representation_contract).items()}
            required = {
                "representation_family", "model_id", "hook_name",
                "source_activation_preprocessing", "sae_input_normalization",
                "activation_reshape", "activation_layout",
            }
            missing = sorted(required - set(contract))
            if missing:
                raise ValueError(
                    "SAELens representation_contract is incomplete; missing "
                    f"{missing}"
                )
            if contract.get("representation_family") != "internal_activation":
                raise ValueError(
                    "SAELens representation_contract must declare "
                    "representation_family='internal_activation'"
                )
            for coordinate in ("model_id", "hook_name"):
                value = contract.get(coordinate)
                if not isinstance(value, str) or not value:
                    raise ValueError(
                        f"SAELens representation_contract {coordinate} must be a "
                        "non-empty string"
                    )
            if contract.get("source_activation_preprocessing") != "raw_hook_activation":
                raise ValueError(
                    "SAELens representation_contract requires raw_hook_activation "
                    "source preprocessing"
                )
            supported_normalization = {
                "none", "expected_average_only_in", "layer_norm",
                "constant_norm_rescale",
            }
            if contract.get("sae_input_normalization") not in supported_normalization:
                raise ValueError(
                    "SAELens representation_contract has unsupported "
                    "sae_input_normalization"
                )
            if contract.get("activation_reshape") != "none":
                raise ValueError(
                    "the flat SAELens adapter requires activation_reshape='none'"
                )
            if contract.get("activation_layout") != layout:
                raise ValueError(
                    "SAELens representation_contract activation_layout disagrees "
                    "with item_projection_policy"
                )
            if item_projection_policy == "single_token" and (
                contract.get("item_reduction") != "single_token"
            ):
                raise ValueError(
                    "single-token item projection requires item_reduction='single_token'"
                )
            if item_projection_policy == "forbid" and "item_reduction" in contract:
                raise ValueError(
                    "token-layout SAELens contracts must not declare a pre-SAE "
                    "item_reduction"
                )
            declared_coordinates = {
                "model_id": _value(metadata, "model_name"),
                "hook_name": _value(metadata, "hook_name"),
                "sae_input_normalization": (
                    _value(cfg, "normalize_activations", "none") or "none"),
                "activation_reshape": reshape,
            }
            if reader_model_revision is not None:
                declared_coordinates["model_revision"] = reader_model_revision
            contradictions = [
                key for key, value in declared_coordinates.items()
                if value is not None and contract.get(key) != _json_value(value)
            ]
            if contradictions:
                raise ValueError(
                    "SAELens representation_contract contradicts checkpoint metadata: "
                    f"{sorted(contradictions)}"
                )

        contract = dict(validate_portable_mapping(
            contract, where="SAELens representation contract"))

        self.sae = sae
        self.release = release
        self.sae_id = sae_id
        self.input_rep = input_rep
        self.item_projection_policy = item_projection_policy
        self.supports_item_projection = item_projection_policy == "single_token"
        self.batch_size = batch_size
        self.max_output_bytes = max_output_bytes
        self.input_dim = int(d_in)
        self.m_total = int(d_sae)
        self.k = None
        self.architecture = architecture
        self.activation_polarity = polarity
        self.code_semantics = "numerical_activity"
        self.selection_rule = f"saelens:{architecture}"
        self.representation_contract = contract
        self.coordinate_pin_status = (
            "reader_revision_declared_sae_unpinned" if reader_model_revision
            else "reader_and_sae_unpinned"
        )
        self.device = str(getattr(sae, "device", _value(cfg, "device", "cpu")))
        self.dtype = str(getattr(sae, "dtype", _value(cfg, "dtype", "float32"))).replace(
            "torch.", ""
        )
        try:
            version = importlib_metadata.version("sae-lens")
        except importlib_metadata.PackageNotFoundError:
            version = None
        cfg_dict = cfg.to_dict() if callable(getattr(cfg, "to_dict", None)) else {
            "architecture": architecture,
            "d_in": self.input_dim,
            "d_sae": self.m_total,
        }
        cfg_payload = json.dumps(
            _json_value(cfg_dict), sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        config_fingerprint = hashlib.sha256(cfg_payload).hexdigest()
        self.projector_provenance = {
            key: value for key, value in {
                "backend": "saelens",
                "saelens_version": version,
                "release": release,
                "sae_id": sae_id,
                "architecture": architecture,
                "d_in": self.input_dim,
                "d_sae": self.m_total,
                "coordinate_pin_status": self.coordinate_pin_status,
                "sae_config_fingerprint": config_fingerprint,
                "item_projection_policy": item_projection_policy,
                "representation_contract": contract,
            }.items() if value is not None
        }
        parameters_fn = getattr(self.sae, "parameters", None)
        if callable(parameters_fn):
            for parameter in parameters_fn():
                requires_grad_fn = getattr(parameter, "requires_grad_", None)
                if callable(requires_grad_fn):
                    requires_grad_fn(False)
        eval_fn = getattr(self.sae, "eval", None)
        if callable(eval_fn):
            eval_fn()

    @classmethod
    def from_pretrained(
        cls,
        release: str,
        sae_id: str,
        *,
        device: str = "cpu",
        dtype: str = "float32",
        force_download: bool = False,
        input_rep: str = "individual",
        batch_size: int = 1024,
        max_output_bytes: int = 256 * 1024 * 1024,
        activation_polarity: str | None = None,
        reader_model_revision: str | None = None,
        item_projection_policy: str = "forbid",
        allow_unregistered_release: bool = False,
    ) -> "SAELensProjector":
        """Load a registered SAELens v6.50+ checkpoint and wrap its inference path."""
        if not isinstance(release, str) or not release:
            raise ValueError("release must be a non-empty string")
        if not isinstance(sae_id, str) or not sae_id:
            raise ValueError("sae_id must be a non-empty string")
        if not isinstance(allow_unregistered_release, bool):
            raise ValueError("allow_unregistered_release must be boolean")
        try:
            from sae_lens import SAE
            from sae_lens.loading.pretrained_saes_directory import (
                get_pretrained_saes_directory,
            )
        except ModuleNotFoundError as exc:
            if exc.name == "sae_lens" or str(exc.name).startswith("sae_lens."):
                raise ImportError(
                    "SAELens support is optional; install 'prefscope[saelens]'"
                ) from exc
            raise
        registered = get_pretrained_saes_directory()
        if release not in registered and not allow_unregistered_release:
            raise ValueError(
                f"SAELens release {release!r} is not in the installed trusted registry; "
                "set allow_unregistered_release=True only for a repository you trust"
            )
        sae = SAE.from_pretrained(
            release=release,
            sae_id=sae_id,
            device=device,
            dtype=dtype,
            force_download=force_download,
        )
        return cls(
            sae,
            release=release,
            sae_id=sae_id,
            input_rep=input_rep,
            batch_size=batch_size,
            max_output_bytes=max_output_bytes,
            activation_polarity=activation_polarity,
            reader_model_revision=reader_model_revision,
            item_projection_policy=item_projection_policy,
        )

    def _activation_matrix(self, values) -> np.ndarray:
        source = np.asarray(values)
        if source.ndim != 2 or source.shape[1] != self.input_dim:
            raise ValueError(
                f"SAELens activations must have shape (n, {self.input_dim}), "
                f"got {source.shape}"
            )
        if not np.issubdtype(source.dtype, np.number) or np.iscomplexobj(source):
            raise ValueError("SAELens activations must be a real numeric matrix")
        matrix = np.asarray(source, dtype=np.float32)
        if not np.isfinite(matrix).all():
            raise ValueError("SAELens activations must be finite")
        return matrix

    def _resolved_batch(self, requested: int | None) -> int:
        chunk_size = self.batch_size if requested is None else requested
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, Integral):
            raise ValueError("batch must be a positive integer or None")
        chunk_size = int(chunk_size)
        if chunk_size < 1:
            raise ValueError("batch must be a positive integer or None")
        bytes_per_row = self.m_total * np.dtype(np.float32).itemsize
        if bytes_per_row > self.max_output_bytes:
            raise ValueError(
                "one dense SAELens feature row exceeds max_output_bytes; choose a "
                "narrower pretrained SAE or raise the explicit memory budget"
            )
        return min(chunk_size, max(1, self.max_output_bytes // bytes_per_row))

    def _encoded_chunks(self, matrix: np.ndarray, *, batch: int | None = None):
        chunk_size = self._resolved_batch(batch)
        torch = _torch_module()
        torch_dtype = getattr(torch, self.dtype, None)
        with torch.inference_mode():
            for start in range(0, matrix.shape[0], chunk_size):
                tensor_kwargs = {"device": self.device}
                if torch_dtype is not None:
                    tensor_kwargs["dtype"] = torch_dtype
                tensor = torch.as_tensor(matrix[start:start + chunk_size], **tensor_kwargs)
                encoded = self.sae.encode(tensor)
                if bool(getattr(encoded, "is_sparse", False)):
                    encoded = encoded.to_dense()
                encoded = encoded.detach()
                to_fn = getattr(encoded, "to", None)
                if callable(to_fn) and hasattr(torch, "float32"):
                    encoded = to_fn(dtype=torch.float32)
                chunk = np.asarray(encoded.cpu().numpy(), dtype=np.float32)
                expected = (min(chunk_size, matrix.shape[0] - start), self.m_total)
                if chunk.shape != expected:
                    raise ValueError(
                        f"SAELens encode returned shape {chunk.shape}; expected {expected}"
                    )
                if not np.isfinite(chunk).all():
                    raise ValueError("SAELens encode returned non-finite feature activity")
                if self.activation_polarity == "nonnegative" and (chunk < 0).any():
                    raise ValueError(
                        "SAELens encode returned negative values despite a "
                        "nonnegative activation contract"
                    )
                yield start, chunk

    def project(self, values: np.ndarray, *, batch: int | None = None) -> np.ndarray:
        """Encode exact-hook activation rows into a bounded dense feature matrix."""
        matrix = self._activation_matrix(values)
        output_bytes = matrix.shape[0] * self.m_total * np.dtype(np.float32).itemsize
        if output_bytes > self.max_output_bytes:
            raise ValueError(
                "dense SAELens output exceeds max_output_bytes; use "
                "project_grouped(..., feature_ids=...) for token aggregation or "
                "raise the explicit memory budget"
            )
        out = np.empty((matrix.shape[0], self.m_total), dtype=np.float32)
        for start, chunk in self._encoded_chunks(matrix, batch=batch):
            out[start:start + len(chunk)] = chunk
        return out

    def validate_activation_contract(
        self, contract: Mapping[str, object]
    ) -> dict[str, object]:
        """Compare independently declared source coordinates with the SAE contract."""
        actual = dict(validate_portable_mapping(
            contract, where="SAELens activation source contract"))
        mismatches = []
        for key, expected in self.representation_contract.items():
            if key not in actual:
                mismatches.append(f"missing {key}")
            elif actual[key] != expected:
                mismatches.append(
                    f"{key}: expected {expected!r}, got {actual[key]!r}"
                )
        if mismatches:
            raise ValueError(
                "activation source is incompatible with this SAELens checkpoint: "
                + "; ".join(mismatches)
            )
        return {
            "status": (
                "matched_declared_reader_revision_sae_unpinned"
                if self.coordinate_pin_status == "reader_revision_declared_sae_unpinned"
                else "matched_declared_unpinned"
            ),
            "coordinate_pin_status": self.coordinate_pin_status,
        }

    def project_grouped(
        self,
        values: np.ndarray,
        token_row_ids,
        *,
        row_ids,
        feature_ids=None,
        batch: int | None = None,
    ) -> np.ndarray:
        """Encode tokens, then max-pool selected features into declared item rows."""
        from prefscope.core.features import validate_feature_ids
        from prefscope.core.representation import validate_row_ids

        if self.activation_polarity != "nonnegative":
            raise ValueError(
                "post-SAE max pooling currently requires nonnegative feature activity"
            )
        matrix = self._activation_matrix(values)
        rows = validate_row_ids(row_ids)
        token_ids = tuple(str(value) for value in token_row_ids)
        if len(token_ids) != len(matrix):
            raise ValueError("token_row_ids must contain one item ID per activation row")
        positions = {row_id: index for index, row_id in enumerate(rows)}
        unknown = sorted(set(token_ids) - set(positions))
        if unknown:
            raise ValueError(f"token_row_ids contain unknown item IDs: {unknown[:5]}")
        counts = {row_id: 0 for row_id in rows}
        for row_id in token_ids:
            counts[row_id] += 1
        missing = [row_id for row_id, count in counts.items() if count == 0]
        if missing:
            raise ValueError(f"token activations are missing item IDs: {missing[:5]}")
        selected = (
            tuple(range(self.m_total))
            if feature_ids is None
            else validate_feature_ids(feature_ids)
        )
        if not selected:
            raise ValueError("feature_ids must select at least one feature")
        if min(selected) < 0 or max(selected) >= self.m_total:
            raise ValueError("feature_ids contain an index outside this SAELens SAE")
        output_bytes = len(rows) * len(selected) * np.dtype(np.float32).itemsize
        if output_bytes > self.max_output_bytes:
            raise ValueError(
                "grouped SAELens output exceeds max_output_bytes; select fewer features"
            )
        selected_index = np.asarray(selected, dtype=int)
        out = np.full((len(rows), len(selected)), -np.inf, dtype=np.float32)
        group_positions = np.asarray([positions[value] for value in token_ids], dtype=int)
        for start, chunk in self._encoded_chunks(matrix, batch=batch):
            stop = start + len(chunk)
            np.maximum.at(out, group_positions[start:stop], chunk[:, selected_index])
        if not np.isfinite(out).all():
            raise ValueError("grouped SAELens feature activity is not finite")
        return out


class SAELensTextBackend(LensBackend):
    """Use one reader model and SAE for prompt and response text views.

    Text views are independent documents. Responses are not silently concatenated with
    prompts or formatted as chats. The backend extracts the exact declared hook, applies
    the SAE to each real token, and max-pools feature activity within each item.
    """

    input_rep = "individual"

    def __init__(
        self,
        projector: SAELensProjector,
        *,
        device: str = "cpu",
        text_batch_size: int = 8,
        long_text_policy: str = "truncate",
        include_bos: bool = False,
        reader_factory=None,
    ) -> None:
        if not isinstance(projector, SAELensProjector):
            raise ValueError("projector must be a SAELensProjector")
        if (
            isinstance(text_batch_size, bool)
            or not isinstance(text_batch_size, Integral)
            or int(text_batch_size) < 1
        ):
            raise ValueError("text_batch_size must be a positive integer")
        if long_text_policy not in {"truncate", "error"}:
            raise ValueError("long_text_policy must be 'truncate' or 'error'")
        if not isinstance(include_bos, bool):
            raise ValueError("include_bos must be boolean")
        self.projector = projector
        self.input_rep = projector.input_rep
        self.device = str(device)
        self.text_batch_size = int(text_batch_size)
        self.long_text_policy = long_text_policy
        self.include_bos = include_bos
        self.reader_factory = reader_factory
        self._reader = None
        self._capabilities = LensCapabilities(
            ("prompt", "response_a", "response_b", "response_difference"),
            shared_feature_space=True,
            difference="a_minus_b_after_encoding",
        )

    @property
    def capabilities(self) -> LensCapabilities:
        return self._capabilities

    @property
    def m_total(self) -> int:
        return self.projector.m_total

    @property
    def activation_polarity(self) -> str:
        return self.projector.activation_polarity

    @property
    def code_semantics(self) -> str:
        return self.projector.code_semantics

    def _metadata_value(self, name, default=None):
        cfg = getattr(self.projector.sae, "cfg", None)
        return _value(_value(cfg, "metadata", {}) or {}, name, default)

    def _reader_model(self):
        if self._reader is not None:
            return self._reader
        model_id = str(self.projector.representation_contract["model_id"])
        kwargs = dict(self._metadata_value("model_from_pretrained_kwargs", {}) or {})
        revision = self.projector.representation_contract.get("model_revision")
        if revision is not None:
            if "revision" in kwargs and kwargs["revision"] != revision:
                raise ValueError(
                    "reader_model_revision contradicts checkpoint model kwargs"
                )
            kwargs["revision"] = revision
        model_class = str(
            self._metadata_value("model_class_name") or "HookedTransformer"
        )
        hook_name = str(self.projector.representation_contract["hook_name"])
        if self.reader_factory is None:
            try:
                from sae_lens.load_model import load_model
            except ModuleNotFoundError as exc:
                if exc.name == "sae_lens" or str(exc.name).startswith("sae_lens."):
                    raise ImportError(
                        "SAELens text encoding needs the optional 'saelens' extra"
                    ) from exc
                raise
            self._reader = load_model(
                model_class_name=model_class,
                model_name=model_id,
                device=self.device,
                model_from_pretrained_kwargs=kwargs,
                hook_names=[hook_name],
            )
        else:
            self._reader = self.reader_factory(
                model_class_name=model_class,
                model_name=model_id,
                device=self.device,
                model_from_pretrained_kwargs=kwargs,
                hook_names=[hook_name],
            )
        eval_fn = getattr(self._reader, "eval", None)
        if callable(eval_fn):
            eval_fn()
        return self._reader

    def _activation_batch(self, texts: list[str]):
        torch = _torch_module()
        model = self._reader_model()
        prepend_bos = bool(self._metadata_value("prepend_bos", False))
        context_size = self._metadata_value("context_size")
        seqpos_slice = self._metadata_value("seqpos_slice") or (None,)
        hook_head_index = self._metadata_value("hook_head_index")
        token_rows = []
        model_class = str(
            self._metadata_value("model_class_name") or "HookedTransformer")
        is_proxy = model_class == "AutoModelForCausalLM" or (
            type(model).__name__ == "HookedProxyLM")
        for text in texts:
            if is_proxy:
                tokens = model.to_tokens(
                    str(text), truncate=False, prepend_bos=False,
                    move_to_device=False)
                if prepend_bos:
                    bos_id = getattr(getattr(model, "tokenizer", None),
                                     "bos_token_id", None)
                    if bos_id is None:
                        raise ValueError(
                            "reader metadata requests BOS but tokenizer has no BOS ID")
                    bos = torch.full(
                        (tokens.shape[0], 1), int(bos_id), dtype=tokens.dtype)
                    tokens = torch.cat((bos, tokens), dim=1)
                tokens = tokens.to(self.device)
            else:
                tokens = model.to_tokens(
                    str(text), truncate=False, prepend_bos=prepend_bos)
            if context_size is not None and int(tokens.shape[1]) > int(context_size):
                if self.long_text_policy == "error":
                    raise ValueError(
                        f"text has {int(tokens.shape[1])} tokens but this SAE declares "
                        f"context_size={int(context_size)}; use "
                        "long_text_policy='truncate' explicitly to keep the first window"
                    )
                tokens = tokens[:, : int(context_size)]
            if int(tokens.shape[1]) == 0:
                raise ValueError("one text contains no tokens")
            token_rows.append(tokens[0])

        # Bucket by exact length. This avoids relying on padding behavior in generic
        # HookedProxyLM readers while retaining batching for equal-length documents.
        by_length = {}
        for index, tokens in enumerate(token_rows):
            by_length.setdefault(int(tokens.shape[0]), []).append((index, tokens))
        extracted = [None] * len(texts)
        hook_name = str(self.projector.representation_contract["hook_name"])
        try:
            from sae_lens.util import extract_stop_at_layer_from_tlens_hook_name
            stop_at_layer = extract_stop_at_layer_from_tlens_hook_name(hook_name)
        except ImportError:
            # Preserve standard TransformerLens block-hook behavior in lightweight
            # installs where the optional SAELens package is not present.
            parts = hook_name.split(".")
            stop_at_layer = (
                int(parts[1]) + 1
                if len(parts) >= 3 and parts[0] == "blocks" and parts[1].isdigit()
                else None
            )
        except ValueError:
            stop_at_layer = None
        for length, entries in by_length.items():
            stacked = torch.stack([tokens for _, tokens in entries], dim=0)
            run_options = {
                "names_filter": [hook_name],
                "prepend_bos": False,
            }
            if stop_at_layer is not None:
                run_options["stop_at_layer"] = stop_at_layer
            if not is_proxy:
                run_options["return_type"] = None
            with torch.inference_mode():
                _, cache = model.run_with_cache(stacked, **run_options)
            hook = cache[hook_name]
            if hook_head_index is not None:
                if hook.ndim != 4:
                    raise ValueError(
                        "SAELens hook_head_index requires a four-dimensional hook"
                    )
                hook = hook[:, :, int(hook_head_index), :]
            positions = list(range(length))[slice(*tuple(seqpos_slice))]
            if prepend_bos and not self.include_bos:
                positions = [position for position in positions if position != 0]
            excluded = self._metadata_value("exclude_special_tokens", False)
            if excluded is True:
                tokenizer = getattr(model, "tokenizer", None)
                if tokenizer is None:
                    raise ValueError(
                        "exclude_special_tokens=True requires a reader tokenizer")
                try:
                    from sae_lens.util import get_special_token_ids
                    special_ids = set(get_special_token_ids(tokenizer))
                except ImportError:
                    # Match SAELens' structural-token policy without importing the
                    # optional package. Additional/chat-template tokens stay included.
                    structural_attrs = (
                        "bos_token_id", "eos_token_id", "pad_token_id",
                        "sep_token_id", "decoder_start_token_id",
                    )
                    special_ids = {
                        token_id
                        for attr in structural_attrs
                        if (token_id := getattr(tokenizer, attr, None)) is not None
                    }
            elif excluded in (False, None):
                special_ids = set()
            elif isinstance(excluded, (list, tuple)) and all(
                isinstance(value, Integral) and not isinstance(value, bool)
                for value in excluded
            ):
                special_ids = {int(value) for value in excluded}
            else:
                raise ValueError(
                    "SAELens exclude_special_tokens must be boolean or integer list")
            for local, (original, tokens) in enumerate(entries):
                selected_positions = [
                    position for position in positions
                    if int(tokens[position]) not in special_ids
                ]
                if not selected_positions:
                    raise ValueError(
                        "one text contains no analyzed tokens after "
                        "BOS/seqpos/special-token selection")
                value = hook[local, selected_positions]
                if value.ndim != 2 or int(value.shape[1]) != self.projector.input_dim:
                    raise ValueError(
                        f"reader hook {hook_name!r} returned token shape "
                        f"{tuple(value.shape)}; expected (*, {self.projector.input_dim})"
                    )
                extracted[original] = np.asarray(
                    value.detach().float().cpu().numpy(), dtype=np.float32
                )
        activations = []
        local_ids = []
        for index, array in enumerate(extracted):
            activations.append(array)
            local_ids.extend([str(index)] * len(array))
        return np.concatenate(activations, axis=0), tuple(local_ids)

    def _encode_view(
        self,
        texts: list[str],
        row_ids: tuple[str, ...],
        *,
        feature_ids,
        batch_size: int,
    ) -> np.ndarray:
        chunks = []
        for start in range(0, len(texts), batch_size):
            stop = min(start + batch_size, len(texts))
            activations, memberships = self._activation_batch(texts[start:stop])
            local_rows = tuple(str(index) for index in range(stop - start))
            chunk = self.projector.project_grouped(
                activations,
                memberships,
                row_ids=local_rows,
                feature_ids=feature_ids,
            )
            chunks.append(chunk)
        return np.concatenate(chunks, axis=0)

    @staticmethod
    def _item_metadata(items) -> dict[str, tuple[object, ...]]:
        reserved = {
            "prompt",
            "response_a",
            "response_b",
            "pref",
            "model_a",
            "model_b",
            "response_length_a",
            "response_length_b",
            "response_length_difference",
        }
        custom = set()
        for item in items:
            if not isinstance(item.meta, dict):
                raise ValueError("PairItem.meta must be a mapping")
            collisions = reserved & set(item.meta)
            if collisions:
                raise ValueError(
                    f"PairItem.meta collides with canonical fields: {sorted(collisions)}"
                )
            custom.update(item.meta)
        lengths_a = tuple(len(str(item.y_a).split()) for item in items)
        lengths_b = tuple(
            None if item.y_b is None else len(str(item.y_b).split()) for item in items
        )
        metadata = {
            "prompt": tuple(str(item.x) for item in items),
            "response_a": tuple(str(item.y_a) for item in items),
            "response_b": tuple(item.y_b for item in items),
            "pref": tuple(item.pref for item in items),
            "model_a": tuple(item.model_a for item in items),
            "model_b": tuple(item.model_b for item in items),
            "response_length_a": lengths_a,
            "response_length_b": lengths_b,
            "response_length_difference": tuple(
                None if b is None else a - b for a, b in zip(lengths_a, lengths_b)
            ),
        }
        metadata.update(
            {
                name: tuple(item.meta.get(name) for item in items)
                for name in sorted(custom)
            }
        )
        return metadata

    def featurize(
        self,
        items,
        *,
        views=None,
        feature_ids=None,
        batch_size=None,
    ):
        from prefscope.core.features import FeatureBatch, validate_feature_ids

        rows = list(items)
        row_ids = validate_row_ids(tuple(item.id for item in rows))
        requested = tuple(views or self.capabilities.views)
        unknown = set(requested) - set(self.capabilities.views)
        if unknown:
            raise ValueError(
                f"SAELens backend does not support views {sorted(unknown)}"
            )
        selected = (
            tuple(range(self.m_total))
            if feature_ids is None
            else validate_feature_ids(feature_ids)
        )
        if not selected:
            raise ValueError("feature_ids must select at least one feature")
        text_batch = self.text_batch_size if batch_size is None else batch_size
        if (
            isinstance(text_batch, bool)
            or not isinstance(text_batch, Integral)
            or int(text_batch) < 1
        ):
            raise ValueError("batch_size must be a positive integer or None")
        text_batch = int(text_batch)
        width = len(selected)
        materialized_views = len(requested)
        if "response_difference" in requested:
            materialized_views += int("response_a" not in requested)
            materialized_views += int("response_b" not in requested)
        output_bytes = len(rows) * width * 4 * materialized_views
        if output_bytes > self.projector.max_output_bytes:
            raise ValueError(
                "requested SAELens feature views exceed max_output_bytes; select "
                "fewer feature_ids or raise the explicit memory budget"
            )
        arrays = {}
        if "prompt" in requested:
            arrays["z_prompt"] = self._encode_view(
                [str(item.x) for item in rows],
                row_ids,
                feature_ids=selected,
                batch_size=text_batch,
            )
        need_a = bool({"response_a", "response_difference"} & set(requested))
        need_b = bool({"response_b", "response_difference"} & set(requested))
        z_a = None
        z_b = None
        if need_a:
            z_a = self._encode_view(
                [str(item.y_a) for item in rows],
                row_ids,
                feature_ids=selected,
                batch_size=text_batch,
            )
            if "response_a" in requested:
                arrays["z_a"] = z_a
        if need_b:
            if any(item.y_b is None for item in rows):
                raise ValueError("response_b views require y_b on every PairItem")
            z_b = self._encode_view(
                [str(item.y_b) for item in rows],
                row_ids,
                feature_ids=selected,
                batch_size=text_batch,
            )
            if "response_b" in requested:
                arrays["z_b"] = z_b
        if "response_difference" in requested:
            arrays["z_diff"] = z_a - z_b
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
        compatibility = self.projector.validate_activation_contract(
            self.projector.representation_contract
        )
        return FeatureBatch(
            row_ids=row_ids,
            arrays=arrays,
            roles={name: roles[name] for name in arrays},
            orientations={name: orientations[name] for name in arrays},
            feature_ids=selected,
            metadata=self._item_metadata(rows),
            activation_polarity=self.activation_polarity,
            code_semantics=self.code_semantics,
            provenance={
                "representation_source": dict(self.projector.representation_contract),
                "text_context": "independent_documents",
                "long_text_policy": self.long_text_policy,
                "include_bos_in_pool": self.include_bos,
                "seqpos_slice": _json_value(self._metadata_value("seqpos_slice")),
                "token_reduction": "post_sae_max",
                "padding": "none_exact_length_buckets",
                "views": {
                    name: {
                        "role": roles[name],
                        "orientation": orientations[name],
                        "activation_polarity": (
                            "signed" if name == "z_diff" else self.activation_polarity
                        ),
                        "code_semantics": (
                            "activity_difference"
                            if name == "z_diff"
                            else self.code_semantics
                        ),
                        "derivation": (
                            "a_minus_b_after_post_sae_max"
                            if name == "z_diff"
                            else "post_sae_max"
                        ),
                    }
                    for name in arrays
                },
                "lens": {
                    "backend": "saelens",
                    "m_total": self.m_total,
                    "selected_features": len(selected),
                    "representation_compatibility": compatibility,
                    "projector": self.projector.projector_provenance,
                },
            },
        )


def project_saelens_tokens(
    lens,
    *,
    row_ids,
    token_activations: Mapping[str, np.ndarray],
    token_row_ids: Mapping[str, object],
    representation_contract: Mapping[str, object],
    feature_ids=None,
    metadata: Mapping[str, object] | None = None,
    batch: int | None = None,
):
    """Create a typed FeatureBatch by encoding tokens before max pooling."""
    from prefscope.core.features import FeatureBatch, validate_feature_ids
    from prefscope.core.representation import validate_row_ids

    projector = getattr(lens, "projector", None)
    if not isinstance(projector, SAELensProjector):
        raise ValueError("project_saelens_tokens requires a SAELensProjector lens")
    rows = validate_row_ids(row_ids)
    activations = dict(token_activations)
    memberships = dict(token_row_ids)
    if set(activations) != set(memberships):
        raise ValueError(
            "token_activations and token_row_ids must name the same arrays"
        )
    if lens.input_rep == "prompt":
        required = {"prompt"}
    else:
        required = {"response_a"}
        if "response_b" in activations:
            required.add("response_b")
    if set(activations) != required:
        raise ValueError(
            f"token arrays for a {lens.input_rep} lens must be {sorted(required)}, "
            f"got {sorted(activations)}"
        )
    compatibility = projector.validate_activation_contract(representation_contract)
    selected = (
        None if feature_ids is None else validate_feature_ids(tuple(feature_ids))
    )
    selected_width = projector.m_total if selected is None else len(selected)
    output_views = 3 if "response_b" in activations else 1
    estimated_output_bytes = (
        len(rows) * selected_width * np.dtype(np.float32).itemsize * output_views
    )
    if estimated_output_bytes > projector.max_output_bytes:
        raise ValueError(
            "typed SAELens feature views exceed max_output_bytes; select fewer features"
        )
    arrays = {}
    if lens.input_rep == "prompt":
        arrays["z_prompt"] = projector.project_grouped(
            activations["prompt"], memberships["prompt"], row_ids=rows,
            feature_ids=selected, batch=batch)
    else:
        arrays["z_a"] = projector.project_grouped(
            activations["response_a"], memberships["response_a"], row_ids=rows,
            feature_ids=selected, batch=batch)
        if "response_b" in activations:
            arrays["z_b"] = projector.project_grouped(
                activations["response_b"], memberships["response_b"], row_ids=rows,
                feature_ids=selected, batch=batch)
            arrays["z_diff"] = arrays["z_a"] - arrays["z_b"]
    selected = (
        tuple(range(projector.m_total)) if selected is None else selected
    )
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
    return FeatureBatch(
        row_ids=rows,
        arrays=arrays,
        roles={name: roles[name] for name in arrays},
        orientations={name: orientations[name] for name in arrays},
        feature_ids=selected,
        metadata=dict(metadata or {}),
        activation_polarity=projector.activation_polarity,
        code_semantics=projector.code_semantics,
        provenance={
            "representation_source": dict(representation_contract),
            "token_reduction": "post_sae_max",
            "views": {
                name: {
                    "role": roles[name],
                    "orientation": orientations[name],
                    "activation_polarity": (
                        "signed" if name == "z_diff" else projector.activation_polarity
                    ),
                    "code_semantics": (
                        "activity_difference"
                        if name == "z_diff" else projector.code_semantics
                    ),
                    "derivation": (
                        "a_minus_b_after_post_sae_max"
                        if name == "z_diff" else "post_sae_max"
                    ),
                }
                for name in arrays
            },
            "lens": {
                "backend": "saelens",
                "input_rep": lens.input_rep,
                "m_total": projector.m_total,
                "selected_features": len(selected),
                "representation_compatibility": compatibility,
                "projector": projector.projector_provenance,
            },
        },
    )


__all__ = [
    "SAELensProjector", "SAELensTextBackend", "project_saelens_tokens",
]
