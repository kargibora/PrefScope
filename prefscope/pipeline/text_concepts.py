"""Apply published prompt/response lenses to one text pair."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from prefscope.analysis.presence import annotation_flag
from prefscope.api.loaded_lens import Lens


def resolve_device(requested: str) -> str:
    """Resolve ``auto`` and reject unavailable explicitly requested accelerators."""
    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("device must be one of: auto, cpu, cuda, mps")
    if requested == "cpu":
        return requested
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "concept extraction needs PyTorch; install 'prefscope[torch]' or a "
            "hardware-specific PyTorch build") from exc

    cuda = bool(torch.cuda.is_available())
    mps = bool(
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    )
    if requested == "cuda":
        if not cuda:
            backend = "HIP/ROCm" if torch.version.hip else "CUDA"
            raise ValueError(
                f"--device cuda requested, but no {backend} GPU is visible; request a GPU "
                "allocation or use --device cpu"
            )
        return requested
    if requested == "mps":
        if not mps:
            raise ValueError("--device mps requested, but Apple MPS is unavailable")
        return requested
    if cuda:
        return "cuda"
    if mps:
        return "mps"
    return "cpu"


def extract_present_concepts(
    lens: Lens,
    codes,
    *,
    policy: str = "mixed",
    fidelity_only: bool = True,
    top: int | None = 20,
) -> list[dict]:
    """Return named positive-pole concepts present in one sparse code vector."""
    values = np.atleast_2d(np.asarray(codes, dtype=np.float32))
    if len(values) != 1:
        raise ValueError("single-text extraction expects exactly one code vector")
    annotations = lens.feature_table.drop_duplicates("feature_id", keep="last").copy()
    annotations["feature_id"] = pd.to_numeric(
        annotations["feature_id"], errors="raise"
    ).astype(int)
    if "concept" not in annotations.columns:
        return []
    named = annotations[
        annotations["concept"].notna()
        & annotations["concept"].astype(str).str.strip().ne("")
    ].copy()
    if fidelity_only:
        if "fidelity_pass" not in named.columns:
            raise ValueError(
                "verified-only extraction needs feature_fidelity.csv bundled with the lens; "
                "pass fidelity_only=False to inspect unverified names"
            )
        named = named[named["fidelity_pass"].map(annotation_flag)]
    feature_ids = [
        feature_id
        for feature_id in named["feature_id"].astype(int)
        if 0 <= feature_id < values.shape[1]
    ]
    presence = lens.presence(values, feature_ids=feature_ids, policy=policy)
    by_id = named.set_index("feature_id")
    rows = []
    for column, feature_id in enumerate(presence.feature_ids):
        if not bool(presence.values[0, column]):
            continue
        annotation = by_id.loc[int(feature_id)]
        basis = str(presence.basis[column])
        row = {
            "feature_id": int(feature_id),
            "concept": str(annotation["concept"]),
            "activation": float(values[0, int(feature_id)]),
            "presence_basis": basis,
            "semantic_threshold": (
                float(presence.thresholds[column])
                if basis == "semantic_threshold"
                else None
            ),
            "fidelity_pass": annotation_flag(
                annotation.get("fidelity_pass"), default=False
            ),
        }
        for name in ("correlation", "agreement", "semantic_role", "requested_share"):
            if name in annotation and pd.notna(annotation[name]):
                value = annotation[name]
                row[name] = value.item() if hasattr(value, "item") else value
        rows.append(row)
    rows.sort(key=lambda item: (-item["activation"], item["feature_id"]))
    return rows if top is None else rows[: int(top)]


def _load_source(source, *, device: str, revision=None, annotations=None) -> Lens:
    if isinstance(source, Lens):
        return source
    value = str(source)
    if value.startswith("hf://"):
        from prefscope.api.hub import split_hf_source

        repo_id, subfolder = split_hf_source(value)
        return Lens.from_pretrained(
            repo_id,
            revision=revision,
            subfolder=subfolder,
            device=device,
            annotations=annotations,
        )
    return Lens.load(Path(value), device=device, annotations=annotations)


def extract_text_concepts(
    prompt: str,
    completion: str | None = None,
    *,
    repo_id: str | None = None,
    prompt_lens=None,
    completion_lens=None,
    prompt_subfolder: str | None = None,
    completion_subfolder: str | None = None,
    revision: str | None = None,
    device: str = "auto",
    presence_policy: str = "calibrated",
    fidelity_only: bool = True,
    top: int | None = 20,
) -> dict:
    """Download/load either or both lenses and report concepts for one text pair."""
    if repo_id and (prompt_lens is not None or completion_lens is not None):
        raise ValueError("use either repo_id+subfolders or explicit lens sources")
    if repo_id and not (prompt_subfolder or completion_subfolder):
        raise ValueError(
            "repo_id needs prompt_subfolder and/or completion_subfolder; explicit "
            "subfolders make multi-lens repositories unambiguous")
    if not repo_id and prompt_lens is None and completion_lens is None:
        raise ValueError("provide prompt_lens and/or completion_lens")
    if completion_lens is not None and completion is None:
        raise ValueError("completion_lens needs completion text")
    if (repo_id and completion_subfolder) and completion is None:
        raise ValueError("completion_subfolder needs completion text")
    if top is not None and top < 0:
        raise ValueError("top must be non-negative or None")
    resolved = resolve_device(device)
    question_lens = None
    response_lens = None
    if repo_id is not None:
        if completion_subfolder:
            response_lens = Lens.from_pretrained(
                repo_id,
                revision=revision,
                subfolder=completion_subfolder,
                device=resolved,
            )
        if prompt_subfolder:
            question_lens = Lens.from_pretrained(
                repo_id, revision=revision, subfolder=prompt_subfolder, device=resolved
            )
    else:
        if completion_lens is not None:
            response_lens = _load_source(
                completion_lens, device=resolved, revision=revision
            )
        if prompt_lens is not None:
            question_lens = _load_source(
                prompt_lens, device=resolved, revision=revision)
    if question_lens is not None and response_lens is not None:
        prompt_contract = question_lens.embedder
        response_contract = response_lens.embedder
        fields = (
            "model_id", "model_revision", "max_tokens", "pooling",
            "normalization", "effective_dtype_name",
        )
        mismatches = []
        for field in fields:
            left = getattr(prompt_contract, field)
            right = getattr(response_contract, field)
            left = left() if callable(left) else left
            right = right() if callable(right) else right
            if left != right:
                mismatches.append(f"{field}: {left!r} != {right!r}")
        if mismatches:
            raise ValueError(
                "prompt and completion lenses have incompatible embedding contracts: "
                + "; ".join(mismatches))
        # Share one set of model weights without discarding the prompt lens's exact
        # training-time prompt instruction.
        response_contract.prompt_embed_instruction = (
            prompt_contract.prompt_embed_instruction)
        question_lens.embedder = response_contract

    result = {
        "source": repo_id or ",".join(
            str(source) for source in (prompt_lens, completion_lens)
            if source is not None),
        "revision": revision,
        "resolved_revision": next(
            (getattr(lens, "pretrained_resolved_revision", None)
             for lens in (question_lens, response_lens)
             if lens is not None and getattr(
                 lens, "pretrained_resolved_revision", None)),
            None,
        ),
        "device": resolved,
        "presence_policy": presence_policy,
    }
    if question_lens is not None:
        result["prompt"] = extract_present_concepts(
            question_lens,
            question_lens.encode_one(prompt),
            policy=presence_policy,
            fidelity_only=fidelity_only,
            top=top,
        )
    if response_lens is not None:
        result["completion"] = extract_present_concepts(
            response_lens,
            response_lens.encode_one(prompt, completion),
            policy=presence_policy,
            fidelity_only=fidelity_only,
            top=top,
        )
    return result


__all__ = ["extract_present_concepts", "extract_text_concepts", "resolve_device"]
