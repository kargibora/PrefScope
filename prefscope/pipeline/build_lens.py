"""Build a frozen SAE lens from a judge annotation corpus.

By default (``input_rep="difference"``) the SAE is trained on the contrast
vector ``e_a - e_b`` (WIMHF-style): features are contrast directions that
capture how response A differs from response B.  In this mode only ``z_diff``
is written, because projecting individual unit embeddings through a
difference-trained SAE is out-of-distribution.

The optional ``input_rep="individual"`` mode pools completion embeddings into
the training matrix and writes per-response codes. For paired data it writes
``z_a``, ``z_b``, and ``z_diff = z_a - z_b``; for homogeneous single-response
data it writes ``z_a``. Difference lenses always require pairs.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from prefscope.core.provenance import ordered_dataset_hash as _ordered_dataset_hash
from prefscope.encode.sae import SAEProjector
from prefscope.pipeline.lens_rep import get_lens_rep
from prefscope.sae.train import train_sae

logger = logging.getLogger(__name__)

_META_COLS = ["instruction_id", "group_id", "model_a", "model_b", "y_judge", "lang",
              "source", "language"]
_SINGLE_TEXT_COLS = ["prompt", "completion_a"]
_EMBEDDING_MANIFEST = "embedding_manifest.json"
_WHITEN_FNAME = "whiten.npz"


def _remove_tree(path: Path) -> None:
    """Remove one staged/backup path without following a directory symlink."""
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _publication_lock(destination: Path):
    """Load the shared advisory publication lock lazily.

    The lazy import avoids pulling the public API package into module import on
    embedding/training workers, while keeping one lock implementation.
    """
    from prefscope.api._lens_publication import _publication_lock as shared_lock

    return shared_lock(destination)


def _recover_orphan_backup(destination: Path) -> None:
    """Delegate orphan recovery to the shared publication implementation."""
    from prefscope.api._lens_publication import (
        _recover_orphan_backup as shared_recover,
    )

    shared_recover(destination)


def _transactional_build(out_dir, builder):
    """Build in a clean sibling directory, then atomically replace ``out_dir``.

    The staging directory is created under the destination's parent, so both rename
    operations stay on one filesystem. An existing lens is first renamed to a private
    backup and restored if the staging-to-destination rename fails. Build failures never
    modify the prior destination.
    """
    destination = Path(out_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _publication_lock(destination):
        _recover_orphan_backup(destination)
        if destination.exists() and not destination.is_dir():
            raise FileExistsError(
                f"lens destination exists and is not a directory: {destination}")

        staging = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
        staging.mkdir()
        backup = destination.parent / f".{destination.name}.bak-{uuid.uuid4().hex}"
        try:
            result = builder(staging)
            had_destination = destination.exists()
            if had_destination:
                os.replace(destination, backup)
            try:
                os.replace(staging, destination)
            except BaseException:
                if had_destination and backup.exists():
                    os.replace(backup, destination)
                raise
            if backup.exists():
                _remove_tree(backup)
            return result
        finally:
            if staging.exists() or staging.is_symlink():
                _remove_tree(staging)


def _completion_metadata(battles: pd.DataFrame, *, single: bool) -> pd.DataFrame:
    cols = [
        column for column in _META_COLS + (_SINGLE_TEXT_COLS if single else [])
        if column in battles.columns
    ]
    return battles[cols].reset_index(drop=True)


def _prompt_metadata(battles: pd.DataFrame) -> pd.DataFrame:
    cols = [
        column for column in (
            "battle_id", "instruction_id", "group_id", "model_a", "model_b",
            "source", "language", "human_pref")
        if column in battles.columns
    ]
    return battles[cols].reset_index(drop=True)


def _validated_manifest(out_dir: Path, manifest_data: dict, projector,
                        expected_files: set[str]) -> dict:
    """Validate the complete staged directory, write its manifest, and re-read it."""
    from prefscope.core.manifest import LensManifest

    typed = LensManifest.from_dict(manifest_data, strict=True)
    typed.validate_projector(projector).validate_arrays(out_dir)
    if not typed.dataset_hash or len(typed.dataset_hash) != 64:
        raise ValueError("lens manifest dataset_hash must be a SHA-256 hex digest")
    try:
        int(typed.dataset_hash, 16)
    except ValueError:
        raise ValueError("lens manifest dataset_hash must be a SHA-256 hex digest") from None

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(typed.to_dict(), indent=2))
    expected = set(expected_files) | {"manifest.json"}
    actual = {path.name for path in out_dir.iterdir()}
    if actual != expected:
        raise ValueError(
            "staged lens contains missing or undeclared artifacts: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")

    # Validate the exact serialized contract that consumers will load after the swap.
    reloaded = LensManifest.from_dict(json.loads(manifest_path.read_text()), strict=True)
    reloaded.validate_projector(projector).validate_arrays(out_dir)
    return reloaded.to_dict()


def _validated_matrix(values, name: str, *, n_rows: int | None = None,
                      n_cols: int | None = None) -> np.ndarray:
    """Return a finite float32 matrix with the declared row/feature alignment."""
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-D matrix; got shape {array.shape}")
    if array.shape[1] == 0:
        raise ValueError(f"{name} must have at least one feature column")
    if n_rows is not None and array.shape[0] != n_rows:
        raise ValueError(
            f"{name} has {array.shape[0]} rows but metadata has {n_rows}; "
            "embeddings, projections, and metadata must align 1:1")
    if n_cols is not None and array.shape[1] != n_cols:
        raise ValueError(
            f"{name} has feature dim {array.shape[1]} but expected {n_cols}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _validated_metadata(battles, n_rows: int, *, name: str = "metadata") -> pd.DataFrame:
    if not isinstance(battles, pd.DataFrame):
        raise ValueError(f"{name} must be a pandas DataFrame")
    if len(battles) != n_rows:
        raise ValueError(
            f"{name} has {len(battles)} rows but embeddings have {n_rows}; "
            "embeddings and metadata must align 1:1")
    return battles


def _remove_stale_whitener(out_dir: Path) -> None:
    """Ensure an unwhitened build cannot reuse preprocessing from an older lens."""
    path = out_dir / _WHITEN_FNAME
    if path.exists():
        path.unlink()


def _embedding_provenance(embedder, *, prompt: bool = False) -> dict:
    """Extract the numerical/text preprocessing contract from an embedder."""
    if hasattr(embedder, "provenance"):
        return dict(embedder.provenance(prompt=prompt))
    dtype = getattr(embedder, "dtype", None)
    return {
        "embed_model_id": getattr(embedder, "model_id", None),
        "embed_model_revision": getattr(embedder, "model_revision", None),
        "max_tokens": getattr(embedder, "max_tokens", None),
        "embed_instruction": getattr(
            embedder,
            "prompt_embed_instruction" if prompt else "embed_instruction",
            None,
        ),
        "pooling": getattr(embedder, "pooling", None),
        "normalization": getattr(embedder, "normalization", None),
        "dtype": str(dtype).removeprefix("torch.") if dtype is not None else None,
        "backend": getattr(embedder, "backend", None),
    }


def _load_embedding_provenance(emb_dir, embed_model_id=None) -> dict:
    path = Path(emb_dir) / _EMBEDDING_MANIFEST
    provenance = json.loads(path.read_text()) if path.is_file() else {}
    recorded = provenance.get("embed_model_id")
    if recorded and embed_model_id and recorded != embed_model_id:
        raise ValueError(
            f"embedding dump was produced by {recorded!r}, not requested "
            f"embed_model_id={embed_model_id!r}")
    if embed_model_id and not recorded:
        provenance["embed_model_id"] = embed_model_id
    if not path.is_file():
        logger.warning(
            "embedding dump %s has no %s; preprocessing provenance is unknown",
            emb_dir, _EMBEDDING_MANIFEST)
    return provenance


def _cap_train_indices(train_mask: np.ndarray, max_train_rows: int | None,
                       seed: int) -> np.ndarray:
    """Reservoir cap over the train rows.

    Returns integer row indices selecting the train split. When ``max_train_rows``
    is set and the split is larger, a seeded subsample of ``max_train_rows`` rows
    is kept (sorted, so memmap reads stay sequential); otherwise every train row
    is returned. A small dictionary rarely needs the full corpus, so capping the
    train rows is what bounds how much of the memmap materializes.
    """
    idx = np.flatnonzero(train_mask)
    if max_train_rows is not None and idx.size > max_train_rows:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(idx, size=max_train_rows, replace=False))
    return idx


def _val_mask(instruction_ids, val_frac: float) -> np.ndarray:
    """Deterministic per-battle val assignment by hashing instruction_id."""
    thresh = int(val_frac * 1000)
    out = np.zeros(len(instruction_ids), dtype=bool)
    for i, iid in enumerate(instruction_ids):
        bucket = int(hashlib.sha1(str(iid).encode()).hexdigest(), 16) % 1000
        out[i] = bucket < thresh
    return out


def _dump_embeddings(emb_dir, e_a, e_b, battles, *, provenance=None) -> None:
    """Save assembled embeddings + aligned meta so SAEs can be retrained without
    re-reading the per-completion cache (read cache once, sweep M/K cheaply)."""
    emb_dir = Path(emb_dir)
    emb_dir.mkdir(parents=True, exist_ok=True)
    e_a = _validated_matrix(e_a, "e_a")
    _validated_metadata(battles, e_a.shape[0])
    np.save(emb_dir / "e_a.npy", e_a)
    if e_b is not None:
        e_b = _validated_matrix(
            e_b, "e_b", n_rows=e_a.shape[0], n_cols=e_a.shape[1])
        np.save(emb_dir / "e_b.npy", e_b)
    elif (emb_dir / "e_b.npy").exists():
        (emb_dir / "e_b.npy").unlink()
    cols = [c for c in _META_COLS + (_SINGLE_TEXT_COLS if e_b is None else [])
            if c in battles.columns]
    battles[cols].reset_index(drop=True).to_parquet(emb_dir / "meta.parquet")
    if provenance:
        (emb_dir / _EMBEDDING_MANIFEST).write_text(
            json.dumps(provenance, indent=2))


def build_lens_from_embeddings(emb_dir, out_dir, *,
                               m_total: int = 128, k: int = 16,
                               matryoshka_prefix=(), input_rep: str = "difference",
                               val_frac: float = 0.1, device: str = "cuda",
                               embed_model_id: str | None = None,
                               max_train_rows: int | None = None,
                               **train_kwargs) -> dict:
    """Train + save an SAE lens from a previously dumped embedding set.

    Reads ``e_a.npy``/optional ``e_b.npy``/``meta.parquet`` once (no corpus, no
    cache scan, no embedding), then trains for the given M/K. A missing ``e_b``
    is valid only for ``input_rep="individual"``.

    The embedding dumps are memory-mapped (``mmap_mode="r"``) so the full arrays
    never become RAM-resident; only the rows selected by the train/val masks (and
    the projection chunks) materialize.
    """
    emb_dir = Path(emb_dir)
    e_a = np.load(emb_dir / "e_a.npy", mmap_mode="r")
    e_b_path = emb_dir / "e_b.npy"
    e_b = np.load(e_b_path, mmap_mode="r") if e_b_path.exists() else None
    battles = pd.read_parquet(emb_dir / "meta.parquet")
    provenance = _load_embedding_provenance(emb_dir, embed_model_id)
    return _train_and_save(
        e_a, e_b, battles, out_dir, m_total=m_total, k=k,
        matryoshka_prefix=matryoshka_prefix, input_rep=input_rep,
        val_frac=val_frac, device=device, embedding_provenance=provenance,
        max_train_rows=max_train_rows, **train_kwargs)


def build_prompt_lens(emb_dir, out_dir, *,
                      m_total: int = 64, k: int = 8, matryoshka_prefix=(),
                      val_frac: float = 0.1, device: str = "cuda",
                      embed_model_id: str | None = None,
                      max_train_rows: int | None = None,
                      **train_kwargs) -> dict:
    """Transactionally train a prompt lens and replace ``out_dir`` as one unit."""
    return _transactional_build(
        out_dir,
        lambda staging: _build_prompt_lens_in_dir(
            emb_dir, staging, m_total=m_total, k=k,
            matryoshka_prefix=matryoshka_prefix, val_frac=val_frac,
            device=device, embed_model_id=embed_model_id,
            max_train_rows=max_train_rows, **train_kwargs),
    )


def _build_prompt_lens_in_dir(emb_dir, out_dir, *,
                              m_total: int, k: int, matryoshka_prefix,
                              val_frac: float, device: str,
                              embed_model_id: str | None,
                              max_train_rows: int | None,
                              **train_kwargs) -> dict:
    """Build a complete prompt lens inside an already-clean staging directory."""
    emb_dir = Path(emb_dir)
    # memmap so the full prompt matrix never becomes RAM-resident; the mask /
    # projection chunks materialize the rows they need. (.astype here would copy
    # the whole array, defeating the memmap — np.asarray on a float32 memmap is a
    # no-op, so let _train rows / the projector cast lazily instead.)
    e = np.load(emb_dir / "e_prompt.npy", mmap_mode="r")
    battles = pd.read_parquet(emb_dir / "meta.parquet").reset_index(drop=True)
    e = _validated_matrix(e, "e_prompt")
    _validated_metadata(battles, e.shape[0], name="prompt metadata")
    provenance = _load_embedding_provenance(emb_dir, embed_model_id)
    id_col = next(
        (name for name in ("group_id", "instruction_id", "battle_id")
         if name in battles.columns), None)
    if id_col is None:
        raise ValueError("prompt meta needs a group_id, instruction_id, or battle_id")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    val = _val_mask(battles[id_col].tolist(), val_frac)
    train = ~val
    if train.sum() == 0 or val.sum() == 0:
        raise ValueError(f"need both train and val: {int(train.sum())}/{int(val.sum())}")

    seed = int(train_kwargs.get("seed", 0))
    train_idx = _cap_train_indices(train, max_train_rows, seed)
    n_train_rows_used = int(train_idx.size)

    model, config, log = train_sae(
        np.asarray(e[train_idx], dtype=np.float32),
        np.asarray(e[val], dtype=np.float32), m_total=m_total, k=k,
        matryoshka_prefix=matryoshka_prefix, device=device, input_rep="prompt",
        max_train_rows=max_train_rows, **train_kwargs)
    ckpt = out_dir / "sae_model.pt"
    torch.save({"state_dict": model.state_dict(), "config": config}, ckpt)
    pd.DataFrame(log).to_csv(out_dir / "sae_training_log.csv", index=False)

    configured_dim = int(config["input_dim"])
    if configured_dim != e.shape[1]:
        raise ValueError(
            f"trained SAE input_dim {configured_dim} does not match e_prompt feature "
            f"dim {e.shape[1]}")
    # Prompt lenses never use whitening. The staging directory is clean, but keep this
    # guard local so a private direct call cannot accidentally load a stale transform.
    _remove_stale_whitener(out_dir)
    proj = SAEProjector(ckpt, device=device)
    z = _validated_matrix(
        proj.project(e), "z_prompt", n_rows=len(battles), n_cols=m_total)
    np.save(out_dir / "z_prompt.npy", z)
    metadata = _prompt_metadata(battles)
    metadata.to_parquet(out_dir / "battles.parquet")
    dataset_hash = _ordered_dataset_hash(metadata, {"e_prompt": e})

    best_val = config["best_val_norm_mse"]
    from prefscope.core.manifest import SCHEMA_VERSION

    manifest_data = {
        "schema_version": SCHEMA_VERSION,
        "n_prompts": int(len(battles)),
        "n_train": int(train.sum()), "n_val": int(val.sum()),
        "n_train_rows_used": n_train_rows_used,
        "m_total": int(m_total), "k": int(k),
        "sae_type": config["sae_type"],
        "activation_polarity": config["activation_polarity"],
        "code_semantics": config["code_semantics"],
        "selection_rule": config["selection_rule"],
        "input_dim": int(config["input_dim"]),
        **provenance,
        "dataset_hash": dataset_hash,
        "best_val_norm_mse": float(best_val) if np.isfinite(best_val) else None,
        "best_val_select_norm_mse": config.get("best_val_select_norm_mse"),
        "best_val_explained_variance": config.get("best_val_explained_variance"),
        "deployment_val_norm_mse": config.get("deployment_val_norm_mse"),
        "deployment_val_explained_variance": config.get(
            "deployment_val_explained_variance"),
        "deployment_val_active": config.get("deployment_val_active"),
        "deployment_dead_neurons": config.get("deployment_dead_neurons"),
        "deployment_rare_neurons": config.get("deployment_rare_neurons"),
        "target_l0": config.get("target_l0"),
        "calibration_l0": config.get("calibration_l0"),
        "threshold_calibration_rows": config.get("threshold_calibration_rows"),
        "optimizer": config.get("optimizer"),
        "weight_decay": config.get("weight_decay"),
        "seed": config.get("seed"),
        "matryoshka_prefix_lengths": config["matryoshka_prefix_lengths"],
        "n_epochs_trained": len(log),
        "input_rep": "prompt",
        "output_arrays": ["z_prompt"],
    }
    return _validated_manifest(
        out_dir, manifest_data, proj,
        {"sae_model.pt", "sae_training_log.csv", "z_prompt.npy", "battles.parquet"},
    )


def build_lens(battles: pd.DataFrame, embedder, out_dir, *,
               m_total: int = 128, k: int = 16, matryoshka_prefix=(),
               input_rep: str = "difference",
               val_frac: float = 0.1, device: str = "cuda",
               embed_model_id: str | None = None,
               max_train_rows: int | None = None,
               dump_embeddings=None, **train_kwargs) -> dict:
    # fail fast BEFORE the costly embed: reject an unknown input_rep, and reject a
    # non-contrastive rep (e.g. prompt — those go through build_prompt_lens).
    rep = get_lens_rep(input_rep)
    if not rep.contrastive:
        raise ValueError(
            f"build_lens needs a contrastive lens (difference/individual); "
            f"{input_rep!r} is not — use build-prompt-lens for prompt lenses")

    required = ["prompt", "completion_a", "instruction_id"]
    missing = [c for c in required if c not in battles.columns]
    if missing:
        raise ValueError(f"battles missing required columns: {missing}")
    if "group_id" not in battles.columns:
        battles = battles.copy()
        battles["group_id"] = [
            hashlib.sha1(str(value).encode()).hexdigest()[:16]
            for value in battles["prompt"]
        ]

    has_b_col = "completion_b" in battles.columns
    has_b = (battles["completion_b"].notna() if has_b_col
             else pd.Series(False, index=battles.index))
    if bool(has_b.any()) and not bool(has_b.all()):
        raise ValueError(
            "mixed paired/single rows are not supported: completion_b must be present "
            "for every row or absent for every row")
    paired = bool(has_b.all())
    if not paired and not rep.per_side:
        raise ValueError(
            f"input_rep={input_rep!r} requires paired data with completion_b; "
            "use input_rep='individual' for single-response data")

    prompts = battles["prompt"].tolist()
    logger.info("embedding completion A…")
    e_a = embedder.encode(prompts, battles["completion_a"].tolist())
    if paired:
        logger.info("embedding completion B…")
        e_b = embedder.encode(prompts, battles["completion_b"].tolist())
    else:
        e_b = None
    e_a = _validated_matrix(e_a, "e_a", n_rows=len(battles))
    if e_b is not None:
        e_b = _validated_matrix(
            e_b, "e_b", n_rows=len(battles), n_cols=e_a.shape[1])

    # Free the embedder's GPU memory before training the SAE — otherwise the
    # embedding phase's retained/fragmented allocations can OOM the SAE step.
    if hasattr(embedder, "unload"):
        embedder.unload()

    if dump_embeddings:
        _dump_embeddings(
            dump_embeddings, e_a, e_b, battles,
            provenance=_embedding_provenance(embedder))

    provenance = _embedding_provenance(embedder)
    actual_model = provenance.get("embed_model_id")
    if embed_model_id and actual_model not in (None, embed_model_id):
        raise ValueError(
            f"embedder model {actual_model!r} does not match "
            f"embed_model_id={embed_model_id!r}")
    if embed_model_id and not actual_model:
        provenance["embed_model_id"] = embed_model_id
    return _train_and_save(
        e_a, e_b, battles, out_dir, m_total=m_total, k=k,
        matryoshka_prefix=matryoshka_prefix, input_rep=input_rep,
        val_frac=val_frac, device=device, embedding_provenance=provenance,
        max_train_rows=max_train_rows, **train_kwargs)


def _train_and_save(e_a, e_b, battles, out_dir, *,
                    m_total, k, matryoshka_prefix, input_rep,
                    val_frac, device, embedding_provenance=None,
                    embed_model_id=None, whiten="none",
                    whiten_eps=1e-5, max_train_rows: int | None = None,
                    **train_kwargs) -> dict:
    """Transactionally train a completion lens and replace ``out_dir`` as one unit."""
    return _transactional_build(
        out_dir,
        lambda staging: _train_and_save_in_dir(
            e_a, e_b, battles, staging, m_total=m_total, k=k,
            matryoshka_prefix=matryoshka_prefix, input_rep=input_rep,
            val_frac=val_frac, device=device,
            embedding_provenance=embedding_provenance,
            embed_model_id=embed_model_id, whiten=whiten,
            whiten_eps=whiten_eps, max_train_rows=max_train_rows,
            **train_kwargs),
    )


def _train_and_save_in_dir(e_a, e_b, battles, out_dir, *,
                           m_total, k, matryoshka_prefix, input_rep,
                           val_frac, device, embedding_provenance=None,
                           embed_model_id=None, whiten="none",
                           whiten_eps=1e-5,
                           max_train_rows: int | None = None,
                           **train_kwargs) -> dict:
    """Build a complete completion lens inside an already-clean staging directory."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = get_lens_rep(input_rep)   # the single home for the input_rep branch logic
    e_a = _validated_matrix(e_a, "e_a")
    _validated_metadata(battles, e_a.shape[0])
    single = e_b is None
    if single and not rep.per_side:
        raise ValueError(
            f"input_rep={input_rep!r} requires paired embeddings (e_b.npy is missing); "
            "single-response training requires input_rep='individual'")
    if e_b is not None:
        e_b = _validated_matrix(
            e_b, "e_b", n_rows=e_a.shape[0], n_cols=e_a.shape[1])

    if "group_id" in battles.columns:
        split_ids = battles["group_id"].tolist()
    elif "prompt" in battles.columns:
        split_ids = [hashlib.sha1(str(value).encode()).hexdigest()[:16]
                     for value in battles["prompt"]]
    elif "instruction_id" in battles.columns:
        split_ids = battles["instruction_id"].tolist()
    else:
        raise ValueError(
            "metadata needs a group_id, prompt, or instruction_id for train/val splitting")
    val = _val_mask(split_ids, val_frac)
    train = ~val
    if train.sum() == 0 or val.sum() == 0:
        raise ValueError(
            f"need both train and val battles: got {int(train.sum())} train / "
            f"{int(val.sum())} val from {len(battles)} battles at "
            f"val_frac={val_frac}")

    # Reservoir cap on the train *battle* indices: with memmap'd e_a/e_b only the
    # selected rows materialize, so a capped run never pulls the full corpus into
    # RAM. Val is left full here and bounded inside train_sae.
    seed = int(train_kwargs.get("seed", 0))
    train_idx = _cap_train_indices(train, max_train_rows, seed)
    n_train_rows_used = int(train_idx.size)

    # training rows per the lens representation (difference: e_a-e_b; individual: pooled
    # [e_a; e_b]). Masks are applied here so the strategy stays a pure (e_a, e_b) -> X fn.
    if single:
        X_train = e_a[train_idx]
        X_val = e_a[val]
    else:
        X_train = rep.training_matrix(e_a[train_idx], e_b[train_idx])
        X_val = rep.training_matrix(e_a[val], e_b[val])
    X_train = _validated_matrix(X_train, "training matrix")
    X_val = _validated_matrix(X_val, "validation matrix", n_cols=X_train.shape[1])

    # Bound val to match the trimmed train set. We cap it HERE (not via train_sae's
    # max_train_rows) because the individual rep doubles the row count
    # (2 x battles), which would make train_sae's >cap test re-subsample an
    # already-capped X_train. So we pass max_train_rows=None to train_sae below.
    if max_train_rows is not None:
        val_cap = max(2000, max_train_rows // 9)
        if X_val.shape[0] > val_cap:
            vrng = np.random.default_rng(seed)
            vkeep = np.sort(vrng.choice(X_val.shape[0], size=val_cap, replace=False))
            X_val = np.ascontiguousarray(X_val[vkeep])

    # optional input whitening (anisotropic embeddings -> de-correlated). Fit on the
    # train rows only, transform both splits, and save the transform BEFORE the
    # projector is built so it re-applies the same whitening at projection time.
    if whiten and whiten != "none":
        from prefscope.sae.whiten import Whitener
        whitener = Whitener.fit(X_train, method=whiten, eps=whiten_eps)
        X_train = _validated_matrix(
            whitener.transform(X_train), "whitened training matrix",
            n_rows=X_train.shape[0], n_cols=X_train.shape[1])
        X_val = _validated_matrix(
            whitener.transform(X_val), "whitened validation matrix",
            n_rows=X_val.shape[0], n_cols=X_val.shape[1])
        whitener.save(out_dir)
    else:
        _remove_stale_whitener(out_dir)

    model, config, log = train_sae(
        X_train, X_val, m_total=m_total, k=k,
        matryoshka_prefix=matryoshka_prefix, device=device,
        input_rep=input_rep, **train_kwargs)

    configured_dim = int(config["input_dim"])
    if configured_dim != X_train.shape[1]:
        raise ValueError(
            f"trained SAE input_dim {configured_dim} does not match embedding feature "
            f"dim {X_train.shape[1]}")
    ckpt_path = out_dir / "sae_model.pt"
    torch.save({"state_dict": model.state_dict(), "config": config}, ckpt_path)
    pd.DataFrame(log).to_csv(out_dir / "sae_training_log.csv", index=False)

    proj = SAEProjector(ckpt_path, device=device)   # auto-loads whiten.npz if present
    # the strategy owns which codes to save (difference: z_diff; individual: z_a/z_b/z_diff)
    arrays = (rep.single_output_arrays(proj, e_a) if single
              else rep.output_arrays(proj, e_a, e_b))
    arrays = {
        name: _validated_matrix(
            array, name, n_rows=len(battles), n_cols=m_total)
        for name, array in arrays.items()
    }
    for stale in {"z_a", "z_b", "z_diff"} - set(arrays):
        stale_path = out_dir / f"{stale}.npy"
        if stale_path.exists():
            stale_path.unlink()
    for name, arr in arrays.items():
        np.save(out_dir / f"{name}.npy", arr)
    output_arrays = list(arrays)

    # A paired lens can re-attach text from its source corpus. A single-response
    # artifact is itself the only general source contract, so retain its text.
    metadata = _completion_metadata(battles, single=single)
    metadata.to_parquet(out_dir / "battles.parquet")
    source_arrays = {"e_a": e_a}
    if e_b is not None:
        source_arrays["e_b"] = e_b
    dataset_hash = _ordered_dataset_hash(metadata, source_arrays)

    # best_val starts at inf; if no epoch ever improved it stays non-finite,
    # which json writes as `Infinity` (invalid JSON for strict downstream readers)
    best_val = config["best_val_norm_mse"]
    best_val = float(best_val) if np.isfinite(best_val) else None

    from prefscope.core.manifest import SCHEMA_VERSION

    provenance = dict(embedding_provenance or {})
    if embed_model_id and not provenance.get("embed_model_id"):
        provenance["embed_model_id"] = embed_model_id
    manifest_data = {
        "schema_version": SCHEMA_VERSION,
        "n_battles": int(len(battles)),
        "n_items": int(len(battles)),
        "dataset_mode": "single" if single else "paired",
        "n_train_battles": int(train.sum()),
        "n_train_rows_used": n_train_rows_used,
        "n_val_battles": int(val.sum()),
        "m_total": int(m_total),
        "k": int(k),
        "sae_type": config.get("sae_type", "batchtopk"),
        "activation_polarity": config.get("activation_polarity"),
        "code_semantics": config.get("code_semantics"),
        "selection_rule": config.get("selection_rule"),
        "input_dim": int(config["input_dim"]),
        **provenance,
        "dataset_hash": dataset_hash,
        "best_val_norm_mse": best_val,
        "best_val_select_norm_mse": config.get("best_val_select_norm_mse"),
        "best_val_explained_variance": config.get("best_val_explained_variance"),
        "deployment_val_norm_mse": config.get("deployment_val_norm_mse"),
        "deployment_val_explained_variance": config.get(
            "deployment_val_explained_variance"),
        "deployment_val_active": config.get("deployment_val_active"),
        "deployment_dead_neurons": config.get("deployment_dead_neurons"),
        "deployment_rare_neurons": config.get("deployment_rare_neurons"),
        "target_l0": config.get("target_l0"),
        "calibration_l0": config.get("calibration_l0"),
        "threshold_calibration_rows": config.get("threshold_calibration_rows"),
        "optimizer": config.get("optimizer"),
        "weight_decay": config.get("weight_decay"),
        "seed": config.get("seed"),
        "matryoshka_prefix_lengths": config["matryoshka_prefix_lengths"],
        "n_epochs_trained": len(log),
        "input_rep": input_rep,
        "whiten": whiten,
        "output_arrays": output_arrays,
    }
    expected_files = {
        "sae_model.pt", "sae_training_log.csv", "battles.parquet",
        *(f"{name}.npy" for name in output_arrays),
    }
    if whiten and whiten != "none":
        expected_files.add(_WHITEN_FNAME)
    return _validated_manifest(out_dir, manifest_data, proj, expected_files)
