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
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from prefscope.encode.sae import SAEProjector
from prefscope.pipeline.lens_rep import get_lens_rep
from prefscope.sae.train import train_sae

logger = logging.getLogger(__name__)

_META_COLS = ["instruction_id", "model_a", "model_b", "y_judge", "lang",
              "source", "language"]
_SINGLE_TEXT_COLS = ["prompt", "completion_a"]


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


def _dump_embeddings(emb_dir, e_a, e_b, battles) -> None:
    """Save assembled embeddings + aligned meta so SAEs can be retrained without
    re-reading the per-completion cache (read cache once, sweep M/K cheaply)."""
    emb_dir = Path(emb_dir)
    emb_dir.mkdir(parents=True, exist_ok=True)
    np.save(emb_dir / "e_a.npy", np.asarray(e_a, dtype=np.float32))
    if e_b is not None:
        np.save(emb_dir / "e_b.npy", np.asarray(e_b, dtype=np.float32))
    elif (emb_dir / "e_b.npy").exists():
        (emb_dir / "e_b.npy").unlink()
    cols = [c for c in _META_COLS + (_SINGLE_TEXT_COLS if e_b is None else [])
            if c in battles.columns]
    battles[cols].reset_index(drop=True).to_parquet(emb_dir / "meta.parquet")


def build_lens_from_embeddings(emb_dir, out_dir, *,
                               m_total: int = 128, k: int = 16,
                               matryoshka_prefix=(8,), input_rep: str = "difference",
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
    return _train_and_save(
        e_a, e_b, battles, out_dir, m_total=m_total, k=k,
        matryoshka_prefix=matryoshka_prefix, input_rep=input_rep,
        val_frac=val_frac, device=device, embed_model_id=embed_model_id,
        max_train_rows=max_train_rows, **train_kwargs)


def build_prompt_lens(emb_dir, out_dir, *,
                      m_total: int = 64, k: int = 8, matryoshka_prefix=(8,),
                      val_frac: float = 0.1, device: str = "cuda",
                      embed_model_id: str | None = None,
                      max_train_rows: int | None = None,
                      **train_kwargs) -> dict:
    """Train a standard SAE on prompt-only embeddings (the paper's prompt matrix).

    Reads ``e_prompt.npy`` + ``meta.parquet`` from an ``embed-prompts`` dump and
    trains a (non-difference) SAE on the single prompt vectors. Saves ``z_prompt``
    so the prompt features = what the request asks for (task / intent / topic).
    """
    emb_dir = Path(emb_dir)
    # memmap so the full prompt matrix never becomes RAM-resident; the mask /
    # projection chunks materialize the rows they need. (.astype here would copy
    # the whole array, defeating the memmap — np.asarray on a float32 memmap is a
    # no-op, so let _train rows / the projector cast lazily instead.)
    e = np.load(emb_dir / "e_prompt.npy", mmap_mode="r")
    battles = pd.read_parquet(emb_dir / "meta.parquet").reset_index(drop=True)
    id_col = "instruction_id" if "instruction_id" in battles.columns else "battle_id"
    if id_col not in battles.columns:
        raise ValueError("prompt meta needs an instruction_id or battle_id column")

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
        matryoshka_prefix=matryoshka_prefix, device=device,
        max_train_rows=max_train_rows, **train_kwargs)
    ckpt = out_dir / "sae_model.pt"
    torch.save({"state_dict": model.state_dict(), "config": config}, ckpt)
    pd.DataFrame(log).to_csv(out_dir / "sae_training_log.csv", index=False)

    proj = SAEProjector(ckpt, device=device)
    z = proj.project(e)
    np.save(out_dir / "z_prompt.npy", z)
    cols = [c for c in ("battle_id", "instruction_id", "model_a", "model_b",
                        "source", "language", "human_pref") if c in battles.columns]
    battles[cols].to_parquet(out_dir / "battles.parquet")

    best_val = config["best_val_norm_mse"]
    from prefscope.core.manifest import LensManifest
    manifest = LensManifest.from_dict({
        "n_prompts": int(len(battles)),
        "n_train": int(train.sum()), "n_val": int(val.sum()),
        "n_train_rows_used": n_train_rows_used,
        "m_total": int(m_total), "k": int(k),
        "input_dim": int(config["input_dim"]),
        "embed_model_id": embed_model_id,
        "best_val_norm_mse": float(best_val) if np.isfinite(best_val) else None,
        "best_val_select_norm_mse": config.get("best_val_select_norm_mse"),
        "matryoshka_prefix_lengths": config["matryoshka_prefix_lengths"],
        "n_epochs_trained": len(log),
        "input_rep": "prompt",
        "output_arrays": ["z_prompt"],
    }).validate_arrays(out_dir).to_dict()
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def build_lens(battles: pd.DataFrame, embedder, out_dir, *,
               m_total: int = 128, k: int = 16, matryoshka_prefix=(8,),
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
    e_a = np.asarray(e_a, dtype=np.float32)
    if e_b is not None:
        e_b = np.asarray(e_b, dtype=np.float32)

    # Free the embedder's GPU memory before training the SAE — otherwise the
    # embedding phase's retained/fragmented allocations can OOM the SAE step.
    if hasattr(embedder, "unload"):
        embedder.unload()

    if dump_embeddings:
        _dump_embeddings(dump_embeddings, e_a, e_b, battles)

    return _train_and_save(
        e_a, e_b, battles, out_dir, m_total=m_total, k=k,
        matryoshka_prefix=matryoshka_prefix, input_rep=input_rep,
        val_frac=val_frac, device=device, embed_model_id=embed_model_id,
        max_train_rows=max_train_rows, **train_kwargs)


def _train_and_save(e_a, e_b, battles, out_dir, *,
                    m_total, k, matryoshka_prefix, input_rep,
                    val_frac, device, embed_model_id, whiten="none",
                    whiten_eps=1e-5, max_train_rows: int | None = None,
                    **train_kwargs) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = get_lens_rep(input_rep)   # the single home for the input_rep branch logic
    e_a = np.asarray(e_a, dtype=np.float32)
    single = e_b is None
    if single and not rep.per_side:
        raise ValueError(
            f"input_rep={input_rep!r} requires paired embeddings (e_b.npy is missing); "
            "single-response training requires input_rep='individual'")
    if e_b is not None:
        e_b = np.asarray(e_b, dtype=np.float32)

    val = _val_mask(battles["instruction_id"].tolist(), val_frac)
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
        X_train = whitener.transform(X_train)
        X_val = whitener.transform(X_val)
        whitener.save(out_dir)

    model, config, log = train_sae(
        X_train, X_val, m_total=m_total, k=k,
        matryoshka_prefix=matryoshka_prefix, device=device, **train_kwargs)

    ckpt_path = out_dir / "sae_model.pt"
    torch.save({"state_dict": model.state_dict(), "config": config}, ckpt_path)
    pd.DataFrame(log).to_csv(out_dir / "sae_training_log.csv", index=False)

    proj = SAEProjector(ckpt_path, device=device)   # auto-loads whiten.npz if present
    # the strategy owns which codes to save (difference: z_diff; individual: z_a/z_b/z_diff)
    arrays = (rep.single_output_arrays(proj, e_a) if single
              else rep.output_arrays(proj, e_a, e_b))
    for stale in {"z_a", "z_b", "z_diff"} - set(arrays):
        stale_path = out_dir / f"{stale}.npy"
        if stale_path.exists():
            stale_path.unlink()
    for name, arr in arrays.items():
        np.save(out_dir / f"{name}.npy", arr)
    output_arrays = list(arrays)

    # A paired lens can re-attach text from its source corpus. A single-response
    # artifact is itself the only general source contract, so retain its text.
    cols = [c for c in _META_COLS + (_SINGLE_TEXT_COLS if single else [])
            if c in battles.columns]
    battles[cols].reset_index(drop=True).to_parquet(out_dir / "battles.parquet")

    # best_val starts at inf; if no epoch ever improved it stays non-finite,
    # which json writes as `Infinity` (invalid JSON for strict downstream readers)
    best_val = config["best_val_norm_mse"]
    best_val = float(best_val) if np.isfinite(best_val) else None

    from prefscope.core.manifest import LensManifest
    manifest = LensManifest.from_dict({
        "n_battles": int(len(battles)),
        "n_items": int(len(battles)),
        "dataset_mode": "single" if single else "paired",
        "n_train_battles": int(train.sum()),
        "n_train_rows_used": n_train_rows_used,
        "n_val_battles": int(val.sum()),
        "m_total": int(m_total),
        "k": int(k),
        "sae_type": config.get("sae_type", "batchtopk"),
        "input_dim": int(config["input_dim"]),
        "embed_model_id": embed_model_id,
        "best_val_norm_mse": best_val,
        "best_val_select_norm_mse": config.get("best_val_select_norm_mse"),
        "matryoshka_prefix_lengths": config["matryoshka_prefix_lengths"],
        "n_epochs_trained": len(log),
        "input_rep": input_rep,
        "whiten": whiten,
        "output_arrays": output_arrays,
    }).validate_arrays(out_dir).to_dict()
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
