"""Train a BatchTopK SAE on pooled completion embeddings (pure compute)."""
from __future__ import annotations

import logging
import math
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from prefscope.core import registry
from prefscope.sae.model import (  # noqa: F401  (registers built-ins on import)
    BatchTopKSAE, JumpReLUSAE, NonnegativeBatchTopKSAE, SimpleTopKSAE,
    resolve_sae_type, sae_semantics)

logger = logging.getLogger(__name__)


def _threshold_stats(model) -> tuple[float, float, float]:
    """Min/median/max deployment thresholds for scalar and per-feature gates."""
    if isinstance(model, JumpReLUSAE):
        values = model._thresholds().detach().float()
    else:
        values = model.threshold.detach().float().reshape(1)
    return tuple(float(x) for x in (
        values.min().item(), values.median().item(), values.max().item()))


@torch.no_grad()
def _calibrate_batchtopk_threshold(
    model,
    X: torch.Tensor,
    *,
    batch: int,
    device: torch.device,
    target_l0: int,
    seed: int,
    max_score_elements: int,
) -> tuple[int, float]:
    """Fit BatchTopK's frozen scalar threshold to an average deployment L0.

    Calibration is bounded by ``max_score_elements`` so a wide lens cannot
    materialize an unbounded N x M score matrix.  The validation rows are sampled
    deterministically and are only used to set this deployment parameter; the
    training checkpoint has already been selected.
    """
    if isinstance(model, (SimpleTopKSAE, JumpReLUSAE)):
        return min(len(X), max(0, max_score_elements // max(1, model.m_total))), float("nan")

    max_rows = max(1, max_score_elements // max(1, model.m_total))
    n_rows = min(len(X), max_rows)
    if n_rows < len(X):
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(X), size=n_rows, replace=False))
        Xcal = X[torch.from_numpy(keep)]
    else:
        Xcal = X

    score_parts: list[torch.Tensor] = []
    for start in range(0, n_rows, batch):
        xb = Xcal[start:start + batch].to(device, non_blocking=True)
        pre = model._selection_pre(model.encode_pre(xb))
        score_parts.append(model._threshold_scores(pre).flatten().cpu())
    scores = torch.cat(score_parts)
    wanted = min(max(1, int(target_l0) * n_rows), scores.numel())
    # Put the threshold in the largest available gap around the Kth boundary.
    # A threshold equal to one observed preactivation is numerically brittle:
    # GEMM reduction order can move that value by a few ULPs when projection is
    # chunked differently. The midpoint keeps frozen support batch-invariant.
    n_boundary = min(wanted + 1, scores.numel())
    boundary = torch.topk(scores, n_boundary, sorted=True).values
    kth = boundary[wanted - 1].item()
    next_score = boundary[wanted].item() if wanted < scores.numel() else None
    if kth <= 0:
        threshold = 0.0
    elif next_score is not None and next_score < kth:
        threshold = float((np.float64(kth) + np.float64(next_score)) / 2.0)
    else:
        # Exact ties cannot be split by a scalar threshold; include the boundary.
        threshold = float(np.nextafter(np.float32(kth), np.float32(-np.inf)))
    model.threshold.fill_(threshold)

    active = 0
    for start in range(0, n_rows, batch):
        xb = Xcal[start:start + batch].to(device, non_blocking=True)
        active += int((model.encode(xb) != 0).sum().item())
    return n_rows, active / max(1, n_rows)


@torch.no_grad()
def _deployment_metrics(model, X: torch.Tensor, *, batch: int,
                        device: torch.device) -> tuple[float, float, int, int]:
    """Metrics for the actual frozen inference path after threshold calibration."""
    recon_parts: list[torch.Tensor] = []
    fire_count = torch.zeros(model.m_total, dtype=torch.long)
    active = 0
    for start in range(0, len(X), batch):
        xb = X[start:start + batch].to(device, non_blocking=True)
        recon, info = model(xb)
        acts = info["activations"].cpu()
        recon_parts.append(recon.cpu())
        active += int((acts != 0).sum())
        fire_count += (acts != 0).sum(dim=0)
    recon = torch.cat(recon_parts, dim=0)
    norm_mse = float(model._normalized_mse(recon, X))
    mean_l0 = active / max(1, len(X))
    dead = int((fire_count == 0).sum())
    rare_cutoff = max(1, int(math.ceil(0.001 * len(X))))
    rare = int((fire_count <= rare_cutoff).sum())
    return norm_mse, mean_l0, dead, rare


def train_sae(
    X_train: np.ndarray,
    X_val: np.ndarray,
    *,
    m_total: int = 128,
    k: int = 16,
    matryoshka_prefix: Sequence[int] = (),
    aux_k: int | None = None,
    dead_threshold_steps: int = 256,
    lr: float = 5e-4,
    batch: int = 512,
    n_epochs: int = 200,
    min_epochs: int = 10,
    patience: int = 5,
    aux_coef: float = 1.0 / 32.0,
    clip_grad: float = 1.0,
    seed: int = 0,
    device: str = "cuda",
    sae_type: str = "auto",
    input_rep: str | None = None,
    sparsity_coef: float = 1e-3,
    bandwidth: float = 1e-3,
    sparsity_warmup_steps: int = 0,
    threshold_calibration_elements: int = 10_000_000,
    log_every_batches: int = 0,
    max_train_rows: int | None = None,
):
    """Return (model, config_dict, log_rows) — best-val weights restored.

    config_dict matches the 'config' block written by the legacy pipeline so
    SAEProjector can read it. No disk I/O happens here.

    Note: a 'simple-topk' checkpoint keeps threshold=0.0, but its frozen
    inference path selects the top-k features per example (``_threshold_select`` →
    per-example top-k), so as a lens it activates exactly k — deployable, though
    batch-topk remains the default for lenses.

    Memory: the training matrix stays resident on CPU; only the per-batch slice
    (and per-batch validation chunks) are moved to ``device``. This keeps the GPU
    footprint O(batch) rather than O(N), so the trainer scales to large N.

    ``max_train_rows`` is a reservoir cap: when set and ``X_train`` has more rows
    than the cap, a seeded subsample of ``max_train_rows`` rows is taken before
    training (a small dictionary rarely needs the full corpus). ``X_val`` is
    capped to ``min(len(X_val), max(2000, max_train_rows // 9))`` so validation
    cannot dominate the trimmed train set. ``max_train_rows=None`` (default)
    disables all capping — behavior is unchanged.
    """
    sae_type = resolve_sae_type(sae_type, input_rep)
    if input_rep == "difference" and sae_type in ("batchtopk-relu", "jumprelu"):
        logger.warning(
            "%s produces presence-style non-negative codes on direct differences; "
            "signed batchtopk is the recommended difference-lens architecture",
            sae_type)
    if input_rep in ("individual", "prompt") and sae_type in (
            "batchtopk", "simple-topk"):
        logger.warning(
            "%s produces signed axes for %s data; single-text interpretation covers "
            "only one pole unless explicitly acknowledged", sae_type, input_rep)
    try:
        cls = registry.get("sae", sae_type)
    except KeyError:
        opts = ", ".join(registry.available("sae")) or "(none registered)"
        raise ValueError(
            f"Unknown sae_type {sae_type!r}; expected one of: {opts}") from None

    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)

    X_train = np.ascontiguousarray(X_train, dtype=np.float32)
    X_val = np.ascontiguousarray(X_val, dtype=np.float32)

    # Reservoir cap: a small dictionary rarely needs the full corpus. Subsample
    # train (and bound val) with a seeded RNG before anything is made resident.
    if max_train_rows is not None and X_train.shape[0] > max_train_rows:
        rng = np.random.default_rng(seed)
        keep = rng.choice(X_train.shape[0], size=max_train_rows, replace=False)
        keep.sort()
        X_train = np.ascontiguousarray(X_train[keep])
        val_cap = max(2000, max_train_rows // 9)
        if X_val.shape[0] > val_cap:
            vkeep = rng.choice(X_val.shape[0], size=val_cap, replace=False)
            vkeep.sort()
            X_val = np.ascontiguousarray(X_val[vkeep])

    d_in = X_train.shape[1]

    requested_prefix = tuple(matryoshka_prefix or ())
    if sae_type == "jumprelu" and requested_prefix:
        raise ValueError(
            "Matryoshka training is not supported for jumprelu; remove "
            "matryoshka_prefix or select a BatchTopK architecture")
    if sparsity_warmup_steps < 0:
        raise ValueError("sparsity_warmup_steps must be >= 0")
    if threshold_calibration_elements < 1:
        raise ValueError("threshold_calibration_elements must be >= 1")

    # Instantiate from the union of params every SAE flavor might want; each class
    # absorbs what it doesn't use via `**_` (mirrors the clusterer registry).
    model = cls(
        input_dim=d_in, m_total_neurons=m_total, k_active_neurons=k,
        aux_k=aux_k, dead_neuron_threshold_steps=dead_threshold_steps,
        matryoshka_prefix_lengths=list(requested_prefix),
        sparsity_coef=sparsity_coef, bandwidth=bandwidth,
    ).to(dev)
    prefix = model.matryoshka_prefix_lengths
    # SAE reference implementations use Adam without implicit L2 shrinkage.  In
    # particular, default AdamW weight decay fights the unit-norm decoder update.
    opt = Adam(model.parameters(), lr=lr)

    # Keep the full matrices on CPU; only per-batch slices go to ``dev`` below.
    # This bounds GPU memory at O(batch) instead of O(N).
    Xt = torch.from_numpy(X_train)
    Xv = torch.from_numpy(X_val)
    n_train = Xt.shape[0]
    n_val = Xv.shape[0]
    n_batches = (n_train + batch - 1) // batch
    n_val_batches = (n_val + batch - 1) // batch

    log_rows: list[dict] = []
    best_val = float("inf")          # selection objective (Matryoshka-averaged norm MSE)
    best_val_full = float("inf")     # full-reconstruction norm MSE at the selected epoch
    best_state: dict | None = None
    patience_left = patience
    global_step = 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=dev)
        epoch_main = 0.0
        epoch_aux = 0.0
        for bi in range(n_batches):
            global_step += 1
            if isinstance(model, JumpReLUSAE):
                scale = (min(1.0, global_step / sparsity_warmup_steps)
                         if sparsity_warmup_steps else 1.0)
                model.set_sparsity_scale(scale)
            idx = perm[bi * batch:(bi + 1) * batch]
            # ``perm``/``idx`` live on ``dev`` (unchanged RNG → identical order);
            # move them to CPU to index the CPU-resident matrix, then ship the
            # gathered slice to ``dev``.
            x = Xt[idx.cpu()].to(dev, non_blocking=True)
            recon, info = model(x)
            coef = 0.0 if sae_type in ("simple-topk", "jumprelu") else aux_coef
            loss, parts = model.compute_loss(x, recon, info, aux_coef=coef)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            model.adjust_decoder_gradient_()
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            opt.step()
            model.normalize_decoder_()
            epoch_main += parts["main"]
            epoch_aux += parts["aux"]
            if log_every_batches and (bi + 1) % log_every_batches == 0:
                logger.info("    epoch %3d  batch %d/%d  main=%.4f  aux=%.4f",
                            epoch, bi + 1, n_batches, parts["main"], parts["aux"])

        model.eval()
        with torch.no_grad():
            # Validate in batches so the full X_val is never device-resident: run
            # each chunk's forward on ``dev``, gather recon + activations back to
            # CPU, then evaluate the metrics on the concatenated CPU tensors —
            # identical reduction to a single full-batch pass.
            # The baseline SSE for normalized MSE uses the global per-feature mean —
            # the same denominator for every Matryoshka prefix level — so compute once.
            mu = Xv.mean(dim=0, keepdim=True)
            sse_base = float(((Xv - mu) ** 2).sum())
            sse_prefix = {L: 0.0 for L in prefix}   # per prefix level (incl. full == m_total)
            recon_parts: list[torch.Tensor] = []
            act_parts: list[torch.Tensor] = []
            for vi in range(n_val_batches):
                xv = Xv[vi * batch:(vi + 1) * batch].to(dev, non_blocking=True)
                r, vinfo = model(xv)
                recon_parts.append(r.cpu())
                act_parts.append(vinfo["activations"].cpu())
                # Accumulate the Matryoshka selection objective chunk-by-chunk (each
                # prefix level's SSE is exact once summed over chunks), so the full X_val
                # never lands on ``dev`` — same memory bound as the metrics above.
                for L in prefix:
                    if L == m_total:
                        pr = r                       # full recon == last prefix term
                    else:
                        partial = vinfo["activations"].clone()
                        partial[:, L:] = 0
                        pr = model.decoder(partial) + model.input_bias
                    sse_prefix[L] += float(((pr - xv) ** 2).sum())
            v_recon = torch.cat(recon_parts, dim=0)
            v_acts = torch.cat(act_parts, dim=0)
            v_mse = F.mse_loss(v_recon, Xv).item()
            v_norm_mse = float(model._normalized_mse(v_recon, Xv))
            # Normalized MSE already divides by the per-feature mean baseline.
            # Using X.var() here mixed a global scalar-mean variance with that
            # baseline and substantially overstated explained variance.
            ev = 1.0 - v_norm_mse
            # Selection metric = the objective TRAINING actually minimizes: mean
            # normalized-MSE across prefix levels for a Matryoshka lens (identical to
            # v_norm_mse when there are no prefixes). Early-stopping/checkpoint choice
            # uses THIS so it can't pick a lens strong on the full code but weak on the
            # coarse prefixes the Matryoshka terms shape.
            v_select = (sum(sse_prefix[L] / (sse_base + 1e-8) for L in prefix) / len(prefix)
                        if prefix else v_norm_mse)
            v_active = float((v_acts != 0).float().sum(dim=-1).mean())
            fire_count = (v_acts != 0).sum(dim=0)
            v_dead = int((fire_count == 0).sum().item())
            rare_cutoff = max(1, int(math.ceil(0.001 * n_val)))
            v_rare = int((fire_count <= rare_cutoff).sum().item())
            thr_min, thr_median, thr_max = _threshold_stats(model)

        log_rows.append({
            "epoch": epoch,
            "train_main_mean": epoch_main / max(1, n_batches),
            "train_aux_mean": epoch_aux / max(1, n_batches),
            "val_mse": v_mse,
            "val_norm_mse": v_norm_mse,
            "val_select_norm_mse": v_select,
            "val_ev": ev,
            "val_active": v_active,
            "dead_neurons": v_dead,
            "rare_neurons": v_rare,
            "threshold": thr_median,
            "threshold_min": thr_min,
            "threshold_median": thr_median,
            "threshold_max": thr_max,
        })
        logger.info("  epoch %3d/%d  main=%.4f  aux=%.4f  val_norm_mse=%.4f  "
                    "val_select=%.4f  EV=%.3f  active=%.1f  dead=%d  thr=%.4f",
                    epoch, n_epochs, epoch_main / max(1, n_batches),
                    epoch_aux / max(1, n_batches), v_norm_mse, v_select, ev,
                    v_active, v_dead, thr_median)

        if v_select + 1e-6 < best_val:
            best_val = v_select
            best_val_full = v_norm_mse
            best_state = {kk: vv.detach().cpu().clone()
                          for kk, vv in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if epoch >= min_epochs and patience_left <= 0:
                logger.info("  early stop at epoch %d (val plateaued)", epoch)
                break

    if best_state is not None:
        model.load_state_dict({kk: vv.to(dev) for kk, vv in best_state.items()})

    model.eval()
    calibration_rows = 0
    calibration_l0 = float("nan")
    if not isinstance(model, (SimpleTopKSAE, JumpReLUSAE)):
        calibration_rows, calibration_l0 = _calibrate_batchtopk_threshold(
            model, Xv, batch=batch, device=dev, target_l0=k, seed=seed,
            max_score_elements=threshold_calibration_elements)
    inference_norm_mse, inference_l0, inference_dead, inference_rare = \
        _deployment_metrics(model, Xv, batch=batch, device=dev)
    inference_ev = 1.0 - inference_norm_mse
    thr_min, thr_median, thr_max = _threshold_stats(model)
    if log_rows:
        log_rows[-1].update({
            "deployment_val_norm_mse": inference_norm_mse,
            "deployment_val_ev": inference_ev,
            "deployment_val_active": inference_l0,
            "deployment_dead_neurons": inference_dead,
            "deployment_rare_neurons": inference_rare,
            "calibrated_threshold": thr_median,
            "calibration_l0": calibration_l0,
            "calibration_rows": calibration_rows,
        })

    config = {
        "sae_type": sae_type,
        **sae_semantics(sae_type),
        "input_dim": d_in,
        "m_total_neurons": m_total,
        "k_active_neurons": k,
        "aux_k": model.aux_k,
        "dead_neuron_threshold_steps": dead_threshold_steps,
        "matryoshka_prefix_lengths": prefix,
        "lr": lr,
        "batch": batch,
        "seed": seed,
        "optimizer": "adam",
        "weight_decay": 0.0,
        # full-reconstruction quality at the selected epoch (comparable across lens types,
        # unchanged meaning for downstream readers); the checkpoint is chosen by the
        # Matryoshka-averaged objective recorded separately below.
        "best_val_norm_mse": best_val_full,
        "best_val_select_norm_mse": best_val,
        "best_val_explained_variance": 1.0 - best_val_full,
        "deployment_val_norm_mse": inference_norm_mse,
        "deployment_val_explained_variance": inference_ev,
        "deployment_val_active": inference_l0,
        "deployment_dead_neurons": inference_dead,
        "deployment_rare_neurons": inference_rare,
        "target_l0": k if sae_type != "jumprelu" else None,
        "threshold_calibration_rows": calibration_rows,
        "calibration_l0": (calibration_l0 if math.isfinite(calibration_l0) else None),
        "threshold_min": thr_min,
        "threshold_median": thr_median,
        "threshold_max": thr_max,
    }
    if sae_type == "jumprelu":
        config["sparsity_coef"] = sparsity_coef
        config["bandwidth"] = bandwidth
        config["sparsity_warmup_steps"] = sparsity_warmup_steps
    return model, config, log_rows
