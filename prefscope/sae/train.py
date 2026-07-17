"""Train a BatchTopK SAE on pooled completion embeddings (pure compute)."""
from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from prefscope.core import registry
from prefscope.sae.model import BatchTopKSAE, JumpReLUSAE, SimpleTopKSAE  # noqa: F401  (registers "sae" components on import)

logger = logging.getLogger(__name__)


def train_sae(
    X_train: np.ndarray,
    X_val: np.ndarray,
    *,
    m_total: int = 128,
    k: int = 16,
    matryoshka_prefix: Sequence[int] = (8,),
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
    sae_type: str = "batchtopk",
    sparsity_coef: float = 1e-3,
    bandwidth: float = 1e-3,
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

    # Instantiate from the union of params every SAE flavor might want; each class
    # absorbs what it doesn't use via `**_` (mirrors the clusterer registry).
    model = cls(
        input_dim=d_in, m_total_neurons=m_total, k_active_neurons=k,
        aux_k=aux_k, dead_neuron_threshold_steps=dead_threshold_steps,
        matryoshka_prefix_lengths=list(matryoshka_prefix),
        sparsity_coef=sparsity_coef, bandwidth=bandwidth,
    ).to(dev)
    prefix = model.matryoshka_prefix_lengths
    opt = AdamW(model.parameters(), lr=lr)

    # Keep the full matrices on CPU; only per-batch slices go to ``dev`` below.
    # This bounds GPU memory at O(batch) instead of O(N).
    Xt = torch.from_numpy(X_train)
    Xv = torch.from_numpy(X_val)
    var_x_val = Xv.var().item()

    n_train = Xt.shape[0]
    n_val = Xv.shape[0]
    n_batches = (n_train + batch - 1) // batch
    n_val_batches = (n_val + batch - 1) // batch

    log_rows: list[dict] = []
    best_val = float("inf")          # selection objective (Matryoshka-averaged norm MSE)
    best_val_full = float("inf")     # full-reconstruction norm MSE at the selected epoch
    best_state: dict | None = None
    patience_left = patience

    for epoch in range(1, n_epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=dev)
        epoch_main = 0.0
        epoch_aux = 0.0
        for bi in range(n_batches):
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
            ev = 1.0 - v_mse / var_x_val
            v_norm_mse = float(model._normalized_mse(v_recon, Xv))
            # Selection metric = the objective TRAINING actually minimizes: mean
            # normalized-MSE across prefix levels for a Matryoshka lens (identical to
            # v_norm_mse when there are no prefixes). Early-stopping/checkpoint choice
            # uses THIS so it can't pick a lens strong on the full code but weak on the
            # coarse prefixes the Matryoshka terms shape.
            v_select = (sum(sse_prefix[L] / (sse_base + 1e-8) for L in prefix) / len(prefix)
                        if prefix else v_norm_mse)
            v_active = float((v_acts != 0).float().sum(dim=-1).mean())
            v_dead = int((model.steps_since_activation > dead_threshold_steps).sum().item())

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
            "threshold": float(model.threshold.item()),
        })
        logger.info("  epoch %3d/%d  main=%.4f  aux=%.4f  val_norm_mse=%.4f  "
                    "val_select=%.4f  EV=%.3f  active=%.1f  dead=%d  thr=%.4f",
                    epoch, n_epochs, epoch_main / max(1, n_batches),
                    epoch_aux / max(1, n_batches), v_norm_mse, v_select, ev,
                    v_active, v_dead, model.threshold.item())

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

    config = {
        "sae_type": sae_type,
        "input_dim": d_in,
        "m_total_neurons": m_total,
        "k_active_neurons": k,
        "aux_k": model.aux_k,
        "dead_neuron_threshold_steps": dead_threshold_steps,
        "matryoshka_prefix_lengths": prefix,
        "lr": lr,
        "batch": batch,
        # full-reconstruction quality at the selected epoch (comparable across lens types,
        # unchanged meaning for downstream readers); the checkpoint is chosen by the
        # Matryoshka-averaged objective recorded separately below.
        "best_val_norm_mse": best_val_full,
        "best_val_select_norm_mse": best_val,
    }
    if sae_type == "jumprelu":
        config["sparsity_coef"] = sparsity_coef
        config["bandwidth"] = bandwidth
    return model, config, log_rows
