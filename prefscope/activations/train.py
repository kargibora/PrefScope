"""Stream-train a BatchTopK SAE from an ActivationCache memmap.

prefscope/sae/train.py loads the whole training matrix onto the GPU, which is
impossible at token scale (tens of millions of rows x hidden). This trainer
keeps activations on disk and copies only per-batch slices to the GPU, reusing
the same BatchTopKSAE model, loss, decoder housekeeping, and early-stopping
logic as the embedding pipeline.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from prefscope.activations.cache import train_val_row_indices
from prefscope.sae.model import BatchTopKSAE

logger = logging.getLogger(__name__)


def train_token_sae(cache, *, m_total: int, k: int,
                    matryoshka_prefix: Sequence[int] = (8,),
                    val_frac: float = 0.05, max_val_tokens: int | None = 200_000,
                    max_train_tokens: int | None = None,
                    n_epochs: int = 2, batch: int = 4096, lr: float = 5e-4,
                    aux_coef: float = 1.0 / 32.0, dead_threshold_steps: int = 256,
                    clip_grad: float = 1.0, seed: int = 0, device: str = "cuda",
                    min_epochs: int = 1, patience: int = 2):
    """Return (model, config, log_rows) — best-val weights restored."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)

    train_idx, val_idx = train_val_row_indices(cache.n_tokens, val_frac,
                                               max_train_tokens, seed)
    if max_val_tokens is not None and len(val_idx) > max_val_tokens:
        # validation is only for early-stopping / EV monitoring; cap it so the
        # whole val set fits on the GPU as a single tensor regardless of scale.
        cap_rng = np.random.default_rng(seed + 1)
        val_idx = np.sort(cap_rng.choice(val_idx, size=max_val_tokens, replace=False))
    d_in = cache.hidden_dim
    model = BatchTopKSAE(input_dim=d_in, m_total_neurons=m_total,
                         k_active_neurons=k, dead_neuron_threshold_steps=dead_threshold_steps,
                         matryoshka_prefix_lengths=list(matryoshka_prefix)).to(dev)
    prefix = model.matryoshka_prefix_lengths
    opt = AdamW(model.parameters(), lr=lr)

    # validation rows are small enough to hold on the GPU
    Xv = torch.from_numpy(np.asarray(cache.acts[val_idx], dtype=np.float32)).to(dev)
    var_x_val = Xv.var().item()

    rng = np.random.default_rng(seed)
    n_train = len(train_idx)
    n_batches = (n_train + batch - 1) // batch
    log_rows: list[dict] = []
    best_val = float("inf")
    best_state = None
    patience_left = patience
    v_dead = 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        order = train_idx[rng.permutation(n_train)]
        epoch_main = epoch_aux = 0.0
        for bi in range(n_batches):
            rows = order[bi * batch:(bi + 1) * batch]
            xb = torch.from_numpy(
                np.asarray(cache.acts[rows], dtype=np.float32)).to(dev)
            recon, info = model(xb)
            loss, parts = model.compute_loss(xb, recon, info, aux_coef=aux_coef)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            model.adjust_decoder_gradient_()
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            opt.step()
            model.normalize_decoder_()
            epoch_main += parts["main"]
            epoch_aux += parts["aux"]

        model.eval()
        with torch.no_grad():
            v_recon, v_info = model(Xv)
            v_mse = F.mse_loss(v_recon, Xv).item()
            ev = 1.0 - v_mse / var_x_val
            v_norm_mse = float(model._normalized_mse(v_recon, Xv))
            v_active = float((v_info["activations"] != 0).float().sum(dim=-1).mean())
            v_dead = int((model.steps_since_activation > dead_threshold_steps).sum().item())
        log_rows.append({"epoch": epoch, "train_main_mean": epoch_main / max(1, n_batches),
                         "train_aux_mean": epoch_aux / max(1, n_batches), "val_mse": v_mse,
                         "val_norm_mse": v_norm_mse, "val_ev": ev, "val_active": v_active,
                         "dead_neurons": v_dead, "threshold": float(model.threshold.item())})
        logger.info("  epoch %d/%d  val_norm_mse=%.4f  EV=%.3f  active=%.1f  dead=%d  thr=%.4f",
                    epoch, n_epochs, v_norm_mse, ev, v_active, v_dead, model.threshold.item())

        if v_norm_mse + 1e-6 < best_val:
            best_val = v_norm_mse
            best_state = {kk: vv.detach().cpu().clone() for kk, vv in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if epoch >= min_epochs and patience_left <= 0:
                logger.info("  early stop at epoch %d", epoch)
                break

    if best_state is not None:
        model.load_state_dict({kk: vv.to(dev) for kk, vv in best_state.items()})

    config = {"sae_type": "batchtopk", "input_dim": d_in, "m_total_neurons": m_total,
              "k_active_neurons": k, "aux_k": model.aux_k,
              "dead_neuron_threshold_steps": dead_threshold_steps,
              "matryoshka_prefix_lengths": prefix, "lr": lr, "batch": batch,
              "best_val_norm_mse": best_val,
              "best_val_ev": float(1.0 - best_val) if np.isfinite(best_val) else None,
              "dead_neurons": v_dead, "n_train_tokens": int(n_train),
              "n_val_tokens": int(len(val_idx)), "max_val_tokens": max_val_tokens}
    return model, config, log_rows
