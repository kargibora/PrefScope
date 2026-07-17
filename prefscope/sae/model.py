from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from prefscope.core import registry


# ─────────────────────────────────────────────────────────────────────────────
#  WIMHF-style Batch TopK SAE
# ─────────────────────────────────────────────────────────────────────────────

@registry.register("sae", "batchtopk")
class BatchTopKSAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        m_total_neurons: int,
        k_active_neurons: int,
        *,
        aux_k: int | None = None,
        dead_neuron_threshold_steps: int = 256,
        matryoshka_prefix_lengths: list[int] | None = None,
        **_,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.m_total = m_total_neurons
        self.k = k_active_neurons
        self.aux_k = aux_k if aux_k is not None else min(2 * k_active_neurons,
                                                         m_total_neurons)
        self.dead_threshold_steps = dead_neuron_threshold_steps
        # Matryoshka prefix lengths (WIMHF: the last element MUST equal m_total
        # so the full reconstruction is included as the final prefix term).
        # If user supplies intermediate prefixes only, we append m_total.
        # If empty/None, no Matryoshka — main loss is just the full reconstruction.
        if matryoshka_prefix_lengths:
            pl = sorted(set(int(p) for p in matryoshka_prefix_lengths
                            if 0 < int(p) < m_total_neurons))
            pl.append(m_total_neurons)
            self.matryoshka_prefix_lengths = pl
        else:
            self.matryoshka_prefix_lengths = []
        assert (not self.matryoshka_prefix_lengths
                or self.matryoshka_prefix_lengths[-1] == m_total_neurons), \
            "Last prefix must equal total neurons"

        # encoder / decoder linear maps (no bias on the modules themselves)
        self.encoder = nn.Linear(input_dim, m_total_neurons, bias=False)
        self.decoder = nn.Linear(m_total_neurons, input_dim, bias=False)
        # explicit biases
        self.input_bias = nn.Parameter(torch.zeros(input_dim))
        self.neuron_bias = nn.Parameter(torch.zeros(m_total_neurons))

        # init: kaiming on encoder, decoder = encoder.T, then unit-norm decoder columns
        nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.t())
        self.normalize_decoder_()

        # buffers for dead-neuron tracking + adaptive threshold
        self.register_buffer("steps_since_activation",
                             torch.zeros(m_total_neurons, dtype=torch.long))
        self.register_buffer("threshold", torch.tensor(0.0))

    # ── core math ──

    def encode_pre(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-selection activations: encoder(x - b_in) + b_neuron."""
        return self.encoder(x - self.input_bias) + self.neuron_bias

    def _batch_topk(self, acts: torch.Tensor) -> torch.Tensor:
        """Signed Batch TopK: keep top (k * batch_size) values by |·|, sign preserved."""
        batch = acts.shape[0]
        k_total = min(self.k * batch, acts.numel())
        scores = acts.abs().flatten()
        signed = acts.flatten()
        _, idx = torch.topk(scores, k_total, dim=-1)
        out = torch.zeros_like(signed)
        out.scatter_(0, idx, signed.gather(0, idx))
        return out.view(acts.shape)

    def _threshold_select(self, acts: torch.Tensor) -> torch.Tensor:
        """Inference-time selection: keep values whose |·| exceeds the learned threshold."""
        mask = acts.abs() > self.threshold
        return torch.where(mask, acts, torch.zeros_like(acts))

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Frozen inference codes: pre-activations then the selection rule.

        Polymorphic — subclasses that override ``_threshold_select`` (JumpReLU's
        per-feature gate, SimpleTopK's per-example top-k) get their own selection
        here without overriding ``encode``. This is the single inference entry
        point ``SAEProjector`` calls, so it must mirror ``forward``'s eval branch."""
        return self._threshold_select(self.encode_pre(x))

    @torch.no_grad()
    def _update_threshold_(self, activ: torch.Tensor, lr: float = 1e-2) -> None:
        """EMA update toward the minimum-magnitude active value (training only)."""
        nz = activ[activ != 0]
        if nz.numel() > 0:
            min_positive = nz.abs().min().detach()
            self.threshold.mul_(1 - lr).add_(lr * min_positive)

    def _aux_topk(self, pre_acts: torch.Tensor) -> tuple[torch.Tensor | None,
                                                          torch.Tensor | None]:
        """Per-example top-aux_k activations restricted to DEAD neurons."""
        dead_mask = (self.steps_since_activation > self.dead_threshold_steps)
        n_dead = int(dead_mask.sum().item())
        if n_dead == 0 or self.aux_k == 0:
            return None, None
        aux_k = min(self.aux_k, n_dead)

        # zero out live neurons in a copy
        masked = pre_acts.masked_fill(~dead_mask.unsqueeze(0), 0.0)
        abs_ = masked.abs()
        _, idx = torch.topk(abs_, aux_k, dim=-1)            # (batch, aux_k)
        values = masked.gather(-1, idx)                      # (batch, aux_k)
        return idx, values

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        pre = self.encode_pre(x)
        if self.training:
            activ = self._batch_topk(pre)
            self._update_threshold_(activ)
            fired = (activ != 0).any(dim=0)
            # steps_since_activation: ++ globally, then 0 for fired neurons
            self.steps_since_activation += 1
            self.steps_since_activation[fired] = 0
            aux_idx, aux_vals = self._aux_topk(pre)
            info = {"activations": activ, "aux_indices": aux_idx,
                    "aux_values": aux_vals}
        else:
            activ = self._threshold_select(pre)
            info = {"activations": activ, "aux_indices": None,
                    "aux_values": None}
        recon = self.decoder(activ) + self.input_bias
        return recon, info

    # ── decoder housekeeping ──

    @torch.no_grad()
    def normalize_decoder_(self) -> None:
        norms = self.decoder.weight.norm(dim=0, keepdim=True) + 1e-8
        self.decoder.weight.div_(norms)

    @torch.no_grad()
    def adjust_decoder_gradient_(self) -> None:
        """Remove the radial component of the decoder gradient.

        This keeps the gradient tangent to the unit-norm constraint surface
        so normalize_decoder_ doesn't fight the optimizer.
        """
        g = self.decoder.weight.grad
        if g is None:
            return
        proj = (self.decoder.weight * g).sum(dim=0, keepdim=True)
        g.sub_(proj * self.decoder.weight)

    # ── loss ──

    @staticmethod
    def _normalized_mse(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(recon, target)
        baseline = F.mse_loss(target.mean(dim=0, keepdim=True).expand_as(target),
                              target)
        return mse / (baseline + 1e-8)

    def matryoshka_norm_mse(self, recon: torch.Tensor, activ: torch.Tensor,
                            x: torch.Tensor) -> torch.Tensor:
        """Main reconstruction objective (WIMHF protocol).

        Mean of normalized MSE across the Matryoshka prefix levels — where the last
        prefix length equals ``m_total`` so the full reconstruction IS the last term —
        or plain normalized MSE on the full reconstruction when the lens has no prefixes.

        Deliberately independent of ``self.training`` so validation and early-stopping
        optimize the *same* quantity the training step minimizes. Gating this on train
        mode (as before) let checkpoint selection fall back to full-reconstruction MSE,
        which can pick a lens strong on the full code but weak on the coarse prefixes the
        Matryoshka terms shape. ``recon`` is the full reconstruction; ``activ`` the codes
        that produced it."""
        if not self.matryoshka_prefix_lengths:
            return self._normalized_mse(recon, x)
        terms: list[torch.Tensor] = []
        for L in self.matryoshka_prefix_lengths:
            if L == self.m_total:
                terms.append(self._normalized_mse(recon, x))   # == full recon
            else:
                partial = activ.clone()
                partial[:, L:] = 0
                terms.append(self._normalized_mse(
                    self.decoder(partial) + self.input_bias, x))
        return torch.stack(terms).mean()

    def compute_loss(self, x: torch.Tensor, recon: torch.Tensor, info: dict,
                     aux_coef: float = 1.0 / 32.0) -> tuple[torch.Tensor, dict]:
        main = self.matryoshka_norm_mse(recon, info["activations"], x)

        # Auxiliary loss for dead neurons (separate term, weighted by aux_coef)
        aux = torch.zeros((), device=x.device, dtype=x.dtype)
        if info.get("aux_indices") is not None:
            aux_act = torch.zeros_like(info["activations"])
            aux_act.scatter_(-1, info["aux_indices"], info["aux_values"])
            aux_recon = self.decoder(aux_act)
            residual = (x - recon).detach()
            aux = self._normalized_mse(aux_recon, residual)

        total = main + aux_coef * aux
        return total, {
            "main": float(main.detach()),
            "aux": float(aux.detach()),
            "total": float(total.detach()),
        }


@registry.register("sae", "simple-topk")
class SimpleTopKSAE(BatchTopKSAE):
    """Plain signed TopK SAE ablation.

    This keeps the same encoder/decoder/loss interface as BatchTopKSAE but
    replaces batch-level sparsity allocation with ordinary per-example TopK
    selection. It also disables Matryoshka and auxiliary dead-neuron loss by
    construction. The goal is a conservative ablation: do the paper's local
    reliability signals depend on BatchTopK, or do they also appear with a
    simpler sparse dictionary?
    """

    def __init__(
        self,
        input_dim: int,
        m_total_neurons: int,
        k_active_neurons: int,
        **_,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            m_total_neurons=m_total_neurons,
            k_active_neurons=k_active_neurons,
            aux_k=0,
            dead_neuron_threshold_steps=10**12,
            matryoshka_prefix_lengths=None,
        )

    def _batch_topk(self, acts: torch.Tensor) -> torch.Tensor:
        k = min(self.k, acts.shape[1])
        _, idx = torch.topk(acts.abs(), k, dim=-1)
        values = acts.gather(-1, idx)
        out = torch.zeros_like(acts)
        out.scatter_(-1, idx, values)
        return out

    def _threshold_select(self, acts: torch.Tensor) -> torch.Tensor:
        return self._batch_topk(acts)

    @torch.no_grad()
    def _update_threshold_(self, activ: torch.Tensor, lr: float = 1e-2) -> None:
        self.threshold.zero_()

    def _aux_topk(self, pre_acts: torch.Tensor) -> tuple[torch.Tensor | None,
                                                          torch.Tensor | None]:
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
#  JumpReLU SAE  (Rajamanoharan et al. 2024, "Jumping Ahead", arXiv:2407.14435)
# ─────────────────────────────────────────────────────────────────────────────
#
#   f(x) = JumpReLU_θ(π(x)),   JumpReLU_θ(z) = z · H(z − θ),   θ = exp(log θ) > 0
#   loss = ‖x − x̂‖²  +  λ‖f(x)‖₀,    ‖f(x)‖₀ = Σ_i H(π_i − θ_i)
#
# The Heaviside step has no useful gradient, so gradients w.r.t. θ use the paper's
# straight-through estimator with a centred rectangle kernel K(s)=𝟙(|s|<½) and a
# fixed bandwidth ε (Eqs. for ∂/∂θ in §4 of the paper):
#   ∂/∂θ JumpReLU_θ(z) = −(θ/ε) K((z−θ)/ε)      ∂/∂θ H(z−θ) = −(1/ε) K((z−θ)/ε)
# We feed θ = exp(log θ) into the autograd Functions, so the extra ×θ chain to the
# log-space parameter is produced by autograd's own `exp` backward — exactly the
# paper's log-threshold parameterization. (STE math cross-checked against the
# reference impl in saprmarks/dictionary_learning.)


class _JumpReLU(torch.autograd.Function):
    """z·H(z−θ): gradient to z is the gate; gradient to the per-feature θ is the STE."""

    @staticmethod
    def forward(ctx, z, threshold, bandwidth):
        ctx.save_for_backward(z, threshold)
        ctx.bw = float(bandwidth)
        return z * (z > threshold).to(z.dtype)

    @staticmethod
    def backward(ctx, g):
        z, threshold = ctx.saved_tensors
        bw = ctx.bw
        z_grad = (z > threshold).to(z.dtype) * g
        rect = (((z - threshold) / bw).abs() < 0.5).to(z.dtype)         # K((z−θ)/ε)
        thr_grad = (-(threshold / bw) * rect * g).sum(dim=0)            # per-feature (M,)
        return z_grad, thr_grad, None


class _Step(torch.autograd.Function):
    """H(z−θ) for the L0 count: no gradient to z, STE pseudo-derivative for θ."""

    @staticmethod
    def forward(ctx, z, threshold, bandwidth):
        ctx.save_for_backward(z, threshold)
        ctx.bw = float(bandwidth)
        return (z > threshold).to(z.dtype)

    @staticmethod
    def backward(ctx, g):
        z, threshold = ctx.saved_tensors
        bw = ctx.bw
        rect = (((z - threshold) / bw).abs() < 0.5).to(z.dtype)
        thr_grad = (-(1.0 / bw) * rect * g).sum(dim=0)
        return None, thr_grad, None


@registry.register("sae", "jumprelu")
class JumpReLUSAE(BatchTopKSAE):
    """JumpReLU SAE: a learned per-feature threshold gates each feature (no top-k).

    Sparsity is set by ``sparsity_coef`` (the L0 penalty λ), not ``k`` — ``k`` is
    accepted only for interface/config compatibility and is unused. Codes are
    one-sided non-negative (z_i = π_i when π_i > θ_i, else 0), so this pairs
    naturally with ``--input-rep individual`` (per-side codes, then differenced)
    rather than the signed ``difference`` representation. Reuses the base encoder/
    decoder, unit-norm decoder housekeeping, and normalized-MSE; disables
    Matryoshka and the dead-neuron auxiliary loss.
    """

    def __init__(self, input_dim, m_total_neurons, k_active_neurons=0, *,
                 sparsity_coef: float = 1e-3, bandwidth: float = 1e-3,
                 threshold_init: float = 1e-3, **_) -> None:
        super().__init__(
            input_dim=input_dim, m_total_neurons=m_total_neurons,
            k_active_neurons=max(1, int(k_active_neurons)),
            aux_k=0, dead_neuron_threshold_steps=10**12,
            matryoshka_prefix_lengths=None)
        self.sparsity_coef = float(sparsity_coef)
        self.bandwidth = float(bandwidth)
        # per-feature threshold in log space (positivity-safe), paper init θ=1e-3
        self.log_threshold = nn.Parameter(
            torch.full((m_total_neurons,), math.log(threshold_init)))

    def _thresholds(self) -> torch.Tensor:
        return torch.exp(self.log_threshold)

    def _threshold_select(self, pre: torch.Tensor) -> torch.Tensor:
        """Inference selection: keep π_i where it clears the feature's threshold."""
        thr = self._thresholds()
        return torch.where(pre > thr, pre, torch.zeros_like(pre))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        pre = self.encode_pre(x)
        if self.training:
            activ = _JumpReLU.apply(pre, self._thresholds(), self.bandwidth)
        else:
            activ = self._threshold_select(pre)
        recon = self.decoder(activ) + self.input_bias
        return recon, {"activations": activ, "pre": pre,
                       "aux_indices": None, "aux_values": None}

    def compute_loss(self, x, recon, info, aux_coef: float = 0.0):
        # main = reconstruction (codebase normalized-MSE, for a comparable val metric);
        # sparsity = λ · mean L0. aux_coef is ignored — λ is self.sparsity_coef.
        main = self._normalized_mse(recon, x)
        l0 = _Step.apply(info["pre"], self._thresholds(),
                         self.bandwidth).sum(dim=-1).mean()
        total = main + self.sparsity_coef * l0
        return total, {"main": float(main.detach()), "aux": float(l0.detach()),
                       "total": float(total.detach())}


def encode_in_batches(model: BatchTopKSAE, X: np.ndarray, batch: int,
                      device: torch.device) -> np.ndarray:
    model.eval()
    out = np.empty((X.shape[0], model.m_total), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, X.shape[0], batch):
            xb = torch.from_numpy(X[i:i + batch]).to(device, dtype=torch.float32)
            pre = model.encode_pre(xb)
            z = model._threshold_select(pre)
            out[i:i + batch] = z.float().cpu().numpy()
    return out
