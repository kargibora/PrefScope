"""Frozen SAE projector.

Loads a trained SAE from its saved checkpoint and runs the *frozen inference*
path. The architecture is resolved by name from the registry (``config["sae_type"]``),
the class is rebuilt from the saved ``config``, and selection is delegated to the
model's ``encode`` — so the inference rule lives in one place (the model) instead of
being re-implemented here. Accepts either a path to ``sae_model.pt`` or a directory
containing it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

# Import for the registration side effect: loading the SAE classes populates the
# "sae" registry bucket so get("sae", ...) below can resolve any built-in name.
import prefscope.sae.model  # noqa: F401
from prefscope.core import registry

# config keys whose names match an SAE __init__ parameter. Training-only params
# (lr/batch/sparsity_coef/bandwidth/...) are NOT passed: inference doesn't need
# them and each class defaults them, so we map only the structural ones. Classes
# absorb any extra via **_, so this list can be a superset.
_INIT_KEYS = ("input_dim", "m_total_neurons", "k_active_neurons", "aux_k",
              "dead_neuron_threshold_steps", "matryoshka_prefix_lengths")


class SAEProjector:
    def __init__(self, model_path: str | Path, device: str = "cpu") -> None:
        p = Path(model_path)
        if p.is_dir():
            p = p / "sae_model.pt"
        # weights_only=True: a lens is a shared artifact, so never run its pickle.
        # Our checkpoint is {state_dict: tensors, config: plain dict}, both of which
        # the safe unpickler handles; a checkpoint that needs arbitrary globals is
        # exactly what we want to reject.
        ckpt = torch.load(p, map_location=device, weights_only=True)
        sd = ckpt["state_dict"]
        self.config = ckpt.get("config", {})
        self.device = device
        self.sae_type = self.config.get("sae_type", "batchtopk")

        # Rebuild the SAE from its checkpoint and reuse its frozen inference path.
        cls = registry.get("sae", self.sae_type)
        kwargs = {k: self.config[k] for k in _INIT_KEYS if k in self.config}
        # input_dim / m_total may be absent from older configs — recover from the
        # weight matrices (encoder.weight is (M, D)).
        W_enc = sd["encoder.weight"]
        kwargs.setdefault("input_dim", W_enc.shape[1])
        kwargs.setdefault("m_total_neurons", W_enc.shape[0])
        kwargs.setdefault("k_active_neurons", self.config.get("k_active_neurons", 1))
        self._model = cls(**kwargs)
        # strict=False: checkpoints carry training-only buffers (e.g. an extra
        # threshold/steps buffer a given flavor may not register) and old configs
        # may differ in incidental keys; the inference path only needs the encoder/
        # decoder/bias/threshold weights, which load by name regardless. We assert
        # the load isn't silently dropping a *required* inference weight below.
        missing, unexpected = self._model.load_state_dict(sd, strict=False)
        required = {"encoder.weight", "decoder.weight", "input_bias", "neuron_bias"}
        dropped = required & set(missing)
        if dropped:
            raise ValueError(
                f"checkpoint {p} is missing required SAE weights {sorted(dropped)}")
        self._model.to(device).eval()

        self.input_dim = int(W_enc.shape[1])
        self.m_total = int(W_enc.shape[0])
        # kept for back-compat / introspection (JumpReLU lenses expose a per-feature gate)
        self.feature_threshold = (
            self._model._thresholds().detach()
            if self.sae_type == "jumprelu" else None)

        # if the lens was trained on whitened inputs, apply the same transform here
        from prefscope.sae.whiten import Whitener
        self.whitener = Whitener.load(p.parent)

    @torch.no_grad()
    def project(self, x: np.ndarray, *, batch: int = 16384) -> np.ndarray:
        """Embeddings (N, D) -> sparse signed codes (N, M).

        Processed in row chunks of ``batch`` so GPU memory stays O(batch x D); the
        encoder is row-independent, so the result is identical to a single pass."""
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"embedding dim {x.shape[-1] if x.ndim else x.shape} != lens "
                f"input_dim {self.input_dim}. The embedder and lens disagree — this "
                f"lens expects {self.input_dim}-dim vectors; check that the embedder "
                f"matches the lens manifest's embed_model_id.")
        out = np.empty((x.shape[0], self.m_total), dtype=np.float32)
        for s in range(0, x.shape[0], batch):
            chunk = x[s:s + batch]
            if self.whitener is not None:             # match training-time whitening
                chunk = self.whitener.transform(chunk)
            elif not chunk.flags.writeable:
                # np.load(..., mmap_mode="r") yields read-only views. PyTorch warns that
                # wrapping those is undefined even though inference does not mutate them;
                # copy one bounded chunk, not the full memory-mapped corpus.
                chunk = np.array(chunk, copy=True)
            xt = torch.as_tensor(chunk, device=self.device)
            out[s:s + batch] = self._model.encode(xt).cpu().numpy()  # selection in the model
        return out

    @torch.no_grad()
    def reconstruct(self, z: np.ndarray, *, batch: int = 16384) -> np.ndarray:
        """Sparse codes (N, M) -> reconstructed embeddings (N, D), in row chunks."""
        z = np.asarray(z, dtype=np.float32)
        out = np.empty((z.shape[0], self.input_dim), dtype=np.float32)
        for s in range(0, z.shape[0], batch):
            zt = torch.as_tensor(z[s:s + batch], device=self.device)
            rec = (self._model.decoder(zt) + self._model.input_bias).cpu().numpy()
            if self.whitener is not None:             # back to embedding space
                rec = self.whitener.inverse_transform(rec)
            out[s:s + batch] = rec
        return out

    def residual_norm(self, x: np.ndarray) -> np.ndarray:
        """Per-row L2 norm of the SAE reconstruction residual ||x - recon|| (N,).

        A coverage/confidence signal: large residual = the SAE represents this
        input poorly (off-dictionary behavior).
        """
        x = np.asarray(x, dtype=np.float32)
        recon = self.reconstruct(self.project(x))
        return np.linalg.norm(x - recon, axis=1).astype(np.float32)
