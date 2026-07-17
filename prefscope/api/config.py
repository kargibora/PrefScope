"""Configuration dataclasses for the public PrefScope API.

``SAEConfig`` is the *architecture* — the part that defines the frozen lens
(width, sparsity, input representation). ``TrainConfig`` is the run-time wrapper:
it nests an ``SAEConfig`` and adds the knobs that don't change what the lens *is*
(embedder choice, validation split, device, row cap, extra training kwargs).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SAEConfig:
    """SAE architecture — the part that defines the frozen lens."""
    m: int = 128
    k: int = 16
    input_rep: str = "individual"        # "individual" | "difference" | "prompt"
    matryoshka_prefix: tuple[int, ...] = (8,)


@dataclass
class TrainConfig:
    """Run-time training configuration; nests the architecture in ``sae``."""
    sae: SAEConfig = field(default_factory=SAEConfig)
    embed_model_id: str | None = None    # None -> embedder/config default
    val_frac: float = 0.1
    # A safe library default: callers can opt into "cuda"/"mps" explicitly.
    # This makes Lens.train(data, out=...) work on an ordinary CPU installation.
    device: str = "cpu"
    max_train_rows: int | None = None
    train_kwargs: dict = field(default_factory=dict)   # -> build_lens **train_kwargs
