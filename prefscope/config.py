"""Shared paths and defaults for PrefScope."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Paths assume the package runs in-place: [tool.uv] package = false ensures
# __file__ resolves inside the repo rather than site-packages.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Config:
    # frozen SAE used for capabilities (1) and (2)
    frozen_sae_dir: Path = PROJECT_ROOT / "features_m128_k16"
    cache_dir: Path = PROJECT_ROOT / "data" / "cache"
    embed_model_id: str = "Qwen/Qwen3-Embedding-8B"
    max_tokens: int = 4096
    # GPU-friendly default; raise via --embed-batch-size on large-VRAM cards
    embed_batch_size: int = 32
    # quarter-tie judge preferences collapse to a tie
    quarter_ties: tuple[float, ...] = (0.25, 0.75)
    # WIMHF embedding instruction (verbatim) — Qwen/Nemotron embedders are
    # instruction-aware and prepending this helps. Each side's exchange is
    # embedded as: f"{embed_instruction}\n\nUser: {prompt}\n\nAssistant: {response}".
    embed_instruction: str = (
        "Represent this user-assistant exchange for predicting which assistant "
        "response humans would prefer, focusing on helpfulness, correctness, "
        "harmlessness, relevance, and style."
    )
    # prompt-only instruction for the prompt-concept lens — embeds f"{...}\n\n
    # User: {prompt}" to capture what the request asks for (task / intent / topic).
    prompt_embed_instruction: str = (
        "Represent this user request, focusing on what task it asks for "
        "(e.g. coding, math, summarization, clarification, factual question, "
        "creative writing, translation) and its topic."
    )


CONFIG = Config()
