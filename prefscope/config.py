"""Shared defaults for PrefScope."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType


def _default_cache_dir() -> Path:
    """Return a user-writable, platform-appropriate embedding cache directory."""
    if value := os.environ.get("PREFSCOPE_CACHE_DIR"):
        return Path(value).expanduser()
    if value := os.environ.get("XDG_CACHE_HOME"):
        return Path(value).expanduser() / "prefscope"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "prefscope"
    if os.name == "nt" and (value := os.environ.get("LOCALAPPDATA")):
        return Path(value).expanduser() / "prefscope" / "Cache"
    return Path.home() / ".cache" / "prefscope"


@dataclass(frozen=True)
class Config:
    cache_dir: Path = _default_cache_dir()
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


VIEWER_EXPORT_DEFAULTS = MappingProxyType({
    "examples_per_feature": 12,
    "examples_per_group": 2,
    "examples_random": 4,
    "examples_boundary": 4,
    "prompt_examples_per_feature": 8,
    "prompt_examples_per_group": 2,
    "prompt_examples_random": 4,
    "prompt_examples_boundary": 4,
    "map_sample": 2500,
    "map_sample_mode": "hybrid",
    "coactivation_top_k": 20,
    "coactivation_max_pairs": 20000,
})


CONFIG = Config()
