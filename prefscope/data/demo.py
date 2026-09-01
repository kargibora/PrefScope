"""Self-contained synthetic data for the installed PrefScope quickstart."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import yaml


TEMPLATES = [
    (
        "Explain {t} to a beginner.",
        "Here's {t}, step by step:\n1. the core idea\n2. a worked example\n"
        "3. a common pitfall. For instance, imagine {t} like sorting books on a shelf.",
        "{t} is just a thing you learn. Look it up.",
    ),
    (
        "Write a function to {t}.",
        "```python\ndef solve():\n    # handles {t} with edge cases\n    return result\n```\n"
        "It also validates the input first.",
        "just write a loop that does {t}.",
    ),
    (
        "What are the trade-offs of {t}?",
        "Pros: it is fast and clear. Cons: it uses more memory. Use {t} when latency "
        "matters; avoid it when memory is tight. On balance it depends on your constraints.",
        "{t} is good sometimes and bad sometimes.",
    ),
    (
        "Summarize {t}.",
        "In short, {t} means three things: scope, method, and result — each matters "
        "for the conclusion.",
        "{t} is about stuff.",
    ),
]

TOPICS = [
    "binary search",
    "gradient descent",
    "HTTP caching",
    "the water cycle",
    "recursion",
    "database indexing",
    "photosynthesis",
    "load balancing",
    "the Fourier transform",
    "garbage collection",
    "TCP handshakes",
    "entropy",
    "dynamic programming",
    "DNS resolution",
    "backpropagation",
]


def _battle_id(row: dict) -> str:
    fields = ("prompt", "model_a", "model_b", "completion_a", "completion_b")
    blob = "|".join(str(row[column]) for column in fields)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def make_demo_corpus() -> pd.DataFrame:
    """Return the deterministic 60-battle synthetic tutorial corpus."""
    rows = []
    for i, topic in enumerate(TOPICS):
        for j, (question, detailed, terse) in enumerate(TEMPLATES):
            flip = (i + j) % 2 == 1
            completion_a, completion_b = (
                (terse, detailed) if flip else (detailed, terse)
            )
            rows.append(
                {
                    "prompt": question.format(t=topic),
                    "model_a": "model-terse" if flip else "model-detailed",
                    "model_b": "model-detailed" if flip else "model-terse",
                    "completion_a": completion_a.format(t=topic),
                    "completion_b": completion_b.format(t=topic),
                    "human_pref": 0.0 if flip else 1.0,
                    "source": "sample",
                    "language": "en",
                }
            )
    frame = pd.DataFrame(rows)
    fields = ["prompt", "model_a", "model_b", "completion_a", "completion_b"]
    frame["battle_id"] = [_battle_id(row) for row in frame[fields].to_dict("records")]
    return frame[["battle_id", "source", "language", *fields, "human_pref"]]


def create_demo(directory, *, force: bool = False) -> dict[str, Path]:
    """Write a complete pip-installable quickstart workspace.

    The generated config uses absolute paths so it works regardless of the caller's
    current directory. Existing non-empty directories are refused unless ``force`` is
    explicit.
    """
    root = Path(directory).expanduser().resolve()
    if root.exists() and any(root.iterdir()) and not force:
        raise FileExistsError(
            f"{root} exists and is not empty; pass force=True to replace demo files"
        )
    root.mkdir(parents=True, exist_ok=True)
    corpus = root / "sample_corpus.parquet"
    config = root / "quickstart.yaml"
    lens = root / "lens"
    results = root / "results"
    make_demo_corpus().to_parquet(corpus, index=False)
    payload = {
        "lens_dir": str(lens),
        "corpus": str(corpus),
        "out_dir": str(results),
        "stages": ["name", "verify", "cluster", "win-relevance"],
        "llm": {
            "backend": "openai",
            "model": "deepseek/deepseek-v3.2",
            "api_base": "https://openrouter.ai/api/v1",
            "api_key_env": "OPENROUTER_API_KEY",
        },
        "interpreter": {"name": "auto", "n_active": 3, "n_zero": 3},
        "verifier": {"name": "auto", "n_per_bucket": 3},
        "clusterer": {"name": "spherical-kmeans", "n_clusters": 4},
        "win_relevance": {"all_features": False},
    }
    config.write_text(yaml.safe_dump(payload, sort_keys=False))
    return {
        "root": root,
        "corpus": corpus,
        "config": config,
        "lens": lens,
        "results": results,
    }


__all__ = ["create_demo", "make_demo_corpus"]
