"""Generate a tiny synthetic battle corpus for the smoke test and the first-lens tutorial.

This is NOT real data. It is 60 templated battles with a little structure — the "A"
answers are detailed/structured (lists, examples, code), the "B" answers are terse —
and `human_pref` favours A. Enough for the pipeline to run end to end on CPU and
produce non-empty (if not scientifically meaningful) concepts.

Regenerate the parquet with:  python examples/make_sample_corpus.py
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


def make_battle_id(row: dict) -> str:
    """Content hash over the battle fields (standalone — no prefscope import needed)."""
    blob = "|".join(str(row[c]) for c in
                    ("prompt", "model_a", "model_b", "completion_a", "completion_b"))
    return hashlib.sha1(blob.encode()).hexdigest()[:16]

# (prompt, detailed/structured A answer, terse B answer)
TEMPLATES = [
    ("Explain {t} to a beginner.",
     "Here's {t}, step by step:\n1. the core idea\n2. a worked example\n3. a common pitfall. "
     "For instance, imagine {t} like sorting books on a shelf.",
     "{t} is just a thing you learn. Look it up."),
    ("Write a function to {t}.",
     "```python\ndef solve():\n    # handles {t} with edge cases\n    return result\n```\n"
     "It also validates the input first.",
     "just write a loop that does {t}."),
    ("What are the trade-offs of {t}?",
     "Pros: it is fast and clear. Cons: it uses more memory. Use {t} when latency matters; "
     "avoid it when memory is tight. On balance it depends on your constraints.",
     "{t} is good sometimes and bad sometimes."),
    ("Summarize {t}.",
     "In short, {t} means three things: scope, method, and result — each matters for the conclusion.",
     "{t} is about stuff."),
]
TOPICS = ["binary search", "gradient descent", "HTTP caching", "the water cycle",
          "recursion", "database indexing", "photosynthesis", "load balancing",
          "the Fourier transform", "garbage collection", "TCP handshakes",
          "entropy", "dynamic programming", "DNS resolution", "backpropagation"]


def build() -> pd.DataFrame:
    rows = []
    for i, topic in enumerate(TOPICS):
        for j, (q, a, b) in enumerate(TEMPLATES):
            # alternate which slot is the detailed answer so A/B aren't degenerate
            flip = (i + j) % 2 == 1
            ca, cb = (b, a) if flip else (a, b)
            pref = 0.0 if flip else 1.0          # human_pref = P(A preferred); A wins when detailed
            rows.append({
                "prompt": q.format(t=topic),
                "model_a": "model-detailed" if not flip else "model-terse",
                "model_b": "model-terse" if not flip else "model-detailed",
                "completion_a": ca.format(t=topic),
                "completion_b": cb.format(t=topic),
                "human_pref": pref,
                "source": "sample",
                "language": "en",
            })
    df = pd.DataFrame(rows)
    content = ["prompt", "model_a", "model_b", "completion_a", "completion_b"]
    df["battle_id"] = [make_battle_id(r) for r in df[content].to_dict("records")]
    return df[["battle_id", "source", "language", *content, "human_pref"]]


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "sample_corpus.parquet"
    df = build()
    df.to_parquet(out, index=False)
    print(f"wrote {len(df)} sample battles -> {out}")
