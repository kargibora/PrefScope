#!/usr/bin/env python
"""Example 02 — name + verify a lens's features (Python API).

Mirrors `prefscope interpret name` then `interpret verify`. This requires a
configured LLM backend. For OpenRouter, set `OPENROUTER_API_KEY` first.

    python scripts/examples/02_interpret_lens.py \
        --lens-dir artifacts/lenses/completion \
        --corpus   data/corpus.parquet \
        --out-dir  artifacts/interpretation/completion
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prefscope.interpret.io import load_lens_battles      # noqa: E402
from prefscope.interpret.llm import LLMClient             # noqa: E402
from prefscope.interpret.name import name_features        # noqa: E402
from prefscope.interpret.verify import verify_features    # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="name + verify SAE features")
    ap.add_argument("--lens-dir", required=True)
    ap.add_argument("--corpus", required=True, help="corpus the lens was built from (for example text)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="deepseek/deepseek-v3.2")
    ap.add_argument("--backend", default="openai", choices=["openai", "claude-cli", "codex-cli"])
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--n-per-bucket", type=int, default=20, help="verify examples per bucket")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # battles row-aligned to z_diff (= f(e_a)-f(e_b) for the completion lens, so
    # z>0 means completion A expresses the concept more than B).
    battles, z_diff, _ = load_lens_battles(args.lens_dir, corpus=args.corpus)
    client = LLMClient(backend=args.backend, model=args.model)

    # 1) name: LLM labels each feature from its most-positive / most-negative pairs
    names = name_features(battles, z_diff, client, concurrency=args.concurrency)
    names.to_csv(out / "feature_names.csv", index=False)
    print(f"named {len(names)} features -> {out/'feature_names.csv'}")

    # 2) verify: held-out detection of each named concept; keeps a fidelity_pass flag
    #    (|corr| >= threshold AND Bonferroni p < 0.05). Raise --n-per-bucket for power.
    fid = verify_features(battles, z_diff, names, client,
                          n_per_bucket=args.n_per_bucket, concurrency=args.concurrency)
    fid.to_csv(out / "feature_fidelity.csv", index=False)
    print(f"{int(fid['fidelity_pass'].sum())}/{len(fid)} passed -> {out/'feature_fidelity.csv'}")


if __name__ == "__main__":
    main()
