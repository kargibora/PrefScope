#!/usr/bin/env python
"""Example 04 — name the prompt-lens features (Python API).

Mirrors `prefscope name-prompts`: re-attaches each battle's prompt text by
battle_id, then LLM-names the prompt SAE features from their top-activating
prompts ("what kind of prompt is this?"). This requires a configured LLM backend.

    python scripts/examples/04_name_prompts.py \
        --lens-dir artifacts/lenses/prompt \
        --corpus   data/corpus.parquet \
        --out      artifacts/interpretation/prompt/prompt_feature_names.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prefscope.data.corpus import load_corpus                       # noqa: E402
from prefscope.interpret.llm import LLMClient                       # noqa: E402
from prefscope.interpret.prompt_name import name_prompt_features    # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="name prompt-lens features")
    ap.add_argument("--lens-dir", required=True, help="prompt lens dir (z_prompt.npy)")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True, help="output prompt_feature_names.csv")
    ap.add_argument("--model", default="deepseek/deepseek-v3.2")
    ap.add_argument("--backend", default="openai")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    lens = Path(args.lens_dir)
    z = np.load(lens / "z_prompt.npy")
    meta = pd.read_parquet(lens / "battles.parquet")
    # the lens keeps battle_id or instruction_id (it sets instruction_id = battle_id)
    bid = (meta["battle_id"] if "battle_id" in meta.columns
           else meta["instruction_id"]).astype(str)
    corp = load_corpus(args.corpus)
    corp["battle_id"] = corp["battle_id"].astype(str)
    prompts = bid.map(corp.set_index("battle_id")["prompt"]).fillna("").tolist()

    client = LLMClient(backend=args.backend, model=args.model)
    df = name_prompt_features(prompts, z, client, concurrency=args.concurrency,
                              instruction_ids=bid.tolist())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"named {len(df)} prompt features -> {args.out}")


if __name__ == "__main__":
    main()
