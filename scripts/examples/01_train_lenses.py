#!/usr/bin/env python
"""Example 01 — train SAE lenses from a dumped embedding set (Python API).

Mirrors the CLI `prefscope build-lens --from-embeddings` / `build-prompt-lens`,
but shows the Python entry points so you can call them from your own code.

The embeddings are read once from the dump (e_a/e_b/e_prompt.npy + meta), so this
is just the SAE fit. Use a GPU (`--device cuda`) or `--device cpu`.

    python scripts/examples/01_train_lenses.py \
        --dump            artifacts/embeddings \
        --out-completion  artifacts/lenses/completion \
        --out-prompt      artifacts/lenses/prompt \
        --m-total 64 --k 16 --device cuda
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prefscope.pipeline.build_lens import (  # noqa: E402
    build_lens_from_embeddings, build_prompt_lens)


def main() -> None:
    ap = argparse.ArgumentParser(description="train SAE lenses from an embedding dump")
    ap.add_argument("--dump", required=True, help="dump dir: e_a/e_b/e_prompt.npy + meta.parquet")
    ap.add_argument("--out-completion", required=True, help="completion (individual) lens dir")
    ap.add_argument("--out-prompt", default=None, help="prompt lens dir (optional)")
    ap.add_argument("--m-total", type=int, default=64)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--matryoshka-prefix", type=int, nargs="+", default=[8, 32],
                    help="intermediate prefixes; m_total is auto-appended")
    ap.add_argument("--n-epochs", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--embed-model-id", default="Qwen/Qwen3-Embedding-8B")
    args = ap.parse_args()

    # Completion lens: input_rep="individual" trains the SAE on pooled [e_a; e_b],
    # so its encoder f scores a single completion. It saves z_a=f(e_a), z_b=f(e_b),
    # and z_diff = f(e_a) - f(e_b) (the directional A-vs-B signal used downstream).
    man = build_lens_from_embeddings(
        args.dump, args.out_completion,
        m_total=args.m_total, k=args.k,
        matryoshka_prefix=tuple(args.matryoshka_prefix),
        input_rep="individual", device=args.device,
        embed_model_id=args.embed_model_id, n_epochs=args.n_epochs)
    print(f"completion lens -> {args.out_completion} "
          f"(val norm-MSE {man.get('best_val_norm_mse')})")

    # Prompt lens: a single-text SAE over e_prompt.npy — "what concepts are in the
    # prompt". Used to condition the prompt-conditioned delta (example 05).
    if args.out_prompt:
        pman = build_prompt_lens(
            args.dump, args.out_prompt,
            m_total=64, k=8, matryoshka_prefix=(8,),
            device=args.device, embed_model_id=args.embed_model_id,
            n_epochs=args.n_epochs)
        print(f"prompt lens -> {args.out_prompt} "
              f"(val norm-MSE {pman.get('best_val_norm_mse')})")


if __name__ == "__main__":
    main()
