#!/usr/bin/env python
"""Project already-embedded prompts through an EXISTING prompt lens -> z_prompt.

The BYO prompt-concept path in two steps (the completion-side `encode-dataset`
has no prompt-lens mode):

    # 1. embed the prompts alone (GPU) — writes e_prompt.npy + battle-id meta
    prefscope embed-prompts --corpus my.parquet --embed-model-id <same as lens> \
        --out enc/prompt --device cuda
    # 2. THIS: apply the frozen prompt-lens SAE -> z_prompt.npy (cheap, CPU)
    python scripts/project_prompt_lens.py --e-prompt enc/prompt --prompt-lens lenses/prompt_8b

z_prompt is battle-id aligned to e_prompt's meta.parquet; argmax(z_prompt) is the
per-battle prompt type used by the conditional/elicitation/omission analyses.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prefscope.encode.sae import SAEProjector  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="project e_prompt through a prompt lens")
    ap.add_argument("--e-prompt", required=True, dest="e_prompt",
                    help="embed-prompts output dir (e_prompt.npy + meta.parquet)")
    ap.add_argument("--prompt-lens", required=True, dest="prompt_lens")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None, help="z_prompt.npy path (default: <e-prompt>/z_prompt.npy)")
    args = ap.parse_args()

    e = np.load(Path(args.e_prompt) / "e_prompt.npy")
    proj = SAEProjector(args.prompt_lens, device=args.device)
    z = np.asarray(proj.project(e), dtype=np.float32)
    out = Path(args.out) if args.out else Path(args.e_prompt) / "z_prompt.npy"
    np.save(out, z)
    print(f"wrote z_prompt {z.shape} (L0 mean {float((z != 0).sum(1).mean()):.1f}) to {out}")


if __name__ == "__main__":
    main()
