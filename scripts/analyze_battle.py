#!/usr/bin/env python
"""Analyze how two responses to one prompt differ, through a trained lens.

Loads a frozen PrefScope lens (the SAE encoder + its interpreted feature axes),
embeds (prompt, response_A) and (prompt, response_B), projects the A-minus-B
contrast through the SAE, and prints the concept axes on which the two responses
differ most. A positive score (z > 0) means response A expresses that concept
more than response B; negative means B expresses it more.

This is a thin driver over the framework: it just loads a `LoadedLens`, wraps
your inputs in a one-battle `Dataset`, and calls `lens.project(...)`.

Examples
--------
    python scripts/analyze_battle.py --lens-dir lenses/arena_m32 \\
        --prompt "Explain entropy to a 10-year-old." \\
        --response-a "Entropy is how messy things get ..." \\
        --response-b "Entropy is the logarithm of the number of microstates ..." \\
        --top 15 --device cuda

    # read any field from a file instead (handy for long responses):
    python scripts/analyze_battle.py --lens-dir lenses/arena_m32 \\
        --prompt-file prompt.txt --response-a-file a.txt --response-b-file b.txt
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# scripts/ runs as a loose file (the package isn't installed); the repo root is
# one level up — add it so `import prefscope` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prefscope.api.loaded_lens import LoadedLens  # noqa: E402
from prefscope.core.dataset import Dataset  # noqa: E402
from prefscope.core.types import PairItem  # noqa: E402

log = logging.getLogger(__name__)


class _SingleBattle(Dataset):
    """A one-item dataset: a single custom (prompt, A, B) battle."""

    def __init__(self, prompt: str, response_a: str, response_b: str) -> None:
        self._item = PairItem(id="custom", x=prompt, y_a=response_a, y_b=response_b)

    def __iter__(self):
        yield self._item


def _resolve(value: str | None, path: str | None, what: str) -> str:
    if path:
        return Path(path).read_text()
    if value:
        return value
    raise SystemExit(f"error: missing {what} (pass --{what} or --{what}-file)")


def _concept(names, feature_id: int) -> str:
    if names is not None and "concept" in names.columns:
        hit = names.loc[names["feature_id"] == feature_id, "concept"]
        if len(hit) and isinstance(hit.iloc[0], str):
            return hit.iloc[0]
    return f"(unnamed feature {feature_id})"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lens-dir", required=True,
                    help="trained lens directory (sae_model.pt + manifest.json [+ feature_names.csv])")
    ap.add_argument("--prompt"); ap.add_argument("--prompt-file")
    ap.add_argument("--response-a"); ap.add_argument("--response-a-file")
    ap.add_argument("--response-b"); ap.add_argument("--response-b-file")
    ap.add_argument("--top", type=int, default=15, help="how many differing axes to show")
    ap.add_argument("--device", default="cpu", help="cpu or cuda (for the embedding model)")
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, datefmt="%H:%M:%S",
        format="%(asctime)s %(levelname)s %(name)s | %(message)s")

    prompt = _resolve(args.prompt, args.prompt_file, "prompt")
    response_a = _resolve(args.response_a, args.response_a_file, "response-a")
    response_b = _resolve(args.response_b, args.response_b_file, "response-b")

    log.info("loading lens from %s (device=%s) and projecting the battle",
             args.lens_dir, args.device)
    lens = LoadedLens.from_dir(args.lens_dir, device=args.device)
    codes, _ = lens.project(_SingleBattle(prompt, response_a, response_b))   # (1, M)
    z = codes[0]

    order = np.argsort(np.abs(z))[::-1]
    fired = [int(i) for i in order if z[i] != 0][: args.top]
    if not fired:
        print("No feature fired on this contrast — the lens sees the two responses "
              "as effectively identical along its concept axes.")
        return

    print(f"\nTop {len(fired)} axes where the two responses differ "
          f"(z>0 = A expresses it more, z<0 = B):\n")
    for fid in fired:
        side = "A" if z[fid] > 0 else "B"
        print(f"  {z[fid]:+8.3f}  [{side} more]  {_concept(lens.names, fid)}")
    print()


if __name__ == "__main__":
    main()
