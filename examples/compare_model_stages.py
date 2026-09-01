#!/usr/bin/env python
"""Compose PrefScope's generic paired API for a pre/post-training experiment.

This file intentionally contains the experiment vocabulary.  The framework functions it
calls know only about aligned response side A and side B.
"""

from __future__ import annotations

import argparse
import json
from prefscope import compare_encoded_responses


def main() -> None:
    parser = argparse.ArgumentParser(
        description="compare pretrained and post-trained responses on shared prompts"
    )
    parser.add_argument(
        "--responses", required=True, help="individual-lens encoded A/B bundle"
    )
    parser.add_argument(
        "--features", required=True, help="response feature annotation directory/CSV"
    )
    parser.add_argument(
        "--prompts", default=None, help="optional aligned prompt-lens encoded bundle"
    )
    parser.add_argument("--prompt-features", default=None)
    parser.add_argument("--prompt-clusters", default=None)
    parser.add_argument("--pretrained-name", default="pretrained")
    parser.add_argument("--posttrained-name", default="posttrained")
    parser.add_argument(
        "--presence-policy",
        default="calibrated",
        choices=["calibrated", "positive_nonzero", "mixed"],
    )
    parser.add_argument(
        "--prompt-presence-policy",
        default="calibrated",
        choices=["calibrated", "positive_nonzero", "mixed"],
    )
    parser.add_argument(
        "--group-col",
        default=None,
        help="group repeated generations by their shared prompt id",
    )
    parser.add_argument("--min-context-pairs", type=int, default=30)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    result = compare_encoded_responses(
        args.responses,
        features=args.features,
        prompt_dir=args.prompts,
        prompt_features=args.prompt_features,
        prompt_clusters=args.prompt_clusters,
        side_a_name=args.pretrained_name,
        side_b_name=args.posttrained_name,
        presence_policy=args.presence_policy,
        prompt_presence_policy=args.prompt_presence_policy,
        min_context_pairs=args.min_context_pairs,
        group_col=args.group_col,
    )
    result.save(args.out)
    print(json.dumps(result.manifest, indent=2))


if __name__ == "__main__":
    main()
