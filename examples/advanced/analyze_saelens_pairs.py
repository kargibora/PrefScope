"""Featurize a paired preference table with any configured Lens backend."""

from __future__ import annotations

import argparse
from pathlib import Path

from prefscope import (
    Lens,
    TableDataset,
    save_feature_batch,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lens-config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prompt-col", default="prompt")
    parser.add_argument("--response-a-col", default="response_a")
    parser.add_argument("--response-b-col", default="response_b")
    parser.add_argument("--preference-col", default="preference")
    parser.add_argument("--item-id-col", default=None)
    parser.add_argument("--group-id-col", default=None)
    parser.add_argument(
        "--feature-id",
        action="append",
        type=int,
        dest="feature_ids",
        help="Retain one feature ID; repeat the option for multiple IDs.",
    )
    args = parser.parse_args()

    items = TableDataset(
        args.data,
        prompt=args.prompt_col,
        a=args.response_a_col,
        b=args.response_b_col,
        pref=args.preference_col,
        id=args.item_id_col,
        group_id=args.group_id_col,
    )
    lens = Lens.from_config(args.lens_config)
    features = lens.featurize(items, feature_ids=args.feature_ids)
    out = save_feature_batch(features, args.out)
    relevance = lens.preference_relevance(
        features,
        group_column="group_id" if args.group_id_col else None,
    )
    relevance_path = Path(out).with_name(f"{Path(out).name}-preference.csv")
    relevance.to_csv(relevance_path, index=False)
    print(
        f"saved {len(features.row_ids)} rows and {len(features.feature_ids)} features"
    )
    print(f"feature bundle: {out}")
    print(f"preference analysis: {relevance_path}")


if __name__ == "__main__":
    main()
