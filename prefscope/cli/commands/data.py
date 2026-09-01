"""Register dataset preparation and inspection commands."""

from __future__ import annotations

from prefscope.cli.data import (
    _cmd_build_corpus,
    _cmd_init_demo,
    _cmd_inspect,
    _cmd_prepare_dataset,
)


def register_data_commands(sub) -> None:
    pi = sub.add_parser(
        "inspect", help="battle-table sanity summary (corpus or annotations)"
    )
    pi.add_argument(
        "--corpus",
        default=None,
        help="merged corpus parquet from build-corpus (label-free)",
    )
    pi.add_argument(
        "--annotations", nargs="+", default=None, help="OpenJury annotation JSON(s)"
    )
    pi.set_defaults(func=_cmd_inspect)

    pdemo = sub.add_parser(
        "init-demo",
        help="write a self-contained synthetic corpus and quickstart config",
    )
    pdemo.add_argument("--out", required=True, help="new demo workspace directory")
    pdemo.add_argument(
        "--force",
        action="store_true",
        help="replace the generated corpus/config inside a non-empty directory",
    )
    pdemo.set_defaults(func=_cmd_init_demo)

    pc = sub.add_parser(
        "build-corpus", help="build a merged label-free battle corpus from HF arenas"
    )
    pc.add_argument(
        "--source",
        nargs="+",
        required=True,
        help="arena sources: lmarena-100k lmarena-140k comparia",
    )
    pc.add_argument("--out", required=True, help="output corpus parquet")
    pc.add_argument("--split", default="train")
    pc.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap battles per source (for quick trials)",
    )
    pc.add_argument(
        "--hf-token-env",
        default="HF_TOKEN",
        dest="hf_token_env",
        help="env var holding an HF token (needed for gated comparia)",
    )
    pc.add_argument(
        "--keep-labels",
        action="store_true",
        dest="keep_labels",
        help="carry the human vote as human_pref (y=P(A preferred)) "
        "for win-relevance analysis",
    )
    pc.set_defaults(func=_cmd_build_corpus)

    ppd = sub.add_parser(
        "prepare-dataset",
        help="map a local or Hugging Face dataset into PrefScope's canonical "
        "single/pair schema",
    )
    source_group = ppd.add_mutually_exclusive_group()
    source_group.add_argument(
        "--data", default=None, help="local .parquet, .csv, .jsonl, or .json source"
    )
    source_group.add_argument(
        "--hf-dataset",
        default=None,
        dest="hf_dataset",
        help="Hugging Face dataset repository id (owner/name)",
    )
    ppd.add_argument(
        "--spec",
        "--mapping",
        default=None,
        dest="spec",
        help="reusable YAML/JSON source + column-mapping specification",
    )
    ppd.add_argument("--out", required=True, help="canonical output table")
    ppd.add_argument(
        "--hf-name",
        default=None,
        dest="hf_name",
        help="optional Hugging Face dataset configuration/subset",
    )
    ppd.add_argument("--split", default=None, help="dataset split (default: train)")
    ppd.add_argument(
        "--hf-revision",
        default=None,
        dest="hf_revision",
        help="Hub branch, tag, or commit",
    )
    ppd.add_argument(
        "--hf-token-env",
        default=None,
        dest="hf_token_env",
        help="environment variable containing a token for a gated dataset",
    )
    ppd.add_argument(
        "--streaming",
        action="store_true",
        help="stream a bounded Hugging Face sample (requires --limit)",
    )
    ppd.add_argument(
        "--limit", type=int, default=None, help="take only the first N source rows"
    )
    ppd.add_argument("--prompt-col", default=None, dest="prompt_col")
    ppd.add_argument(
        "--response-col",
        default=None,
        dest="response_col",
        help="response A / the only response",
    )
    ppd.add_argument(
        "--response-2-col",
        default=None,
        dest="response2_col",
        help="response B; canonical completion_b/response_2 is auto-detected",
    )
    ppd.add_argument(
        "--label-col",
        default=None,
        dest="label_col",
        help="optional preference label column",
    )
    ppd.add_argument("--model-col", default=None, dest="model_col")
    ppd.add_argument("--model-2-col", default=None, dest="model2_col")
    ppd.add_argument("--id-col", default=None, dest="id_col")
    ppd.add_argument("--language-col", default=None, dest="language_col")
    ppd.add_argument(
        "--keep-column", action="append", default=None, dest="keep_columns",
        help="retain a scalar metadata/rating column; repeat as needed",
    )
    ppd.add_argument(
        "--prompt-role",
        default=None,
        dest="prompt_role",
        help="structured-message selector, e.g. user:first",
    )
    ppd.add_argument(
        "--response-role",
        default=None,
        dest="response_role",
        help="structured-message selector, e.g. assistant:last",
    )
    ppd.add_argument(
        "--response-2-role",
        default=None,
        dest="response2_role",
        help="structured-message selector for response B",
    )
    ppd.add_argument(
        "--label-mode",
        choices=["probability", "winner", "a-wins"],
        default=None,
        dest="label_mode",
        help="probability=P(A), winner=map explicit tokens, a-wins=chosen/rejected layout",
    )
    ppd.add_argument(
        "--a-wins-value",
        action="append",
        default=None,
        dest="a_wins_values",
        help="winner token meaning A won; repeat or comma-separate",
    )
    ppd.add_argument(
        "--b-wins-value",
        action="append",
        default=None,
        dest="b_wins_values",
        help="winner token meaning B won; repeat or comma-separate",
    )
    ppd.add_argument(
        "--tie-value",
        action="append",
        default=None,
        dest="tie_values",
        help="winner token meaning tie; repeat or comma-separate",
    )
    ppd.add_argument(
        "--single",
        action="store_true",
        help="force single-response mode instead of auto-detecting completion_b",
    )
    ppd.add_argument(
        "--fail-on-empty",
        action="store_true",
        dest="fail_on_empty",
        help="raise instead of dropping rows with empty prompt/response text",
    )
    ppd.set_defaults(func=_cmd_prepare_dataset)



__all__ = ["register_data_commands"]
