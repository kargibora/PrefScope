"""Register high-level, config-driven workflows."""
from __future__ import annotations

from prefscope.cli.workflow import _cmd_analyze


def register_workflow_commands(sub) -> None:
    analyze = sub.add_parser(
        "analyze",
        help="apply published lenses to a local or Hugging Face dataset from one config",
    )
    analyze.add_argument("--config", required=True, help="analysis YAML/JSON config")
    analyze.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE",
        help="override any config field; repeatable (for example --set data.source.limit=1000)",
    )
    source = analyze.add_mutually_exclusive_group()
    source.add_argument("--data", default=None, help="override with a local dataset file")
    source.add_argument(
        "--hf-dataset", default=None, dest="hf_dataset",
        help="override with a Hugging Face dataset id",
    )
    analyze.add_argument("--out", default=None, help="override out_dir")
    analyze.add_argument("--repo", default=None, help="override the shared Hub lens repo")
    analyze.add_argument(
        "--completion-lens", default=None, dest="completion_lens",
        help="override with a local or hf:// completion lens",
    )
    analyze.add_argument(
        "--prompt-lens", default=None, dest="prompt_lens",
        help="override with a local or hf:// prompt lens",
    )
    analyze.add_argument(
        "--completion-subfolder", default=None, dest="completion_subfolder")
    analyze.add_argument("--prompt-subfolder", default=None, dest="prompt_subfolder")
    analyze.add_argument("--revision", default=None, help="override Hub lens revision")
    analyze.add_argument("--device", default=None, help="override cpu/cuda device")
    analyze.add_argument(
        "--presence-policy", choices=["calibrated", "positive_nonzero", "mixed"],
        default=None, dest="presence_policy",
    )
    analyze.add_argument("--top-k", type=int, default=None, dest="top_k")
    viewer = analyze.add_mutually_exclusive_group()
    viewer.add_argument("--viewer", action="store_true", dest="viewer", default=None)
    viewer.add_argument("--no-viewer", action="store_false", dest="viewer")
    analyze.add_argument(
        "--fresh", action="store_true",
        help="replace this analysis output; otherwise a matching partial run resumes",
    )
    analyze.set_defaults(func=_cmd_analyze)


__all__ = ["register_workflow_commands"]
