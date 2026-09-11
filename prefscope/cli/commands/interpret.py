"""Register config-driven and individual interpretation commands."""

from __future__ import annotations

from prefscope.interpret.llm import DEFAULT_API_BASE, DEFAULT_MODEL
from prefscope.cli.interpret import (
    _cmd_interpret_name,
    _cmd_interpret_verify,
)


def register_interpret_commands(sub) -> None:
    pn = sub.add_parser(
        "interpret",
        help="run feature naming or verification",
    )
    isub = pn.add_subparsers(dest="interpret_command", required=True)

    def _add_common(p):
        p.add_argument("--lens-dir", required=True)
        p.add_argument(
            "--annotations",
            nargs="+",
            default=None,
            help="annotation JSON(s) the lens was built from",
        )
        p.add_argument(
            "--corpus",
            default=None,
            help="merged corpus parquet the lens was built from",
        )
        p.add_argument("--out", required=True)
        p.add_argument(
            "--backend", choices=["openai", "claude-cli", "codex-cli"], default="openai"
        )
        p.add_argument("--model", default=DEFAULT_MODEL)
        p.add_argument(
            "--api-base",
            default=DEFAULT_API_BASE,
            help="OpenAI-compatible base URL (OpenRouter default; "
            "set to a local vLLM endpoint to run offline)",
        )
        p.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
        p.add_argument(
            "--max-tokens",
            type=int,
            default=2000,
            dest="max_tokens",
            help="maximum output-token budget per interpretation request",
        )
        p.add_argument("--verify-frac", type=float, default=0.2, dest="verify_frac")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument(
            "--concurrency",
            type=int,
            default=1,
            help="number of features to send to the LLM in parallel "
            "(thread pool; 1 = sequential)",
        )
        p.add_argument(
            "--reasoning-effort",
            default=None,
            dest="reasoning_effort",
            choices=["none", "minimal", "low", "medium", "high"],
            help="disable reasoning with 'none', or reduce it with minimal/low; "
            "omit to use the provider default",
        )
        restart = p.add_mutually_exclusive_group()
        restart.add_argument(
            "--resume",
            dest="fresh",
            action="store_false",
            help="resume matching rows already checkpointed at --out (default)",
        )
        restart.add_argument(
            "--fresh",
            dest="fresh",
            action="store_true",
            help="discard the prior output/checkpoint/usage ledger and start from scratch",
        )
        p.set_defaults(fresh=False)

    pnn = isub.add_parser("name", help="label each feature from top pairs")
    _add_common(pnn)
    pnn.add_argument("--features", type=int, nargs="*", default=None)
    pnn.add_argument(
        "--lens-kind",
        choices=["completion", "prompt"],
        default="completion",
        dest="lens_kind",
        help="completion lens (default) or prompt lens (reads z_prompt.npy and needs --corpus)",
    )
    pnn.add_argument(
        "--name-mode",
        default="auto",
        dest="name_mode",
        help="interpreter strategy: auto (default) picks individual vs pairwise "
        "from the lens manifest's input_rep (and single-text for prompt "
        "lenses); or name any registered strategy (built-in: individual, "
        "pairwise, single-text). Unknown -> error lists all.",
    )
    pnn.add_argument("--n-active", type=int, default=10, dest="n_active")
    pnn.add_argument("--n-zero", type=int, default=10, dest="n_zero")
    pnn.add_argument(
        "--pole",
        choices=["positive", "negative"],
        default=None,
        help="name one pole of a signed lens (negative is supported for prompt lenses)",
    )
    pnn.add_argument(
        "--negatives",
        choices=["random", "close"],
        default="random",
        help="non-activating controls: 'random' silent responses (default) or "
        "'close' HARD negatives — silent responses whose other concepts "
        "resemble the activators, so the name isolates THIS feature instead "
        "of a generic trait (e.g. formatting) the controls also share",
    )
    pnn.add_argument(
        "--abbreviate",
        action="store_true",
        help="run the WIMHF abbreviate-concept step",
    )
    pnn.add_argument(
        "--debug-responses",
        default=None,
        dest="debug_responses",
        help="dir to dump each feature's raw LLM response (feature_<id>.txt) "
        "for debugging empty/garbage concepts",
    )
    pnn.set_defaults(func=_cmd_interpret_name)

    pnv = isub.add_parser("verify", help="held-out fidelity of named axes")
    _add_common(pnv)
    pnv.add_argument("--names", required=True, help="feature_names.csv from `name`")
    pnv.add_argument(
        "--features",
        type=int,
        nargs="*",
        default=None,
        help="optional feature IDs; by default verify every row in --names",
    )
    pnv.add_argument(
        "--verify-mode",
        default="auto",
        dest="verify_mode",
        help="verifier strategy: auto (default) picks individual vs pairwise from "
        "the lens manifest's input_rep; or name any registered strategy "
        "(built-in: individual, pairwise, prompt). --lens-kind prompt forces prompt.",
    )
    pnv.add_argument("--n-per-bucket", type=int, default=10, dest="n_per_bucket")
    pnv.add_argument(
        "--sampling",
        choices=[
            "extremes",
            "random-active",
            "quantile-stratified",
        ],
        default="extremes",
        help="activation cases to verify: strongest activations (default), uniform random "
        "nonzero activations, or activation-quantile-stratified cases",
    )
    pnv.add_argument(
        "--n-examples",
        type=int,
        default=None,
        dest="n_examples",
        help="total per-feature label budget; overrides --n-per-bucket allocation",
    )
    pnv.add_argument(
        "--min-success-rate",
        type=float,
        default=0.8,
        dest="min_success_rate",
        help="minimum fraction of parseable verifier responses",
    )
    pnv.add_argument(
        "--min-bucket",
        type=int,
        default=5,
        dest="min_bucket",
        help="minimum successfully labelled examples in each required bucket",
    )
    pnv.add_argument(
        "--pole",
        choices=["positive"],
        default=None,
        help="acknowledge positive-pole-only verification of a signed "
        "individual or prompt lens",
    )
    pnv.add_argument(
        "--fidelity-threshold",
        type=float,
        default=0.3,
        dest="fidelity_threshold",
        help="min POSITIVE correlation to pass (with Bonferroni p<0.05); a "
        "flipped-polarity name (negative correlation) fails",
    )
    pnv.add_argument(
        "--lens-kind",
        choices=["completion", "prompt"],
        default="completion",
        dest="lens_kind",
        help="'prompt' verifies prompt-lens concepts on z_prompt + prompt text "
        "(needs --corpus; folds the old verify_prompts.py)",
    )
    pnv.add_argument(
        "--negatives",
        default="random",
        help="prompt verify: 'random' silent prompts or 'close' controls",
    )
    pnv.set_defaults(func=_cmd_interpret_verify)



__all__ = ["register_interpret_commands"]
