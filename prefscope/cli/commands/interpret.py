"""Register config-driven and individual interpretation commands."""

from __future__ import annotations

from prefscope.interpret.llm import DEFAULT_API_BASE, DEFAULT_MODEL
from prefscope.cli.interpret import (
    _cmd_interpret_calibrate_presence,
    _cmd_interpret_classify_role,
    _cmd_interpret_name,
    _cmd_interpret_verify,
    _cmd_name_prompts,
    _cmd_run,
)


def register_interpret_commands(sub) -> None:
    pnp = sub.add_parser(
        "name-prompts",
        help="compatibility alias for `interpret name --lens-kind prompt`",
    )
    pnp.add_argument("--lens-dir", required=True, help="prompt lens dir (z_prompt.npy)")
    pnp.add_argument(
        "--corpus", required=True, help="corpus parquet (prompt text by battle_id)"
    )
    pnp.add_argument("--out", required=True, help="output prompt_feature_names.csv")
    pnp.add_argument("--features", type=int, nargs="*", default=None)
    pnp.add_argument("--n-active", type=int, default=10, dest="n_active")
    pnp.add_argument("--n-zero", type=int, default=10, dest="n_zero")
    pnp.add_argument(
        "--backend", choices=["openai", "claude-cli", "codex-cli"], default="openai"
    )
    pnp.add_argument("--model", default=DEFAULT_MODEL)
    pnp.add_argument("--api-base", default=DEFAULT_API_BASE)
    pnp.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    pnp.add_argument("--max-tokens", type=int, default=2000, dest="max_tokens")
    pnp.add_argument("--concurrency", type=int, default=1)
    pnp.add_argument("--verify-frac", type=float, default=0.2, dest="verify_frac")
    pnp.add_argument("--seed", type=int, default=0)
    pnp.add_argument("--negatives", choices=["random", "close"], default="random")
    pnp.add_argument("--debug-responses", default=None, dest="debug_responses")
    pnp.add_argument(
        "--pole",
        choices=["positive", "negative"],
        default=None,
        help="name the positive or negative pole of a legacy signed prompt lens",
    )
    pnp.add_argument(
        "--reasoning-effort",
        default=None,
        dest="reasoning_effort",
        choices=["none", "minimal", "low", "medium", "high"],
        help="disable reasoning (none) or curb thinking tokens (minimal/low) to avoid "
        "truncation and cut cost on this simple task",
    )
    prompt_restart = pnp.add_mutually_exclusive_group()
    prompt_restart.add_argument(
        "--resume",
        dest="fresh",
        action="store_false",
        help="resume matching rows already checkpointed at --out (default)",
    )
    prompt_restart.add_argument(
        "--fresh",
        dest="fresh",
        action="store_true",
        help="discard the prior output/checkpoint/usage ledger and start from scratch",
    )
    pnp.set_defaults(fresh=False)
    pnp.set_defaults(func=_cmd_name_prompts)

    prun = sub.add_parser(
        "run",
        help="run a config-driven pipeline (name/verify/cluster/win-relevance) from a "
        "YAML/JSON file; every component is selected by name + params in the config",
    )
    prun.add_argument(
        "--config", required=True, help="pipeline config (.yaml/.yml/.json)"
    )
    prun.set_defaults(func=_cmd_run)

    pn = sub.add_parser(
        "interpret",
        help="run individual naming, verification, or calibration stages",
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
            "stratified-random",
        ],
        default="extremes",
        help="activation cases to verify: strongest activations (default), uniform random "
        "nonzero activations, or activation-quantile-stratified cases; "
        "stratified-random is a deprecated alias for random-active",
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
        help="prompt verify: 'random' silent prompts or 'close' (needs --embeddings)",
    )
    pnv.add_argument(
        "--embeddings",
        default=None,
        help="deprecated compatibility flag; prompt 'close' controls now use "
        "the aligned prompt SAE code space (no raw embedding matrix needed)",
    )
    pnv.set_defaults(func=_cmd_interpret_verify)

    pcal = isub.add_parser(
        "calibrate-presence",
        help="learn per-feature thresholds for corpus-level semantic presence",
    )
    _add_common(pcal)
    pcal.add_argument("--names", required=True, help="feature_names.csv from `name`")
    pcal.add_argument(
        "--fidelity",
        default=None,
        help="feature_fidelity.csv; by default calibrate only passing names",
    )
    pcal.add_argument(
        "--all-named",
        action="store_true",
        help="calibrate all non-abstained names, including fidelity failures",
    )
    pcal.add_argument("--features", type=int, nargs="*", default=None)
    pcal.add_argument(
        "--lens-kind",
        choices=["completion", "prompt"],
        default="completion",
        dest="lens_kind",
    )
    pcal.add_argument(
        "--pole",
        choices=["positive"],
        default=None,
        help="required acknowledgement for a signed lens",
    )
    pcal.add_argument(
        "--n-per-bin",
        type=int,
        default=4,
        dest="n_per_bin",
        help="labels sampled from each non-top positive activation stratum",
    )
    pcal.add_argument(
        "--n-top",
        type=int,
        default=20,
        dest="n_top",
        help="labels sampled from the top one-percent activation stratum",
    )
    pcal.add_argument(
        "--n-zero",
        type=int,
        default=10,
        dest="n_zero",
        help="silent controls used to detect concept undercoverage",
    )
    pcal.add_argument(
        "--batch-size",
        type=int,
        default=8,
        dest="batch_size",
        help="examples labelled in one LLM request",
    )
    pcal.add_argument(
        "--target-precision",
        type=float,
        default=0.8,
        dest="target_precision",
        help="minimum Wilson lower bound required above the learned threshold",
    )
    pcal.add_argument(
        "--min-above",
        type=int,
        default=20,
        dest="min_above",
        help="minimum labelled active examples supporting a threshold",
    )
    pcal.add_argument(
        "--max-silent-rate",
        type=float,
        default=0.2,
        dest="max_silent_rate",
        help="maximum concept rate in z=0 controls for presence_pass",
    )
    pcal.set_defaults(func=_cmd_interpret_calibrate_presence)

    prole = isub.add_parser(
        "classify-role",
        help="experimentally classify named response features as behavioral or specific",
    )
    _add_common(prole)
    prole.add_argument(
        "--names",
        required=True,
        help="feature_fidelity.csv (or another feature_id,concept annotation table)",
    )
    prole.add_argument(
        "--linkage",
        default=None,
        help="optional feature_prompt_linkage.csv for conservative combined scope",
    )
    prole.add_argument("--features", type=int, nargs="*", default=None)
    prole.add_argument(
        "--all-named",
        action="store_true",
        help="include names without fidelity_pass=True",
    )
    prole.add_argument(
        "--pole",
        choices=["positive"],
        default=None,
        help="required acknowledgement for a signed individual-response lens",
    )
    prole.add_argument(
        "--n-top",
        type=int,
        default=6,
        dest="n_top",
        help="strongest unique-prompt response examples per feature",
    )
    prole.add_argument(
        "--n-random",
        type=int,
        default=2,
        dest="n_random",
        help="additional random nonzero response examples per feature",
    )
    prole.add_argument(
        "--min-valid-examples",
        type=int,
        default=4,
        dest="min_valid_examples",
        help="minimum examples where the named concept is observed",
    )
    prole.add_argument(
        "--batch-size",
        type=int,
        default=4,
        dest="batch_size",
        help="evidence examples per LLM request; incomplete batches are retried once",
    )
    prole.set_defaults(func=_cmd_interpret_classify_role)



__all__ = ["register_interpret_commands"]
