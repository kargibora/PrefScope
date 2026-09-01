"""Register model- and concept-level analysis commands."""

from __future__ import annotations

from prefscope.config import CONFIG
from prefscope.interpret.llm import DEFAULT_API_BASE, DEFAULT_MODEL
from prefscope.cli.analysis import (
    _cmd_associate_outcomes,
    _cmd_cluster_features,
    _cmd_conditional_delta,
    _cmd_context_profile,
    _cmd_diagnose,
    _cmd_elicit,
    _cmd_feature_relations,
    _cmd_report,
    _cmd_sae_metrics,
    _cmd_select_lens,
    _cmd_screen_confounds,
    _cmd_win_relevance,
)


def register_analysis_commands(sub) -> None:
    pctx = sub.add_parser(
        "context-profile",
        help="measure response-feature prompt scope, optionally with calibrated stability",
    )
    pctx.add_argument(
        "--completion-lens",
        required=True,
        dest="completion_lens",
        help="individual response lens containing z_a.npy and z_b.npy",
    )
    pctx.add_argument(
        "--prompt-lens",
        required=True,
        dest="prompt_lens",
        help="aligned prompt lens containing z_prompt.npy",
    )
    pctx.add_argument(
        "--calibration",
        default=None,
        help="optional feature_calibration.csv; omit for LLM-free prompt-link testing",
    )
    pctx.add_argument("--names", default=None, help="optional response feature names")
    pctx.add_argument(
        "--prompt-names",
        default=None,
        dest="prompt_names",
        help="optional prompt feature names for readable context evidence",
    )
    pctx.add_argument(
        "--prompt-fidelity",
        default=None,
        dest="prompt_fidelity",
        help="optional prompt_feature_fidelity.csv; restricts context axes",
    )
    pctx.add_argument(
        "--prompt-calibration",
        default=None,
        dest="prompt_calibration",
        help="calibrated mode: optional prompt feature semantic thresholds",
    )
    pctx.add_argument(
        "--prompt-clusters",
        default=None,
        dest="prompt_clusters",
        help="optional feature_id,cluster_id CSV; contexts remain overlapping",
    )
    pctx.add_argument(
        "--prompt-presence-policy",
        choices=["calibrated", "positive_nonzero", "mixed"],
        default="mixed",
        dest="prompt_presence_policy",
        help="calibrated mode: presence rule for overlapping prompt contexts",
    )
    pctx.add_argument("--out", required=True, help="feature_context.csv")
    pctx.add_argument(
        "--model-out",
        default=None,
        dest="model_out",
        help="model_feature_context.parquet (calibrated mode only)",
    )
    pctx.add_argument("--chunk-rows", type=int, default=50_000, dest="chunk_rows")
    pctx.add_argument(
        "--min-context-occurrences",
        type=int,
        default=10,
        dest="min_context_occurrences",
    )
    pctx.add_argument(
        "--top-n",
        type=int,
        default=100,
        dest="top_n",
        help="LLM-free mode: strongest prompt rows retained per response feature",
    )
    pctx.add_argument(
        "--min-top-examples",
        type=int,
        default=30,
        dest="min_top_examples",
        help="LLM-free mode: minimum positive-pole prompts needed to classify scope",
    )
    pctx.add_argument(
        "--prompt-tail-fractions",
        type=float,
        nargs="+",
        default=(0.005, 0.01, 0.02),
        dest="prompt_tail_fractions",
        help="LLM-free mode: high-activation prompt tails tested for stable enrichment",
    )
    pctx.add_argument(
        "--min-tail-overlap",
        type=int,
        default=5,
        dest="min_tail_overlap",
        help="LLM-free mode: minimum overlap at a prompt activation tail",
    )
    pctx.add_argument(
        "--min-link-lift",
        type=float,
        default=2.0,
        dest="min_link_lift",
        help="LLM-free mode: minimum enrichment over the prompt-tail corpus rate",
    )
    pctx.add_argument(
        "--link-q-threshold",
        type=float,
        default=0.05,
        dest="link_q_threshold",
        help="LLM-free mode: BH-adjusted enrichment threshold within each tail",
    )
    pctx.add_argument(
        "--min-link-scales",
        type=int,
        default=2,
        dest="min_link_scales",
        help="LLM-free mode: tails where the same prompt link must pass",
    )
    pctx.add_argument(
        "--min-model-context-battles",
        type=int,
        default=20,
        dest="min_model_context_battles",
    )
    pctx.add_argument(
        "--min-model-context-discordant",
        type=int,
        default=3,
        dest="min_model_context_discordant",
    )
    pctx.add_argument(
        "--min-stable-contexts", type=int, default=3, dest="min_stable_contexts"
    )
    pctx.add_argument(
        "--consistency-threshold",
        type=float,
        default=0.75,
        dest="consistency_threshold",
    )
    pctx.add_argument("--q-threshold", type=float, default=0.05, dest="q_threshold")
    pctx.add_argument(
        "--general-min-contexts", type=int, default=5, dest="general_min_contexts"
    )
    pctx.add_argument(
        "--general-max-context-share",
        type=float,
        default=0.5,
        dest="general_max_context_share",
    )
    pctx.add_argument(
        "--general-max-prompt-dependence",
        type=float,
        default=0.5,
        dest="general_max_prompt_dependence",
    )
    pctx.add_argument(
        "--min-choice-ratio", type=float, default=0.15, dest="min_choice_ratio"
    )
    pctx.add_argument(
        "--prompt-content-max-choice",
        type=float,
        default=0.15,
        dest="prompt_content_max_choice",
    )
    pctx.set_defaults(func=_cmd_context_profile)

    pd_ = sub.add_parser(
        "diagnose",
        help="aggregate a target model's contrast codes into per-feature tendencies",
    )
    pd_.add_argument("--lens-dir", required=True, help="frozen lens directory")
    pd_.add_argument(
        "--annotations",
        nargs="+",
        required=True,
        help="OpenJury annotation JSON(s) containing the target model",
    )
    pd_.add_argument("--model", required=True, help="target model name to diagnose")
    pd_.add_argument("--out", required=True, help="output diagnosis CSV")
    pd_.add_argument(
        "--battles-out",
        default=None,
        dest="battles_out",
        help="optional parquet of per-battle evidence (target vs "
        "opponent text, outcome, per-axis activation) for the viewer",
    )
    pd_.add_argument(
        "--bank",
        default=None,
        help="oriented-code bank dir (from `build-bank`); adds the "
        "inside-vs-outside Welch contrast vs the model pool and "
        "sorts by distinctiveness (delta_vs_pool)",
    )
    pd_.add_argument(
        "--fidelity",
        default=None,
        help="feature_fidelity.csv from `interpret verify`; attaches "
        "concept names and (by default) restricts to passing axes",
    )
    pd_.add_argument(
        "--win-relevance",
        default=None,
        dest="win_relevance",
        help="win-relevance CSV (from `win-relevance`); merges the global "
        "length-controlled delta_win_rate as the headline `helps_win` "
        "signal",
    )
    pd_.add_argument(
        "--all-features",
        action="store_true",
        help="diagnose every feature, not just fidelity-passing ones",
    )
    pd_.add_argument(
        "--top",
        type=int,
        default=10,
        help="how many over/under-expressed features to print",
    )
    pd_.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    pd_.add_argument(
        "--embed-model-id",
        default=None,
        dest="embed_model_id",
        help="override the embedder; defaults to the lens manifest's "
        "embed_model_id (recommended — leave unset)",
    )
    pd_.add_argument(
        "--embed-batch-size",
        type=int,
        default=CONFIG.embed_batch_size,
        dest="embed_batch_size",
    )
    pd_.add_argument(
        "--max-tokens", type=int, default=CONFIG.max_tokens, dest="max_tokens"
    )
    pd_.add_argument("--cache-dir", default=None, dest="cache_dir")
    pd_.add_argument("--cache-workers", type=int, default=32, dest="cache_workers")
    pd_.add_argument(
        "--embed-backend",
        choices=["hf", "vllm", "vllm-server"],
        default="hf",
        dest="embed_backend",
        help="MUST match the lens's embedder model (vectors are cached "
        "by model_id, so the backend can differ if cache is warm)",
    )
    pd_.add_argument(
        "--tensor-parallel-size", type=int, default=1, dest="tensor_parallel_size"
    )
    pd_.add_argument(
        "--embed-api-base",
        default=None,
        dest="embed_api_base",
        help="vllm-server: OpenAI-compatible /v1 URL",
    )
    pd_.add_argument(
        "--embed-api-key-env", default="OPENAI_API_KEY", dest="embed_api_key_env"
    )
    pd_.set_defaults(func=_cmd_diagnose)

    prp = sub.add_parser(
        "report",
        help="human-readable per-model concept report card (markdown) over the diagnosis",
    )
    prp.add_argument("--lens-dir", required=True, help="frozen lens directory")
    prp.add_argument("--model", required=True, help="target model to report on")
    prp.add_argument(
        "--annotations",
        nargs="+",
        default=None,
        help="OpenJury annotation JSON(s) containing the target model",
    )
    prp.add_argument(
        "--corpus",
        default=None,
        help="merged corpus parquet (alternative to --annotations)",
    )
    prp.add_argument(
        "--names",
        default=None,
        help="feature_fidelity/feature_names CSV (concept names; by default "
        "restricts to fidelity-passing axes)",
    )
    prp.add_argument(
        "--win-relevance",
        default=None,
        dest="win_relevance",
        help="win-relevance CSV -> surfaces preference-associated gaps (helps_win)",
    )
    prp.add_argument(
        "--prompt-lens",
        default=None,
        dest="prompt_lens",
        help="prompt lens dir (z_prompt.npy) -> adds the strong/weak "
        "prompt-types section",
    )
    prp.add_argument(
        "--prompt-names",
        default=None,
        dest="prompt_names",
        help="prompt_feature_names.csv to label prompt concepts",
    )
    prp.add_argument(
        "--bank",
        default=None,
        help="oriented-code bank dir (build-bank); under-expression "
        "measured vs the pool (delta_vs_pool)",
    )
    prp.add_argument("--out", required=True, help="output markdown report path")
    prp.add_argument(
        "--top", type=int, default=15, help="how many concepts to list per section"
    )
    prp.add_argument(
        "--min-battles",
        type=int,
        default=20,
        dest="min_battles",
        help="min battles per prompt concept for the prompt-types section",
    )
    prp.add_argument(
        "--min-prompt-activation",
        type=float,
        default=0.0,
        dest="min_prompt_activation",
        help="raw positive activation required for prompt-concept membership",
    )
    prp.add_argument(
        "--all-features",
        action="store_true",
        help="report on every feature, not just fidelity-passing ones",
    )
    prp.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    prp.add_argument(
        "--embed-model-id",
        default=None,
        dest="embed_model_id",
        help="override the embedder; defaults to the lens manifest's value",
    )
    prp.add_argument(
        "--embed-batch-size",
        type=int,
        default=CONFIG.embed_batch_size,
        dest="embed_batch_size",
    )
    prp.add_argument(
        "--max-tokens", type=int, default=CONFIG.max_tokens, dest="max_tokens"
    )
    prp.add_argument("--cache-dir", default=None, dest="cache_dir")
    prp.add_argument("--cache-workers", type=int, default=32, dest="cache_workers")
    prp.add_argument(
        "--embed-backend",
        choices=["hf", "vllm", "vllm-server"],
        default="hf",
        dest="embed_backend",
    )
    prp.add_argument(
        "--tensor-parallel-size", type=int, default=1, dest="tensor_parallel_size"
    )
    prp.add_argument("--embed-api-base", default=None, dest="embed_api_base")
    prp.add_argument(
        "--embed-api-key-env", default="OPENAI_API_KEY", dest="embed_api_key_env"
    )
    prp.set_defaults(func=_cmd_report)

    pw = sub.add_parser(
        "win-relevance",
        help="features associated with preference (activation vs human_pref)",
    )
    pw.add_argument("--lens-dir", default=None)
    pw.add_argument(
        "--corpus",
        default=None,
        help="corpus parquet WITH human_pref (build-corpus --keep-labels)",
    )
    pw.add_argument(
        "--encoded-dir",
        default=None,
        dest="encoded_dir",
        help="codes bundle from encode-dataset (z_diff.npy + canonical meta.parquet); "
        "alternative to --lens-dir/--corpus",
    )
    pw.add_argument(
        "--names",
        default=None,
        help="feature_names/fidelity CSV to attach concepts + filter",
    )
    pw.add_argument(
        "--all-features",
        action="store_true",
        help="score every feature, not just fidelity-passing ones",
    )
    pw.add_argument(
        "--clusters",
        default=None,
        help="feature_clusters.csv -> ALSO emit cluster-level win-relevance "
        "(<out>_clusters.csv): Anatomy-style per-behavior Δwin-rate",
    )
    pw.add_argument(
        "--group-col",
        default=None,
        dest="group_col",
        help="independent prompt-group column; defaults to group_id or a stable prompt hash",
    )
    pw.add_argument("--out", required=True, help="output win-relevance CSV")
    pw.set_defaults(func=_cmd_win_relevance)

    pout = sub.add_parser(
        "associate-outcomes",
        help="associate sparse concepts with binary, preference, continuous, or "
        "multi-attribute outcomes",
    )
    pout.add_argument(
        "--encoded-dir", required=True, dest="encoded_dir",
        help="encode-dataset bundle containing meta.parquet and sparse code arrays",
    )
    pout.add_argument(
        "--outcome-col", required=True, action="append", dest="outcome_col",
        help="metadata outcome column; repeat for multi_continuous",
    )
    pout.add_argument(
        "--outcome-kind", required=True,
        choices=["binary", "probability", "preference", "continuous", "multi_continuous"],
        dest="outcome_kind",
    )
    pout.add_argument(
        "--normalization", choices=["auto", "none", "zscore"], default="auto",
    )
    pout.add_argument(
        "--code-array", choices=["auto", "z_a", "z_diff", "z_prompt"], default="auto",
        dest="code_array",
    )
    pout.add_argument("--names", default=None, help="optional feature names CSV")
    pout.add_argument(
        "--group-col", default=None, dest="group_col",
        help="independent group column; defaults to group_id or a stable prompt hash",
    )
    pout.add_argument(
        "--no-grouping", action="store_true", dest="no_grouping",
        help="use row-level descriptive associations even when prompt groups are available",
    )
    pout.add_argument("--min-units", type=int, default=3, dest="min_units")
    pout.add_argument("--out", required=True, help="output long-form association CSV")
    pout.set_defaults(func=_cmd_associate_outcomes)

    psc = sub.add_parser(
        "screen-confounds",
        help="screen preference-associated response concepts for length entanglement",
    )
    psc.add_argument(
        "--lens-dir", required=True, help="completion lens containing z_diff.npy"
    )
    psc.add_argument(
        "--corpus",
        required=True,
        help="aligned corpus with human_pref and both completions",
    )
    psc.add_argument(
        "--names", default=None, help="optional feature annotation CSV to attach"
    )
    psc.add_argument("--out", required=True, help="output confound-screen CSV")
    psc.add_argument(
        "--confound-threshold",
        type=float,
        default=0.3,
        dest="confound_threshold",
        help="minimum absolute feature/length correlation (default: 0.3)",
    )
    psc.add_argument(
        "--collapse-fraction",
        type=float,
        default=0.5,
        dest="collapse_fraction",
        help="maximum residual/original outcome-correlation ratio (default: 0.5)",
    )
    psc.add_argument(
        "--permute",
        type=int,
        default=0,
        metavar="N",
        help="optional human_pref permutation-null repetitions",
    )
    psc.add_argument("--seed", type=int, default=0)
    psc.add_argument(
        "--group-col", default=None, dest="group_col",
        help="independent prompt-group column; defaults to group_id or prompt hash",
    )
    psc.set_defaults(func=_cmd_screen_confounds)

    pel = sub.add_parser(
        "elicit",
        help="prompt-concept -> response-concept co-activation lift (preference-independent): "
        "which response concepts appear when a prompt concept is present",
    )
    pel.add_argument(
        "--completion-lens",
        required=True,
        dest="completion_lens",
        help="individual lens dir (z_a.npy; z_b.npy optional for paired data)",
    )
    pel.add_argument(
        "--prompt-lens",
        required=True,
        dest="prompt_lens",
        help="prompt lens dir (z_prompt.npy)",
    )
    pel.add_argument(
        "--completion-names",
        default=None,
        dest="completion_names",
        help="feature_names.csv (response concepts)",
    )
    pel.add_argument(
        "--completion-fidelity",
        default=None,
        dest="completion_fidelity",
        help="feature_fidelity.csv -> restrict to verified response axes",
    )
    pel.add_argument(
        "--prompt-names",
        default=None,
        dest="prompt_names",
        help="prompt_feature_names.csv",
    )
    pel.add_argument(
        "--prompt-fidelity",
        default=None,
        dest="prompt_fidelity",
        help="prompt_feature_fidelity.csv -> restrict to verified prompt axes",
    )
    pel.add_argument(
        "--min-support",
        type=int,
        default=30,
        dest="min_support",
        help="min responses where the prompt feature fires to test a cell",
    )
    pel.add_argument(
        "--min-cooccur",
        type=int,
        default=5,
        dest="min_cooccur",
        help="min co-occurrences to test a cell",
    )
    pel.add_argument(
        "--group-col",
        default=None,
        dest="group_col",
        help="independent prompt-group column in completion-lens battles metadata",
    )
    pel.add_argument("--out", required=True, help="output elicitation CSV")
    pel.set_defaults(func=_cmd_elicit)

    pcd = sub.add_parser(
        "conditional-delta",
        help="prompt-conditioned completion delta Δ_{k,f} (which response properties "
        "distinguish the winner per prompt type) + optional conditional δ_{f,k}",
    )
    pcd.add_argument(
        "--completion-lens",
        required=True,
        dest="completion_lens",
        help="completion lens dir (z_diff.npy)",
    )
    pcd.add_argument(
        "--prompt-lens",
        required=True,
        dest="prompt_lens",
        help="prompt lens dir (z_prompt.npy)",
    )
    pcd.add_argument(
        "--corpus",
        default=None,
        help="corpus WITH human_pref — orients z_diff toward the winner "
        "(without it Δ ~ 0; required for --conditional-out)",
    )
    pcd.add_argument("--completion-names", default=None, dest="completion_names")
    pcd.add_argument("--prompt-names", default=None, dest="prompt_names")
    pcd.add_argument(
        "--prompt-clusters",
        default=None,
        dest="prompt_clusters",
        help="prompt_feature_clusters.csv -> condition on prompt CLUSTERS "
        "(fewer Bonferroni tests, more power) instead of raw concepts",
    )
    pcd.add_argument(
        "--conditional-out",
        default=None,
        dest="conditional_out",
        help="ALSO emit the length-controlled conditional win-rate δ_{f,k}",
    )
    pcd.add_argument(
        "--completion-fidelity",
        default=None,
        dest="completion_fidelity",
        help="feature_fidelity.csv -> restrict the conditional table to verified axes",
    )
    pcd.add_argument(
        "--prompt-fidelity",
        default=None,
        dest="prompt_fidelity",
        help="prompt_feature_fidelity.csv -> use only verified prompt axes",
    )
    pcd.add_argument(
        "--min-prompt-activation",
        type=float,
        default=0.0,
        dest="min_prompt_activation",
        help="raw positive activation required for prompt membership",
    )
    pcd.add_argument(
        "--min-prompt-support",
        type=int,
        default=30,
        dest="min_prompt_support",
        help="minimum battles containing a prompt concept/cluster",
    )
    pcd.add_argument("--seed", type=int, default=0)
    pcd.add_argument(
        "--group-col", default=None, dest="group_col",
        help="independent prompt-group column; defaults to group_id or prompt hash",
    )
    pcd.add_argument(
        "--permute",
        type=int,
        default=0,
        metavar="N",
        help="label-permutation null: shuffle prompt-concept labels N times",
    )
    pcd.add_argument(
        "--jobs",
        type=int,
        default=1,
        metavar="N",
        help="parallelize the permutation null across N processes",
    )
    pcd.add_argument("--out", required=True, help="output Δ_{k,f} CSV")
    pcd.set_defaults(func=_cmd_conditional_delta)

    psm = sub.add_parser(
        "sae-metrics",
        help="redundancy + fit health metrics for a lens (decoder cosine, MI, FVU, "
        "dead-frac, L0) — track across an M sweep. NOT an absorption score.",
    )
    psm.add_argument("--lens-dir", required=True)
    psm.add_argument(
        "--out",
        default=None,
        help="CSV to append a metrics row to (for M-sweep tables)",
    )
    psm.set_defaults(func=_cmd_sae_metrics)

    psel = sub.add_parser(
        "select-lens",
        help="pick a width/sparsity from a sae-metrics sweep CSV: rejects dead, "
        "duplicated, and memorisation-prone configurations, then takes the best fit",
    )
    psel.add_argument("--sweep", required=True,
                      help="CSV written by repeated `sae-metrics --out`")
    psel.add_argument("--n-rows", type=int, default=None,
                      help="training rows, to bound width by rows-per-feature")
    psel.add_argument("--input-dim", type=int, default=None,
                      help="representation dimension, to report the expansion ratio")
    psel.add_argument("--out", default=None, help="annotated sweep table CSV")
    psel.set_defaults(func=_cmd_select_lens)

    pfr = sub.add_parser(
        "feature-relations",
        help="find duplicate, specialized, coactive, and same-name feature axes "
        "without merging them",
    )
    pfr.add_argument("--lens-dir", required=True)
    pfr.add_argument(
        "--names",
        default=None,
        help="feature names/fidelity CSV; enables label-collision diagnostics",
    )
    pfr.add_argument(
        "--lens-kind",
        choices=["completion", "prompt"],
        default="completion",
        dest="lens_kind",
    )
    pfr.add_argument(
        "--cluster-on",
        choices=["difference", "individual"],
        default="individual",
        dest="cluster_on",
        help="completion relationship space; individual is recommended",
    )
    pfr.add_argument(
        "--cofire-pole",
        choices=["positive", "negative", "nonzero"],
        default="positive",
        dest="cofire_pole",
    )
    pfr.add_argument("--min-cooccur", type=int, default=30, dest="min_cooccur")
    pfr.add_argument("--min-jaccard", type=float, default=0.05, dest="min_jaccard")
    pfr.add_argument(
        "--min-containment", type=float, default=0.50, dest="min_containment"
    )
    pfr.add_argument("--min-phi", type=float, default=0.05, dest="min_phi")
    pfr.add_argument("--min-lift", type=float, default=1.50, dest="min_lift")
    pfr.add_argument(
        "--min-name-similarity", type=float, default=0.80,
        dest="min_name_similarity",
    )
    pfr.add_argument(
        "--min-decoder-cosine", type=float, default=0.70,
        dest="min_decoder_cosine",
    )
    pfr.add_argument(
        "--fidelity-only", action="store_true", dest="fidelity_only",
        help="restrict relations to fidelity-passing axes",
    )
    pfr.add_argument(
        "--no-decoder", action="store_true", dest="no_decoder",
        help="skip decoder-direction similarity (torch is then unnecessary)",
    )
    pfr.add_argument("--out", required=True, help="output feature relationships CSV")
    pfr.set_defaults(func=_cmd_feature_relations)

    pcl = sub.add_parser(
        "cluster-features",
        help="group co-activating SAE features into corpus-specific groups",
    )
    pcl.add_argument("--lens-dir", required=True, help="lens dir with z_diff.npy")
    pcl.add_argument(
        "--names",
        default=None,
        help="feature_fidelity/feature_names CSV (concepts + fidelity)",
    )
    pcl.add_argument("--n-clusters", type=int, default=10, dest="n_clusters")
    pcl.add_argument(
        "--method",
        default="spherical-kmeans",
        help="clusterer component (built-in: cofire-leiden, mi-leiden, "
        "spherical-kmeans, agglomerative; or any registered). "
        "cofire-leiden is recommended for interpreted signed features.",
    )
    pcl.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="mi-leiden resolution (higher -> more, smaller communities)",
    )
    pcl.add_argument(
        "--knn",
        type=int,
        default=0,
        help="Leiden: sparsify graph to each feature's top-knn edges (0=dense)",
    )
    pcl.add_argument(
        "--knn-mode",
        choices=["mutual", "union"],
        default="mutual",
        dest="knn_mode",
        help="cofire-leiden: retain mutual neighbors (recommended) or the "
        "symmetric union of one-sided neighbors",
    )
    pcl.add_argument(
        "--min-cluster-size",
        type=int,
        default=1,
        dest="min_cluster_size",
        help="community-size diagnostic threshold; cofire-leiden preserves "
        "small communities instead of merging unrelated features",
    )
    pcl.add_argument(
        "--small-community-policy",
        choices=["preserve", "merge"],
        default="preserve",
        dest="small_community_policy",
        help="preserve is safe; merge exists only to reproduce legacy artifacts",
    )
    pcl.add_argument(
        "--affinity-metric",
        choices=["phi", "cosine", "npmi"],
        default="phi",
        dest="affinity_metric",
        help="cofire-leiden positive co-presence edge weight",
    )
    pcl.add_argument(
        "--cofire-pole",
        choices=["positive", "negative", "nonzero"],
        default="positive",
        dest="cofire_pole",
        help="which signed activation pole counts as concept presence",
    )
    pcl.add_argument(
        "--min-cooccur",
        type=int,
        default=30,
        dest="min_cooccur",
        help="minimum co-firing support for a graph edge",
    )
    pcl.add_argument(
        "--stability-runs",
        type=int,
        default=5,
        dest="stability_runs",
        help="Leiden seeds used for adjusted-Rand stability diagnostics",
    )
    pcl.add_argument(
        "--super-resolution",
        type=float,
        default=None,
        dest="super_resolution",
        help="optional second Leiden pass over fine communities",
    )
    pcl.add_argument(
        "--super-knn",
        type=int,
        default=4,
        dest="super_knn",
        help="mutual-kNN sparsity for optional superclusters",
    )
    pcl.add_argument(
        "--fidelity-only",
        action="store_true",
        dest="fidelity_only",
        help="cluster only fidelity-passing features",
    )
    pcl.add_argument(
        "--cluster-on",
        choices=["difference", "individual"],
        default="difference",
        dest="cluster_on",
        help="co-firing space: 'difference' (z_diff, default) or 'individual' "
        "(z_a/z_b stacked — Anatomy-style semantic co-occurrence; avoids "
        "merging antonym features that co-fire only in the contrast)",
    )
    pcl.add_argument(
        "--lens-kind",
        choices=["completion", "prompt"],
        default="completion",
        dest="lens_kind",
        help="'prompt' clusters z_prompt.npy on a prompt lens (folds the old "
        "cluster_prompts.py); 'completion' (default) uses --cluster-on",
    )
    pcl.add_argument(
        "--name-clusters",
        action="store_true",
        dest="name_clusters",
        help="LLM-name each behavior from its member concepts",
    )
    pcl.add_argument(
        "--backend", choices=["openai", "claude-cli", "codex-cli"], default="openai"
    )
    pcl.add_argument("--model", default=DEFAULT_MODEL)
    pcl.add_argument("--api-base", default=DEFAULT_API_BASE)
    pcl.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    pcl.add_argument("--concurrency", type=int, default=1)
    pcl.add_argument("--out", required=True, help="output feature_clusters.csv")
    pcl.set_defaults(func=_cmd_cluster_features)



__all__ = ["register_analysis_commands"]
