"""Register lens building, packaging, encoding, and application commands."""

from __future__ import annotations

from prefscope.config import CONFIG
from prefscope.cli.data import (
    _cmd_build_lens,
    _cmd_build_prompt_lens,
    _cmd_compare_responses,
    _cmd_concepts,
    _cmd_embed_corpus,
    _cmd_embed_prompts,
    _cmd_encode_dataset,
)
from prefscope.cli.lens import _cmd_extract_concepts, _cmd_package_lens


def register_lens_commands(sub) -> None:
    pb = sub.add_parser("build-lens", help="embed + train a frozen SAE lens")
    pb.add_argument(
        "--annotations", nargs="+", default=None, help="OpenJury annotation JSON(s)"
    )
    pb.add_argument(
        "--corpus",
        default=None,
        help="merged corpus parquet from build-corpus (label-free)",
    )
    pb.add_argument(
        "--dump-embeddings",
        default=None,
        dest="dump_embeddings",
        help="also save assembled embeddings here (e_a/e_b/meta) so "
        "later --from-embeddings can retrain without re-embedding",
    )
    pb.add_argument(
        "--from-embeddings",
        default=None,
        dest="from_embeddings",
        help="train from a dumped embedding set (skip corpus + cache "
        "scan + embedding); for fast M/K sweeps",
    )
    pb.add_argument("--out", required=True, help="output lens directory")
    pb.add_argument("--m-total", type=int, default=128, dest="m_total")
    pb.add_argument("--k", type=int, default=16)
    pb.add_argument(
        "--matryoshka-prefix",
        type=int,
        nargs="+",
        default=[],
        dest="matryoshka_prefix",
        help="nested Matryoshka prefix lengths; m_total is appended "
        "automatically; omitted by default",
    )
    pb.add_argument(
        "--whiten",
        choices=["none", "standardize", "pca"],
        default="none",
        help="whiten inputs before the SAE (anisotropic embeddings): "
        "'standardize' per-dim, 'pca' full PCA whitening (arXiv:2511.13981). "
        "Stored with the lens and re-applied at projection.",
    )
    pb.add_argument("--whiten-eps", type=float, default=1e-5, dest="whiten_eps")
    pb.add_argument(
        "--sae-type",
        default="auto",
        dest="sae_type",
        help="SAE architecture: auto (signed batchtopk for differences; "
        "non-negative batchtopk-relu for individual data), or any "
        "registered SAE (batchtopk, signed-batchtopk, batchtopk-relu, "
        "jumprelu, simple-topk). jumprelu "
        "(JumpReLU SAE, arXiv:2407.14435 — learned per-feature thresholds; "
        "pair with --input-rep individual); simple-topk is an ablation.",
    )
    pb.add_argument(
        "--sparsity-coef",
        type=float,
        default=1e-3,
        dest="sparsity_coef",
        help="jumprelu: L0 sparsity penalty lambda (tune this; higher = sparser)",
    )
    pb.add_argument(
        "--bandwidth",
        type=float,
        default=1e-3,
        help="jumprelu: straight-through-estimator rectangle-kernel bandwidth epsilon",
    )
    pb.add_argument(
        "--sparsity-warmup-steps",
        type=int,
        default=0,
        dest="sparsity_warmup_steps",
        help="jumprelu: linearly warm lambda over this many optimizer steps",
    )
    pb.add_argument(
        "--input-rep",
        choices=["difference", "individual"],
        default="difference",
        dest="input_rep",
        help="SAE input: 'difference' (e_a-e_b, WIMHF-style, default) "
        "or 'individual' (pooled e_a,e_b)",
    )
    pb.add_argument("--val-frac", type=float, default=0.1, dest="val_frac")
    pb.add_argument("--batch", type=int, default=512)
    pb.add_argument("--n-epochs", type=int, default=200, dest="n_epochs")
    pb.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        dest="max_train_rows",
        help="reservoir cap: train the SAE on at most N randomly-sampled "
        "rows (the dataset is usually far larger than a small "
        "dictionary needs)",
    )
    pb.add_argument("--seed", type=int, default=0)
    pb.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    pb.add_argument(
        "--embed-model-id", default=CONFIG.embed_model_id, dest="embed_model_id"
    )
    pb.add_argument("--embed-model-revision", default=None, dest="embed_model_revision")
    pb.add_argument(
        "--embed-batch-size",
        type=int,
        default=CONFIG.embed_batch_size,
        dest="embed_batch_size",
    )
    pb.add_argument(
        "--cache-workers",
        type=int,
        default=32,
        dest="cache_workers",
        help="parallel threads for reading cached embeddings "
        "(big speedup on parallel filesystems)",
    )
    pb.add_argument(
        "--embed-backend",
        choices=["hf", "vllm", "vllm-server"],
        default="hf",
        dest="embed_backend",
        help="'hf' (default transformers), 'vllm' (in-process), or "
        "'vllm-server' (HTTP to a vLLM OpenAI server, e.g. a "
        "Singularity container) — no GPU torch needed host-side",
    )
    pb.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        dest="tensor_parallel_size",
        help="vLLM tensor-parallel GPUs (model must be split; for an "
        "8B model that fits on one GPU, prefer data-parallel sharding)",
    )
    pb.add_argument(
        "--embed-api-base",
        default=None,
        dest="embed_api_base",
        help="vllm-server: OpenAI-compatible /v1 URL (e.g. http://localhost:8000/v1)",
    )
    pb.add_argument(
        "--embed-api-key-env",
        default="OPENAI_API_KEY",
        dest="embed_api_key_env",
        help="env var holding the server API key (vLLM ignores the value)",
    )
    pb.add_argument(
        "--max-tokens", type=int, default=CONFIG.max_tokens, dest="max_tokens"
    )
    pb.add_argument("--cache-dir", default=None, dest="cache_dir")
    pb.set_defaults(func=_cmd_build_lens)

    pe = sub.add_parser(
        "embed-corpus",
        help="embed one shard of a corpus into the cache (parallel multi-GPU "
        "pre-pass; then run build-lens to train from the warm cache)",
    )
    pe.add_argument("--corpus", required=True, help="merged corpus parquet")
    pe.add_argument(
        "--shard", type=int, default=0, help="this shard index in [0, num-shards)"
    )
    pe.add_argument(
        "--num-shards",
        type=int,
        default=1,
        dest="num_shards",
        help="total shards (= number of parallel GPU processes)",
    )
    pe.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    pe.add_argument(
        "--embed-model-id", default=CONFIG.embed_model_id, dest="embed_model_id"
    )
    pe.add_argument("--embed-model-revision", default=None, dest="embed_model_revision")
    pe.add_argument(
        "--embed-batch-size",
        type=int,
        default=CONFIG.embed_batch_size,
        dest="embed_batch_size",
    )
    pe.add_argument(
        "--max-tokens", type=int, default=CONFIG.max_tokens, dest="max_tokens"
    )
    pe.add_argument("--cache-dir", default=None, dest="cache_dir")
    pe.add_argument("--cache-workers", type=int, default=32, dest="cache_workers")
    pe.add_argument(
        "--embed-backend",
        choices=["hf", "vllm", "vllm-server"],
        default="hf",
        dest="embed_backend",
    )
    pe.add_argument(
        "--tensor-parallel-size", type=int, default=1, dest="tensor_parallel_size"
    )
    pe.add_argument(
        "--embed-api-base",
        default=None,
        dest="embed_api_base",
        help="vllm-server: OpenAI-compatible /v1 URL",
    )
    pe.add_argument(
        "--embed-api-key-env", default="OPENAI_API_KEY", dest="embed_api_key_env"
    )
    pe.set_defaults(func=_cmd_embed_corpus)

    pep = sub.add_parser(
        "embed-prompts",
        help="embed prompts alone -> battle_id-aligned e_prompt.npy for the prompt lens",
    )
    pep.add_argument("--corpus", required=True, help="merged corpus parquet")
    pep.add_argument(
        "--out", required=True, help="output dir for e_prompt.npy + meta.parquet"
    )
    pep.add_argument("--shard", type=int, default=0)
    pep.add_argument(
        "--num-shards",
        type=int,
        default=1,
        dest="num_shards",
        help=">1: only warm the cache for this shard (multi-GPU pre-pass)",
    )
    pep.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    pep.add_argument(
        "--embed-model-id", default=CONFIG.embed_model_id, dest="embed_model_id"
    )
    pep.add_argument("--embed-model-revision", default=None, dest="embed_model_revision")
    pep.add_argument(
        "--embed-batch-size",
        type=int,
        default=CONFIG.embed_batch_size,
        dest="embed_batch_size",
    )
    pep.add_argument(
        "--max-tokens", type=int, default=CONFIG.max_tokens, dest="max_tokens"
    )
    pep.add_argument("--cache-dir", default=None, dest="cache_dir")
    pep.add_argument("--cache-workers", type=int, default=32, dest="cache_workers")
    pep.add_argument(
        "--embed-backend",
        choices=["hf", "vllm", "vllm-server"],
        default="hf",
        dest="embed_backend",
    )
    pep.add_argument(
        "--tensor-parallel-size", type=int, default=1, dest="tensor_parallel_size"
    )
    pep.add_argument(
        "--embed-api-base",
        default=None,
        dest="embed_api_base",
        help="vllm-server: OpenAI-compatible /v1 URL",
    )
    pep.add_argument(
        "--embed-api-key-env", default="OPENAI_API_KEY", dest="embed_api_key_env"
    )
    pep.set_defaults(func=_cmd_embed_prompts)

    ppl = sub.add_parser(
        "build-prompt-lens",
        help="train a standard SAE on prompt embeddings (the prompt-concept matrix)",
    )
    ppl.add_argument(
        "--from-embeddings",
        required=True,
        dest="from_embeddings",
        help="embed-prompts dump dir (e_prompt.npy + meta.parquet)",
    )
    ppl.add_argument("--out", required=True, help="output prompt-lens directory")
    ppl.add_argument("--m-total", type=int, default=64, dest="m_total")
    ppl.add_argument("--k", type=int, default=8)
    ppl.add_argument(
        "--matryoshka-prefix", type=int, nargs="+", default=[], dest="matryoshka_prefix"
    )
    ppl.add_argument(
        "--sae-type",
        default="auto",
        dest="sae_type",
        help="SAE architecture; auto uses non-negative batchtopk-relu",
    )
    ppl.add_argument(
        "--sparsity-coef",
        type=float,
        default=1e-3,
        dest="sparsity_coef",
        help="jumprelu L0 penalty lambda",
    )
    ppl.add_argument(
        "--bandwidth", type=float, default=1e-3, help="jumprelu STE bandwidth epsilon"
    )
    ppl.add_argument(
        "--sparsity-warmup-steps", type=int, default=0, dest="sparsity_warmup_steps"
    )
    ppl.add_argument("--val-frac", type=float, default=0.1, dest="val_frac")
    ppl.add_argument("--batch", type=int, default=512)
    ppl.add_argument("--n-epochs", type=int, default=200, dest="n_epochs")
    ppl.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        dest="max_train_rows",
        help="reservoir cap: train the SAE on at most N randomly-sampled "
        "rows (the dataset is usually far larger than a small "
        "dictionary needs)",
    )
    ppl.add_argument("--seed", type=int, default=0)
    ppl.add_argument("--device", default="cpu", choices=["cuda", "mps", "cpu"])
    ppl.add_argument(
        "--embed-model-id",
        default=CONFIG.embed_model_id,
        dest="embed_model_id",
        help="label only (recorded in manifest)",
    )
    ppl.set_defaults(func=_cmd_build_prompt_lens)

    ppack = sub.add_parser(
        "package-lens",
        help="create a compact, validated inference-only lens for publishing",
    )
    ppack.add_argument(
        "--lens-dir", required=True, help="source trained lens directory"
    )
    ppack.add_argument(
        "--annotations",
        nargs="*",
        default=None,
        help="interpretation directory or CSV files to bundle",
    )
    ppack.add_argument("--out", required=True, help="destination artifact directory")
    ppack.add_argument(
        "--model-card",
        default=None,
        help="optional Markdown file copied to the artifact as README.md",
    )
    ppack.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    ppack.add_argument("--overwrite", action="store_true")
    ppack.set_defaults(func=_cmd_package_lens)

    ped = sub.add_parser(
        "encode-dataset",
        help="encode an arbitrary prompt/response dataset into reusable sparse "
        "vectors with a trained prompt, individual, or difference lens",
    )
    lens_group = ped.add_mutually_exclusive_group(required=True)
    lens_group.add_argument(
        "--lens-dir",
        default=None,
        dest="lens_dir",
        help="local trained lens dir (supports custom embedding backend/cache flags)",
    )
    lens_group.add_argument(
        "--lens",
        default=None,
        help="local lens directory or hf://owner/repository[/subfolder]",
    )
    ped.add_argument(
        "--revision",
        default=None,
        help="Hub branch, tag, or commit (--lens hf:// only)",
    )
    ped.add_argument(
        "--subfolder", default=None, help="lens subfolder inside the Hub repository"
    )
    ped.add_argument("--hub-cache-dir", default=None, dest="hub_cache_dir")
    ped.add_argument(
        "--hf-token-env",
        default=None,
        dest="hf_token_env",
        help="environment variable containing a token for a private lens",
    )
    ped.add_argument(
        "--local-files-only",
        action="store_true",
        dest="local_files_only",
        help="use only an already-cached Hub lens snapshot",
    )
    ped.add_argument(
        "--data",
        required=True,
        help="prepared/local dataset file (.parquet / .csv / .jsonl / .json)",
    )
    ped.add_argument(
        "--out", required=True, help="output dir for codes + meta + manifest"
    )
    ped.add_argument(
        "--overwrite", action="store_true",
        help="replace an existing non-empty output bundle as one validated directory",
    )
    ped.add_argument("--prompt-col", default="prompt", dest="prompt_col")
    ped.add_argument("--response-col", default="response", dest="response_col")
    ped.add_argument(
        "--response-2-col",
        default=None,
        dest="response2_col",
        help="second response column; its presence switches to battle mode",
    )
    ped.add_argument(
        "--model-col",
        default=None,
        dest="model_col",
        help="optional; copied to meta.parquet for later phases",
    )
    ped.add_argument(
        "--model-2-col",
        default=None,
        dest="model2_col",
        help="optional; the second model's name (battle mode)",
    )
    ped.add_argument(
        "--label-col",
        default=None,
        dest="label_col",
        help="optional preference/winner column; copied to meta.parquet",
    )
    ped.add_argument(
        "--metadata-col",
        action="append",
        default=[],
        dest="metadata_cols",
        help="extra source column to preserve in meta.parquet; repeat as needed",
    )
    # embed knobs — model id is read from the lens manifest, NOT a flag
    ped.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "mps", "cpu"],
        help="device for the embedder (cuda also covers ROCm builds)",
    )
    ped.add_argument(
        "--embed-batch-size",
        type=int,
        default=CONFIG.embed_batch_size,
        dest="embed_batch_size",
    )
    ped.add_argument("--cache-dir", default=None, dest="cache_dir")
    ped.add_argument("--cache-workers", type=int, default=32, dest="cache_workers")
    ped.add_argument(
        "--embed-backend",
        choices=["hf", "vllm", "vllm-server"],
        default="hf",
        dest="embed_backend",
    )
    ped.add_argument(
        "--tensor-parallel-size", type=int, default=1, dest="tensor_parallel_size"
    )
    ped.add_argument(
        "--embed-api-base",
        default=None,
        dest="embed_api_base",
        help="vllm-server: OpenAI-compatible /v1 URL",
    )
    ped.add_argument(
        "--embed-api-key-env", default="OPENAI_API_KEY", dest="embed_api_key_env"
    )
    ped.set_defaults(func=_cmd_encode_dataset)

    pcompare = sub.add_parser(
        "compare-responses",
        help="paired, label-free concept shifts between response A and response B",
    )
    pcompare.add_argument(
        "--encoded-dir",
        required=True,
        dest="encoded_dir",
        help="individual-lens encode-dataset bundle containing z_a.npy and z_b.npy",
    )
    pcompare.add_argument(
        "--features",
        required=True,
        help="response lens/interpret directory or merged feature annotation CSV",
    )
    pcompare.add_argument(
        "--prompt-encoded-dir",
        default=None,
        dest="prompt_encoded_dir",
        help="optional aligned prompt-lens encode-dataset bundle",
    )
    pcompare.add_argument(
        "--prompt-features",
        default=None,
        dest="prompt_features",
        help="prompt lens/interpret directory or merged prompt annotation CSV",
    )
    pcompare.add_argument(
        "--prompt-clusters",
        default=None,
        dest="prompt_clusters",
        help="optional feature_id,cluster_id CSV; otherwise contexts are prompt concepts",
    )
    pcompare.add_argument("--side-a-name", default="A", dest="side_a_name")
    pcompare.add_argument("--side-b-name", default="B", dest="side_b_name")
    pcompare.add_argument(
        "--presence-policy",
        choices=["calibrated", "positive_nonzero", "mixed"],
        default="calibrated",
        dest="presence_policy",
        help="response presence rule; calibrated is the statistically safe default",
    )
    pcompare.add_argument(
        "--prompt-presence-policy",
        choices=["calibrated", "positive_nonzero", "mixed"],
        default="calibrated",
        dest="prompt_presence_policy",
    )
    pcompare.add_argument(
        "--include-unverified",
        action="store_true",
        dest="include_unverified",
        help="include named features that did not pass fidelity verification",
    )
    pcompare.add_argument(
        "--include-unnamed",
        action="store_true",
        dest="include_unnamed",
        help="include raw feature ids without a concept name",
    )
    pcompare.add_argument(
        "--min-context-pairs", type=int, default=30, dest="min_context_pairs"
    )
    pcompare.add_argument(
        "--group-col",
        default=None,
        dest="group_col",
        help="optional metadata column grouping repeated generations of one prompt",
    )
    pcompare.add_argument(
        "--examples-per-direction", type=int, default=3, dest="examples_per_direction"
    )
    pcompare.add_argument("--confidence", type=float, default=0.95)
    pcompare.add_argument("--out", required=True)
    pcompare.set_defaults(func=_cmd_compare_responses)

    pconcept = sub.add_parser(
        "concepts",
        help="export every active concept and raw activation for arbitrary prompts/"
        "responses using a local or Hugging Face lens",
    )
    pconcept.add_argument(
        "--lens",
        required=True,
        help="local lens directory or hf://owner/repository[/subfolder]",
    )
    pconcept.add_argument(
        "--revision",
        default=None,
        help="Hub branch, tag, or commit (hf:// lenses only)",
    )
    pconcept.add_argument(
        "--subfolder", default=None, help="lens subfolder inside the Hub repository"
    )
    pconcept.add_argument("--hub-cache-dir", default=None, dest="hub_cache_dir")
    pconcept.add_argument(
        "--hf-token-env",
        default=None,
        dest="hf_token_env",
        help="environment variable containing a token for a private repo",
    )
    pconcept.add_argument(
        "--local-files-only",
        action="store_true",
        dest="local_files_only",
        help="use only an already-cached Hub snapshot",
    )
    pconcept.add_argument(
        "--annotations",
        nargs="*",
        default=None,
        help="optional interpretation CSV(s) or directory to merge with the lens bundle",
    )
    pconcept.add_argument(
        "--data", required=True, help="input .parquet, .csv, .jsonl, or .json"
    )
    pconcept.add_argument(
        "--out", required=True, help="long concept table (.parquet, .csv, or .jsonl)"
    )
    pconcept.add_argument("--prompt-col", default="prompt", dest="prompt_col")
    pconcept.add_argument(
        "--response-col",
        default="response",
        dest="response_col",
        help="ignored for a prompt lens",
    )
    pconcept.add_argument(
        "--response-2-col",
        default=None,
        dest="response2_col",
        help="optional second response; exports side=a and side=b",
    )
    pconcept.add_argument("--batch-size", type=int, default=128, dest="batch_size")
    pconcept.add_argument("--device", default="cpu", choices=["cuda", "mps", "cpu"])
    pconcept.add_argument(
        "--pole", default="any", choices=["any", "positive", "negative"]
    )
    pconcept.add_argument(
        "--min-abs-activation", type=float, default=0.0, dest="min_abs_activation"
    )
    pconcept.add_argument(
        "--top-k",
        type=int,
        default=None,
        dest="top_k",
        help="optional per-item cap; default exports every active feature",
    )
    pconcept.add_argument(
        "--include-zero",
        action="store_true",
        dest="include_zero",
        help="also export zero-valued features (usually very large)",
    )
    pconcept.add_argument(
        "--fidelity-only",
        action="store_true",
        dest="fidelity_only",
        help="keep only features passing bundled verification",
    )
    pconcept.add_argument(
        "--semantic-presence-only",
        action="store_true",
        dest="semantic_presence_only",
        help="apply bundled per-feature semantic thresholds",
    )
    pconcept.add_argument(
        "--include-text",
        action="store_true",
        dest="include_text",
        help="duplicate prompt/completion text into each concept row",
    )
    pconcept.set_defaults(func=_cmd_concepts)

    ptext = sub.add_parser(
        "extract-concepts",
        help="print named concepts expressed by one prompt and optional response",
    )
    ptext.add_argument(
        "--repo", help="Hugging Face repository containing prompt/response lenses"
    )
    ptext.add_argument(
        "--prompt-lens", help="local path or hf:// source for the prompt lens"
    )
    ptext.add_argument(
        "--completion-lens",
        help="local path or hf:// source for the response lens (required with a "
        "completion unless --repo is used)",
    )
    ptext.add_argument("--prompt", required=True)
    ptext.add_argument("--completion", default=None)
    ptext.add_argument("--revision", default=None)
    ptext.add_argument("--prompt-subfolder", default=None)
    ptext.add_argument("--completion-subfolder", default=None)
    ptext.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda", "mps"]
    )
    ptext.add_argument(
        "--presence-policy",
        default="calibrated",
        choices=["calibrated", "mixed", "positive_nonzero"],
        help="calibrated is strict; mixed explicitly falls back to exploratory activity",
    )
    ptext.add_argument(
        "--include-unverified",
        action="store_true",
        help="include named concepts that failed or lack fidelity verification",
    )
    ptext.add_argument(
        "--top",
        type=int,
        default=20,
        help="maximum concepts per text; use 0 for every present concept",
    )
    ptext.add_argument("--json", action="store_true")
    ptext.set_defaults(func=_cmd_extract_concepts)



__all__ = ["register_lens_commands"]
