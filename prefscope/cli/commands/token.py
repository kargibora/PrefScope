"""Register token-activation and token-SAE commands."""

from __future__ import annotations

from prefscope.cli.token import (
    _cmd_extract_activations,
    _cmd_summarize_activations,
    _cmd_train_token_sae,
)


def register_token_commands(sub) -> None:
    pxa = sub.add_parser(
        "extract-activations",
        help="extract layer-L token activations from any HF causal LM into a memmap cache",
    )
    pxa.add_argument("--corpus", required=True)
    pxa.add_argument("--out", required=True, help="output cache dir")
    pxa.add_argument(
        "--model-id", default="meta-llama/Llama-3.1-8B-Instruct", dest="model_id"
    )
    pxa.add_argument(
        "--layer",
        type=int,
        default=24,
        help="hidden layer to extract (default 24, as in Anatomy of "
        "Post-Training for Llama-3.1-8B-Instruct; ~0.75x depth)",
    )
    pxa.add_argument(
        "--n-battles",
        type=int,
        default=30000,
        dest="n_battles",
        help="random subsample size; 0 = all",
    )
    pxa.add_argument("--max-tokens", type=int, default=512, dest="max_tokens")
    pxa.add_argument(
        "--outlier-norm-mult", type=float, default=6.0, dest="outlier_norm_mult"
    )
    pxa.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    pxa.add_argument("--dtype", default="bfloat16")
    pxa.add_argument(
        "--attn-implementation",
        default="sdpa",
        dest="attn_implementation",
        help="HF attn backend; 'sdpa' (default) works on CUDA+ROCm, "
        "'eager' is the safe fallback on AMD/ROCm",
    )
    pxa.add_argument("--seed", type=int, default=0)
    pxa.set_defaults(func=_cmd_extract_activations)

    pts = sub.add_parser(
        "train-token-sae", help="stream-train a BatchTopK SAE from an activation cache"
    )
    pts.add_argument("--cache", required=True, help="extract-activations cache dir")
    pts.add_argument("--out", required=True, help="output SAE dir")
    pts.add_argument(
        "--expansion",
        type=int,
        default=8,
        help="m_total = expansion * hidden_dim (ignored if --m-total set)",
    )
    pts.add_argument(
        "--m-total",
        type=int,
        default=0,
        dest="m_total",
        help="explicit feature count; overrides --expansion",
    )
    pts.add_argument("--k", type=int, default=64)
    pts.add_argument(
        "--matryoshka-prefix", type=int, nargs="+", default=[], dest="matryoshka_prefix"
    )
    pts.add_argument("--val-frac", type=float, default=0.05, dest="val_frac")
    pts.add_argument(
        "--max-train-tokens",
        type=int,
        default=40_000_000,
        dest="max_train_tokens",
        help="reservoir cap on training rows",
    )
    pts.add_argument("--epochs", type=int, default=2)
    pts.add_argument("--batch", type=int, default=4096)
    pts.add_argument("--seed", type=int, default=0)
    pts.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    pts.set_defaults(func=_cmd_train_token_sae)

    psa = sub.add_parser(
        "summarize-activations",
        help="project cached activations through the SAE -> per-span X^max/X^freq",
    )
    psa.add_argument("--cache", required=True)
    psa.add_argument("--sae", required=True, help="train-token-sae output dir")
    psa.add_argument("--out", required=True, help="output summaries dir")
    psa.add_argument("--batch", type=int, default=8192)
    psa.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    psa.set_defaults(func=_cmd_summarize_activations)


__all__ = ["register_token_commands"]
