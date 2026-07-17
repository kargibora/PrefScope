from prefscope.__main__ import build_parser


def _parse(argv):
    return build_parser().parse_args(argv)


def test_extract_activations_args():
    a = _parse(["extract-activations", "--corpus", "c.parquet", "--out", "o",
                "--model-id", "meta-llama/Llama-3.1-8B-Instruct", "--layer", "20",
                "--n-battles", "30000", "--max-tokens", "512"])
    assert a.command == "extract-activations"
    assert a.model_id == "meta-llama/Llama-3.1-8B-Instruct"
    assert a.layer == 20 and a.n_battles == 30000 and a.max_tokens == 512
    assert a.outlier_norm_mult == 6.0
    assert a.attn_implementation == "sdpa"   # default; works on CUDA + ROCm
    assert callable(a.func)


def test_extract_activations_attn_implementation_override():
    a = _parse(["extract-activations", "--corpus", "c", "--out", "o",
                "--attn-implementation", "eager"])
    assert a.attn_implementation == "eager"


def test_train_token_sae_args():
    a = _parse(["train-token-sae", "--cache", "cache", "--out", "sae",
                "--expansion", "8", "--k", "64", "--epochs", "2"])
    assert a.command == "train-token-sae"
    assert a.expansion == 8 and a.k == 64 and a.epochs == 2
    assert a.max_train_tokens == 40_000_000
    assert callable(a.func)


def test_summarize_activations_args():
    a = _parse(["summarize-activations", "--cache", "cache", "--sae", "sae",
                "--out", "summaries"])
    assert a.command == "summarize-activations"
    assert a.cache == "cache" and a.sae == "sae" and a.out == "summaries"
    assert callable(a.func)
