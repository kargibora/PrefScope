"""Inspect one prompt with a registered pretrained SAELens SAE.

Network downloads: GPT-2 Small, its SAE checkpoint, and optional Neuronpedia labels.
The labels are external proposals, not PrefScope-verified semantic presence.
"""
from __future__ import annotations

import argparse
import json
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np

from prefscope import Lens


def _explanation(model_id: str, layer: str, feature_id: int) -> tuple[str, str]:
    url = (
        "https://www.neuronpedia.org/api/feature/"
        f"{quote(model_id, safe='')}/{quote(layer, safe='')}/{feature_id}"
    )
    page = f"https://www.neuronpedia.org/{model_id}/{layer}/{feature_id}"
    try:
        request = Request(url, headers={"User-Agent": "PrefScope-SAELens-example/1"})
        with urlopen(request, timeout=10) as response:
            explanations = json.load(response).get("explanations") or []
    except Exception:
        return "no explanation retrieved", page
    if not explanations:
        return "no explanation available", page

    def rank(item):
        scores = [score.get("value") for score in item.get("scores", [])]
        scores = [float(score) for score in scores if score is not None]
        return (bool(scores), max(scores, default=float("-inf")))

    best = max(explanations, key=rank)
    return str(best.get("description") or "unnamed feature"), page


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        default="Explain why the sky appears blue during the day.",
    )
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()
    if args.top < 1:
        raise ValueError("--top must be positive")
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error(
            "--device cuda needs a CUDA-enabled PyTorch build; use --device cpu "
            "on this machine"
        )
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("--device mps is not available; use --device cpu")

    # Small public checkpoint used by the SAELens tutorial.
    lens = Lens.from_saelens(
        "gpt2-small-res-jb",
        "blocks.8.hook_resid_pre",
        input_rep="prompt",
        device=args.device,
    )
    sae = lens.projector.sae
    metadata = sae.cfg.metadata

    # Load the reader exactly as requested by the checkpoint metadata.
    from sae_lens import HookedSAETransformer

    reader_model_id = str(metadata.model_name)
    model_kwargs = dict(metadata.model_from_pretrained_kwargs or {})
    model = HookedSAETransformer.from_pretrained_no_processing(
        reader_model_id,
        device=args.device,
        **model_kwargs,
    )
    tokens = model.to_tokens(
        args.prompt,
        prepend_bos=bool(metadata.prepend_bos),
    )[:, : int(metadata.context_size)]
    _, cache = model.run_with_cache(
        tokens,
        names_filter=[str(metadata.hook_name)],
    )

    # Exclude BOS from the prompt summary.
    start = 1 if metadata.prepend_bos else 0
    activations = (
        cache[str(metadata.hook_name)][0, start:]
        .detach().float().cpu().numpy()
    )
    token_text = model.to_str_tokens(tokens)[start:]
    if len(activations) == 0:
        raise ValueError("the prompt contains no analyzed tokens")

    # This contract is derived from the model request and the loaded SAE config.
    extraction_contract = {
        "representation_family": "internal_activation",
        "model_id": reader_model_id,
        "hook_name": str(metadata.hook_name),
        "source_activation_preprocessing": "raw_hook_activation",
        "sae_input_normalization": str(sae.cfg.normalize_activations),
        "activation_reshape": str(sae.cfg.reshape_activations),
        "activation_layout": "token",
        "context_size": int(metadata.context_size),
        "prepend_bos": bool(metadata.prepend_bos),
        "model_from_pretrained_kwargs": model_kwargs,
    }
    features = lens.project_saelens_tokens(
        row_ids=("prompt-1",),
        token_activations={"prompt": activations},
        token_row_ids={"prompt": ("prompt-1",) * len(activations)},
        representation_contract=extraction_contract,
    )
    prompt_codes = features.array("z_prompt")[0]
    token_codes = lens.projector.project(activations)
    top = np.argsort(prompt_codes)[-args.top:][::-1]

    neuronpedia_id = getattr(metadata, "neuronpedia_id", None)
    if not neuronpedia_id or "/" not in neuronpedia_id:
        raise ValueError("this checkpoint does not declare a Neuronpedia source")
    neuronpedia_model, neuronpedia_layer = neuronpedia_id.split("/", 1)

    print(f"\nPrompt: {args.prompt}")
    print("External explanations below are proposals, not verified presence.\n")
    for feature_id in top:
        value = float(prompt_codes[feature_id])
        if value <= 0:
            continue
        token_index = int(np.argmax(token_codes[:, feature_id]))
        description, page = _explanation(
            neuronpedia_model, neuronpedia_layer, int(feature_id))
        print(
            f"feature={int(feature_id):5d}  activation={value:8.3f}  "
            f"top_token={token_text[token_index]!r}\n"
            f"  {description}\n  {page}"
        )


if __name__ == "__main__":
    main()
