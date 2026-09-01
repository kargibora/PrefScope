"""Token-level activation extraction from any HF causal LM.

Pure helpers (filter_outlier_rows, build_chat_inputs) import only NumPy and are
unit-tested without a model. ActivationExtractor (added in a later task) imports
torch/transformers lazily inside __init__ so importing this module stays cheap and
the pure helpers remain testable on machines without a GPU stack.
"""
from __future__ import annotations

import numpy as np


def filter_outlier_rows(vectors: np.ndarray, mult: float) -> np.ndarray:
    """Boolean keep-mask: drop rows whose L2 norm exceeds ``mult`` x the MEAN norm.

    Mirrors the paper's "norm > 6x the average norm" filter. Applied per response
    span; with realistic span lengths (hundreds of tokens) a single outlier does
    not dominate the mean, so it is correctly dropped. (For very short spans the
    mean is less robust, which is acceptable for the pilot.)
    """
    norms = np.linalg.norm(np.asarray(vectors, dtype=np.float32), axis=1)
    if norms.size == 0:
        return np.zeros(0, dtype=bool)
    return norms <= mult * float(norms.mean())


def build_chat_inputs(tokenizer, prompt: str, completion: str, max_tokens: int) -> dict:
    """Tokenize a (user=prompt, assistant=completion) chat and locate the response span.

    Returns ``input_ids`` (truncated to ``max_tokens``) and the half-open response
    token span ``[resp_start, resp_end)``.

    The full-chat tokenization is authoritative. The prompt-only chat with a
    generation prompt must be its prefix; its length is therefore the response
    boundary. This avoids concatenating the assistant header twice (the full chat
    already contains that header for standard Hugging Face chat templates).
    """
    prompt_ids = list(tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, tokenize=True))
    prompt_ids_no_gen = list(tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=False, tokenize=True))
    full_ids = list(tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt},
         {"role": "assistant", "content": completion}],
        add_generation_prompt=False, tokenize=True))

    if list(full_ids[:len(prompt_ids_no_gen)]) != prompt_ids_no_gen:
        raise ValueError(
            "chat template is not prefix-consistent: the prompt-only tokenization "
            "is not a prefix of the full prompt+completion tokenization, so the "
            "response-span boundary cannot be located by length. This tokenizer "
            "needs a different boundary method (e.g. return_assistant_tokens_mask).")
    if list(full_ids[:len(prompt_ids)]) != prompt_ids:
        raise ValueError(
            "chat template is not generation-prefix-consistent: the prompt with an "
            "assistant generation prompt is not a prefix of the full chat. This "
            "tokenizer needs a response-boundary method such as "
            "return_assistant_tokens_mask.")
    resp_start = len(prompt_ids)
    assembled = full_ids[:max_tokens]
    resp_end = len(assembled)
    resp_start = min(resp_start, resp_end)
    return {"input_ids": assembled, "resp_start": resp_start, "resp_end": resp_end}


class ActivationExtractor:
    """Load any HF causal LM and yield layer-L token activations per battle span.

    Two forward passes per battle: (prompt, completion_a) and (prompt, completion_b).
    Under causal attention the prompt-token states are identical across passes, so
    the prompt span is emitted once (from the A pass). Outlier tokens are dropped.
    """

    def __init__(self, model_id: str, layer: int, *, max_tokens: int = 512,
                 outlier_norm_mult: float = 6.0, device: str = "cuda",
                 dtype: str = "bfloat16", attn_implementation: str = "sdpa") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.layer = int(layer)
        self.max_tokens = int(max_tokens)
        self.outlier_norm_mult = float(outlier_norm_mult)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        torch_dtype = getattr(torch, dtype)
        # attn_implementation defaults to "sdpa" (works on both CUDA and ROCm);
        # use "eager" if a backend's FlashAttention/SDPA path misbehaves (e.g. ROCm).
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, output_hidden_states=True,
            attn_implementation=attn_implementation,
        ).to(device).eval()
        self.hidden_dim = int(self.model.config.hidden_size)
        n_layers = int(self.model.config.num_hidden_layers)
        if not 0 <= self.layer <= n_layers:
            raise ValueError(f"layer {self.layer} out of range for {n_layers}-layer model")

    def _layer_states(self, input_ids):
        """Run ONE forward pass; return layer-L hidden states for all tokens (float16)."""
        import torch
        with torch.no_grad():
            ids = torch.tensor([input_ids], device=self.device)
            out = self.model(ids)
            # hidden_states[0] is the embedding layer; layer L is index L
            h = out.hidden_states[self.layer][0]  # (seq, hidden)
        return h.float().cpu().numpy().astype(np.float16)

    def iter_battle_activations(self, battles):
        """Yield (vectors float16 [n,hidden], rows list[dict]) per battle.

        Two forward passes per battle (one per completion). Under causal attention
        the prompt-token states are identical across passes, so the prompt span is
        emitted once, sliced from the A pass. Outlier tokens are dropped.
        """
        for _, b in battles.iterrows():
            bid = str(b["battle_id"])
            ia = build_chat_inputs(self.tokenizer, b["prompt"], b["completion_a"],
                                   self.max_tokens)
            ib = build_chat_inputs(self.tokenizer, b["prompt"], b.get("completion_b", ""),
                                   self.max_tokens)
            ha = self._layer_states(ia["input_ids"])   # 1 forward (completion A)
            hb = self._layer_states(ib["input_ids"])   # 1 forward (completion B)
            spans = (
                ("prompt", ha[0:ia["resp_start"]]),
                ("a", ha[ia["resp_start"]:ia["resp_end"]]),
                ("b", hb[ib["resp_start"]:ib["resp_end"]]),
            )
            for span, h in spans:
                if h.shape[0] == 0:
                    continue
                keep = filter_outlier_rows(h, self.outlier_norm_mult)
                h = h[keep]
                if h.shape[0] == 0:
                    continue
                rows = [{"battle_id": bid, "span": span, "token_idx": int(i)}
                        for i in np.where(keep)[0]]
                yield h, rows
