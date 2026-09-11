"""Convenience text-to-array wrappers built on the public featurization path."""
from __future__ import annotations

import numpy as np

from prefscope.core.types import PairItem


def encode_texts(lens, prompts, completions=None) -> np.ndarray:
    """Return one prompt or response view as an array for a specialized recipe."""
    prompts = [prompts] if isinstance(prompts, str) else list(prompts)
    if lens.input_rep == "prompt":
        items = [PairItem(str(index), str(prompt), "") for index, prompt in enumerate(prompts)]
        return lens.featurize(items, views=("prompt",)).array("z_prompt")
    if lens.input_rep != "individual":
        raise ValueError("text encoding needs a prompt or individual-response lens")
    if completions is None:
        raise ValueError("an individual-response lens needs aligned completions")
    completions = [completions] if isinstance(completions, str) else list(completions)
    if len(prompts) != len(completions):
        raise ValueError("prompts and completions must have the same length")
    items = [
        PairItem(str(index), str(prompt), str(completion))
        for index, (prompt, completion) in enumerate(zip(prompts, completions))
    ]
    return lens.featurize(items, views=("response_a",)).array("z_a")


def encode_text(lens, prompt, completion=None) -> np.ndarray:
    """Return one recipe-oriented feature vector."""
    return encode_texts(lens, prompt, completion)[0]
