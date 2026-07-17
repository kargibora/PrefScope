"""Interpreter strategies — registry-driven, config-selectable feature naming.

A ``NameStrategy`` is a registered component (kind ``"interpreter"``) selected by name
from the CLI (``--name-mode``) or a config file. Its tunables are constructor args (so a
YAML ``interpreter: {name: individual, n_active: 12}`` maps straight to it), and the
runtime data arrives as a typed ``LensCodes`` — so a third-party strategy never touches
the on-disk lens layout (``z_a.npy`` etc.). Resolve via ``registry.make("interpreter",
name, **params)``; importing this module registers the built-ins.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from prefscope.artifacts import BATTLES, MANIFEST, Z_A, Z_B, Z_PROMPT, lens_battle_ids
from prefscope.core import registry
from prefscope.interpret.io import load_lens_battles


def _load_prompt_lens(lens_dir, corpus):
    """``(input_rep, meta, instruction_ids, z_prompt, prompts)`` for a prompt lens.

    A prompt lens has no A/B pair, so its codes live in ``z_prompt`` and the text is the
    prompt itself — mapped from the corpus by battle id. Shared by ``LensCodes`` (naming)
    and ``VerifyCodes`` (verification) so the two never drift."""
    if not corpus:
        raise ValueError("prompt-lens stages need a corpus (to fetch prompt text).")
    from prefscope.data.corpus import load_corpus

    lens_dir = Path(lens_dir)
    input_rep = json.loads((lens_dir / MANIFEST).read_text()).get("input_rep", "prompt")
    z_prompt = np.load(lens_dir / Z_PROMPT)
    meta = pd.read_parquet(lens_dir / BATTLES)
    bid = lens_battle_ids(meta)
    corp = load_corpus(corpus)
    corp["battle_id"] = corp["battle_id"].astype(str)
    prompts = pd.Series(bid).map(corp.set_index("battle_id")["prompt"]).fillna("").tolist()
    return input_rep, meta, list(bid), z_prompt, prompts


@dataclass
class LensCodes:
    """Typed, manifest-described view of a lens's codes + text for a naming run.

    A completion lens carries ``z_diff`` (difference contrast) or ``z_a``/``z_b`` (per-side)
    + the A/B battles; a prompt lens (``lens_kind='prompt'``) carries ``z_prompt`` + the
    prompt texts. Strategies read these without knowing the on-disk layout. ``input_rep`` is
    the manifest's, used to auto-pick a completion strategy.
    """
    lens_dir: Path
    input_rep: str
    battles: pd.DataFrame
    z_diff: np.ndarray | None
    z_a: np.ndarray | None
    z_b: np.ndarray | None
    lens_kind: str = "completion"
    z_prompt: np.ndarray | None = None
    prompts: list | None = None
    instruction_ids: list | None = None

    @classmethod
    def load(cls, lens_dir, annotations=None, *, corpus=None,
             lens_kind: str = "completion") -> "LensCodes":
        lens_dir = Path(lens_dir)
        if lens_kind == "prompt":
            input_rep, meta, ids, z_prompt, prompts = _load_prompt_lens(lens_dir, corpus)
            return cls(lens_dir=lens_dir, input_rep=input_rep, battles=meta, z_diff=None,
                       z_a=None, z_b=None, lens_kind="prompt", z_prompt=z_prompt,
                       prompts=prompts, instruction_ids=ids)
        manifest = json.loads((lens_dir / MANIFEST).read_text())   # a lens always has one
        single = manifest.get("dataset_mode") == "single"
        if single:
            battles = pd.read_parquet(lens_dir / BATTLES)
            z_diff = None
        else:
            battles, z_diff, _ = load_lens_battles(lens_dir, annotations, corpus=corpus)
        z_a = np.load(lens_dir / Z_A) if (lens_dir / Z_A).exists() else None
        z_b = np.load(lens_dir / Z_B) if (lens_dir / Z_B).exists() else None
        return cls(lens_dir=lens_dir, input_rep=manifest.get("input_rep", "difference"),
                   battles=battles, z_diff=z_diff, z_a=z_a, z_b=z_b,
                   lens_kind="completion",
                   instruction_ids=battles["instruction_id"].astype(str).tolist())


_OPT_KEYS = ("features", "n_active", "n_zero", "verify_frac", "seed",
             "abbreviate", "concurrency", "debug_dir", "negatives",
             "n_candidates", "candidate_pool_factor")


class NameStrategy(ABC):
    """Names a lens's features. Tunables in __init__; runtime data via LensCodes."""

    def __init__(self, *, features=None, n_active: int = 12, n_zero: int = 8,
                 verify_frac: float = 0.2, seed: int = 0, abbreviate: bool = False,
                 concurrency: int = 1, debug_dir=None, negatives: str = "random",
                 n_candidates: int = 1, candidate_pool_factor: int = 3) -> None:
        if n_candidates < 1:
            raise ValueError("n_candidates must be >= 1")
        if candidate_pool_factor < 1:
            raise ValueError("candidate_pool_factor must be >= 1")
        self.opts = {k: v for k, v in locals().items() if k in _OPT_KEYS}

    @abstractmethod
    def name(self, codes: LensCodes, client) -> pd.DataFrame:
        ...


@registry.register("interpreter", "pairwise")
class PairwiseNameStrategy(NameStrategy):
    """Difference lens: name each feature from the A/B contrast (z_diff)."""

    def name(self, codes: LensCodes, client) -> pd.DataFrame:
        from prefscope.interpret.name import name_features
        return name_features(codes.battles, codes.z_diff, client, **self.opts)


@registry.register("interpreter", "individual")
class IndividualNameStrategy(NameStrategy):
    """Individual lens: name each feature by the SHARED trait of top single responses."""

    def name(self, codes: LensCodes, client) -> pd.DataFrame:
        from prefscope.interpret.name import name_individual_features
        if codes.z_a is None:
            raise ValueError(
                "individual naming needs z_a (and z_b for paired data) — this lens has none (use an "
                "--input-rep individual lens, or --name-mode pairwise).")
        return name_individual_features(codes.battles, codes.z_a, codes.z_b, client, **self.opts)


@registry.register("interpreter", "single-text")
class SingleTextNameStrategy(NameStrategy):
    """Prompt lens: name each feature from its top-activating prompts (there is no A/B pair)."""

    def name(self, codes: LensCodes, client) -> pd.DataFrame:
        from prefscope.interpret.prompt_name import name_prompt_features
        if codes.z_prompt is None or codes.prompts is None:
            raise ValueError(
                "single-text naming needs a prompt lens (z_prompt + prompt text) — load "
                "LensCodes with lens_kind='prompt' and a corpus.")
        o = self.opts
        return name_prompt_features(
            codes.prompts, codes.z_prompt, client, features=o["features"],
            n_active=o["n_active"], n_zero=o["n_zero"], verify_frac=o["verify_frac"],
            seed=o["seed"], concurrency=o["concurrency"],
            negatives=o.get("negatives", "random"),
            n_candidates=o["n_candidates"],
            candidate_pool_factor=o["candidate_pool_factor"],
            instruction_ids=codes.instruction_ids)


def resolve_name_mode(mode: str, input_rep: str, lens_kind: str = "completion") -> str:
    """Pick the naming strategy: a prompt lens -> 'single-text'; else 'auto' maps to
    individual/pairwise off the manifest input_rep (not file probing)."""
    if lens_kind == "prompt":
        return "single-text"
    if mode == "auto":
        return "individual" if input_rep == "individual" else "pairwise"
    return mode


# ---------------------------------------------------------------------------------------
# Verifier strategies — same registry-driven contract, kind "verifier".
# ---------------------------------------------------------------------------------------

@dataclass
class VerifyCodes:
    """Typed codes + text a verifier needs, built per lens-kind so strategies don't load
    files. completion lenses carry z_diff/z_a/z_b + battles; the prompt lens carries
    z_prompt + the prompt texts (mapped from the corpus)."""
    lens_kind: str
    input_rep: str
    battles: pd.DataFrame
    instruction_ids: list
    z_diff: np.ndarray | None = None
    z_a: np.ndarray | None = None
    z_b: np.ndarray | None = None
    z_prompt: np.ndarray | None = None
    prompts: list | None = None

    @classmethod
    def load(cls, lens_dir, annotations=None, *, corpus=None,
             lens_kind: str = "completion") -> "VerifyCodes":
        lens_dir = Path(lens_dir)
        if lens_kind == "prompt":
            input_rep, meta, ids, z_prompt, prompts = _load_prompt_lens(lens_dir, corpus)
            return cls("prompt", input_rep, meta, ids, z_prompt=z_prompt, prompts=prompts)
        input_rep = json.loads((lens_dir / MANIFEST).read_text()).get("input_rep", "difference")
        manifest = json.loads((lens_dir / MANIFEST).read_text())
        single = manifest.get("dataset_mode") == "single"
        if single:
            battles = pd.read_parquet(lens_dir / BATTLES)
            z_diff = None
        else:
            battles, z_diff, _ = load_lens_battles(lens_dir, annotations, corpus=corpus)
        z_a = np.load(lens_dir / Z_A) if (lens_dir / Z_A).exists() else None
        z_b = np.load(lens_dir / Z_B) if (lens_dir / Z_B).exists() else None
        ids = battles["instruction_id"].astype(str).tolist()
        return cls("completion", input_rep, battles, ids, z_diff=z_diff, z_a=z_a, z_b=z_b)


_VOPT = ("n_per_bucket", "verify_frac", "seed", "fidelity_threshold", "concurrency",
         "negatives", "embeddings", "min_success_rate", "min_bucket", "sampling",
         "n_examples")


class VerifyStrategy(ABC):
    """Held-out fidelity check of named axes. Tunables in __init__; data via VerifyCodes."""

    def __init__(self, *, n_per_bucket: int = 10, verify_frac: float = 0.2, seed: int = 0,
                 fidelity_threshold: float = 0.3, concurrency: int = 1,
                 negatives: str = "random", embeddings=None,
                 min_success_rate: float = 0.8, min_bucket: int = 5,
                 sampling: str = "extremes", n_examples: int | None = None) -> None:
        if sampling not in ("extremes", "stratified-random"):
            raise ValueError("sampling must be 'extremes' or 'stratified-random'")
        if n_examples is not None and n_examples < 2:
            raise ValueError("n_examples must be >= 2")
        if not 0 <= min_success_rate <= 1:
            raise ValueError("min_success_rate must be in [0, 1]")
        if min_bucket < 1:
            raise ValueError("min_bucket must be >= 1")
        self.opts = {k: v for k, v in locals().items() if k in _VOPT}

    @abstractmethod
    def verify(self, codes: VerifyCodes, names: pd.DataFrame, client) -> pd.DataFrame:
        ...


@registry.register("verifier", "pairwise")
class PairwiseVerifyStrategy(VerifyStrategy):
    """Difference lens: A/B pairwise z_diff verifier (WIMHF single-concept annotate)."""

    def verify(self, codes, names, client):
        from prefscope.interpret.verify import verify_features
        o = self.opts
        return verify_features(codes.battles, codes.z_diff, names, client,
                               n_per_bucket=o["n_per_bucket"], verify_frac=o["verify_frac"],
                               seed=o["seed"], fidelity_threshold=o["fidelity_threshold"],
                               concurrency=o["concurrency"],
                               min_success_rate=o["min_success_rate"],
                               min_bucket=o["min_bucket"], sampling=o["sampling"],
                               n_examples=o["n_examples"])


@registry.register("verifier", "individual")
class IndividualVerifyStrategy(VerifyStrategy):
    """Individual lens: single-response presence verify on stacked z_a/z_b."""

    def verify(self, codes, names, client):
        from prefscope.interpret.verify import verify_single_text_features
        if codes.z_a is None:
            raise ValueError("individual verify needs z_a (an --input-rep individual lens).")
        paired = codes.z_b is not None
        texts = codes.battles["completion_a"].tolist()
        z_stack = codes.z_a
        if paired:
            texts += codes.battles["completion_b"].tolist()
            z_stack = np.concatenate([codes.z_a, codes.z_b], axis=0)
        # Pass the user prompts as CONTEXT so context-dependent concepts (refuses an unsafe
        # request, follows the requested format, answers in the requested language) can be
        # judged in the response — the naming step already sees the context (#6).
        contexts = codes.battles["prompt"].tolist() * (2 if paired else 1)
        o = self.opts
        # forward the configured negatives strategy (was silently dropped, so the individual
        # lens always used random controls even under --negatives similar) (#6). Use the SAE
        # codes as the similarity space — code-space, always aligned with the 2N stacked
        # texts, and consistent with how naming picks hard negatives (no raw embeddings needed).
        neg = o.get("negatives", "random")
        return verify_single_text_features(
            texts, z_stack, names, client, negatives=neg,
            embeddings=(z_stack if neg != "random" else None),
            n_active=o["n_per_bucket"], n_zero=o["n_per_bucket"],
            verify_frac=o["verify_frac"], seed=o["seed"],
            fidelity_threshold=o["fidelity_threshold"], concurrency=o["concurrency"],
            min_success_rate=o["min_success_rate"], min_bucket=o["min_bucket"],
            sampling=o["sampling"], n_examples=o["n_examples"],
            instruction_ids=list(codes.instruction_ids) * (2 if paired else 1),
            contexts=contexts)


@registry.register("verifier", "prompt")
class PromptVerifyStrategy(VerifyStrategy):
    """Prompt lens: single-text presence verify on z_prompt + the prompt texts."""

    def verify(self, codes, names, client):
        from prefscope.interpret.verify import verify_single_text_features
        if codes.z_prompt is None or codes.prompts is None:
            raise ValueError("prompt verify needs a prompt lens (z_prompt) loaded with --corpus.")
        o = self.opts
        emb = np.load(o["embeddings"]) if o["embeddings"] else None
        return verify_single_text_features(
            codes.prompts, codes.z_prompt, names, client, negatives=o["negatives"],
            embeddings=emb, n_active=o["n_per_bucket"], n_zero=o["n_per_bucket"],
            verify_frac=o["verify_frac"], seed=o["seed"],
            fidelity_threshold=o["fidelity_threshold"], concurrency=o["concurrency"],
            min_success_rate=o["min_success_rate"], min_bucket=o["min_bucket"],
            sampling=o["sampling"], n_examples=o["n_examples"],
            instruction_ids=list(codes.instruction_ids))


def resolve_verify_mode(mode: str, input_rep: str, lens_kind: str = "completion") -> str:
    """Pick the verifier strategy: a prompt lens -> 'prompt'; else 'auto' maps to
    individual/pairwise off the manifest input_rep."""
    if lens_kind == "prompt":
        return "prompt"
    if mode == "auto":
        return "individual" if input_rep == "individual" else "pairwise"
    return mode
