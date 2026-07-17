"""Prompt-concept → response-concept elicitation over two lenses.

Pipeline wrapper around ``analysis.elicitation.prompt_response_association``: loads the
completion (individual) lens z_a/z_b and the prompt lens z_prompt, row-aligns them by
battle id, stacks both responses (each paired with its prompt), restricts to the
verified axes, and returns the co-activation lift edge table. Preference-independent.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from prefscope.analysis.elicitation import prompt_response_association
from prefscope.artifacts import BATTLES, Z_A, Z_B, Z_PROMPT, lens_battle_ids  # noqa: F401


def _verified(path) -> list | None:
    if path and Path(path).exists():
        df = pd.read_csv(path)
        if "fidelity_pass" in df.columns:
            return df.loc[df["fidelity_pass"].astype(bool), "feature_id"].astype(int).tolist()
    return None


def _name_map(path, col: str = "concept") -> dict:
    if path and Path(path).exists():
        df = pd.read_csv(path)
        if "feature_id" in df.columns and col in df.columns:
            return dict(zip(df["feature_id"].astype(int), df[col]))
    return {}


def run_elicitation(completion_lens, prompt_lens, *, completion_names=None,
                    completion_fidelity=None, prompt_names=None, prompt_fidelity=None,
                    min_support: int = 30, min_cooccur: int = 5, log=print) -> pd.DataFrame:
    """Return the prompt→response elicitation edge table for two lenses.

    ``completion_lens`` must be an INDIVIDUAL lens (has z_a/z_b). ``*_fidelity`` restrict
    to verified axes; ``*_names`` attach concept labels. ``log`` is a print-like callback.
    """
    clens, plens = Path(completion_lens), Path(prompt_lens)
    if not (clens / Z_A).exists() or not (clens / Z_B).exists():
        raise ValueError(
            f"{clens} has no {Z_A}/{Z_B} — elicitation needs an INDIVIDUAL completion "
            f"lens (--input-rep individual), not the difference lens.")
    za = np.load(clens / Z_A)
    zb = np.load(clens / Z_B)
    zp = np.load(plens / Z_PROMPT)
    cb, pb = lens_battle_ids(clens), lens_battle_ids(plens)

    # row-align the two lenses by battle id (built from the same dump, so usually exact)
    if not (len(cb) == len(pb) and bool((cb == pb).all())):
        common = pd.Index(cb).intersection(pd.Index(pb))
        cpos = {b: i for i, b in enumerate(cb)}
        ppos = {b: i for i, b in enumerate(pb)}
        ic = np.array([cpos[b] for b in common])
        ip = np.array([ppos[b] for b in common])
        za, zb, zp = za[ic], zb[ic], zp[ip]
        log(f"aligned on {len(common)} shared battles (of {len(cb)} / {len(pb)})")

    # stack both responses; each row carries its prompt's concepts. The two rows of a
    # battle share a prompt -> n_clusters=N corrects the χ² for within-battle dependence.
    n_battles = za.shape[0]
    resp_fire = np.vstack([za, zb])
    prompt_fire = np.vstack([zp, zp])
    pverif, cverif = _verified(prompt_fidelity), _verified(completion_fidelity)
    log(f"{prompt_fire.shape[0]} responses ({n_battles} battles) | "
        f"prompt axes: {len(pverif) if pverif else prompt_fire.shape[1]} "
        f"({'verified' if pverif else 'all'}) | "
        f"response axes: {len(cverif) if cverif else resp_fire.shape[1]} "
        f"({'verified' if cverif else 'all'})")

    edges = prompt_response_association(
        prompt_fire, resp_fire, prompt_features=pverif, resp_features=cverif,
        min_support=min_support, min_cooccur=min_cooccur, n_clusters=n_battles)

    pnames, cnames = _name_map(prompt_names), _name_map(completion_names)
    if pnames:
        edges["prompt_feature_name"] = edges["prompt_feature"].map(pnames)
    if cnames:
        edges["completion_feature_name"] = edges["completion_feature"].map(cnames)
    return edges
