"""Prompt-concept → response-concept elicitation over two lenses.

Pipeline wrapper around ``analysis.elicitation.prompt_response_association``: loads an
individual completion lens (``z_a`` and optional ``z_b``) plus the prompt lens,
row-aligns them, restricts to verified axes, and returns the co-activation lift edge
table. Preference-independent, so a single ``(prompt, completion)`` dataset is enough.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from prefscope.analysis.elicitation import (
    prompt_response_association,
    prompt_response_association_paired,
)
from prefscope.analysis.grouping import resolve_group_ids
from prefscope.analysis.presence import annotation_flag
from prefscope.artifacts import BATTLES, Z_A, Z_B, Z_PROMPT, lens_battle_ids  # noqa: F401


def _annotation_table(source) -> pd.DataFrame | None:
    if isinstance(source, pd.DataFrame):
        return source
    if source and Path(source).exists():
        return pd.read_csv(source)
    return None


def _verified(source) -> list | None:
    df = _annotation_table(source)
    if df is not None:
        if "fidelity_pass" in df.columns:
            return df.loc[
                df["fidelity_pass"].map(annotation_flag), "feature_id"
            ].astype(int).tolist()
    return None


def _name_map(source, col: str = "concept") -> dict:
    df = _annotation_table(source)
    if df is not None:
        if "feature_id" in df.columns and col in df.columns:
            return dict(zip(df["feature_id"].astype(int), df[col]))
    return {}


def run_elicitation(
    completion_lens,
    prompt_lens,
    *,
    completion_names=None,
    completion_fidelity=None,
    prompt_names=None,
    prompt_fidelity=None,
    min_support: int = 30,
    min_cooccur: int = 5,
    group_ids=None,
    group_col: str | None = None,
    log=print,
) -> pd.DataFrame:
    """Return the prompt→response elicitation edge table for two lenses.

    ``completion_lens`` must be an INDIVIDUAL lens with ``z_a``; paired artifacts may
    additionally contain ``z_b``. ``*_fidelity`` restricts to verified axes and
    ``*_names`` attaches concept labels. Optional ``group_ids`` must follow the original
    completion-lens battle order. Repeated IDs make inference operate on independent
    prompt groups; selected prompt membership must be constant within each group.
    """
    clens, plens = Path(completion_lens), Path(prompt_lens)
    if not (clens / Z_A).exists():
        raise ValueError(
            f"{clens} has no {Z_A} — elicitation needs an INDIVIDUAL completion "
            f"lens (--input-rep individual), not a contrast-only difference lens.")
    za = np.load(clens / Z_A, mmap_mode="r")
    paired = (clens / Z_B).exists()
    zb = np.load(clens / Z_B, mmap_mode="r") if paired else None
    zp = np.load(plens / Z_PROMPT, mmap_mode="r")
    cb, pb = lens_battle_ids(clens), lens_battle_ids(plens)
    if len(za) != len(cb) or len(zp) != len(pb) or (zb is not None and len(zb) != len(cb)):
        raise ValueError("lens code row counts must match their battle-ID vectors")
    if pd.Index(cb).has_duplicates or pd.Index(pb).has_duplicates:
        raise ValueError(
            "completion and prompt lens battle IDs must each be unique for alignment")
    if group_ids is not None and group_col is not None:
        raise ValueError("pass group_ids or group_col, not both")
    if group_ids is None:
        completion_meta = pd.read_parquet(clens / BATTLES)
        group_ids = resolve_group_ids(completion_meta, group_col=group_col)
    if group_ids is not None:
        group_ids = np.asarray(group_ids, dtype=object)
        if group_ids.ndim != 1 or len(group_ids) != len(cb):
            raise ValueError(
                f"group_ids must be 1-D with length {len(cb)} (completion-lens order)"
            )

    # Row-align the two lenses by battle id (built from the same dump, so usually exact).
    if not (len(cb) == len(pb) and bool((cb == pb).all())):
        common = pd.Index(cb).intersection(pd.Index(pb))
        cpos = {b: i for i, b in enumerate(cb)}
        ppos = {b: i for i, b in enumerate(pb)}
        ic = np.array([cpos[b] for b in common])
        ip = np.array([ppos[b] for b in common])
        za, zp = za[ic], zp[ip]
        if paired:
            zb = zb[ic]
        if group_ids is not None:
            group_ids = group_ids[ic]
        log(f"aligned on {len(common)} shared battles (of {len(cb)} / {len(pb)})")

    # Paired analysis counts both responses without materializing stacked 2N matrices.
    # Its p-values use prompts (or explicit repeated prompt groups) as independent units.
    # Single response rows use group inference only when group_ids repeat.
    n_items = za.shape[0]
    if paired:
        n_response_rows = 2 * n_items
        unit = f"{n_items} battles"
    else:
        n_response_rows = n_items
        unit = f"{n_items} prompt/completion items"
    pverif, cverif = _verified(prompt_fidelity), _verified(completion_fidelity)
    log(f"{n_response_rows} responses ({unit}) | "
        f"prompt axes: {len(pverif) if pverif else zp.shape[1]} "
        f"({'verified' if pverif else 'all'}) | "
        f"response axes: {len(cverif) if cverif else za.shape[1]} "
        f"({'verified' if cverif else 'all'})")

    if paired:
        edges = prompt_response_association_paired(
            zp,
            za,
            zb,
            prompt_features=pverif,
            resp_features=cverif,
            min_support=min_support,
            min_cooccur=min_cooccur,
            group_ids=group_ids,
        )
    else:
        edges = prompt_response_association(
            zp,
            za,
            prompt_features=pverif,
            resp_features=cverif,
            min_support=min_support,
            min_cooccur=min_cooccur,
            group_ids=group_ids,
        )

    pnames, cnames = _name_map(prompt_names), _name_map(completion_names)
    if pnames:
        edges["prompt_feature_name"] = edges["prompt_feature"].map(pnames)
    if cnames:
        edges["completion_feature_name"] = edges["completion_feature"].map(cnames)
    return edges
