"""Prompt-conditioned completion delta Δ_{k,f} (+ optional conditional δ_{f,k}).

Orient the completion lens's z_diff by human preference, condition on the prompt lens's
dominant prompt concept (or prompt clusters), and run the framework's
``region_behavior_contrast`` — which response properties distinguish the winner for each
prompt type. Optionally also the length-controlled conditional win-rate δ_{f,k} and a
label-permutation null. Pipeline wrapper; the statistics live in ``analysis.dataset`` and
``pipeline.winrelevance``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from prefscope.analysis.dataset import region_behavior_contrast
from prefscope.artifacts import Z_DIFF, Z_PROMPT, lens_battle_ids
from prefscope.data.corpus import load_corpus
from prefscope.data.pair_schema import LABEL, RESPONSE_A, RESPONSE_B, orient_by_label

# module-level state shared with permutation workers via fork (no big-array pickling)
_PERM: dict = {}


def _perm_survivors(seed: int) -> int:
    """One permutation: shuffle the prompt-concept labels, recompute Δ, count survivors."""
    pc = np.random.default_rng(seed).permutation(_PERM["concept"])
    perm = region_behavior_contrast(_PERM["z"], pc, seed=_PERM["seed"])
    return int(((perm["p_bonferroni"] < 0.05) & perm["stable"]).sum())


def _name_map(path, col: str = "concept") -> dict:
    if path and Path(path).exists():
        df = pd.read_csv(path)
        if "feature_id" in df.columns and col in df.columns:
            return dict(zip(df["feature_id"].astype(int), df[col]))
    return {}


def run_prompt_conditioned_delta(completion_lens, prompt_lens, out, *, corpus=None,
                                 completion_names=None, prompt_names=None,
                                 prompt_clusters=None, conditional_out=None,
                                 completion_fidelity=None, seed: int = 0,
                                 permute: int = 0, jobs: int = 1, log=print) -> pd.DataFrame:
    """Compute and write Δ_{k,f}; optionally δ_{f,k} (``conditional_out``) and a null."""
    clens, plens = Path(completion_lens), Path(prompt_lens)
    z_diff = np.load(clens / Z_DIFF)
    z_prompt = np.load(plens / Z_PROMPT)
    cb, pb = lens_battle_ids(clens), lens_battle_ids(plens)

    # row-align the two lenses by battle id
    if len(cb) == len(pb) and bool((cb == pb).all()):
        bids = cb
    else:
        common = pd.Index(cb).intersection(pd.Index(pb))
        cpos = {b: i for i, b in enumerate(cb)}
        ppos = {b: i for i, b in enumerate(pb)}
        ic = np.array([cpos[b] for b in common])
        ip = np.array([ppos[b] for b in common])
        z_diff, z_prompt, bids = z_diff[ic], z_prompt[ip], common.to_numpy()
        log(f"aligned on {len(common)} shared battles (of {len(cb)} / {len(pb)})")

    z_raw = y_keep = length_keep = None
    if corpus:
        # Orient z_diff so + = the HUMAN-PREFERRED response expresses the feature more;
        # drop ties. (A/B are arbitrary slots, so unoriented sign(z_diff) averages to ~0.)
        corp = load_corpus(corpus)
        if LABEL not in corp.columns:
            raise ValueError("corpus has no human_pref; rebuild with `build-corpus --keep-labels`")
        corp["battle_id"] = corp["battle_id"].astype(str)
        y = (pd.Series(bids).astype(str)
             .map(corp.set_index("battle_id")[LABEL]).to_numpy(dtype=float))
        z_oriented, keep = orient_by_label(y, z_diff)
        z_raw = z_diff[keep].copy()                 # UNORIENTED — for the conditional logistic
        y_keep = y[keep]
        bids_keep = pd.Series(bids).astype(str).to_numpy()[keep]
        z_diff = z_oriented
        z_prompt = z_prompt[keep]
        if RESPONSE_A in corp.columns:
            ci = corp.set_index("battle_id")
            _wc = lambda c: ci[c].reindex(bids_keep).fillna("").astype(str).str.split().str.len().to_numpy(float)  # noqa: E731
            length_keep = _wc(RESPONSE_A) - _wc(RESPONSE_B)
        else:
            length_keep = np.zeros(int(keep.sum()))
        log(f"oriented by human_pref: kept {int(keep.sum())} decisive battles "
            f"(dropped {int((~keep).sum())} ties/unlabeled)")
    else:
        log("WARNING: no corpus -> z_diff is UNORIENTED; Δ measures positional asymmetry, "
            "NOT the winner, and will be ~0. Pass corpus (with human_pref) to orient.")

    # each battle's dominant prompt-lens feature; optionally fold features -> clusters.
    # Require a POSITIVE max: an all-zero prompt code (silent) or an all-negative row
    # (only opposite poles) has no concept present, so mark it -1/unknown rather than
    # letting argmax assign it to feature 0 / the least-negative feature (#4).
    dom = z_prompt.argmax(axis=1)
    dom = np.where(z_prompt.max(axis=1) > 0, dom, -1)
    if prompt_clusters:
        pc = pd.read_csv(prompt_clusters)
        f2c = dict(zip(pc["feature_id"].astype(int), pc["cluster_id"].astype(int)))
        concept = np.array([f2c.get(int(d), -1) for d in dom])
        pnames = ({int(c): str(b) for c, b in pc.dropna(subset=["behavior"])
                   .groupby("cluster_id")["behavior"].first().items()}
                  if "behavior" in pc.columns else {})
        unit = "prompt clusters"
    else:
        concept = dom
        pnames = _name_map(prompt_names)
        unit = "prompt concepts"
    log(f"{z_diff.shape[0]} battles | {z_diff.shape[1]} completion features "
        f"| {len(np.unique(concept))} active {unit}")

    delta = region_behavior_contrast(z_diff, concept, seed=seed).rename(
        columns={"cluster_id": "prompt_concept", "feature_id": "completion_feature"})
    cnames = _name_map(completion_names)
    if pnames:
        delta["prompt_concept_name"] = delta["prompt_concept"].map(pnames)
    if cnames:
        delta["completion_feature_name"] = delta["completion_feature"].map(cnames)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    delta.to_csv(out, index=False)
    n_obs = int(((delta["p_bonferroni"] < 0.05) & delta["stable"]).sum())
    log(f"wrote {len(delta)} (prompt_concept, completion_feature) rows to {out}; "
        f"{n_obs} significant & split-half stable")

    # Length-controlled conditional win-rate δ_{f,k} (prompt-type × behavior interaction)
    if conditional_out and corpus:
        from prefscope.pipeline.winrelevance import conditional_win_relevance
        feats = None
        if completion_fidelity:
            fdf = pd.read_csv(completion_fidelity)
            if "fidelity_pass" in fdf.columns:
                feats = fdf.loc[fdf["fidelity_pass"].astype(bool), "feature_id"].astype(int).tolist()
        cond = conditional_win_relevance(z_raw, y_keep, length_keep, concept, features=feats)
        if pnames:
            cond["prompt_concept_name"] = cond["prompt_concept"].map(pnames)
        if cnames:
            cond["completion_feature_name"] = cond["feature_id"].map(cnames)
        Path(conditional_out).parent.mkdir(parents=True, exist_ok=True)
        cond.to_csv(conditional_out, index=False)
        nsig = int(cond["cond_significant"].sum()) if len(cond) else 0
        log(f"wrote {len(cond)} conditional (prompt_type x feature) cells to "
            f"{conditional_out}; {nsig} significant (length-controlled)")

    # Label-permutation null: break the prompt<->completion association and count survivors.
    if permute > 0:
        _PERM.update(z=z_diff, concept=concept, seed=seed)
        seeds = [seed + 1 + i for i in range(permute)]
        if jobs > 1:
            import multiprocessing as mp
            with mp.get_context("fork").Pool(jobs) as pool:
                null = np.array(pool.map(_perm_survivors, seeds))
        else:
            null = np.array([_perm_survivors(s) for s in seeds])
        exceed = int((null >= n_obs).sum())
        log(f"\nlabel-permutation null ({permute} shuffles): survivors mean={null.mean():.1f}, "
            f"95th pct={np.percentile(null, 95):.0f}, max={null.max()}")
        log(f"observed={n_obs}  |  empirical p = {(exceed + 1) / (permute + 1):.4f} "
            f"({exceed}/{permute} shuffles matched or beat observed)")
    return delta
