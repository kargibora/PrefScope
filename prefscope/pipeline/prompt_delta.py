"""Prompt-conditioned completion delta Δ_{k,f} (+ optional conditional δ_{f,k}).

Orient the completion lens's z_diff by human preference, condition on every positively
active prompt concept (or cluster), and run the framework's overlapping-region contrast:
which response properties distinguish the winner when each prompt concept is present?
Optionally also compute length-controlled conditional win-rate δ_{f,k} and a
prompt-membership permutation null.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from prefscope.analysis.dataset import region_membership_contrast
from prefscope.analysis.presence import annotation_flag
from prefscope.analysis.prompt_regions import prompt_region_membership
from prefscope.artifacts import (
    Z_PROMPT, lens_battle_ids, require_paired_codes,
)
from prefscope.data.corpus import load_corpus
from prefscope.core.manifest import LensManifest
from prefscope.data.pair_schema import LABEL, RESPONSE_A, RESPONSE_B, orient_by_label


def _require_antisymmetric_codes(lens_dir: Path, *, command: str) -> None:
    manifest_path = lens_dir / "manifest.json"
    manifest = LensManifest.from_dict(
        json.loads(manifest_path.read_text()), strict=False)
    if manifest.input_rep != "individual":
        raise ValueError(
            f"{command} orients A/B codes by preference and therefore needs an "
            "individual lens, where z_diff = f(e_a) - f(e_b). Direct-difference lens "
            "codes are nonlinear and cannot be reversed by negating z_diff.")


# module-level state shared with permutation workers via fork (no big-array pickling)
_PERM: dict = {}


def _perm_survivors(seed: int) -> int:
    """Shuffle whole prompt membership groups, preserving concept co-occurrence."""
    rng = np.random.default_rng(seed)
    group_ids = _PERM.get("group_ids")
    if group_ids is None or len(set(group_ids.tolist())) == len(group_ids):
        membership = rng.permutation(_PERM["membership"], axis=0)
    else:
        ordered = list(dict.fromkeys(group_ids.tolist()))
        group_membership = []
        for group in ordered:
            rows = np.flatnonzero(group_ids == group)
            values = _PERM["membership"][rows]
            if not np.all(values == values[0]):
                raise ValueError(
                    "prompt membership must be constant within groups for permutation")
            group_membership.append(values[0])
        shuffled = rng.permutation(np.asarray(group_membership), axis=0)
        membership = np.empty_like(_PERM["membership"])
        for group, values in zip(ordered, shuffled):
            membership[group_ids == group] = values
    perm = region_membership_contrast(
        _PERM["z"], membership, region_ids=_PERM["region_ids"],
        seed=_PERM["seed"], group_ids=group_ids)
    return int(((perm["p_bonferroni"] < 0.05) & perm["stable"]).sum())


def _name_map(path, col: str = "concept") -> dict:
    if path and Path(path).exists():
        df = pd.read_csv(path)
        if "feature_id" in df.columns and col in df.columns:
            return dict(zip(df["feature_id"].astype(int), df[col]))
    return {}


def _verified_ids(path) -> list[int] | None:
    if path and Path(path).exists():
        frame = pd.read_csv(path)
        if {"feature_id", "fidelity_pass"} <= set(frame.columns):
            return frame.loc[
                frame["fidelity_pass"].map(annotation_flag),
                "feature_id",
            ].astype(int).tolist()
    return None


def run_prompt_conditioned_delta(completion_lens, prompt_lens, out, *, corpus=None,
                                 completion_names=None, prompt_names=None,
                                 prompt_clusters=None, conditional_out=None,
                                 completion_fidelity=None, prompt_fidelity=None,
                                 min_prompt_activation: float = 0.0,
                                 min_prompt_support: int = 30, seed: int = 0,
                                 permute: int = 0, jobs: int = 1,
                                 group_col: str | None = None,
                                 log=print) -> pd.DataFrame:
    """Compute and write Δ_{k,f}; optionally δ_{f,k} (``conditional_out``) and a null."""
    clens, plens = Path(completion_lens), Path(prompt_lens)
    if corpus:
        _require_antisymmetric_codes(clens, command="conditional-delta")
    z_diff = np.load(require_paired_codes(clens, command="conditional-delta"))
    z_prompt = np.load(plens / Z_PROMPT)
    cb, pb = lens_battle_ids(clens), lens_battle_ids(plens)
    if len(z_diff) != len(cb) or len(z_prompt) != len(pb):
        raise ValueError(
            "lens code row counts must match their battle-ID vectors before alignment")
    if pd.Index(cb).has_duplicates or pd.Index(pb).has_duplicates:
        raise ValueError(
            "completion and prompt lens battle IDs must each be unique for alignment")

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

    z_raw = y_keep = length_keep = group_keep = None
    if corpus:
        # Orient z_diff so + = the HUMAN-PREFERRED response expresses the feature more;
        # drop ties. (A/B are arbitrary slots, so unoriented sign(z_diff) averages to ~0.)
        corp = load_corpus(corpus)
        if LABEL not in corp.columns:
            raise ValueError("corpus has no human_pref; rebuild with `build-corpus --keep-labels`")
        from prefscope.analysis.grouping import resolve_group_ids

        corp["battle_id"] = corp["battle_id"].astype(str)
        aligned_corp = corp.set_index("battle_id").reindex(pd.Series(bids).astype(str))
        y = aligned_corp[LABEL].to_numpy(dtype=float)
        all_groups = resolve_group_ids(
            aligned_corp.reset_index(drop=True), group_col=group_col)
        z_oriented, keep = orient_by_label(y, z_diff)
        z_raw = z_diff[keep].copy()                 # UNORIENTED — for the conditional logistic
        y_keep = y[keep]
        group_keep = all_groups[keep] if all_groups is not None else None
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

    # A prompt can express several concepts. Keep every positive activation above the
    # requested raw threshold; optional clusters union all firing member features.
    prompt_features = _verified_ids(prompt_fidelity)
    if prompt_clusters:
        pc = pd.read_csv(prompt_clusters)
        region_ids, membership, _ = prompt_region_membership(
            z_prompt, feature_ids=prompt_features, clusters=pc,
            min_activation=min_prompt_activation)
        pnames = ({int(c): str(b) for c, b in pc.dropna(subset=["behavior"])
                   .groupby("cluster_id")["behavior"].first().items()}
                  if "behavior" in pc.columns else {})
        unit = "prompt clusters"
    else:
        region_ids, membership, _ = prompt_region_membership(
            z_prompt, feature_ids=prompt_features,
            min_activation=min_prompt_activation)
        pnames = _name_map(prompt_names)
        unit = "prompt concepts"
    support = membership.sum(axis=0)
    keep_regions = support >= int(min_prompt_support)
    region_ids, membership = region_ids[keep_regions], membership[:, keep_regions]
    memberships_per_prompt = (
        float(membership.sum(axis=1).mean()) if membership.shape[1] else 0.0)
    log(f"{z_diff.shape[0]} battles | {z_diff.shape[1]} completion features "
        f"| {len(region_ids)} supported {unit} | "
        f"{memberships_per_prompt:.2f} memberships/prompt")

    delta = region_membership_contrast(
        z_diff, membership, region_ids=region_ids, seed=seed,
        min_inside=min_prompt_support, group_ids=group_keep).rename(
        columns={"region_id": "prompt_concept", "feature_id": "completion_feature"})
    delta["prompt_unit_kind"] = "cluster" if prompt_clusters else "feature"
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
                feats = fdf.loc[
                    fdf["fidelity_pass"].map(annotation_flag), "feature_id"
                ].astype(int).tolist()
        cond = conditional_win_relevance(
            z_raw, y_keep, length_keep, membership,
            prompt_region_ids=region_ids, features=feats, group_ids=group_keep)
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
        _PERM.update(
            z=z_diff, membership=membership, region_ids=region_ids, seed=seed,
            group_ids=group_keep)
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
