"""Human-readable per-model concept report card.

A presentation layer over an existing diagnosis (``run_diagnose``): turn the
per-feature DataFrame into a markdown summary of what a model most/least
distinguishes itself from opponents on (the diagnosis fire_rate is a contrast rate,
not absolute prevalence), which rewarded concepts it under-expresses (a gap worth
closing), and — when a prompt lens is supplied — which prompt types it is strong /
weak on.

Everything here is PURE (no embedding, no GPU, no LLM): ``format_report`` renders a
diagnosis frame, ``prompt_concept_winrates`` aggregates a prompt lens's codes into
per-prompt-concept win rates. The CLI (``prefscope report``) does the embedding and
calls these.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from prefscope.artifacts import Z_PROMPT, lens_battle_ids


def _named(diag: pd.DataFrame) -> pd.DataFrame:
    """Rows whose ``concept`` is a real (non-empty) name; everything else is skipped."""
    if "concept" not in diag.columns:
        return diag.iloc[0:0]
    c = diag["concept"]
    mask = c.notna() & (c.astype(str).str.strip() != "")
    return diag[mask]


def _bullet_fire(df: pd.DataFrame) -> list[str]:
    # fire_rate here is the target-minus-opponent CONTRAST disagreement rate (from z_diff),
    # NOT absolute prevalence — so word it as "differs from opponent", not "does X% of the
    # time". Per-model absolute prevalence is available in the viewer's report card (#1).
    return [f"- {r['concept']} — differs from opponent in {r['fire_rate']:.0%} of battles"
            for _, r in df.iterrows()]


def format_report(diag: pd.DataFrame, *, model: str, n_battles: int, win_rate: float,
                  top: int = 15, prompt_winrates: pd.DataFrame | None = None,
                  relations: pd.DataFrame | None = None) -> str:
    """Render a markdown concept report card from a diagnosis DataFrame.

    No I/O, no embedding. ``diag`` is the per-feature frame from ``run_diagnose``
    (needs ``concept`` + ``fire_rate`` + ``net_direction``; ``delta_vs_pool`` /
    ``helps_win`` / ``outcome_assoc_lc`` used when present). Features without a
    ``concept`` name are skipped. ``relations`` (from ``prompt_to_response_winrates``)
    adds the per-model prompt→response section when supplied.
    """
    named = _named(diag)
    wr = float(win_rate)
    wr_s = "n/a" if wr != wr else f"{wr:.0%}"
    lines = [f"# {model} — concept report card", "",
             f"{n_battles} battles · win rate {wr_s}", ""]

    by_fire = named.sort_values("fire_rate", ascending=False)
    lines += ["## Frequently distinguishes from opponents", ""]
    lines += _bullet_fire(by_fire.head(top)) or ["- (no named concepts)"]
    lines += [""]

    lines += ["## Rarely distinguishes from opponents", ""]
    lines += _bullet_fire(by_fire.tail(top).iloc[::-1]) or ["- (no named concepts)"]
    lines += [""]

    lines += ["## Rewarded gaps", ""]
    lines += _rewarded_gaps(named, top)
    lines += [""]

    if prompt_winrates is not None:
        lines += _prompt_types(prompt_winrates, top)

    if relations is not None:
        lines += [""] + _relations(relations, top)

    return "\n".join(lines).rstrip() + "\n"


def _relations(rel: pd.DataFrame, top: int) -> list[str]:
    """Per-model prompt→response edges, strongest |Δwin| first."""
    by = rel.reindex(rel["delta_win"].abs().sort_values(ascending=False).index).head(top)
    lines = ["## Prompt → Response", ""]
    if by.empty:
        return lines + ["- (none — no prompt→response edge clears the support floor)"]
    return lines + [f"- {r['prompt_concept']} ⇒ {r['response_concept']} — "
                    f"{r['delta_win']:+.2f} Δwin (n={int(r['n'])})"
                    for _, r in by.iterrows()]


def _rewarded_gaps(named: pd.DataFrame, top: int) -> list[str]:
    """Concepts the model UNDER-expresses AND that are rewarded.

    Under-expression = ``delta_vs_pool`` (vs the pool, when a bank was passed) else
    ``net_direction`` < 0. Reward = ``helps_win`` (global length-controlled
    Δwin-rate) if present, else the within-model ``outcome_assoc_lc``. Sorted by the
    reward signal, descending. No reward column → a one-line hint instead.
    """
    reward_col = next((c for c in ("helps_win", "outcome_assoc_lc")
                       if c in named.columns), None)
    if reward_col is None:
        return ["(pass --win-relevance to surface rewarded gaps)"]
    under_col = "delta_vs_pool" if "delta_vs_pool" in named.columns else "net_direction"

    gaps = named[(named[under_col] < 0) & (named[reward_col] > 0)]
    gaps = gaps.sort_values(reward_col, ascending=False).head(top)
    if gaps.empty:
        return ["- (none — no rewarded concept is under-expressed)"]
    return [f"- {r['concept']} — under-expressed, +{r[reward_col]:.2f} Δwin "
            f"(length-controlled)" for _, r in gaps.iterrows()]


def _prompt_types(pw: pd.DataFrame, top: int) -> list[str]:
    by_wr = pw.sort_values("win_rate", ascending=False)
    fmt = lambda r: (f"- {r['prompt_concept']} — win rate {r['win_rate']:.0%} "  # noqa: E731
                     f"(n={int(r['n'])})")
    lines = ["## Strong / weak prompt types", "", "Strongest:"]
    lines += [fmt(r) for _, r in by_wr.head(top).iterrows()] or ["- (none)"]
    lines += ["", "Weakest:"]
    lines += [fmt(r) for _, r in by_wr.tail(top).iloc[::-1].iterrows()] or ["- (none)"]
    return lines


def _prompt_name_map(prompt_names) -> dict:
    """Accept either a {feature_id: name} dict or a names DataFrame (feature_id, concept)."""
    if prompt_names is None:
        return {}
    if isinstance(prompt_names, dict):
        return {int(k): v for k, v in prompt_names.items()}
    if isinstance(prompt_names, pd.DataFrame) and \
            {"feature_id", "concept"} <= set(prompt_names.columns):
        return dict(zip(prompt_names["feature_id"].astype(int), prompt_names["concept"]))
    return {}


def prompt_concept_winrates(prompt_lens_dir, battle_ids, win, *, prompt_names=None,
                            min_battles: int = 20) -> pd.DataFrame:
    """Per-prompt-concept win rate for one model.

    Loads the prompt lens's ``z_prompt`` (row-aligned to its ``battles.parquet`` by
    battle id), assigns each of the model's battles its dominant prompt concept
    (``argmax``), and returns ``win`` averaged per concept. Battles whose id is not
    in the prompt lens are dropped; concepts seen in fewer than ``min_battles`` are
    filtered out. Columns: ``prompt_concept`` (name if ``prompt_names`` given, else
    id), ``win_rate``, ``n``.
    """
    plens = Path(prompt_lens_dir)
    z_prompt = np.load(plens / Z_PROMPT)
    pb = lens_battle_ids(plens)
    ppos = {b: i for i, b in enumerate(pb)}

    battle_ids = [str(b) for b in battle_ids]
    win = np.asarray(win, dtype=float)
    rows, keep = [], np.zeros(len(battle_ids), dtype=bool)
    for j, b in enumerate(battle_ids):
        i = ppos.get(b)
        if i is not None:
            rows.append(i)
            keep[j] = True
    if not rows:
        return pd.DataFrame(columns=["prompt_concept", "win_rate", "n"])

    # dominant prompt concept per battle, but require a POSITIVE max — a silent (all-zero)
    # or all-negative prompt code has no concept present, so drop it rather than let argmax
    # assign it to feature 0 / the least-negative feature (#4).
    zc = z_prompt[rows]
    dom = np.where(zc.max(axis=1) > 0, zc.argmax(axis=1), -1)
    pos = dom >= 0
    agg = (pd.DataFrame({"prompt_concept": dom[pos].astype(int), "win": win[keep][pos]})
           .groupby("prompt_concept")["win"]
           .agg(win_rate="mean", n="count").reset_index())
    if agg.empty:
        return pd.DataFrame(columns=["prompt_concept", "win_rate", "n"])
    agg = agg[agg["n"] >= min_battles].reset_index(drop=True)

    names = _prompt_name_map(prompt_names)
    if names:
        agg["prompt_concept"] = agg["prompt_concept"].map(
            lambda c: names.get(int(c), int(c)))
    return agg


def prompt_to_response_winrates(prompt_lens_dir, battle_ids, response_codes, feature_ids,
                                win, *, prompt_names=None, response_names=None,
                                min_support: int = 20, top: int = 15) -> pd.DataFrame:
    """Per-model prompt-concept → response-concept → Δwin edges.

    For one model's battles: assign each battle its dominant prompt concept (prompt
    lens ``z_prompt`` argmax) and, per response feature, contrast the model's win
    rate when that response concept FIRES (oriented code > 0 — the model expressed it
    more than its opponent) against when it does not, WITHIN the same prompt concept.
    ``delta_win`` is that within-prompt contrast (so it answers "given this kind of
    prompt, does producing this concept help this model win"). Each edge needs at
    least ``min_support`` battles on BOTH sides; the strongest ``top`` by |Δwin| are
    returned. Columns: ``prompt_concept``, ``response_concept``, ``delta_win``, ``n``
    (battles where the concept fired). Battles absent from the prompt lens are dropped.

    ``response_codes`` is (n_battles, F) oriented codes aligned to ``battle_ids`` and
    ``feature_ids`` (its F columns). Names map ids → concepts when supplied.
    """
    plens = Path(prompt_lens_dir)
    z_prompt = np.load(plens / Z_PROMPT)
    pb = lens_battle_ids(plens)
    ppos = {b: i for i, b in enumerate(pb)}

    battle_ids = [str(b) for b in battle_ids]
    win = np.asarray(win, dtype=float)
    codes = np.asarray(response_codes, dtype=float)
    feat_ids = [int(f) for f in feature_ids]
    cols = ["prompt_concept", "response_concept", "delta_win", "n"]

    rows, keep = [], np.zeros(len(battle_ids), dtype=bool)
    for j, b in enumerate(battle_ids):
        i = ppos.get(b)
        if i is not None:
            rows.append(i)
            keep[j] = True
    if not rows:
        return pd.DataFrame(columns=cols)

    # dominant prompt concept per battle, but require a POSITIVE max — a silent (all-zero)
    # or all-negative prompt code has no concept present. Drop those (-1) rather than let
    # argmax assign them to feature 0 / the least-negative feature (#4, sibling of
    # prompt_concept_winrates above).
    zc = z_prompt[rows]
    dom = np.where(zc.max(axis=1) > 0, zc.argmax(axis=1), -1).astype(int)   # (n_keep,)
    fired = codes[keep] > 0                                 # (n_keep, F)
    win_keep = win[keep]

    edges = []
    for k in np.unique(dom):
        if k < 0:                                          # no positive prompt concept
            continue
        mask = dom == k
        if mask.sum() < min_support:
            continue
        fk = fired[mask]                                    # (n_k, F)
        wk = win_keep[mask]                                 # (n_k,)
        nf = fk.sum(axis=0)                                 # fired count per feature
        nn = (~fk).sum(axis=0)                              # not-fired count
        wr_f = (fk * wk[:, None]).sum(axis=0) / np.where(nf > 0, nf, 1)
        wr_n = ((~fk) * wk[:, None]).sum(axis=0) / np.where(nn > 0, nn, 1)
        delta = wr_f - wr_n
        valid = (nf >= min_support) & (nn >= min_support)
        for fi in np.nonzero(valid)[0]:
            edges.append((int(k), feat_ids[fi], float(delta[fi]), int(nf[fi])))

    if not edges:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(edges, columns=cols)
    df = df.reindex(df["delta_win"].abs().sort_values(ascending=False).index)
    df = df.head(top).reset_index(drop=True)

    pmap = _prompt_name_map(prompt_names)
    rmap = _prompt_name_map(response_names)
    if pmap:
        df["prompt_concept"] = df["prompt_concept"].map(lambda c: pmap.get(int(c), int(c)))
    if rmap:
        df["response_concept"] = df["response_concept"].map(lambda c: rmap.get(int(c), int(c)))
    return df
