"""Example-battle exports: per-feature shards, per-model drill-ins, and the
report-card sample battles."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from prefscope.artifacts import Z_PROMPT, lens_battle_ids


def export_examples(lens: Path, corpus_path: str, features: pd.DataFrame,
                    n_per: int = 8) -> dict | None:
    """Top-activating example battles per NAMED feature (verified or not).

    Examples are surfaced for every named axis — not just verified ones — so an
    unverified feature can still be inspected/analyzed by its concrete activations.
    Text is capped tightly because the viewer clips to ~1.4k chars anyway; storing
    more is pure payload. The bundle is loaded lazily (only when an examples view
    opens), so covering all ~hundreds of named features doesn't slow startup.
    """
    if not corpus_path:
        return None
    from prefscope.interpret.io import load_lens_battles
    battles, z_diff, _ = load_lens_battles(lens, corpus=corpus_path)
    # all NAMED features (concept present), verified or not — was verified-only.
    if "concept" in features.columns:
        feats = features.loc[features["concept"].notna()
                             & (features["concept"].astype(str).str.strip() != ""),
                             "feature_id"]
    else:
        feats = features["feature_id"]
    feats = feats.astype(int).tolist()

    def trunc(s, n):
        s = str(s)
        return s if len(s) <= n else s[:n] + " …[truncated]"

    out = {}
    for f in feats:
        col = z_diff[:, f]
        order = np.argsort(-np.abs(col))
        picks = [i for i in order if col[i] != 0][: n_per * 2][: n_per]
        rows = []
        for i in picks:
            b = battles.iloc[int(i)]
            rows.append({
                "z": round(float(col[i]), 4),
                "prompt": trunc(b.get("prompt", ""), 800),
                "model_a": str(b.get("model_a", "A")),
                "model_b": str(b.get("model_b", "B")),
                "completion_a": trunc(b.get("completion_a", ""), 2000),
                "completion_b": trunc(b.get("completion_b", ""), 2000),
            })
        out[str(f)] = rows
    return out


def export_report_battles(lens: Path, corpus_path: str, prompt_lens, diag,
                          prompt_names=None, *, per_type: int = 5,
                          max_chars: int = 500) -> dict | None:
    """Per (model × prompt-concept) sample battles, for the report-card drill-in.

    For each model in ``diag`` and each prompt concept that appears in its
    ``prompt_types``, take up to ``per_type`` of that model's battles on that prompt
    concept — the most prompt-typical ones (highest prompt-concept activation) —
    oriented so ``self`` is this model's answer and ``other`` the opponent's, with the
    outcome from this model's perspective. Text is truncated to ``max_chars`` to keep
    the bundle small. Returns ``{model: {concept_name: [{prompt, self, other,
    outcome}]}}`` or None if the inputs aren't available.
    """
    if not corpus_path or prompt_lens is None or diag is None or not diag.get("models"):
        return None
    from prefscope.data.corpus import load_corpus
    from prefscope.data.orient import orient_to_model

    corp = load_corpus(corpus_path)
    if "human_pref" not in corp.columns or corp["human_pref"].isna().all():
        return None
    corp = corp.dropna(subset=["human_pref"]).copy()
    corp["instruction_id"] = corp["instruction_id"].astype(str)
    corp["y_judge"] = corp["human_pref"].astype(float)
    # orient_to_model only accepts decisive/tie labels in {0, 0.5, 1}; drop anything
    # else (e.g. averaged annotators / quarter-ties) so a BYO dataset can't abort the
    # export. This is a sample drill-in, so dropping a few battles is harmless.
    corp = corp[corp["y_judge"].isin([0.0, 0.5, 1.0])]
    if corp.empty:
        return None

    # battle_id -> (dominant prompt concept name, that concept's activation)
    pl = Path(prompt_lens)
    zp = np.load(pl / Z_PROMPT)
    pb = [str(b) for b in lens_battle_ids(pl)]
    # require a POSITIVE max — a silent/all-negative prompt has no concept present, so it
    # gets a None sentinel that never matches a real `wanted` concept (dropped below),
    # rather than argmax mislabelling it as feature 0 / the least-negative pole.
    dom = np.where(zp.max(axis=1) > 0, zp.argmax(axis=1), -1)
    nmap = {}
    if isinstance(prompt_names, pd.DataFrame) and \
            {"feature_id", "concept"} <= set(prompt_names.columns):
        nmap = dict(zip(prompt_names["feature_id"].astype(int), prompt_names["concept"]))
    bid_concept = {b: (str(nmap.get(int(dom[i]), int(dom[i]))) if dom[i] >= 0 else None)
                   for i, b in enumerate(pb)}
    bid_act = {b: (float(zp[i, dom[i]]) if dom[i] >= 0 else 0.0) for i, b in enumerate(pb)}

    def trunc(s):
        s = "" if s is None else str(s)
        return s if len(s) <= max_chars else s[:max_chars] + " …[truncated]"

    out: dict = {}
    for m in diag["models"]:
        wanted = {pt["concept"] for pt in diag["rows"].get(m, {}).get("prompt_types", [])}
        if not wanted:
            continue
        ob = orient_to_model(corp, m)
        if ob.empty:
            continue
        ids = ob["instruction_id"].astype(str)
        ob = ob.assign(_concept=ids.map(bid_concept), _act=ids.map(bid_act).fillna(0.0))
        ob = ob[ob["_concept"].isin(wanted)].sort_values("_act", ascending=False)
        per_concept = {}
        for c, g in ob.groupby("_concept"):
            per_concept[str(c)] = [{
                "prompt": trunc(r.prompt),
                "self": trunc(r.self_completion),
                "other": trunc(r.other_completion),
                "outcome": str(r.outcome),
            } for r in g.head(per_type).itertuples()]
        if per_concept:
            out[m] = per_concept
    return out or None


def export_examples_by_model(lens: Path, corpus_path: str, features: pd.DataFrame,
                             diag, *, n_per: int = 4, max_chars: int = 1500) -> dict | None:
    """Per (model × feature) example answers — the model's OWN responses that most strongly
    exhibit the feature, so the report-card drill-in never falls back to "sampled across
    models". Uses the individual lens's per-side codes z_a/z_b (activation of the feature on
    each model's answer) + the corpus text; outcome is from the model's perspective.

    Returns ``{model: {feature_id: [{z, prompt, answer, outcome}]}}`` (outcome ∈
    win/loss/tie/?), or None without a corpus / per-side codes / diagnosis."""
    if not corpus_path or diag is None or not diag.get("models"):
        return None
    za_p, zb_p = lens / "z_a.npy", lens / "z_b.npy"
    if not (za_p.exists() and zb_p.exists()):
        return None
    from prefscope.interpret.io import load_lens_battles
    battles, _z, _ = load_lens_battles(lens, corpus=corpus_path)
    za = np.load(za_p, mmap_mode="r")
    zb = np.load(zb_p, mmap_mode="r")
    if len(battles) != len(za) or len(battles) != len(zb):
        return None

    feats = features.loc[features.get("fidelity_pass", False) == True, "feature_id"] \
        if "fidelity_pass" in features else features["feature_id"]
    feats = feats.astype(int).tolist()
    if not feats:
        return None
    models = set(diag["models"])
    ma = battles["model_a"].astype(str).to_numpy()
    mb = battles["model_b"].astype(str).to_numpy()
    prompts = battles["prompt"].astype(str).to_numpy()
    ca = battles["completion_a"].astype(str).to_numpy()
    cb = battles["completion_b"].astype(str).to_numpy()
    ycol = next((c for c in ("y_judge", "human_pref") if c in battles.columns), None)
    y = battles[ycol].to_numpy(dtype=float) if ycol else None

    def outcome(side: str, i: int) -> str:
        if y is None or np.isnan(y[i]) or y[i] not in (0.0, 0.5, 1.0):
            return "?"                      # non-decisive (averaged annotators / quarter-ties)
        if y[i] == 0.5:
            return "tie"
        a_won = y[i] == 1.0
        return ("win" if a_won else "loss") if side == "a" else ("win" if not a_won else "loss")

    def trunc(s, n):
        s = "" if s is None else str(s)
        return s if len(s) <= n else s[:n] + " …[truncated]"

    # candidate firing events (model, feature, activation, battle, side), model in universe.
    # Signed SAE: the concept NAME describes the POSITIVE pole, so we keep only positive-pole
    # firings (act > 0) — a strongly-negative activation is the OPPOSITE pole (a different
    # concept) and must NOT be surfaced under "answers exhibiting <concept>".
    parts = []
    for f in feats:
        for act, marr, side in ((np.asarray(za[:, f]), ma, "a"),
                                (np.asarray(zb[:, f]), mb, "b")):
            mask = act > 0
            if not mask.any():
                continue
            idx = np.nonzero(mask)[0]
            parts.append(pd.DataFrame({"f": f, "m": marr[idx], "act": act[idx],
                                       "i": idx, "side": side}))
    if not parts:
        return None
    rec = pd.concat(parts, ignore_index=True)
    rec = rec[rec["m"].isin(models)]
    if rec.empty:
        return None
    # strongest concept-pole expression first (signed, not |·|)
    top = rec.sort_values("act", ascending=False).groupby(["m", "f"], sort=False).head(n_per)

    out: dict = {}
    for r in top.itertuples():
        i, side = int(r.i), r.side
        out.setdefault(r.m, {}).setdefault(str(int(r.f)), []).append({
            "z": round(float(r.act), 4),
            "prompt": trunc(prompts[i], 500),
            "answer": trunc(ca[i] if side == "a" else cb[i], max_chars),
            "outcome": outcome(side, i),
        })
    return out or None
