"""Example-battle exports: per-feature shards, per-model drill-ins, and the
report-card sample battles."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from prefscope.analysis.presence import annotation_flag
from prefscope.artifacts import BATTLES, Z_A, Z_B, Z_DIFF, Z_PROMPT, lens_battle_ids

from .presence import feature_thresholds

EXAMPLE_GROUP_COLUMNS = ("language", "lang", "source")


def _percentile(sorted_reference: np.ndarray, value: float) -> float | None:
    """Exact within-axis percentile for one exported activation strength."""
    if not len(sorted_reference):
        return None
    rank = np.searchsorted(sorted_reference, value, side="right")
    return round(100.0 * float(rank) / len(sorted_reference), 2)


def _group_fields(frame: pd.DataFrame, row: int) -> dict:
    """Optional corpus slice carried with an example for viewer-side filtering."""
    for column in EXAMPLE_GROUP_COLUMNS:
        if column in frame.columns:
            value = frame.iloc[row].get(column)
            if pd.notna(value) and str(value).strip():
                return {"group": str(value), "group_column": column}
    return {}


def export_examples(lens: Path, corpus_path: str, features: pd.DataFrame,
                    n_per: int = 8, n_per_group: int = 2,
                    n_random: int = 0, n_boundary: int = 0,
                    seed: int = 0) -> dict | None:
    """Top-activating example battles for every available SAE feature.

    Unnamed and failed-verification axes are deliberately included: concrete examples
    are how a user can inspect those points in the feature atlas and decide whether a
    missing/bad label merits another interpretation pass.
    Text is capped tightly because the viewer clips to ~1.4k chars anyway; storing
    more is pure payload. The bundle is loaded lazily (only when an examples view
    opens), so covering all ~hundreds of named features doesn't slow startup.
    """
    if not corpus_path:
        return None
    from prefscope.interpret.io import load_lens_battles
    battles, z_diff, _ = load_lens_battles(lens, corpus=corpus_path)
    contrast = (Path(lens) / Z_DIFF).exists()
    group_column = next((column for column in EXAMPLE_GROUP_COLUMNS
                         if column in battles.columns), None)
    group_values = (battles[group_column].fillna("").astype(str).to_numpy()
                    if group_column else None)
    feats = pd.to_numeric(features["feature_id"], errors="coerce").dropna().astype(int)
    feats = [f for f in feats.drop_duplicates().tolist() if 0 <= f < z_diff.shape[1]]
    thresholds, calibrated = feature_thresholds(features, feats)

    def trunc(s, n):
        s = str(s)
        return s if len(s) <= n else s[:n] + " …[truncated]"

    out = {}
    for feature_pos, f in enumerate(feats):
        col = np.asarray(z_diff[:, f], dtype=np.float32)
        order = np.argsort(-np.abs(col))
        picks = [int(i) for i in order if col[i] != 0][:n_per]
        selection = {row: "strongest" for row in picks}
        if group_values is not None and n_per_group > 0:
            for group in sorted(set(group_values) - {""}):
                members = np.flatnonzero(group_values == group)
                ranked = members[np.argsort(-np.abs(col[members]))]
                extra = [int(i) for i in ranked if col[i] != 0
                         and int(i) not in picks[:n_per]][:n_per_group]
                picks.extend(extra)
                for row in extra:
                    selection.setdefault(row, "group_strongest")
        active = np.flatnonzero(col != 0 if contrast else col > 0)
        available = np.asarray([row for row in active if int(row) not in selection], dtype=int)
        if n_random > 0 and len(available):
            rng = np.random.default_rng(seed + int(f))
            random_rows = rng.choice(available, size=min(n_random, len(available)),
                                     replace=False).astype(int).tolist()
            picks.extend(random_rows)
            for row in random_rows:
                selection.setdefault(row, "random_contrast" if contrast else "random_present")
        remaining = np.asarray([row for row in active if int(row) not in selection], dtype=int)
        if n_boundary > 0 and len(remaining):
            if not contrast and calibrated[feature_pos]:
                threshold = float(thresholds[feature_pos])
                eligible = remaining[col[remaining] >= threshold]
                ranked_boundary = eligible[np.argsort(np.abs(col[eligible] - threshold))]
                boundary_kind = "near_threshold"
            else:
                ranked_boundary = remaining[np.argsort(np.abs(col[remaining]))]
                boundary_kind = "near_boundary"
            boundary_rows = ranked_boundary[:n_boundary].astype(int).tolist()
            picks.extend(boundary_rows)
            for row in boundary_rows:
                selection.setdefault(row, boundary_kind)
        picks = sorted(set(picks), key=lambda i: -abs(float(col[i])))
        reference = np.sort(np.abs(col[active]) if contrast else col[active])
        rows = []
        for i in picks:
            b = battles.iloc[int(i)]
            strength = abs(float(col[i])) if contrast else float(col[i])
            rows.append({
                "z": round(float(col[i]), 4),
                "activation_percentile": _percentile(reference, strength),
                "activation_reference": "absolute_contrast" if contrast else "positive_activation",
                "selection_kind": selection[i],
                "prompt": trunc(b.get("prompt", ""), 800),
                "model_a": str(b.get("model_a", "A")),
                "model_b": str(b.get("model_b", "B")),
                "completion_a": trunc(b.get("completion_a", ""), 2000),
                "completion_b": trunc(b.get("completion_b", ""), 2000),
                **_group_fields(battles, int(i)),
            })
        out[str(f)] = rows
    return out


def export_prompt_examples(prompt_lens: Path, corpus_path: str,
                           features: pd.DataFrame, n_per: int = 8,
                           n_per_group: int = 2,
                           n_random: int = 0, n_boundary: int = 0,
                           seed: int = 0,
                           max_chars: int = 1400) -> dict | None:
    """Top positive-pole prompt activators for every prompt SAE axis.

    Prompt examples are independent of preference labels and prompt→response linkage,
    so unnamed, unverified, and unlinked axes remain inspectable. An empty list means
    the axis has no positive activation in this corpus; the exporter never substitutes
    a least-negative or unrelated prompt merely to fill the UI.
    """
    plens = Path(prompt_lens)
    if not corpus_path or not (plens / Z_PROMPT).exists() or not (plens / BATTLES).exists():
        return None
    z = np.load(plens / Z_PROMPT, mmap_mode="r")
    lens_rows = pd.read_parquet(plens / BATTLES)
    if len(lens_rows) != len(z):
        raise ValueError("prompt battles/z_prompt row mismatch for prompt examples")

    if "prompt" in lens_rows and lens_rows["prompt"].fillna("").astype(str).str.len().gt(0).any():
        prompts = lens_rows["prompt"].fillna("").astype(str).to_numpy()
        group_frame = lens_rows
    else:
        from prefscope.artifacts import battle_id_col
        from prefscope.data.corpus import load_corpus

        corpus = load_corpus(corpus_path)
        lens_ids = lens_battle_ids(lens_rows)
        cid = battle_id_col(corpus)
        corpus_ids = corpus[cid].astype(str)
        aligned = corpus.assign(_id=corpus_ids).drop_duplicates("_id").set_index("_id") \
            .reindex(lens_ids).reset_index(drop=True)
        prompts = aligned["prompt"].fillna("").astype(str).to_numpy()
        group_frame = aligned

    feats = pd.to_numeric(features["feature_id"], errors="coerce").dropna().astype(int)
    feats = [f for f in feats.drop_duplicates().tolist() if 0 <= f < z.shape[1]]

    def trunc(value):
        text = "" if value is None else str(value)
        return text if len(text) <= max_chars else text[:max_chars] + " …[truncated]"

    out: dict[str, list[dict]] = {}
    for feature_id in feats:
        column = np.asarray(z[:, feature_id], dtype=np.float32)
        active = np.flatnonzero(column > 0)
        if len(active):
            order = list(active[np.argsort(-column[active])[:n_per]])
            selection = {int(row): "strongest" for row in order}
            if n_per_group > 0:
                group_column = next((name for name in EXAMPLE_GROUP_COLUMNS
                                     if name in group_frame.columns), None)
                if group_column:
                    group_values = group_frame[group_column].fillna("").astype(str).to_numpy()
                    for group in sorted(set(group_values) - {""}):
                        members = np.flatnonzero((group_values == group) & (column > 0))
                        ranked = members[np.argsort(-column[members])[:n_per_group]]
                        order.extend(int(row) for row in ranked)
                        for row in ranked:
                            selection.setdefault(int(row), "group_strongest")
            available = np.asarray([row for row in active if int(row) not in selection], dtype=int)
            if n_random > 0 and len(available):
                rng = np.random.default_rng(seed + int(feature_id))
                random_rows = rng.choice(available, size=min(n_random, len(available)),
                                         replace=False).astype(int).tolist()
                order.extend(random_rows)
                for row in random_rows:
                    selection.setdefault(row, "random_present")
            remaining = np.asarray([row for row in active if int(row) not in selection], dtype=int)
            if n_boundary > 0 and len(remaining):
                boundary_rows = remaining[np.argsort(column[remaining])[:n_boundary]] \
                    .astype(int).tolist()
                order.extend(boundary_rows)
                for row in boundary_rows:
                    selection.setdefault(row, "near_boundary")
            order = sorted(set(int(row) for row in order), key=lambda row: -float(column[row]))
            reference = np.sort(column[active])
            out[str(feature_id)] = [
                {"z": round(float(column[row]), 4),
                 "activation_percentile": _percentile(reference, float(column[row])),
                 "activation_reference": "positive_activation",
                 "selection_kind": selection[int(row)],
                 "prompt": trunc(prompts[row]),
                 **_group_fields(group_frame, int(row))}
                for row in order
            ]
        else:
            out[str(feature_id)] = []
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


def export_joint_examples(lens: Path, corpus_path: str, prompt_lens, pairs, *,
                          per_pair: int = 3, prompt_chars: int = 700,
                          response_chars: int = 1400) -> dict | None:
    """Top examples where one prompt feature *and* one response feature are active.

    ``pairs`` is an iterable of ``(prompt_feature, response_feature)`` IDs already
    selected for the viewer's elicitation/conditional tables.  Results are sharded by
    prompt feature by the caller, so opening one prompt does not download every
    transcript in the corpus.

    This export deliberately uses raw positive-pole activations rather than semantic
    thresholds.  Within each pair, candidates are ranked by the geometric mean of the
    prompt and response activations after robust, pair-local normalization.  That makes
    an example rank highly only when *both* sides of the relationship are strong; one
    extreme activation cannot fully compensate for a nearly silent counterpart.
    """
    lens, plens = Path(lens), Path(prompt_lens) if prompt_lens else None
    if not corpus_path or plens is None:
        return None
    if not (lens / Z_A).exists() or not (plens / Z_PROMPT).exists():
        return None

    pair_set = sorted({(int(pc), int(cf)) for pc, cf in pairs})
    if not pair_set:
        return None

    from prefscope.interpret.io import load_lens_battles

    battles, _z, _ = load_lens_battles(lens, corpus=corpus_path)
    za = np.load(lens / Z_A, mmap_mode="r")
    zb = np.load(lens / Z_B, mmap_mode="r") if (lens / Z_B).exists() else None
    zp = np.load(plens / Z_PROMPT, mmap_mode="r")
    cb = lens_battle_ids(lens)
    pb = lens_battle_ids(plens)
    if len(battles) != len(za) or len(cb) != len(za) \
            or (zb is not None and len(za) != len(zb)):
        raise ValueError("completion battles/response-code row mismatch for joint examples")

    # Preserve completion-lens order so ``battles.iloc[ic]`` remains aligned with z_a/z_b.
    if len(cb) == len(pb) and bool((cb == pb).all()):
        ic = np.arange(len(cb), dtype=int)
        ip = ic
    else:
        ppos = {b: i for i, b in enumerate(pb)}
        ic = np.array([i for i, b in enumerate(cb) if b in ppos], dtype=int)
        ip = np.array([ppos[cb[i]] for i in ic], dtype=int)
    if not len(ic):
        return None
    battles = battles.iloc[ic].reset_index(drop=True)

    valid_pairs = [(pc, cf) for pc, cf in pair_set
                   if 0 <= pc < zp.shape[1] and 0 <= cf < za.shape[1]]
    if not valid_pairs:
        return None

    by_prompt: dict[int, list[int]] = {}
    for pc, cf in valid_pairs:
        by_prompt.setdefault(pc, []).append(cf)

    prompts = battles["prompt"].fillna("").astype(str).to_numpy()
    ca = battles["completion_a"].fillna("").astype(str).to_numpy()
    cb_text = (battles["completion_b"].fillna("").astype(str).to_numpy()
               if "completion_b" in battles else None)
    ma = battles.get("model_a", pd.Series("A", index=battles.index)).astype(str).to_numpy()
    mb = battles.get("model_b", pd.Series("B", index=battles.index)).astype(str).to_numpy()
    ycol = next((c for c in ("human_pref", "y_judge") if c in battles.columns), None)
    y = battles[ycol].to_numpy(dtype=float) if ycol else None
    group_column = next((column for column in EXAMPLE_GROUP_COLUMNS
                         if column in battles.columns), None)
    group_values = (battles[group_column].astype(str).to_numpy()
                    if group_column else None)

    def trunc(value, limit):
        value = "" if value is None else str(value)
        return value if len(value) <= limit else value[:limit] + " …[truncated]"

    def outcome(side: str, i: int) -> str | None:
        if y is None or np.isnan(y[i]) or y[i] not in (0.0, 0.5, 1.0):
            return None
        if y[i] == 0.5:
            return "tie"
        a_won = y[i] == 1.0
        return ("win" if a_won else "loss") if side == "a" else \
            ("win" if not a_won else "loss")

    out: dict[str, dict] = {}
    for pc, cfs in by_prompt.items():
        # Keep the dense code matrices memory-mapped.  Indexing the entire matrices by
        # ``ic``/``ip`` would silently allocate several GiB on an Arena-sized corpus.
        # These 1-D feature reads are the only data materialized for each relationship.
        p_all = np.asarray(zp[ip, pc], dtype=np.float32)
        pidx = np.flatnonzero(p_all > 0)
        if not len(pidx):
            continue
        p_act = p_all[pidx]
        p_scale = float(np.quantile(p_act, 0.99))
        if not np.isfinite(p_scale) or p_scale <= 0:
            p_scale = float(p_act.max())
        p_norm = np.clip(p_act / max(p_scale, 1e-12), 0.0, 1.0)

        cfs = sorted(set(cfs))
        examples: dict[str, list[dict]] = {}
        response_rows = ic[pidx]
        # Read neighboring response features in modest blocks. One-column-at-a-time
        # access is pathological for row-major memmaps; a full prompt×all-features block
        # can be GiB. 64 columns keeps peak memory bounded while making the scan sequential.
        for start in range(0, len(cfs), 64):
            block_cfs = cfs[start:start + 64]
            a_block = np.asarray(za[np.ix_(response_rows, block_cfs)], dtype=np.float32)
            b_block = (np.asarray(zb[np.ix_(response_rows, block_cfs)], dtype=np.float32)
                       if zb is not None else None)
            for j, cf in enumerate(block_cfs):
                a_act = a_block[:, j]
                b_act = b_block[:, j] if b_block is not None else None
                positive = (np.concatenate([a_act[a_act > 0], b_act[b_act > 0]])
                            if b_act is not None else a_act[a_act > 0])
                if not len(positive):
                    continue
                r_scale = float(np.quantile(positive, 0.99))
                if not np.isfinite(r_scale) or r_scale <= 0:
                    r_scale = float(positive.max())

                candidates = []
                side_activations = [("a", a_act)]
                if b_act is not None:
                    side_activations.append(("b", b_act))
                for side, acts in side_activations:
                    local = np.flatnonzero(acts > 0)
                    if not len(local):
                        continue
                    r_norm = np.clip(acts[local] / max(r_scale, 1e-12), 0.0, 1.0)
                    score = np.sqrt(p_norm[local] * r_norm)
                    candidates.extend((float(score[k]), int(local[k]), side)
                                      for k in range(len(local)))
                candidates.sort(key=lambda row: row[0], reverse=True)

                picked, seen_battles = [], set()
                for score, local_i, side in candidates:
                    if local_i in seen_battles:
                        continue
                    seen_battles.add(local_i)
                    i = int(pidx[local_i])
                    response_act = a_act[local_i] if side == "a" else b_act[local_i]
                    row = {
                        "prompt_activation": round(float(p_act[local_i]), 4),
                        "response_activation": round(float(response_act), 4),
                        "joint_score": round(score, 4),
                        "prompt": trunc(prompts[i], prompt_chars),
                        "response": trunc(ca[i] if side == "a" else cb_text[i], response_chars),
                        "model": str(ma[i] if side == "a" else mb[i]),
                        "side": side,
                    }
                    if group_values is not None:
                        row["group"] = str(group_values[i])
                        row["group_column"] = group_column
                    result = outcome(side, i)
                    if result is not None:
                        row["outcome"] = result
                    picked.append(row)
                    if len(picked) >= per_pair:
                        break
                if picked:
                    examples[str(cf)] = picked
        if examples:
            out[str(pc)] = {"prompt_feature": pc, "examples": examples}
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

    feats = features.loc[features["fidelity_pass"].map(annotation_flag), "feature_id"] \
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
    thresholds, calibrated = feature_thresholds(features, feats)
    threshold_by_feature = dict(zip(feats, thresholds))
    calibrated_by_feature = dict(zip(feats, calibrated))
    for f in feats:
        for act, marr, side in ((np.asarray(za[:, f]), ma, "a"),
                                (np.asarray(zb[:, f]), mb, "b")):
            mask = (act >= threshold_by_feature[f] if calibrated_by_feature[f]
                    else act > 0)
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
