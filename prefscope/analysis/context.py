"""Prompt scope and cross-context stability for response features."""
from __future__ import annotations

import json
import math
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import binomtest, hypergeom

from prefscope.analysis.grouping import typed_group_keys, validate_group_ids
from prefscope.analysis.presence import annotation_flag, feature_thresholds


BEHAVIOR_CATEGORIES = ("general", "context_specific", "prompt_content", "unclassified")
PROMPT_SCOPES = (
    "prompt_linked",
    "no_detected_prompt_link",
    "insufficient_evidence",
)


def _strict_boolean_array(values, *, name: str, ndim: int) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D")
    if raw.dtype == bool:
        return raw
    if not np.issubdtype(raw.dtype, np.number):
        raise ValueError(f"{name} must contain boolean or numeric 0/1 values")
    numeric = np.asarray(raw, dtype=float)
    if not np.isfinite(numeric).all() or not np.isin(numeric, [0.0, 1.0]).all():
        raise ValueError(f"{name} must contain finite boolean or numeric 0/1 values")
    return numeric.astype(bool)


def _entropy(probabilities) -> float:
    p = np.asarray(probabilities, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum()) if len(p) else 0.0


def _prompt_dependence(presence: np.ndarray, contexts: np.ndarray) -> tuple[float, float]:
    """Return I(presence; context)/H(presence) and Jensen-Shannon shift.

    NMI measures how predictable semantic presence is from prompt context. JSD compares
    the prompt-context mixture among present responses with the overall prompt mixture.
    """
    presence = _strict_boolean_array(presence, name="presence", ndim=1)
    contexts = np.asarray(contexts)
    n = len(presence)
    if not n or presence.all() or (~presence).all():
        return float("nan"), float("nan")
    _, inv = np.unique(contexts, return_inverse=True)
    counts = np.bincount(inv).astype(float)
    positive = np.bincount(inv[presence], minlength=len(counts)).astype(float)
    negative = counts - positive
    py = np.array([(~presence).mean(), presence.mean()])
    pk = counts / n
    mi = 0.0
    for y, joint_counts in enumerate((negative, positive)):
        joint = joint_counts / n
        nz = joint > 0
        mi += float(np.sum(joint[nz] * np.log(joint[nz] / (py[y] * pk[nz]))))
    hy = _entropy(py)
    nmi = mi / hy if hy > 0 else float("nan")
    q = positive / positive.sum()
    midpoint = 0.5 * (q + pk)
    qnz, pnz = q > 0, pk > 0
    js = 0.5 * np.sum(q[qnz] * np.log(q[qnz] / midpoint[qnz]))
    js += 0.5 * np.sum(pk[pnz] * np.log(pk[pnz] / midpoint[pnz]))
    return float(nmi), float(js / math.log(2.0))


def _region_dependence(presence: np.ndarray, membership: np.ndarray) -> tuple[float, float]:
    """Dependence for overlapping regions: strongest binary NMI + membership JSD."""
    presence = _strict_boolean_array(presence, name="presence", ndim=1)
    membership = _strict_boolean_array(
        membership, name="region membership", ndim=2)
    if len(membership) != len(presence):
        raise ValueError("region membership must be 2-D and aligned to presence")
    nmis = [_prompt_dependence(presence, membership[:, j].astype(np.int8))[0]
            for j in range(membership.shape[1])]
    finite = [value for value in nmis if np.isfinite(value)]
    max_nmi = max(finite) if finite else float("nan")
    total = membership.sum(axis=0).astype(float)
    positive = membership[presence].sum(axis=0).astype(float)
    if total.sum() == 0 or positive.sum() == 0:
        return float(max_nmi), float("nan")
    p, q = total / total.sum(), positive / positive.sum()
    midpoint = 0.5 * (p + q)
    pnz, qnz = p > 0, q > 0
    js = 0.5 * np.sum(p[pnz] * np.log(p[pnz] / midpoint[pnz]))
    js += 0.5 * np.sum(q[qnz] * np.log(q[qnz] / midpoint[qnz]))
    return float(max_nmi), float(js / math.log(2.0))


def _context_membership(prompt_context, context_ids=None):
    contexts = np.asarray(prompt_context)
    if contexts.ndim == 1:
        values = validate_group_ids(
            prompt_context, len(contexts), name="prompt_context")
        value_keys = typed_group_keys(values)
        if context_ids is None:
            key_list = value_keys.tolist()
            ordered_keys = list(dict.fromkeys(key_list))
            labels = np.empty(len(ordered_keys), dtype=object)
            for index, key in enumerate(ordered_keys):
                labels[index] = values[key_list.index(key)]
        else:
            requested_values = tuple(context_ids)
            requested = validate_group_ids(
                requested_values, len(requested_values), name="prompt_context_ids")
            requested_keys = typed_group_keys(requested)
            if len(set(requested_keys)) != len(requested_keys):
                raise ValueError("prompt_context_ids must be unique")
            ordered_keys = list(requested_keys)
            labels = requested
        membership = np.column_stack([
            np.asarray([key == requested for key in value_keys], dtype=bool)
            for requested in ordered_keys
        ])
        ids = labels
    elif contexts.ndim == 2:
        membership = _strict_boolean_array(
            contexts, name="prompt_context membership", ndim=2)
        if context_ids is None:
            ids = np.arange(membership.shape[1], dtype=int)
        else:
            ids = validate_group_ids(
                context_ids, membership.shape[1], name="prompt_context_ids")
            if len(set(typed_group_keys(ids).tolist())) != len(ids):
                raise ValueError("prompt_context_ids must contain unique values")
    else:
        raise ValueError("prompt_context must be 1-D labels or 2-D overlapping membership")
    return np.asarray(ids), membership


def _bh_adjust(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid):
        return out
    order = valid[np.argsort(p[valid])]
    ranked = p[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out[order] = np.minimum(ranked, 1.0)
    return out


def _feature_type(annotation) -> tuple[str, str, float]:
    """Map an existing semantic-role annotation to a coarse, non-LLM runtime type."""
    role = str(annotation.get("semantic_role", "mixed_or_unclear"))
    requested = pd.to_numeric(annotation.get("requested_share", np.nan), errors="coerce")
    requested = float(requested) if pd.notna(requested) else float("nan")
    if role in {"requested_task", "topic_content"} or (
        role == "language" and np.isfinite(requested) and requested >= 0.5
    ):
        kind = "requested_or_content"
    elif role in {"response_policy", "presentation", "reasoning_strategy", "language"}:
        kind = (
            "response_behavior"
            if not np.isfinite(requested) or requested < 0.5
            else "requested_or_content"
        )
    else:
        kind = "unclassified"
    return kind, role, requested


def profile_prompt_linkage(
    z_a,
    z_b,
    prompt_scores,
    *,
    features: pd.DataFrame | None = None,
    prompt_names: pd.DataFrame | None = None,
    prompt_context_ids=None,
    top_n: int = 100,
    min_top_examples: int = 30,
    prompt_tail_fractions=(0.005, 0.01, 0.02),
    min_tail_overlap: int = 5,
    min_context_lift: float = 2.0,
    q_threshold: float = 0.05,
    min_stable_scales: int = 2,
) -> pd.DataFrame:
    """Test whether strong response activations link to strong prompt activations.

    Each selected prompt concept is represented only by several high-activation tails.
    For every response feature, one-sided hypergeometric tests compare its top prompts
    with those tails. P-values are BH-adjusted across prompt concepts within each tail,
    and a link must recur for the same concept at multiple tail sizes. Absence of a
    detected link is deliberately not called general behavior: the prompt vocabulary
    may be incomplete. No LLM call or semantic-presence threshold is involved.
    """
    z_a = np.asarray(z_a)
    z_b = np.asarray(z_b)
    prompt_scores = np.asarray(prompt_scores)
    if z_a.ndim != 2 or z_b.ndim != 2 or z_a.shape != z_b.shape:
        raise ValueError("z_a and z_b must be aligned 2-D arrays")
    if prompt_scores.ndim != 2 or len(prompt_scores) != len(z_a):
        raise ValueError("prompt scores must be 2-D and align with completion rows")
    if top_n < 1 or min_top_examples < 1 or min_tail_overlap < 1:
        raise ValueError("top_n and minimum example counts must be positive")
    if min_top_examples > top_n:
        raise ValueError("min_top_examples cannot exceed top_n")
    if min_context_lift <= 0:
        raise ValueError("min_context_lift must be positive")
    if not 0 < q_threshold <= 1:
        raise ValueError("q_threshold must be in (0, 1]")
    fractions = np.asarray(sorted(set(float(x) for x in prompt_tail_fractions)))
    if not len(fractions) or (fractions <= 0).any() or (fractions >= 1).any():
        raise ValueError("prompt_tail_fractions must contain values in (0, 1)")
    if not 1 <= min_stable_scales <= len(fractions):
        raise ValueError("min_stable_scales must be between 1 and the number of tails")

    context_ids = (
        np.arange(prompt_scores.shape[1], dtype=int)
        if prompt_context_ids is None
        else np.asarray(prompt_context_ids)
    )
    if len(context_ids) != prompt_scores.shape[1]:
        raise ValueError("prompt_context_ids must have one entry per prompt score column")

    n_rows = len(prompt_scores)
    tail_memberships = []
    tail_sizes = []
    for fraction in fractions:
        target = int(np.ceil(float(fraction) * n_rows))
        membership = np.zeros(prompt_scores.shape, dtype=bool)
        sizes = np.zeros(prompt_scores.shape[1], dtype=int)
        for j in range(prompt_scores.shape[1]):
            values = np.asarray(prompt_scores[:, j], dtype=np.float32)
            positive = np.flatnonzero(values > 0)
            n_tail = min(target, len(positive))
            if not n_tail:
                continue
            if n_tail == len(positive):
                selected = positive
            else:
                local = np.argpartition(values[positive], -n_tail)[-n_tail:]
                selected = positive[local]
            membership[selected, j] = True
            sizes[j] = n_tail
        tail_memberships.append(membership)
        tail_sizes.append(sizes)

    if features is None or features.empty:
        table = pd.DataFrame({"feature_id": np.arange(z_a.shape[1], dtype=int)})
    else:
        if "feature_id" not in features.columns:
            raise ValueError("features need a feature_id column")
        table = features.copy()
        table["feature_id"] = pd.to_numeric(
            table["feature_id"], errors="raise"
        ).astype(int)
        table = table.drop_duplicates("feature_id", keep="last")
        if "fidelity_pass" in table.columns:
            table = table[table["fidelity_pass"].map(annotation_flag)]
    table = table[
        table["feature_id"].between(0, z_a.shape[1] - 1, inclusive="both")
    ].sort_values("feature_id")
    annotations = table.set_index("feature_id", drop=False)
    prompt_map = (
        {
            int(row.feature_id): str(row.concept)
            for row in prompt_names.drop_duplicates("feature_id", keep="last").itertuples()
        }
        if prompt_names is not None
        and {"feature_id", "concept"} <= set(prompt_names.columns)
        else {}
    )

    rows = []
    for feature_id in table["feature_id"].astype(int):
        annotation = annotations.loc[feature_id]
        score = np.maximum(
            np.asarray(z_a[:, feature_id], dtype=np.float32),
            np.asarray(z_b[:, feature_id], dtype=np.float32),
        )
        positive = np.flatnonzero(score > 0)
        if len(positive) > top_n:
            selected_local = np.argpartition(score[positive], -top_n)[-top_n:]
            selected = positive[selected_local]
        else:
            selected = positive
        if len(selected):
            selected = selected[np.argsort(score[selected], kind="stable")[::-1]]
        n_top = int(len(selected))
        counts_by_scale = []
        lift_by_scale = []
        q_by_scale = []
        passed_by_scale = []
        for membership, sizes in zip(tail_memberships, tail_sizes, strict=True):
            counts = (
                membership[selected].sum(axis=0).astype(int)
                if n_top
                else np.zeros(len(context_ids), dtype=int)
            )
            top_share = counts / max(1, n_top)
            corpus_share = sizes / max(1, n_rows)
            lift = np.divide(
                top_share,
                corpus_share,
                out=np.zeros(len(context_ids), dtype=float),
                where=corpus_share > 0,
            )
            p_values = np.ones(len(context_ids), dtype=float)
            eligible = sizes > 0
            p_values[eligible] = hypergeom.sf(
                counts[eligible] - 1,
                n_rows,
                sizes[eligible],
                n_top,
            )
            q_values = _bh_adjust(p_values)
            passed = (
                (counts >= int(min_tail_overlap))
                & (lift >= float(min_context_lift))
                & (q_values <= float(q_threshold))
            )
            counts_by_scale.append(counts)
            lift_by_scale.append(lift)
            q_by_scale.append(q_values)
            passed_by_scale.append(passed)
        counts_by_scale = np.asarray(counts_by_scale)
        lift_by_scale = np.asarray(lift_by_scale)
        q_by_scale = np.asarray(q_by_scale)
        passed_by_scale = np.asarray(passed_by_scale)
        stable_counts = passed_by_scale.sum(axis=0)
        stable_contexts = stable_counts >= int(min_stable_scales)

        if n_top < min_top_examples:
            scope = "insufficient_evidence"
            reason = f"only {n_top} positive-pole prompts; need {min_top_examples}"
        elif stable_contexts.any():
            candidates = np.flatnonzero(stable_contexts)
            best_q = np.nanmin(q_by_scale[:, candidates], axis=0)
            strongest = int(candidates[np.argmin(best_q)])
            scope = "prompt_linked"
            reason = (
                f"prompt context {context_ids[strongest]} is enriched at "
                f"{stable_counts[strongest]}/{len(fractions)} activation tails"
            )
        else:
            scope = "no_detected_prompt_link"
            reason = "no prompt-context enrichment repeats across the required tails"

        reference_scale = int(np.argmin(np.abs(fractions - 0.01)))
        reference_counts = counts_by_scale[reference_scale]
        reference_supported = reference_counts > 0
        count_values = reference_counts[reference_supported].astype(float)
        effective = (
            float(np.exp(_entropy(count_values / count_values.sum())))
            if count_values.sum()
            else 0.0
        )
        strong_threshold = float(score[selected].min()) if n_top else float("nan")
        if n_top:
            pa = np.asarray(z_a[:, feature_id], dtype=np.float32) >= strong_threshold
            pb = np.asarray(z_b[:, feature_id], dtype=np.float32) >= strong_threshold
            any_side = pa | pb
            paired_choice = (
                float((pa ^ pb).sum() / any_side.sum())
                if any_side.any()
                else float("nan")
            )
        else:
            paired_choice = float("nan")
        feature_type, semantic_role, requested_share = _feature_type(annotation)

        context_rows = []
        for j in range(len(context_ids)):
            if not reference_supported[j] and not passed_by_scale[:, j].any():
                continue
            scales = []
            for scale, fraction in enumerate(fractions):
                corpus_share = tail_sizes[scale][j] / max(1, n_rows)
                scales.append(
                    {
                        "tail_fraction": float(fraction),
                        "n_overlap": int(counts_by_scale[scale, j]),
                        "top_share": float(counts_by_scale[scale, j] / max(1, n_top)),
                        "corpus_share": float(corpus_share),
                        "lift": float(lift_by_scale[scale, j]),
                        "q_value": float(q_by_scale[scale, j]),
                        "passes": bool(passed_by_scale[scale, j]),
                    }
                )
            context_rows.append(
                {
                    "prompt_feature_id": int(context_ids[j]),
                    "concept": prompt_map.get(int(context_ids[j]), ""),
                    "n_scales_passed": int(stable_counts[j]),
                    "stable_link": bool(stable_contexts[j]),
                    "best_q_value": float(np.nanmin(q_by_scale[:, j])),
                    "max_lift": float(np.nanmax(lift_by_scale[:, j])),
                    "scales": scales,
                }
            )
        context_rows.sort(
            key=lambda item: (
                not item["stable_link"],
                -item["n_scales_passed"],
                item["best_q_value"],
                -item["max_lift"],
                item["prompt_feature_id"],
            )
        )
        rows.append(
            {
                "feature_id": int(feature_id),
                "concept": str(annotation.get("concept", "")),
                "scope_method": "stable_prompt_tail_enrichment",
                "prompt_scope": scope,
                "scope_reason": reason,
                "feature_type": feature_type,
                "semantic_role": semantic_role,
                "requested_share": requested_share,
                "n_positive_prompts": int(len(positive)),
                "n_top_prompts": n_top,
                "strong_activation_threshold": strong_threshold,
                "paired_choice_ratio": paired_choice,
                "prompt_tail_fractions_json": json.dumps(fractions.tolist()),
                "min_stable_scales": int(min_stable_scales),
                "n_linked_prompt_contexts": int(stable_contexts.sum()),
                "n_supported_prompt_contexts": int(reference_supported.sum()),
                "effective_prompt_contexts": effective,
                "normalized_prompt_breadth": (
                    effective / max(1, len(context_ids))
                ),
                "reference_prompt_tail_fraction": float(fractions[reference_scale]),
                "max_prompt_context_share": (
                    float((reference_counts / max(1, n_top)).max())
                    if len(reference_counts)
                    else float("nan")
                ),
                "max_prompt_context_lift": (
                    float(np.nanmax(lift_by_scale[reference_scale]))
                    if len(context_ids)
                    else float("nan")
                ),
                "top_prompt_contexts_json": json.dumps(
                    context_rows[:8], ensure_ascii=False
                ),
            }
        )
    return pd.DataFrame(rows)


def classify_feature(*, semantic_role: str, requested_share: float,
                     choice_ratio: float, prompt_dependence: float,
                     n_contexts: int, max_context_share: float,
                     general_min_contexts: int = 5,
                     general_max_context_share: float = 0.5,
                     general_max_prompt_dependence: float = 0.5,
                     min_choice_ratio: float = 0.15,
                     prompt_content_max_choice: float = 0.15) -> str:
    """Transparent first-pass taxonomy; thresholds are exposed by the CLI."""
    finite = all(np.isfinite(x) for x in (
        choice_ratio, prompt_dependence, max_context_share))
    if not finite:
        return "unclassified"
    content_role = semantic_role in {"requested_task", "topic_content"}
    requested = np.isfinite(requested_share) and requested_share >= 0.5
    if content_role and (requested or choice_ratio <= prompt_content_max_choice):
        return "prompt_content"
    if (semantic_role == "language" and requested and
            choice_ratio <= prompt_content_max_choice):
        return "prompt_content"
    if (choice_ratio >= min_choice_ratio and n_contexts >= general_min_contexts and
            max_context_share <= general_max_context_share and
            prompt_dependence <= general_max_prompt_dependence and
            semantic_role in {"response_policy", "presentation", "reasoning_strategy",
                              "language"}):
        return "general"
    if choice_ratio >= min_choice_ratio:
        return "context_specific"
    return "unclassified"


def profile_feature_context(z_a, z_b, calibration: pd.DataFrame,
                            prompt_context: np.ndarray, model_a, model_b, *,
                            names: pd.DataFrame | None = None,
                            prompt_names: pd.DataFrame | None = None,
                            prompt_context_ids=None,
                            min_context_occurrences: int = 10,
                            min_model_context_battles: int = 20,
                            min_model_context_discordant: int = 3,
                            min_stable_contexts: int = 3,
                            consistency_threshold: float = 0.75,
                            q_threshold: float = 0.05,
                            general_min_contexts: int = 5,
                            general_max_context_share: float = 0.5,
                            general_max_prompt_dependence: float = 0.5,
                            min_choice_ratio: float = 0.15,
                            prompt_content_max_choice: float = 0.15,
                            on_feature=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Profile calibrated per-side presence without materializing N x M codes."""
    n = z_a.shape[0]
    if z_b.shape[0] != n or len(prompt_context) != n or len(model_a) != n or len(model_b) != n:
        raise ValueError("completion codes, prompt contexts, and model columns must align")
    cal = calibration.copy()
    candidate_ids = pd.to_numeric(cal["feature_id"], errors="raise").astype(int).to_numpy()
    thresholds, calibrated = feature_thresholds(cal, candidate_ids)
    cal = cal.loc[calibrated].copy()
    cal["semantic_threshold"] = thresholds[calibrated]
    name_map = ({int(r.feature_id): str(r.concept) for r in names.itertuples()}
                if names is not None and "concept" in names.columns else {})
    prompt_map = ({int(r.feature_id): str(r.concept) for r in prompt_names.itertuples()}
                  if prompt_names is not None and "concept" in prompt_names.columns else {})
    context_ids, membership = _context_membership(prompt_context, prompt_context_ids)
    model_a = np.asarray(model_a, dtype=str)
    model_b = np.asarray(model_b, dtype=str)

    # Denominators used by every feature; count a model once per side/battle.
    all_models = np.concatenate((model_a, model_b))
    all_membership = np.vstack((membership, membership))
    model_battles = Counter(all_models.tolist())
    model_context_battles = Counter()
    for models_side in (model_a, model_b):
        for i, model in enumerate(models_side):
            for j in np.flatnonzero(membership[i]):
                model_context_battles[(str(model), context_ids[j])] += 1
    context_counts = {context_ids[j]: int(all_membership[:, j].sum())
                      for j in range(len(context_ids))}
    feature_rows, model_rows = [], []

    for crow in cal.itertuples():
        f = int(crow.feature_id)
        if f < 0 or f >= z_a.shape[1] or f >= z_b.shape[1]:
            continue
        threshold = float(crow.semantic_threshold)
        pa = np.asarray(z_a[:, f], dtype=np.float32) >= threshold
        pb = np.asarray(z_b[:, f], dtype=np.float32) >= threshold
        side_presence = np.concatenate((pa, pb))
        present_vector = all_membership[side_presence].sum(axis=0).astype(int)
        present_counts = {context_ids[j]: int(value) for j, value in enumerate(present_vector)
                          if value > 0}
        n_present = int(side_presence.sum())
        supported = {k: v for k, v in present_counts.items()
                     if context_counts[k] >= min_context_occurrences}
        shares = np.asarray(list(present_counts.values()), dtype=float)
        max_share = float(shares.max() / shares.sum()) if len(shares) else float("nan")
        effective = float(np.exp(_entropy(shares / shares.sum()))) if len(shares) else 0.0
        nmi, js = _region_dependence(side_presence, all_membership)
        any_side = pa | pb
        choice_ratio = float((pa ^ pb).sum() / any_side.sum()) if any_side.any() else float("nan")
        requested_share = float(getattr(crow, "requested_share", float("nan")))
        role = str(getattr(crow, "semantic_role", "mixed_or_unclear"))
        category = classify_feature(
            semantic_role=role, requested_share=requested_share,
            choice_ratio=choice_ratio, prompt_dependence=nmi,
            n_contexts=len(supported), max_context_share=max_share,
            general_min_contexts=general_min_contexts,
            general_max_context_share=general_max_context_share,
            general_max_prompt_dependence=general_max_prompt_dependence,
            min_choice_ratio=min_choice_ratio,
            prompt_content_max_choice=prompt_content_max_choice)
        top_contexts = [{"prompt_feature_id": int(k),
                         "concept": prompt_map.get(int(k), ""),
                         "n_present": int(v), "share": float(v / max(1, n_present))}
                        for k, v in sorted(present_counts.items(),
                                           key=lambda item: (-item[1], str(item[0])))[:8]]
        feature_row = {
            "feature_id": f, "concept": name_map.get(f, str(getattr(crow, "concept", ""))),
            "semantic_role": role, "requested_share": requested_share,
            "semantic_threshold": threshold, "semantic_presence_rate": n_present / (2 * n),
            "prompt_dependence_nmi": nmi, "prompt_context_js": js,
            "effective_prompt_contexts": effective,
            "max_prompt_context_share": max_share,
            "n_supported_prompt_contexts": len(supported),
            "paired_choice_ratio": choice_ratio, "behavior_category": category,
            "top_prompt_contexts_json": json.dumps(top_contexts, ensure_ascii=False),
        }
        feature_rows.append(feature_row)
        if on_feature is not None:
            on_feature(feature_row)

        # Only discordant pairs can identify which model chose the feature. This makes
        # prompt-forced properties naturally contribute little evidence.
        d = pa.astype(np.int8) - pb.astype(np.int8)
        discordant = np.flatnonzero(d != 0)
        observations: dict[str, list[tuple[int, int]]] = {}
        for i in discordant:
            observations.setdefault(model_a[i], []).append((int(d[i]), int(i)))
            observations.setdefault(model_b[i], []).append((-int(d[i]), int(i)))
        for model, obs in observations.items():
            direction = np.fromiter((x[0] for x in obs), dtype=np.int8)
            battle_rows = np.fromiter((x[1] for x in obs), dtype=int)
            n_discord = len(direction)
            positive = int((direction > 0).sum())
            p_value = float(binomtest(positive, n_discord, 0.5).pvalue)
            context_effects = []
            for j, ctx in enumerate(context_ids):
                vals = direction[membership[battle_rows, j]]
                total = model_context_battles[(model, ctx)]
                if total < min_model_context_battles or len(vals) < min_model_context_discordant:
                    continue
                context_effects.append({
                    "prompt_feature_id": int(ctx), "concept": prompt_map.get(int(ctx), ""),
                    "n_battles": int(total), "n_discordant": int(len(vals)),
                    "choice_effect": float(vals.mean()),
                })
            global_sign = int(np.sign(direction.sum()))
            signs = [int(np.sign(x["choice_effect"])) for x in context_effects
                     if x["choice_effect"] != 0]
            consistency = (sum(s == global_sign for s in signs) / len(signs)
                           if signs and global_sign else float("nan"))
            model_rows.append({
                "model": model, "feature_id": f, "concept": feature_row["concept"],
                "feature_category": category,
                "n_battles": int(model_battles[model]), "n_discordant": n_discord,
                "net_choice_rate": float(direction.sum() / model_battles[model]),
                "discordant_direction": float(direction.mean()),
                "p_value": p_value, "n_supported_contexts": len(context_effects),
                "cross_context_consistency": consistency,
                "cross_context_stable_raw": bool(
                    len(context_effects) >= min_stable_contexts and
                    np.isfinite(consistency) and consistency >= consistency_threshold),
                "top_contexts_json": json.dumps(
                    sorted(context_effects, key=lambda x: (-x["n_discordant"],
                                                          x["prompt_feature_id"]))[:8],
                    ensure_ascii=False),
            })

    features = pd.DataFrame(feature_rows)
    models = pd.DataFrame(model_rows)
    if not models.empty:
        models["q_value"] = np.nan
        for _, idx in models.groupby("model").groups.items():
            models.loc[idx, "q_value"] = _bh_adjust(models.loc[idx, "p_value"].to_numpy())
        models["cross_context_stable"] = (
            models["cross_context_stable_raw"].astype(bool) &
            (models["q_value"] < q_threshold))
        models["behavior_category"] = "unclassified"
        models.loc[models["feature_category"].eq("prompt_content"),
                   "behavior_category"] = "prompt_content"
        models.loc[models["feature_category"].eq("context_specific"),
                   "behavior_category"] = "context_specific"
        stable_general = (models["feature_category"].eq("general") &
                          models["cross_context_stable"])
        models.loc[stable_general, "behavior_category"] = "general"
        unstable_general = (models["feature_category"].eq("general") &
                            ~models["cross_context_stable"] &
                            (models["q_value"] < q_threshold))
        models.loc[unstable_general, "behavior_category"] = "context_specific"
    return (features.sort_values("feature_id").reset_index(drop=True),
            models.sort_values(["model", "feature_id"]).reset_index(drop=True))
