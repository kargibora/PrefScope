"""Paired concept shifts between any two aligned response sets.

The functions in this module are deliberately agnostic about what A and B mean.  They
may be two model checkpoints, decoding policies, adapters, humans/models, or the two
sides of a preference dataset.  Preference labels are not used here.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from prefscope.analysis.grouping import factorize_group_ids, validate_group_ids
from prefscope.core.features import validate_feature_ids


RESPONSE_SCOPES = (
    "general_tendency",
    "context_specific_tendency",
    "prompt_content",
    "unclassified",
)


def bh_adjust(p_values) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, preserving NaNs."""
    p = np.asarray(p_values, dtype=float)
    out = np.full(len(p), np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid):
        return out
    order = valid[np.argsort(p[valid], kind="stable")]
    ranked = p[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out[order] = np.minimum(ranked, 1.0)
    return out


def _boolean_matrix(values, *, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 2:
        raise ValueError(f"{name} must be a 2-D presence matrix, got {raw.shape}")
    if raw.dtype == bool:
        return raw
    if not np.issubdtype(raw.dtype, np.number):
        raise ValueError(f"{name} must contain only boolean or numeric 0/1 values")
    numeric = np.asarray(raw, dtype=float)
    if not np.isfinite(numeric).all() or not np.isin(numeric, [0.0, 1.0]).all():
        raise ValueError(f"{name} must contain only finite boolean or numeric 0/1 values")
    return numeric.astype(bool)


def _validate_presence(a, b, feature_ids, basis):
    a = _boolean_matrix(a, name="presence_a")
    b = _boolean_matrix(b, name="presence_b")
    if a.shape != b.shape:
        raise ValueError(
            f"paired presence matrices must be aligned, got {a.shape} / {b.shape}")
    ids = (
        tuple(range(a.shape[1])) if feature_ids is None
        else validate_feature_ids(feature_ids, width=a.shape[1])
    )
    ids = np.asarray(ids, dtype=int)
    if basis is None:
        basis = np.full(a.shape[1], "unspecified", dtype=object)
    else:
        basis = np.asarray(basis, dtype=object)
        if basis.ndim != 1 or len(basis) != a.shape[1]:
            raise ValueError("basis must have one entry per presence column")
        if pd.isna(basis).any() or any(not str(value) for value in basis):
            raise ValueError("basis must contain non-empty nonmissing values")
    return a, b, ids, basis


def _group_means(a: np.ndarray, b: np.ndarray, group_ids):
    """Equal-weight independent groups for effects and confidence intervals."""
    if group_ids is None:
        inv = np.arange(len(a), dtype=int)
    else:
        groups = validate_group_ids(group_ids, len(a))
        inv, n_groups = factorize_group_ids(groups)
    n_groups = len(a) if group_ids is None else n_groups
    count = np.bincount(inv, minlength=n_groups).astype(float)
    ga = np.zeros((n_groups, a.shape[1]), dtype=np.float64)
    gb = np.zeros_like(ga)
    np.add.at(ga, inv, a.astype(np.float64))
    np.add.at(gb, inv, b.astype(np.float64))
    if n_groups:
        ga /= count[:, None]
        gb /= count[:, None]
    return ga, gb, n_groups, bool(n_groups == len(a))


def _mean_ci(values: np.ndarray, confidence: float) -> tuple[np.ndarray, np.ndarray]:
    """Distribution-free Hoeffding interval for a mean of values in [-1, 1].

    A t/bootstrap interval collapses to zero width when every observed pair moves in the
    same direction. Hoeffding remains finite-sample valid in that important boundary case
    and applies unchanged to independent prompt-group averages.
    """
    n = len(values)
    if n == 0:
        nan = np.full(values.shape[1], np.nan)
        return nan, nan.copy()
    mean = values.mean(axis=0)
    alpha = 1.0 - float(confidence)
    half = math.sqrt(2.0 * math.log(2.0 / alpha) / n)
    return np.maximum(-1.0, mean - half), np.minimum(1.0, mean + half)


def paired_concept_shift(
    presence_a,
    presence_b,
    *,
    feature_ids=None,
    basis=None,
    group_ids=None,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Estimate B-minus-A prevalence shifts from prompt-aligned responses.

    With one row per independent prompt, ``p_value`` is the exact two-sided McNemar
    test (a binomial test over discordant pairs).  When ``group_ids`` contains repeated
    rows, effects, intervals, and the conservative Hoeffding p-value target the
    equal-group-weight mean of the group-average differences.  ``test`` records which
    path was used so clustered and unclustered evidence are never silently mixed.
    """
    if not 0 < float(confidence) < 1:
        raise ValueError("confidence must be in (0, 1)")
    a, b, ids, basis = _validate_presence(
        presence_a, presence_b, feature_ids, basis)
    ga, gb, n_groups, singleton_groups = _group_means(a, b, group_ids)
    gd = gb - ga
    prevalence_a = ga.mean(axis=0) if n_groups else np.full(a.shape[1], np.nan)
    prevalence_b = gb.mean(axis=0) if n_groups else np.full(a.shape[1], np.nan)
    delta = prevalence_b - prevalence_a
    ci_lo, ci_hi = _mean_ci(gd, float(confidence))

    a_only = (a & ~b).sum(axis=0).astype(int)
    b_only = (~a & b).sum(axis=0).astype(int)
    discordant = a_only + b_only
    nonzero_groups = np.count_nonzero(gd, axis=0).astype(int)
    p_values = np.ones(a.shape[1], dtype=float)
    if singleton_groups:
        for j, n_disc in enumerate(discordant):
            if n_disc:
                p_values[j] = float(
                    binomtest(int(b_only[j]), int(n_disc), 0.5).pvalue)
        test = "exact_mcnemar"
    else:
        # Invert the same Hoeffding bound used for the confidence interval.  Group
        # averages lie in [-1, 1], so under H0 E[gd] = 0 the two-sided tail bound is
        # 2 exp(-n_groups * observed_mean**2 / 2).  Unlike a sign test, this targets
        # the reported equal-group-weight mean rather than the median group direction.
        finite = np.isfinite(delta)
        p_values[finite] = np.minimum(
            1.0, 2.0 * np.exp(-n_groups * np.square(delta[finite]) / 2.0))
        test = "cluster_hoeffding"

    return pd.DataFrame({
        "feature_id": ids,
        "n_pairs": int(len(a)),
        "n_groups": int(n_groups),
        "prevalence_a": prevalence_a,
        "prevalence_b": prevalence_b,
        "delta_b_minus_a": delta,
        "ci_low": ci_lo,
        "ci_high": ci_hi,
        "ci_method": "hoeffding",
        "a_only": a_only,
        "b_only": b_only,
        "n_discordant": discordant,
        "n_nonzero_groups": nonzero_groups,
        "p_value": p_values,
        "q_value": bh_adjust(p_values),
        "test": test,
        "inference_test": test,
        "orientation": "delta_b_minus_a",
        "estimand": (
            "B-minus-A calibrated concept-prevalence shift over aligned rows"
            if singleton_groups else
            "equal-group-weight B-minus-A concept-prevalence shift"
        ),
        "presence_basis": basis,
    })


def paired_concept_shift_by_region(
    presence_a,
    presence_b,
    region_membership,
    *,
    feature_ids=None,
    basis=None,
    region_ids=None,
    group_ids=None,
    min_pairs: int = 30,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Run the paired shift inside every overlapping prompt region."""
    a, b, ids, basis = _validate_presence(
        presence_a, presence_b, feature_ids, basis)
    membership = _boolean_matrix(region_membership, name="region_membership")
    if membership.shape[0] != len(a):
        raise ValueError(
            "region_membership must be a 2-D matrix aligned to paired response rows")
    if (
        not isinstance(min_pairs, int)
        or isinstance(min_pairs, bool)
        or min_pairs < 1
    ):
        raise ValueError("min_pairs must be a positive integer")
    rids = (np.arange(membership.shape[1], dtype=int) if region_ids is None
            else np.asarray(list(region_ids)))
    if len(rids) != membership.shape[1]:
        raise ValueError("region_ids must have one entry per membership column")
    try:
        duplicate_regions = pd.Index(rids).has_duplicates
    except TypeError as exc:
        raise ValueError("region_ids must contain hashable scalar values") from exc
    if duplicate_regions or pd.isna(rids).any():
        raise ValueError("region_ids must contain unique nonmissing values")
    groups = None
    if group_ids is not None:
        values = validate_group_ids(group_ids, len(a))
        groups, n_labels = factorize_group_ids(values)
        labels = [values[np.flatnonzero(groups == group)[0]] for group in range(n_labels)]
        for group, label in enumerate(labels):
            rows_in_group = membership[groups == group]
            if not np.all(rows_in_group == rows_in_group[0]):
                raise ValueError(
                    "region membership must be constant within each independent "
                    f"group; group {label!r} varies")
    rows = []
    for j, region_id in enumerate(rids):
        mask = membership[:, j]
        row_support = int(mask.sum())
        group_support = (
            row_support if groups is None else int(np.unique(groups[mask]).size))
        if group_support < int(min_pairs):
            continue
        frame = paired_concept_shift(
            a[mask], b[mask], feature_ids=ids, basis=basis,
            group_ids=None if groups is None else groups[mask], confidence=confidence)
        frame.insert(0, "region_id", region_id)
        frame.insert(1, "region_support", row_support)
        frame.insert(2, "region_group_support", group_support)
        frame["q_value_within_region"] = frame["q_value"]
        rows.append(frame)
    if not rows:
        return pd.DataFrame(columns=[
            "region_id", "region_support", "region_group_support", "feature_id",
            "n_pairs", "n_groups",
            "prevalence_a", "prevalence_b", "delta_b_minus_a", "ci_low", "ci_high",
            "ci_method",
            "a_only", "b_only", "n_discordant", "n_nonzero_groups",
            "p_value", "q_value", "q_value_within_region", "test",
            "presence_basis",
        ])
    result = pd.concat(rows, ignore_index=True)
    result["q_value"] = bh_adjust(result["p_value"].to_numpy())
    return result


def summarize_response_scope(
    overall: pd.DataFrame,
    conditional: pd.DataFrame | None = None,
    *,
    feature_annotations: pd.DataFrame | None = None,
    q_threshold: float = 0.05,
    min_discordant: int = 20,
    min_contexts: int = 3,
    consistency_threshold: float = 0.75,
) -> pd.DataFrame:
    """Classify each A/B shift as general, context-specific, content, or unknown.

    This is a scope classification, not a value judgment.  ``good`` and ``bad`` are
    intentionally absent because those require an explicit criterion or preference label.
    ``min_discordant`` gates on independent groups with a nonzero group-average shift;
    for singleton groups this is identical to the number of discordant paired rows.
    """
    required = {
        "feature_id", "delta_b_minus_a", "q_value", "n_discordant",
        "n_nonzero_groups",
    }
    if not required <= set(overall.columns):
        raise ValueError(f"overall comparison is missing {sorted(required - set(overall))}")
    result = overall.copy()
    annotations = pd.DataFrame(index=pd.Index([], name="feature_id"))
    if feature_annotations is not None and not feature_annotations.empty:
        if "feature_id" not in feature_annotations:
            raise ValueError("feature_annotations need a feature_id column")
        annotations = feature_annotations.drop_duplicates("feature_id", keep="last") \
            .set_index("feature_id")

    context_lookup = {}
    if conditional is not None and not conditional.empty:
        context_lookup = {int(fid): frame for fid, frame in conditional.groupby("feature_id")}

    records = []
    eligible_roles = {
        "response_policy", "presentation", "reasoning_strategy", "language",
    }
    for row in result.itertuples(index=False):
        feature_id = int(row.feature_id)
        ann = (annotations.loc[feature_id] if feature_id in annotations.index
               else pd.Series(dtype=object))
        role = str(ann.get("semantic_role", "mixed_or_unclear"))
        requested = pd.to_numeric(ann.get("requested_share", np.nan), errors="coerce")
        requested = float(requested) if pd.notna(requested) else float("nan")
        contexts = context_lookup.get(feature_id, pd.DataFrame())
        supported = contexts[
            contexts["n_nonzero_groups"] >= int(min_discordant)
        ] if not contexts.empty else contexts
        global_sign = int(np.sign(float(row.delta_b_minus_a)))
        signs = np.sign(supported["delta_b_minus_a"].to_numpy(dtype=float)) \
            if not supported.empty else np.asarray([], dtype=float)
        signs = signs[signs != 0]
        consistency = (float((signs == global_sign).mean())
                       if len(signs) and global_sign else float("nan"))
        n_contexts = int(len(signs))
        # ``q_value`` is BH over the full feature x region family. The within-region
        # value remains useful for exploration but is too permissive for promoting a
        # concept to a supported context-specific tendency after searching many regions.
        context_q_col = "q_value"
        any_context_signal = bool(
            not supported.empty and context_q_col in supported and
            (pd.to_numeric(supported[context_q_col], errors="coerce")
             < float(q_threshold)).any())
        overall_signal = bool(
            np.isfinite(float(row.q_value)) and float(row.q_value) < float(q_threshold)
            and int(row.n_nonzero_groups) >= int(min_discordant))
        content = role in {"requested_task", "topic_content"} or (
            role == "language" and np.isfinite(requested) and requested >= 0.5)
        if content:
            scope = "prompt_content"
        elif (overall_signal and role in eligible_roles and
              n_contexts >= int(min_contexts) and np.isfinite(consistency) and
              consistency >= float(consistency_threshold) and
              (not np.isfinite(requested) or requested < 0.5)):
            scope = "general_tendency"
        elif overall_signal or any_context_signal:
            scope = "context_specific_tendency"
        else:
            scope = "unclassified"
        records.append({
            "feature_id": feature_id,
            "semantic_role": role,
            "requested_share": requested,
            "n_supported_contexts": n_contexts,
            "cross_context_consistency": consistency,
            "response_scope": scope,
        })
    scope = pd.DataFrame(records)
    return result.merge(scope, on="feature_id", how="left")
