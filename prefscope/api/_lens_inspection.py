"""Internal concept inspection and analysis delegation for the Lens facade."""
from __future__ import annotations

import numpy as np
import pandas as pd

import prefscope.analysis as analysis

def fidelity_feature_ids(lens):
    if lens.names is not None and "fidelity_pass" in lens.names.columns:
        from prefscope.analysis.presence import annotation_flag

        passing = lens.names["fidelity_pass"].map(annotation_flag)
        return lens.names.loc[passing, "feature_id"].astype(int).tolist()
    return None

def concept_names(lens):
    """Series mapping feature_id -> concept name, or None if unnamed.

    De-dups on ``feature_id`` so the index is unique (a duplicated id would
    make ``names.loc[fid]`` a Series and break ``top_concepts``).
    """
    if lens.names is not None and "concept" in lens.names.columns:
        return (lens.names.drop_duplicates("feature_id")
                .set_index("feature_id")["concept"])
    return None

def feature_table(lens) -> pd.DataFrame:
    """One row per feature with every bundled annotation column available."""
    if lens.names is not None:
        return lens.names.copy()
    return pd.DataFrame(
        {"feature_id": np.arange(int(lens.projector.m_total), dtype=int)})

def presence(lens, codes, *, feature_ids=None, policy: str = "calibrated"):
    """Resolve sparse codes into semantically supported concept presence.

    Returns a :class:`prefscope.analysis.presence.PresenceMatrix`.  The default
    omits axes without a passing learned semantic threshold; use ``policy="mixed"``
    explicitly for exploratory threshold-or-positive-nonzero behavior.
    """
    from prefscope.analysis.presence import concept_presence

    return concept_presence(
        codes, lens.feature_table, feature_ids=feature_ids, policy=policy)

def top_concepts(lens, codes, k: int = 5, *, matching_pole_only: bool = True):
    """Per row, the k active named features with the largest |code|.

    Returns a list (one per row) of ``(concept, signed_value)`` pairs sorted
    by ``|value|`` descending. Unnamed and zero features are skipped. For signed
    axes, the default also skips the opposite (negative) pole because the stored
    concept names the positive pole; set ``matching_pole_only=False`` only when
    explicitly inspecting signed axes. Rows may therefore contain fewer than k.
    """
    codes = np.atleast_2d(np.asarray(codes, dtype=np.float32))
    if codes.shape[1] != int(lens.projector.m_total):
        raise ValueError(
            f"codes have {codes.shape[1]} features but lens has "
            f"{lens.projector.m_total}")
    if k <= 0:
        return [[] for _ in range(len(codes))]
    names = lens.concept_names
    out = []
    for row in codes:
        picks: list[tuple] = []
        if names is not None:
            for fid in np.argsort(-np.abs(row)):
                fid = int(fid)
                if np.isnan(row[fid]) or row[fid] == 0:
                    continue
                if (matching_pole_only and lens.activation_polarity == "signed"
                        and row[fid] < 0):
                    continue
                if fid in names.index:
                    name = names.loc[fid]
                    if pd.notna(name):
                        picks.append((name, float(row[fid])))
                        if len(picks) == k:
                            break
        out.append(picks)
    return out

def concept_activations(
    lens,
    codes,
    *,
    row_ids=None,
    active_only: bool = True,
    pole: str = "any",
    min_abs_activation: float = 0.0,
    top_k: int | None = None,
    fidelity_only: bool = False,
    semantic_presence_only: bool = False,
) -> pd.DataFrame:
    """Return sparse codes as a filterable long-form concept table.

    Unlike :meth:`top_concepts`, this keeps feature ids, raw signed activation,
    rank, and all bundled annotation columns.  By default every nonzero activation
    is returned; ``top_k`` is optional rather than implicit.

    ``semantic_presence_only`` uses each feature's ``semantic_threshold`` and
    requires ``presence_pass`` when that column is available. It is meaningful for
    the positive pole; negative signed activations represent the opposite direction
    and therefore cannot satisfy a positive concept-name threshold.
    """
    values = np.atleast_2d(np.asarray(codes, dtype=np.float32))
    if values.shape[1] != int(lens.projector.m_total):
        raise ValueError(
            f"codes have {values.shape[1]} features but lens has "
            f"{lens.projector.m_total}")
    if pole not in {"any", "positive", "negative"}:
        raise ValueError("pole must be one of: any, positive, negative")
    if top_k is not None and int(top_k) <= 0:
        raise ValueError("top_k must be positive or None")
    if float(min_abs_activation) < 0:
        raise ValueError("min_abs_activation must be non-negative")
    if row_ids is None:
        row_ids = np.arange(len(values))
    row_ids = list(row_ids)
    if len(row_ids) != len(values):
        raise ValueError(
            f"row_ids length {len(row_ids)} != codes rows {len(values)}")

    features = lens.feature_table.drop_duplicates("feature_id", keep="last")
    features = features.set_index("feature_id")
    if fidelity_only:
        if "fidelity_pass" not in features.columns:
            raise ValueError(
                "fidelity_only=True needs bundled feature_fidelity.csv")
        from prefscope.analysis.presence import annotation_flag

        allowed = set(
            features.index[features["fidelity_pass"].map(annotation_flag)])
    else:
        allowed = set(int(x) for x in features.index)

    if semantic_presence_only and "semantic_threshold" not in features.columns:
        raise ValueError(
            "semantic_presence_only=True needs bundled feature_calibration.csv")

    records: list[dict] = []
    for row_id, row in zip(row_ids, values):
        mask = np.isfinite(row)
        if active_only:
            mask &= row != 0
        if pole == "positive":
            mask &= row > 0
        elif pole == "negative":
            mask &= row < 0
        mask &= np.abs(row) >= float(min_abs_activation)
        ids = np.flatnonzero(mask)
        if allowed:
            ids = np.asarray([f for f in ids if int(f) in allowed], dtype=int)
        else:
            ids = np.empty(0, dtype=int)
        if len(ids):
            ids = ids[np.argsort(-np.abs(row[ids]), kind="stable")]
        if top_k is not None:
            ids = ids[:int(top_k)]
        for rank, feature_id in enumerate(ids, start=1):
            activation = float(row[feature_id])
            annotation = (
                features.loc[int(feature_id)].to_dict()
                if int(feature_id) in features.index else {})
            threshold = annotation.get("semantic_threshold")
            calibration_pass = (
                analysis.annotation_flag(annotation.get("presence_pass"))
                if "presence_pass" in features.columns else True
            )
            semantic_present = (
                bool(calibration_pass and activation > 0 and pd.notna(threshold)
                     and activation >= float(threshold))
                if threshold is not None else False
            )
            if semantic_presence_only:
                if not semantic_present:
                    continue
            records.append({
                "row_id": row_id,
                "rank": rank,
                "feature_id": int(feature_id),
                "activation": activation,
                "abs_activation": abs(activation),
                "pole": "positive" if activation > 0 else (
                    "negative" if activation < 0 else "zero"),
                "concept_pole_matches_name": bool(
                    activation > 0 or lens.activation_polarity != "signed"),
                "semantic_present": semantic_present,
                **annotation,
            })

    leading = [
        "row_id", "rank", "feature_id", "concept", "activation",
        "abs_activation", "pole", "concept_pole_matches_name",
        "semantic_present",
    ]
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        available = [
            c for c in leading
            if c in {"row_id", "rank", "feature_id", "activation",
                     "abs_activation", "pole", "concept_pole_matches_name",
                     "semantic_present"}
            or c in features.columns
        ]
        return pd.DataFrame(columns=available)
    ordered = [c for c in leading if c in frame.columns]
    return frame[ordered + [c for c in frame.columns if c not in ordered]]

def diagnose(lens, codes, meta, *, fidelity_only: bool = False):
    """See ``prefscope.analysis.diagnose``."""
    return analysis.diagnose(codes, meta, names=lens.names, fidelity_only=fidelity_only)

def feature_preference_relevance(lens, codes, meta):
    """See ``prefscope.analysis.feature_preference_relevance``."""
    return analysis.feature_preference_relevance(codes, meta, names=lens.names)

def evaluate_preference(lens, codes, meta, **kwargs):
    """See ``prefscope.analysis.evaluate_preference``."""
    return analysis.evaluate_preference(codes, meta, names=lens.names, **kwargs)
