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
        return lens.names.drop_duplicates("feature_id").set_index("feature_id")[
            "concept"
        ]
    return None


def feature_table(lens) -> pd.DataFrame:
    """One row per feature with every bundled annotation column available."""
    if lens.names is not None:
        return lens.names.copy()
    return pd.DataFrame(
        {"feature_id": np.arange(int(lens.projector.m_total), dtype=int)}
    )


def presence(lens, codes, *, feature_ids=None, policy: str = "calibrated"):
    """Resolve sparse codes into semantically supported concept presence.

    Returns a :class:`prefscope.analysis.presence.PresenceMatrix`.  The default
    omits axes without a passing learned semantic threshold; use ``policy="mixed"``
    explicitly for exploratory threshold-or-positive-nonzero behavior.
    """
    from prefscope.analysis.presence import concept_presence

    return concept_presence(
        codes, lens.feature_table, feature_ids=feature_ids, policy=policy
    )


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
            f"{lens.projector.m_total}"
        )
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
                if (
                    matching_pole_only
                    and lens.activation_polarity == "signed"
                    and row[fid] < 0
                ):
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
    """Return codes as a filterable long-form table aligned by feature ID."""
    from prefscope.api.feature_activations import feature_activation_table
    from prefscope.core.features import FeatureMatrix

    if pole not in {"any", "positive", "negative"}:
        raise ValueError("pole must be one of: any, positive, negative")
    if top_k is not None and int(top_k) <= 0:
        raise ValueError("top_k must be positive or None")
    if float(min_abs_activation) < 0:
        raise ValueError("min_abs_activation must be non-negative")

    legacy_row_map = None
    finite_values = None
    if isinstance(codes, FeatureMatrix):
        matrix = codes
        if (
            row_ids is not None
            and tuple(str(value) for value in row_ids) != matrix.row_ids
        ):
            raise ValueError("row_ids cannot override FeatureMatrix row IDs")
        invalid = [
            feature_id
            for feature_id in matrix.feature_ids
            if feature_id < 0 or feature_id >= int(lens.projector.m_total)
        ]
        if invalid:
            raise ValueError(f"FeatureMatrix IDs are outside this lens: {invalid[:10]}")
        lens.feature_catalog.validate_for(matrix)
        empty_input = False
    else:
        values = np.atleast_2d(np.asarray(codes, dtype=np.float32))
        if values.shape[1] != int(lens.projector.m_total):
            raise ValueError(
                f"codes have {values.shape[1]} features but lens has "
                f"{lens.projector.m_total}"
            )
        legacy_row_ids = (
            tuple(range(len(values))) if row_ids is None else tuple(row_ids)
        )
        if len(legacy_row_ids) != len(values):
            raise ValueError(
                f"row_ids length {len(legacy_row_ids)} != codes rows {len(values)}"
            )
        surrogate_ids = tuple(
            f"__prefscope_legacy_row_{index}" for index in range(len(values))
        )
        legacy_row_map = dict(zip(surrogate_ids, legacy_row_ids, strict=True))
        finite_values = np.isfinite(values)
        safe_values = np.where(finite_values, values, 0.0).astype(
            np.float32, copy=False
        )
        empty_input = len(values) == 0
        matrix = FeatureMatrix(
            safe_values
            if not empty_input
            else np.zeros((1, values.shape[1]), dtype=np.float32),
            surrogate_ids if not empty_input else ("__empty__",),
            role="custom",
            orientation="unspecified",
            activation_polarity=lens.activation_polarity,
            code_semantics=lens.code_semantics,
        )

    table = feature_activation_table(
        matrix,
        active_only=active_only,
        min_abs_activation=min_abs_activation,
    )
    if empty_input:
        table = table.iloc[:0].copy()
    elif finite_values is not None and not table.empty:
        row_positions = (
            table["row_id"]
            .map({row_id: index for index, row_id in enumerate(matrix.row_ids)})
            .to_numpy(dtype=int)
        )
        feature_positions = table["feature_id"].to_numpy(dtype=int)
        table = table.loc[finite_values[row_positions, feature_positions]].copy()
    if pole == "positive":
        table = table.loc[table["activation"] > 0].copy()
    elif pole == "negative":
        table = table.loc[table["activation"] < 0].copy()

    features = lens.feature_table.drop_duplicates("feature_id", keep="last")
    reserved = set(table.columns) - {"feature_id"}
    collisions = reserved & set(features.columns)
    if collisions:
        raise ValueError(
            f"lens annotations conflict with activation fields: {sorted(collisions)}"
        )
    if fidelity_only:
        if "fidelity_pass" not in features.columns:
            raise ValueError("fidelity_only=True needs bundled feature_fidelity.csv")
        from prefscope.analysis.presence import annotation_flag

        features = features.loc[features["fidelity_pass"].map(annotation_flag)]
    if semantic_presence_only and "semantic_threshold" not in features.columns:
        raise ValueError(
            "semantic_presence_only=True needs bundled feature_calibration.csv"
        )

    table = table.merge(
        features,
        on="feature_id",
        how="inner",
        sort=False,
        validate="many_to_one",
    )
    if "semantic_threshold" in table:
        thresholds = pd.to_numeric(table["semantic_threshold"], errors="coerce")
        if "presence_pass" in table:
            calibration_pass = table["presence_pass"].map(analysis.annotation_flag)
        else:
            calibration_pass = pd.Series(True, index=table.index)
        table["semantic_present"] = (
            calibration_pass
            & (table["activation"] > 0)
            & thresholds.notna()
            & (table["activation"] >= thresholds)
        )
    else:
        table["semantic_present"] = False
    if semantic_presence_only:
        table = table.loc[table["semantic_present"]].copy()

    table["pole"] = np.where(
        table["activation"] > 0,
        "positive",
        np.where(table["activation"] < 0, "negative", "zero"),
    )
    table["concept_pole_matches_name"] = (table["activation"] > 0) | (
        lens.activation_polarity != "signed"
    )
    if top_k is not None:
        table = table.groupby("row_id", sort=False, group_keys=False).head(int(top_k))
    table["rank"] = table.groupby("row_id", sort=False).cumcount() + 1

    if legacy_row_map is not None and not table.empty:
        table["row_id"] = table["row_id"].map(legacy_row_map)

    leading = [
        "row_id",
        "rank",
        "feature_id",
        "concept",
        "activation",
        "abs_activation",
        "pole",
        "concept_pole_matches_name",
        "semantic_present",
    ]
    ordered = [column for column in leading if column in table.columns]
    return table[
        ordered + [column for column in table if column not in ordered]
    ].reset_index(drop=True)


def diagnose(lens, codes, meta, *, fidelity_only: bool = False):
    """See ``prefscope.analysis.diagnose``."""
    return analysis.diagnose(codes, meta, names=lens.names, fidelity_only=fidelity_only)


def feature_preference_relevance(lens, codes, meta):
    """See ``prefscope.analysis.feature_preference_relevance``."""
    return analysis.feature_preference_relevance(codes, meta, names=lens.names)


def evaluate_preference(lens, codes, meta, **kwargs):
    """See ``prefscope.analysis.evaluate_preference``."""
    return analysis.evaluate_preference(codes, meta, names=lens.names, **kwargs)
