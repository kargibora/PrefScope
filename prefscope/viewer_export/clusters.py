"""Export inspectable statistical feature communities for the web viewer."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from prefscope.analysis.presence import annotation_flag

from .sanitize import _read_csv, _round, _sanitize


def _frame(value) -> pd.DataFrame | None:
    if value is None:
        return None
    if isinstance(value, pd.DataFrame):
        return value.copy()
    return _read_csv(Path(value))


def _parse_ids(value) -> list[int]:
    if value is None or pd.isna(value):
        return []
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _meaningful_label(value, cluster_id: int) -> str | None:
    if value is None or pd.isna(value):
        return None
    label = str(value).strip()
    if not label or re.fullmatch(r"cluster\s+\d+", label, flags=re.IGNORECASE):
        return None
    return label if label.casefold() != f"community {cluster_id}".casefold() else None


def export_feature_clusters(
    clusters,
    features: pd.DataFrame,
    *,
    kind: str,
    summary=None,
    diagnostics=None,
) -> dict | None:
    """Return a self-contained response/prompt feature-community payload.

    Communities are statistical co-firing groups, not merged concepts.  Every member
    retains its original feature id and interpretation, including unnamed/unverified
    axes. Optional cluster-run summaries and stability diagnostics are preserved so the
    viewer can communicate the strength and limits of the partition.
    """
    if kind not in {"response", "prompt"}:
        raise ValueError("kind must be 'response' or 'prompt'")
    membership = _frame(clusters)
    if membership is None or not len(membership):
        return None
    required = {"feature_id", "cluster_id"}
    if not required <= set(membership.columns):
        raise ValueError(f"cluster table needs columns {sorted(required)}")

    membership = membership.dropna(subset=["feature_id", "cluster_id"]).copy()
    membership["feature_id"] = membership["feature_id"].astype(int)
    membership["cluster_id"] = membership["cluster_id"].astype(int)
    membership = membership.drop_duplicates("feature_id", keep="last")

    # ``cluster-features`` repeats the interpretation beside each membership for a
    # human-readable CSV. The viewer feature table is the authoritative, richer source
    # for interpretations and annotations. Keeping both copies would produce pandas
    # ``concept_x``/``concept_y`` columns and silently omit member labels from JSON.
    partition_columns = [
        column for column in (
            "feature_id", "cluster_id", "behavior", "supercluster_id",
        ) if column in membership.columns
    ]
    membership = membership[partition_columns]

    feature_table = features.dropna(subset=["feature_id"]).copy()
    feature_table["feature_id"] = feature_table["feature_id"].astype(int)
    # Membership is authoritative. The feature export may already carry cluster_id and
    # behavior columns, so remove them before joining to avoid _x/_y ambiguity.
    feature_table = feature_table.drop(
        columns=[column for column in ("cluster_id", "behavior")
                 if column in feature_table.columns]
    )
    joined = membership.merge(feature_table, on="feature_id", how="left")

    summary_table = _frame(summary)
    summary_by: dict[int, dict] = {}
    if summary_table is not None and "cluster_id" in summary_table.columns:
        summary_table = summary_table.dropna(subset=["cluster_id"]).copy()
        summary_table["cluster_id"] = summary_table["cluster_id"].astype(int)
        summary_by = {
            int(row["cluster_id"]): row.to_dict()
            for _, row in summary_table.drop_duplicates("cluster_id", keep="last").iterrows()
        }

    diagnostic_table = _frame(diagnostics)
    run_diagnostics = None
    if diagnostic_table is not None and len(diagnostic_table):
        run_diagnostics = _round(diagnostic_table.iloc[[0]])[0]

    member_columns = [
        column for column in (
            "feature_id", "concept", "fidelity_pass", "correlation", "agreement",
            "precision", "recall", "f1", "semantic_family", "semantic_role",
            "behavior_category", "generality", "fire_rate", "semantic_presence_rate",
            "n_prompt_types", "feature_summary",
        ) if column in joined.columns
    ]
    communities = []
    for cluster_id, group in joined.groupby("cluster_id", sort=True):
        cid = int(cluster_id)
        summary_row = summary_by.get(cid, {})
        behavior_values = group.get("behavior", pd.Series(dtype=object)).dropna()
        raw_label = (
            summary_row.get("behavior")
            if pd.notna(summary_row.get("behavior"))
            else behavior_values.iloc[0] if len(behavior_values) else None
        )
        representatives = _parse_ids(summary_row.get("representative_feature_ids"))
        if not representatives:
            representatives = group["feature_id"].astype(int).head(6).tolist()
        rep_concepts = summary_row.get("representative_concepts")
        if rep_concepts is None or pd.isna(rep_concepts):
            concept_map = dict(zip(group["feature_id"].astype(int), group.get(
                "concept", pd.Series(index=group.index, dtype=object))))
            rep_concepts = " | ".join(
                str(concept_map[fid]) for fid in representatives
                if fid in concept_map and pd.notna(concept_map[fid])
            )
        fidelity = (group["fidelity_pass"].map(annotation_flag)
                    if "fidelity_pass" in group.columns else pd.Series(False, index=group.index))
        community = {
            "cluster_id": cid,
            "label": _meaningful_label(raw_label, cid),
            "n_features": int(len(group)),
            "n_named": int(group["concept"].notna().sum()) if "concept" in group else 0,
            "n_verified": int(fidelity.sum()),
            "feature_ids": group["feature_id"].astype(int).tolist(),
            "representative_feature_ids": representatives,
            "representative_concepts": str(rep_concepts) if rep_concepts else "",
            "within_affinity_mean": summary_row.get("within_affinity_mean"),
            "external_affinity_mean": summary_row.get("external_affinity_mean"),
            "affinity_separation": summary_row.get("affinity_separation"),
            "within_phi_mean": summary_row.get("within_phi_mean"),
            "negative_pair_fraction": summary_row.get("negative_pair_fraction"),
            "members": _round(group[member_columns].sort_values("feature_id")),
        }
        if "supercluster_id" in group.columns:
            super_ids = group["supercluster_id"].dropna().astype(int).unique()
            community["supercluster_id"] = int(super_ids[0]) if len(super_ids) == 1 else None
        communities.append(community)

    clustered_ids = set(membership["feature_id"].astype(int))
    all_ids = feature_table["feature_id"].astype(int).tolist()
    unclustered = [feature_id for feature_id in all_ids if feature_id not in clustered_ids]
    payload = {
        "kind": kind,
        "method": (run_diagnostics or {}).get("method"),
        "n_total_features": int(len(feature_table)),
        "n_clustered_features": int(len(clustered_ids)),
        "n_unclustered_features": int(len(unclustered)),
        "n_clusters": int(len(communities)),
        "unclustered_feature_ids": unclustered,
        "diagnostics": run_diagnostics,
        "clusters": sorted(
            communities,
            key=lambda row: (-row["n_features"], row["cluster_id"]),
        ),
    }
    return _sanitize(payload)
