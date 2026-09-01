"""Generic paired response comparison over already-encoded lens artifacts."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from prefscope.analysis.grouping import resolve_group_ids
from prefscope.analysis.paired import (
    paired_concept_shift, paired_concept_shift_by_region, summarize_response_scope,
)
from prefscope.analysis.presence import annotation_flag, concept_presence
from prefscope.analysis.prompt_regions import regions_from_feature_presence
from prefscope.artifacts import (
    FEATURE_CALIBRATION, FEATURE_FIDELITY, FEATURE_NAMES,
    PROMPT_FEATURE_FIDELITY, PROMPT_FEATURE_NAMES,
)


@dataclass
class ResponseComparison:
    """Durable, label-free comparison of two aligned response sets."""

    overall: pd.DataFrame
    conditional: pd.DataFrame
    scope: pd.DataFrame
    examples: pd.DataFrame
    manifest: dict

    def save(self, out) -> Path:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        self.overall.to_parquet(out / "concept_shift.parquet", index=False)
        self.conditional.to_parquet(
            out / "concept_shift_by_context.parquet", index=False)
        self.scope.to_parquet(out / "response_scope.parquet", index=False)
        self.examples.to_parquet(out / "paired_examples.parquet", index=False)
        (out / "comparison.json").write_text(
            json.dumps(self.manifest, indent=2, ensure_ascii=False))
        return out


def _read_annotations(source, *, prompt: bool = False) -> pd.DataFrame:
    if source is None:
        return pd.DataFrame(columns=["feature_id"])
    if isinstance(source, pd.DataFrame):
        frames = [source.copy()]
    else:
        path = Path(source)
        if path.is_file():
            frames = [pd.read_csv(path)]
        else:
            names = PROMPT_FEATURE_NAMES if prompt else FEATURE_NAMES
            fidelity = PROMPT_FEATURE_FIDELITY if prompt else FEATURE_FIDELITY
            candidates = [path / names, path / fidelity, path / FEATURE_CALIBRATION]
            frames = [pd.read_csv(candidate) for candidate in candidates if candidate.exists()]
    if not frames:
        return pd.DataFrame(columns=["feature_id"])
    for frame in frames:
        if "feature_id" not in frame.columns:
            raise ValueError("feature annotation tables need a feature_id column")
    # Last non-null value wins while preserving columns that occur in only one artifact.
    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged["feature_id"] = pd.to_numeric(merged["feature_id"], errors="raise").astype(int)
    return merged.groupby("feature_id", as_index=False, sort=True).last()


def _analysis_feature_ids(features: pd.DataFrame, width: int, *,
                          fidelity_only: bool, named_only: bool) -> list[int]:
    if features.empty:
        if fidelity_only or named_only:
            raise ValueError(
                "feature annotations are required for fidelity/named filtering")
        return list(range(width))
    keep = features.copy()
    if fidelity_only:
        if "fidelity_pass" not in keep.columns:
            raise ValueError("fidelity_only needs a fidelity_pass annotation column")
        passing = keep["fidelity_pass"].map(annotation_flag)
        keep = keep[passing]
    if named_only:
        if "concept" not in keep.columns:
            raise ValueError("named_only needs a concept annotation column")
        concept = keep["concept"].fillna("").astype(str).str.strip()
        keep = keep[concept.ne("")]
    ids = keep["feature_id"].astype(int).tolist()
    return [feature_id for feature_id in ids if 0 <= feature_id < width]


def _load_bundle(directory, *, prompt: bool = False):
    directory = Path(directory)
    meta_path = directory / "meta.parquet"
    if not meta_path.exists():
        meta_path = directory / "battles.parquet"
    if not meta_path.exists():
        raise FileNotFoundError(f"no meta.parquet/battles.parquet in {directory}")
    meta = pd.read_parquet(meta_path).reset_index(drop=True)
    if prompt:
        path = directory / "z_prompt.npy"
        if not path.exists():
            raise FileNotFoundError(f"missing {path}")
        codes = np.load(path, mmap_mode="r")
        if len(codes) != len(meta):
            raise ValueError("prompt codes and metadata are misaligned")
        return codes, meta
    za, zb = directory / "z_a.npy", directory / "z_b.npy"
    if not za.exists() or not zb.exists():
        raise ValueError(
            "paired response comparison needs an individual lens bundle with z_a.npy "
            "and z_b.npy")
    a, b = np.load(za, mmap_mode="r"), np.load(zb, mmap_mode="r")
    if a.shape != b.shape or len(a) != len(meta):
        raise ValueError("response A/B codes and metadata are misaligned")
    return a, b, meta


def _id_column(meta: pd.DataFrame, *, unique: bool = False) -> str:
    for column in ("battle_id", "item_id", "row_id"):
        if column in meta.columns and (not unique or not meta[column].astype(str).duplicated().any()):
            return column
    qualifier = " a unique" if unique else ""
    raise ValueError(
        f"encoded metadata needs{qualifier} battle_id, item_id, or row_id for alignment")


def _align_prompt(response_meta: pd.DataFrame, prompt_codes, prompt_meta: pd.DataFrame):
    common = [column for column in ("battle_id", "item_id", "row_id")
              if column in response_meta and column in prompt_meta]
    id_column = next((column for column in common
                      if not response_meta[column].astype(str).duplicated().any()
                      and not prompt_meta[column].astype(str).duplicated().any()), None)
    if id_column is None:
        raise ValueError(
            "prompt/response bundles need a shared unique battle_id, item_id, or row_id")
    left = response_meta[id_column].astype(str)
    right = prompt_meta[id_column].astype(str)
    position = pd.Series(np.arange(len(right)), index=right)
    rows = left.map(position)
    if rows.isna().any():
        missing = left[rows.isna()].head(5).tolist()
        raise ValueError(f"prompt bundle is missing response item ids such as {missing}")
    return np.asarray(prompt_codes[np.asarray(rows, dtype=int)])


def _attach_annotations(frame: pd.DataFrame, annotations: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or annotations.empty:
        return frame
    keep = [column for column in annotations.columns
            if column != "feature_id" and column not in frame.columns]
    return frame.merge(annotations[["feature_id", *keep]], on="feature_id", how="left")


def _paired_examples(meta, z_a, z_b, presence_a, presence_b, feature_ids, *,
                     side_a_name: str, side_b_name: str, per_direction: int = 3):
    columns = [
        "feature_id", "direction", "item_id", "prompt", "response_a", "response_b",
        "activation_a", "activation_b", "side_a_name", "side_b_name",
    ]
    if int(per_direction) <= 0:
        return pd.DataFrame(columns=columns)
    id_col = _id_column(meta)
    prompt_col = "prompt" if "prompt" in meta else None
    a_col = next((c for c in ("completion_a", "response") if c in meta), None)
    b_col = next((c for c in ("completion_b", "response_2") if c in meta), None)
    rows = []
    for j, feature_id in enumerate(feature_ids):
        for direction, mask, score in (
            ("a_only", presence_a[:, j] & ~presence_b[:, j],
             np.asarray(z_a[:, feature_id] - z_b[:, feature_id])),
            ("b_only", ~presence_a[:, j] & presence_b[:, j],
             np.asarray(z_b[:, feature_id] - z_a[:, feature_id])),
        ):
            candidates = np.flatnonzero(mask)
            if not len(candidates):
                continue
            chosen = candidates[np.argsort(-score[candidates], kind="stable")[:per_direction]]
            for i in chosen:
                rows.append({
                    "feature_id": int(feature_id), "direction": direction,
                    "item_id": str(meta.iloc[int(i)][id_col]),
                    "prompt": str(meta.iloc[int(i)][prompt_col]) if prompt_col else "",
                    "response_a": str(meta.iloc[int(i)][a_col]) if a_col else "",
                    "response_b": str(meta.iloc[int(i)][b_col]) if b_col else "",
                    "activation_a": float(z_a[int(i), feature_id]),
                    "activation_b": float(z_b[int(i), feature_id]),
                    "side_a_name": str(side_a_name), "side_b_name": str(side_b_name),
                })
    return pd.DataFrame(rows, columns=columns)


def compare_encoded_responses(
    response_dir,
    *,
    features,
    prompt_dir=None,
    prompt_features=None,
    prompt_clusters=None,
    side_a_name: str = "A",
    side_b_name: str = "B",
    presence_policy: str = "calibrated",
    prompt_presence_policy: str = "calibrated",
    fidelity_only: bool = True,
    named_only: bool = True,
    min_context_pairs: int = 30,
    group_col: str | None = None,
    examples_per_direction: int = 3,
    confidence: float = 0.95,
) -> ResponseComparison:
    """Compare A/B concepts in an encoded individual-response bundle.

    The operation is label-free: a ``human_pref`` column, when present, is ignored.
    Preference association remains a separate analysis that can be joined by feature id.
    """
    z_a, z_b, meta = _load_bundle(response_dir)
    annotations = _read_annotations(features)
    ids = _analysis_feature_ids(
        annotations, z_a.shape[1], fidelity_only=fidelity_only, named_only=named_only)
    pa = concept_presence(z_a, annotations, feature_ids=ids, policy=presence_policy)
    pb = concept_presence(z_b, annotations, feature_ids=ids, policy=presence_policy)
    if not np.array_equal(pa.feature_ids, pb.feature_ids):
        raise AssertionError("A/B semantic-presence features unexpectedly differ")
    if not len(pa.feature_ids):
        raise ValueError(
            f"no features remain under presence_policy={presence_policy!r}; calibrate "
            "semantic presence or explicitly choose --presence-policy mixed")
    try:
        groups = resolve_group_ids(meta, group_col=group_col)
    except ValueError as exc:
        if group_col is not None:
            raise ValueError(
                f"invalid group_col {group_col!r}; preserve it during encoding with "
                f"--metadata-col {group_col}: {exc}") from exc
        raise
    overall = paired_concept_shift(
        pa.values, pb.values, feature_ids=pa.feature_ids, basis=pa.basis,
        group_ids=groups, confidence=confidence)

    conditional = paired_concept_shift_by_region(
        pa.values, pb.values, np.zeros((len(meta), 0), dtype=bool),
        feature_ids=pa.feature_ids, basis=pa.basis, region_ids=[],
        group_ids=groups, min_pairs=min_context_pairs, confidence=confidence)
    region_kind = None
    if prompt_dir is not None:
        z_prompt, prompt_meta = _load_bundle(prompt_dir, prompt=True)
        z_prompt = _align_prompt(meta, z_prompt, prompt_meta)
        prompt_annotations = _read_annotations(prompt_features, prompt=True)
        pids = _analysis_feature_ids(
            prompt_annotations, z_prompt.shape[1],
            fidelity_only=fidelity_only, named_only=named_only)
        pp = concept_presence(
            z_prompt, prompt_annotations, feature_ids=pids,
            policy=prompt_presence_policy)
        if prompt_clusters is not None:
            cluster_table = (prompt_clusters if isinstance(prompt_clusters, pd.DataFrame)
                             else pd.read_csv(prompt_clusters))
            region_ids, membership = regions_from_feature_presence(
                pp.values, pp.feature_ids, clusters=cluster_table)
            region_kind = "prompt_cluster"
        else:
            region_ids, membership = regions_from_feature_presence(
                pp.values, pp.feature_ids)
            region_kind = "prompt_concept"
        conditional = paired_concept_shift_by_region(
            pa.values, pb.values, membership, feature_ids=pa.feature_ids,
            basis=pa.basis, region_ids=region_ids, group_ids=groups,
            min_pairs=min_context_pairs, confidence=confidence)
        conditional["region_kind"] = region_kind
        if region_kind == "prompt_cluster" and "behavior" in cluster_table.columns:
            labels = (cluster_table.dropna(subset=["cluster_id", "behavior"])
                      .groupby("cluster_id", as_index=False, sort=True)["behavior"].last()
                      .rename(columns={"cluster_id": "region_id",
                                       "behavior": "region_concept"}))
            conditional = conditional.merge(labels, on="region_id", how="left")
        elif not prompt_annotations.empty and region_kind == "prompt_concept":
            pnames = prompt_annotations[[c for c in ("feature_id", "concept")
                                         if c in prompt_annotations.columns]]
            if "concept" in pnames:
                conditional = conditional.merge(
                    pnames.rename(columns={"feature_id": "region_id",
                                           "concept": "region_concept"}),
                    on="region_id", how="left")

    scope = summarize_response_scope(
        overall, conditional, feature_annotations=annotations)
    overall = _attach_annotations(overall, annotations)
    conditional = _attach_annotations(conditional, annotations)
    scope = _attach_annotations(scope, annotations)
    examples = _paired_examples(
        meta, z_a, z_b, pa.values, pb.values, pa.feature_ids,
        side_a_name=side_a_name, side_b_name=side_b_name,
        per_direction=examples_per_direction)
    examples = _attach_annotations(examples, annotations)
    manifest = {
        "schema_version": 1,
        "analysis": "paired_response_concept_shift",
        "response_dir": str(Path(response_dir)),
        "prompt_dir": str(Path(prompt_dir)) if prompt_dir is not None else None,
        "side_a_name": str(side_a_name), "side_b_name": str(side_b_name),
        "n_pairs": int(len(meta)), "n_features": int(len(pa.feature_ids)),
        "presence_policy": presence_policy,
        "prompt_presence_policy": prompt_presence_policy if prompt_dir is not None else None,
        "presence_basis_counts": pd.Series(pa.basis).value_counts().to_dict(),
        "region_kind": region_kind,
        "min_context_pairs": int(min_context_pairs),
        "group_col": group_col,
        "confidence": float(confidence),
        "preference_labels_used": False,
        "files": {
            "overall": "concept_shift.parquet",
            "conditional": "concept_shift_by_context.parquet",
            "scope": "response_scope.parquet",
            "examples": "paired_examples.parquet",
        },
    }
    return ResponseComparison(overall, conditional, scope, examples, manifest)
