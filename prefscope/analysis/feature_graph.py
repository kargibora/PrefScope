"""Describe relationships between sparse features without merging them.

Feature clustering is useful for navigation, but a partition loses asymmetric
structure.  A narrow ``code in Greek`` feature can be almost contained in a broad
``Greek`` feature even though the reverse is false.  This module keeps that
information explicit and combines three independent signals:

* overlap between positive-pole firing sets;
* optional cosine similarity between SAE decoder directions;
* optional similarity/collisions between interpreted names.

The returned relationships are diagnostics.  In particular, co-firing is not a
causal or semantic-equivalence claim, and ``candidate_merge`` is deliberately
conservative.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from prefscope.pipeline.cluster import feature_cofire_affinity


RELATION_COLUMNS = [
    "feature_a", "feature_b", "concept_a", "concept_b", "n_a", "n_b",
    "n_both", "jaccard", "containment_a_in_b", "containment_b_in_a",
    "lift", "phi", "decoder_cosine", "name_similarity", "same_name",
    "relation", "candidate_merge", "needs_relabel",
]

_NAME_STOPWORDS = {
    "a", "an", "the", "is", "are", "be", "being", "has", "have", "in",
    "into", "of", "on", "to", "for", "with", "and", "or", "response",
    "prompt", "asks", "requests", "provides", "provide", "contains", "uses",
}


def _name_key(value) -> str:
    """Normalize superficial label variants for collision detection."""
    if value is None or pd.isna(value):
        return ""
    text = re.sub(r"[^\w\s]", " ", str(value).casefold(), flags=re.UNICODE)
    text = " ".join(text.split())
    # The namer alternates between "written in Greek" and "is written in Greek".
    return text[3:] if text.startswith("is ") else text


def _name_tokens(value) -> set[str]:
    return {token for token in _name_key(value).split()
            if token not in _NAME_STOPWORDS}


def _token_jaccard(a, b) -> float:
    aa, bb = _name_tokens(a), _name_tokens(b)
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / len(aa | bb)


def _concept_lookup(names: pd.DataFrame | None) -> dict[int, str]:
    if names is None or not {"feature_id", "concept"} <= set(names.columns):
        return {}
    table = names.dropna(subset=["feature_id"]).copy()
    table["feature_id"] = table["feature_id"].astype(int)
    # Prefer the last non-null annotation when callers concatenate naming/fidelity tables.
    table = table.dropna(subset=["concept"]).drop_duplicates("feature_id", keep="last")
    return {int(row.feature_id): str(row.concept) for row in table.itertuples()}


def _decoder_cosines(decoder: np.ndarray | None, features: np.ndarray) -> np.ndarray | None:
    if decoder is None:
        return None
    weight = np.asarray(decoder, dtype=np.float32)
    if weight.ndim != 2:
        raise ValueError("decoder must be a 2D weight matrix")
    max_feature = int(features.max(initial=-1))
    if weight.shape[1] > max_feature:
        directions = weight[:, features]
    elif weight.shape[0] > max_feature:
        directions = weight[features].T
    else:
        raise ValueError(
            f"decoder shape {weight.shape} does not contain feature {max_feature}"
        )
    norms = np.linalg.norm(directions, axis=0, keepdims=True)
    unit = np.divide(
        directions, norms, out=np.zeros_like(directions), where=norms > 1e-8
    )
    return unit.T @ unit


def load_decoder_directions(lens_dir) -> np.ndarray:
    """Load ``(input_dim, n_features)`` decoder directions from a lens checkpoint.

    Torch remains optional for PrefScope.  Callers that only need activation/name
    relationships can omit decoder directions entirely.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "decoder relationships require torch; install prefscope[cpu] or pass "
            "--no-decoder"
        ) from exc
    path = Path(lens_dir) / "sae_model.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict", checkpoint)
    key = next((key for key in state if key.endswith("decoder.weight")), None)
    if key is None:
        raise ValueError(f"{path} does not contain decoder.weight")
    return state[key].float().cpu().numpy()


def feature_relationships(
    z,
    *,
    names: pd.DataFrame | None = None,
    decoder: np.ndarray | None = None,
    features=None,
    pole: str = "positive",
    min_cooccur: int = 30,
    min_jaccard: float = 0.05,
    min_containment: float = 0.50,
    min_phi: float = 0.05,
    min_lift: float = 1.50,
    min_name_similarity: float = 0.80,
    min_decoder_cosine: float = 0.70,
    specialization_containment: float = 0.80,
    specialization_reverse_max: float = 0.60,
    duplicate_containment: float = 0.80,
    duplicate_decoder_cosine: float = 0.70,
    chunk_size: int = 8192,
) -> pd.DataFrame:
    """Return candidate relationships between feature axes.

    ``containment_a_in_b`` is ``P(b fires | a fires)``.  Consequently, high
    ``containment_a_in_b`` with low reverse containment means that ``a`` is the
    narrower activation pattern and may specialize ``b``.

    Names are used only to surface collisions and nearby labels; they never make a
    pair a merge candidate.  Merging requires both bidirectional activation
    containment and aligned decoder directions.
    """
    shape = getattr(z, "shape", None)
    if shape is None or len(shape) != 2:
        raise ValueError("z must be a 2D code matrix")
    feats = (np.arange(shape[1], dtype=int) if features is None
             else np.asarray([int(feature) for feature in features], dtype=int))
    if len(feats) < 2:
        return pd.DataFrame(columns=RELATION_COLUMNS)
    if feats.min() < 0 or feats.max() >= shape[1]:
        raise ValueError("features contain an index outside z")

    _, stats = feature_cofire_affinity(
        z,
        features=feats,
        pole=pole,
        metric="phi",
        min_cooccur=0,
        chunk_size=chunk_size,
        return_stats=True,
    )
    n = int(stats["n"])
    ones = np.asarray(stats["ones"], dtype=np.float64)
    both = np.asarray(stats["cooccur"], dtype=np.float64)
    phi_matrix = np.asarray(stats["phi"], dtype=np.float64)
    decoder_cos = _decoder_cosines(decoder, feats)
    concepts = _concept_lookup(names)

    ii, jj = np.triu_indices(len(feats), k=1)
    n_a, n_b, n_both = ones[ii], ones[jj], both[ii, jj]
    union = n_a + n_b - n_both
    jaccard = np.divide(n_both, union, out=np.zeros_like(n_both), where=union > 0)
    a_in_b = np.divide(n_both, n_a, out=np.zeros_like(n_both), where=n_a > 0)
    b_in_a = np.divide(n_both, n_b, out=np.zeros_like(n_both), where=n_b > 0)
    if n:
        expected = n_a * n_b / n
        lift = np.divide(n_both, expected, out=np.zeros_like(n_both), where=expected > 0)
    else:
        lift = np.zeros_like(n_both)
    phi = phi_matrix[ii, jj]
    dec = (decoder_cos[ii, jj] if decoder_cos is not None
           else np.full(len(ii), np.nan))

    concept_a = np.asarray([concepts.get(int(feats[i]), "") for i in ii], dtype=object)
    concept_b = np.asarray([concepts.get(int(feats[j]), "") for j in jj], dtype=object)
    key_a = np.asarray([_name_key(value) for value in concept_a], dtype=object)
    key_b = np.asarray([_name_key(value) for value in concept_b], dtype=object)
    same_name = (key_a != "") & (key_a == key_b)
    name_similarity = np.asarray(
        [_token_jaccard(a, b) for a, b in zip(concept_a, concept_b)], dtype=float
    )

    supported = n_both >= int(min_cooccur)
    activation_candidate = supported & (
        (jaccard >= min_jaccard)
        | (np.maximum(a_in_b, b_in_a) >= min_containment)
        | ((phi >= min_phi) & (lift >= min_lift))
    )
    name_candidate = same_name | (name_similarity >= min_name_similarity)
    decoder_candidate = np.isfinite(dec) & (np.abs(dec) >= min_decoder_cosine)
    keep = activation_candidate | name_candidate | decoder_candidate

    rows = []
    for pos in np.where(keep)[0]:
        ca, cb = float(a_in_b[pos]), float(b_in_a[pos])
        signed_decoder = float(dec[pos]) if np.isfinite(dec[pos]) else np.nan
        bidirectional = min(ca, cb)
        merge = (
            bidirectional >= duplicate_containment
            and np.isfinite(signed_decoder)
            and signed_decoder >= duplicate_decoder_cosine
        )
        if merge:
            relation = "near_duplicate"
        elif ca >= specialization_containment and cb <= specialization_reverse_max:
            relation = "a_specializes_b"
        elif cb >= specialization_containment and ca <= specialization_reverse_max:
            relation = "b_specializes_a"
        elif bidirectional >= duplicate_containment:
            relation = "same_firing_region"
        elif same_name[pos]:
            relation = "same_name_collision"
        elif activation_candidate[pos]:
            relation = "coactive_distinct"
        elif decoder_candidate[pos]:
            relation = "decoder_neighbor"
        else:
            relation = "name_neighbor"
        rows.append({
            "feature_a": int(feats[ii[pos]]),
            "feature_b": int(feats[jj[pos]]),
            "concept_a": concept_a[pos] or None,
            "concept_b": concept_b[pos] or None,
            "n_a": int(n_a[pos]),
            "n_b": int(n_b[pos]),
            "n_both": int(n_both[pos]),
            "jaccard": float(jaccard[pos]),
            "containment_a_in_b": ca,
            "containment_b_in_a": cb,
            "lift": float(lift[pos]),
            "phi": float(phi[pos]),
            "decoder_cosine": signed_decoder,
            "name_similarity": float(name_similarity[pos]),
            "same_name": bool(same_name[pos]),
            "relation": relation,
            "candidate_merge": bool(merge),
            "needs_relabel": bool(same_name[pos] and not merge),
        })
    if not rows:
        return pd.DataFrame(columns=RELATION_COLUMNS)
    result = pd.DataFrame(rows, columns=RELATION_COLUMNS)
    order = {
        "near_duplicate": 0,
        "a_specializes_b": 1,
        "b_specializes_a": 1,
        "same_firing_region": 2,
        "same_name_collision": 3,
        "coactive_distinct": 4,
        "decoder_neighbor": 5,
        "name_neighbor": 6,
    }
    result["_order"] = result["relation"].map(order)
    result = result.sort_values(
        ["_order", "needs_relabel", "jaccard", "phi"],
        ascending=[True, False, False, False],
    ).drop(columns="_order").reset_index(drop=True)
    result.attrs["n_rows"] = n
    result.attrs["n_features"] = len(feats)
    return result


def feature_relationship_summary(relations: pd.DataFrame) -> pd.DataFrame:
    """Compact counts for logging and experiment comparison."""
    counts = relations["relation"].value_counts() if len(relations) else pd.Series(dtype=int)
    rows = [{"relation": str(name), "n_pairs": int(count)}
            for name, count in counts.items()]
    rows.extend([
        {"relation": "candidate_merge_total",
         "n_pairs": int(relations.get("candidate_merge", pd.Series(dtype=bool)).sum())},
        {"relation": "needs_relabel_total",
         "n_pairs": int(relations.get("needs_relabel", pd.Series(dtype=bool)).sum())},
    ])
    return pd.DataFrame(rows, columns=["relation", "n_pairs"])
