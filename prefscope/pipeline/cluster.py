"""Group SAE features into higher-level behaviors by co-activation.

A large dictionary tends to split one behavior across many near-duplicate
features; clustering merges features that fire together. Three methods are
provided:

- ``mi-leiden`` — consolidate near-duplicate *features* into concepts: a feature
  graph weighted by the mutual information between features' firing patterns,
  partitioned with the Leiden community-detection algorithm (Traag, Waltman & van
  Eck, "From Louvain to Leiden: guaranteeing well-connected communities", Sci.
  Rep. 9:5233, 2019). The number of concepts follows from the graph structure
  (tuned via ``resolution``). Requires the optional ``igraph`` and ``leidenalg``
  packages.
  NOTE: this clusters FEATURES (to merge redundant concepts). It is *not* the
  data-example clustering of *Anatomy of Post-Training* (App. B), which groups
  training examples by behavior — a different object; see the dataset-diagnosis
  design for example-level grouping.
- ``spherical-kmeans`` — cosine k-means on the unit-normalized feature activation
  columns; a fast, dependency-light approximation that takes a preset
  ``n_clusters``.
- ``agglomerative`` — average-linkage clustering on the co-activation distance
  ``1 - |corr(z_f, z_g)|``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def feature_distance(z: np.ndarray) -> np.ndarray:
    """(M, M) co-activation distance 1 - |corr| between feature columns."""
    z = np.asarray(z, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.corrcoef(z.T)                      # nan for zero-variance columns
    d = 1.0 - np.abs(c)
    d[~np.isfinite(d)] = 1.0                       # never-firing features: maximally far
    np.fill_diagonal(d, 0.0)
    return d


def feature_mi(z: np.ndarray) -> np.ndarray:
    """(M, M) mutual information (nats) between features' binary firing patterns.

    Each feature is reduced to its firing indicator ``z[:, f] != 0`` across items,
    and MI is computed in closed form from the 2x2 co-firing contingency. The
    diagonal is zeroed; never-firing features contribute zero MI.
    """
    fired = (np.asarray(z) != 0).astype(np.float64)        # (N, M)
    n, m = fired.shape
    if n == 0:
        return np.zeros((m, m))
    p1 = fired.mean(axis=0)                                 # P(feature fires)
    p0 = 1.0 - p1
    p11 = (fired.T @ fired) / n                             # P(both fire)
    p10 = p1[:, None] - p11
    p01 = p1[None, :] - p11
    p00 = 1.0 - p11 - p10 - p01

    def _term(joint, pa, pb):
        denom = pa * pb
        with np.errstate(divide="ignore", invalid="ignore"):
            t = joint * np.log(joint / denom)
        return np.where((joint > 0) & (denom > 0), t, 0.0)

    mi = (_term(p11, p1[:, None], p1[None, :])
          + _term(p10, p1[:, None], p0[None, :])
          + _term(p01, p0[:, None], p1[None, :])
          + _term(p00, p0[:, None], p0[None, :]))
    np.fill_diagonal(mi, 0.0)
    return mi


def _leiden_partition(mi: np.ndarray, *, resolution: float, seed: int,
                      knn: int | None = None, min_community_size: int = 1) -> np.ndarray:
    """Partition the MI feature graph with the Leiden algorithm (per-node labels).

    ``knn`` sparsifies the graph to each node's strongest ``knn`` edges (symmetric
    union) before partitioning — running Leiden on a dense MI matrix degrades
    community quality. ``min_community_size`` relabels communities smaller than the
    threshold to a shared bucket (id 0). Labels are compactly renumbered.
    """
    try:
        import igraph as ig
        import leidenalg
    except ImportError as exc:
        raise ImportError(
            "mi-leiden clustering needs igraph + leidenalg — install them with: "
            "uv sync --extra cluster"
        ) from exc

    m = mi.shape[0]
    W = np.array(mi, dtype=float)
    np.fill_diagonal(W, 0.0)
    if knn and knn < m - 1:                       # keep each node's top-knn edges
        keep_mask = np.zeros((m, m), dtype=bool)
        top = np.argsort(-W, axis=1)[:, :knn]
        keep_mask[np.repeat(np.arange(m), knn), top.ravel()] = True
        keep_mask |= keep_mask.T
        W = np.where(keep_mask, W, 0.0)

    iu, ju = np.triu_indices(m, k=1)
    w = W[iu, ju]
    keep = w > 0
    graph = ig.Graph(n=m, edges=list(zip(iu[keep].tolist(), ju[keep].tolist())))
    graph.es["weight"] = w[keep].tolist()
    partition = leidenalg.find_partition(
        graph, leidenalg.RBConfigurationVertexPartition,
        weights="weight", resolution_parameter=resolution, seed=seed)
    labels = np.asarray(partition.membership, dtype=int)

    if min_community_size > 1:                    # fold tiny communities into bucket 0
        counts = np.bincount(labels)
        small = set(np.where(counts < min_community_size)[0].tolist())
        labels = np.array([-1 if c in small else c for c in labels])
    remap = {c: i for i, c in enumerate(sorted(set(labels.tolist())))}
    return np.array([remap[c] for c in labels], dtype=int)


def _spherical_kmeans(points: np.ndarray, k: int) -> np.ndarray:
    """Cosine k-means: unit-normalize each row of ``points`` (zero-norm rows stay
    at the origin) and assign one of ``k`` clusters. Returns (n,) int labels."""
    from sklearn.cluster import KMeans

    x = np.asarray(points, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = np.divide(x, norms, out=np.zeros_like(x), where=norms > 1e-8)
    with np.errstate(all="ignore"):
        return KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(x)


def cluster_features(z: np.ndarray, n_clusters: int = 10,
                     method: str = "spherical-kmeans", *, features=None,
                     resolution: float = 1.0, seed: int = 0,
                     knn: int | None = None, min_community_size: int = 1) -> pd.DataFrame:
    """Group features into behaviors. Returns [feature_id, cluster_id].

    ``mi-leiden`` selects the number of behaviors itself via ``resolution``
    (``n_clusters`` is ignored). ``spherical-kmeans`` and ``agglomerative`` use
    the preset ``n_clusters``; the latter can degenerate into one blob plus
    singletons when correlations are weak (sparse TopK codes).
    """
    z = np.asarray(z, dtype=np.float32)
    feats = list(range(z.shape[1])) if features is None else [int(f) for f in features]
    if not feats:
        return pd.DataFrame(columns=["feature_id", "cluster_id"])
    sub = z[:, feats]

    if method == "mi-leiden":
        labels = _leiden_partition(feature_mi(sub), resolution=resolution, seed=seed,
                                   knn=knn, min_community_size=min_community_size)
    elif method == "agglomerative":
        from sklearn.cluster import AgglomerativeClustering
        k = max(1, min(n_clusters, len(feats)))
        labels = AgglomerativeClustering(
            n_clusters=k, metric="precomputed", linkage="average"
        ).fit_predict(feature_distance(sub))
    elif method == "spherical-kmeans":
        k = max(1, min(n_clusters, len(feats)))
        labels = _spherical_kmeans(sub.T, k)   # cluster feature columns
    else:
        raise ValueError(
            "method must be 'mi-leiden', 'spherical-kmeans', or 'agglomerative', "
            f"got {method!r}")

    return pd.DataFrame({"feature_id": feats, "cluster_id": labels.astype(int)})


def cluster_examples(profile: np.ndarray, n_clusters: int = 10, *,
                     method: str = "spherical-kmeans") -> pd.DataFrame:
    """Cluster the N example *rows* of ``profile`` (N, M) into behavior regions
    B_k — the data-example clustering of *Anatomy of Post-Training* (App. B.1),
    via spherical k-means in normalized activation space. Returns
    [example_index, cluster_id].

    ``profile`` is the per-example activity matrix to cluster on, e.g. the
    symmetric activity ``s`` (see ``analysis.dataset.symmetric_activity``) or
    ``|z_diff|`` for a difference lens.
    """
    if method != "spherical-kmeans":
        raise ValueError(f"cluster_examples supports 'spherical-kmeans', got {method!r}")
    x = np.asarray(profile, dtype=np.float32)
    n = x.shape[0]
    labels = _spherical_kmeans(x, max(1, min(n_clusters, n)))
    return pd.DataFrame({"example_index": np.arange(n), "cluster_id": labels.astype(int)})


def summarize_clusters(clusters: pd.DataFrame,
                       names: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per behavior: size, member feature ids, a representative name.

    When fidelity results are available, only passing positive-pole names can label
    the behavior; the representative is their highest-correlation member. With raw
    unverified names, the first member is used as a provisional label.
    """
    if clusters.empty:
        return pd.DataFrame(columns=[
            "cluster_id", "n_features", "n_verified", "behavior",
            "feature_ids", "member_concepts"])
    df = clusters.copy()
    if names is not None:
        keep = [c for c in ("feature_id", "concept", "correlation", "fidelity_pass")
                if c in names.columns]
        df = df.merge(names[keep], on="feature_id", how="left")

    rows = []
    for cid, g in df.groupby("cluster_id"):
        named = g
        if "fidelity_pass" in g.columns:
            named = g[g["fidelity_pass"].fillna(False).astype(bool)]
        if "correlation" in named.columns and named["correlation"].notna().any():
            rep = named.loc[named["correlation"].idxmax()]
        elif not named.empty:
            rep = named.iloc[0]
        else:
            rep = None
        concepts = (named["concept"].dropna().tolist()
                    if "concept" in named.columns else [])
        n_verified = int(g["fidelity_pass"].fillna(False).astype(bool).sum()) \
            if "fidelity_pass" in g.columns else None
        rows.append({
            "cluster_id": int(cid),
            "n_features": int(len(g)),
            "n_verified": n_verified,
            "behavior": (rep["concept"] if rep is not None and "concept" in g.columns
                         and pd.notna(rep.get("concept")) else f"cluster {cid}"),
            "feature_ids": ",".join(str(int(f)) for f in g["feature_id"]),
            "member_concepts": " | ".join(concepts[:8]),
        })
    return pd.DataFrame(rows).sort_values("n_features", ascending=False).reset_index(drop=True)


def name_clusters(summary: pd.DataFrame, client, *, concurrency: int = 1) -> dict:
    """LLM-name each behavior from its member concepts. Returns {cluster_id: label}.

    The per-feature concepts in a cluster describe one shared behavior; a single
    representative concept mislabels heterogeneous clusters, so we ask an LLM for a
    short umbrella label.
    """
    from prefscope.interpret._parallel import run as _run

    def _one(row) -> tuple:
        cid = int(row["cluster_id"])
        members = row.get("member_concepts", "")
        if not members:
            return cid, ""
        prompt = (
            "The following response-difference features fire together, so they form "
            "ONE higher-level behavior. Member concepts:\n\n"
            f"{members}\n\n"
            "Give a SHORT (3-6 word) umbrella label for the shared behavior. "
            "Reply with only the label, no quotes or punctuation.")
        try:
            out = client.raw([{"role": "user", "content": prompt}], max_tokens=30)
            return cid, out.strip().splitlines()[0].strip().strip('"').strip()
        except Exception:
            return cid, ""

    pairs = _run(_one, [r for _, r in summary.iterrows()], concurrency,
                 desc="naming clusters")
    return {cid: name for cid, name in pairs}


# ---------------------------------------------------------------------------------------
# Clusterer components — swap the clustering algorithm by name (CLI --method / config).
# Each wraps cluster_features with its method fixed; tunables in __init__ (YAML-mappable).
# ---------------------------------------------------------------------------------------
from abc import ABC, abstractmethod  # noqa: E402

from prefscope.core import registry  # noqa: E402


def load_cofiring_codes(lens_dir, *, lens_kind: str = "completion",
                        cluster_on: str = "difference") -> np.ndarray:
    """The (N, M) code matrix a clusterer co-fires over, picked by lens kind.

    A prompt lens clusters ``z_prompt``. A completion lens clusters the difference
    codes ``z_diff`` by default, or the stacked individual codes ``[z_a; z_b]`` when
    ``cluster_on="individual"`` (semantic co-occurrence à la Anatomy — avoids merging
    antonym features that co-fire only in the contrast). Falls back to ``z_diff`` when
    individual codes are absent. Shared by the CLI and the config runner so both pick
    the same space."""
    from pathlib import Path

    from prefscope.artifacts import Z_A, Z_B, Z_DIFF, Z_PROMPT

    ld = Path(lens_dir)
    if lens_kind == "prompt":
        return np.load(ld / Z_PROMPT)
    if (cluster_on == "individual" or not (ld / Z_DIFF).exists()) and (ld / Z_A).exists():
        arrays = [np.load(ld / Z_A)]
        if (ld / Z_B).exists():
            arrays.append(np.load(ld / Z_B))
        return np.concatenate(arrays, axis=0)
    return np.load(ld / Z_DIFF)


class Clusterer(ABC):
    """Group SAE features into behaviors. Returns a [feature_id, cluster_id] frame."""

    @abstractmethod
    def cluster(self, z: np.ndarray, *, features=None) -> pd.DataFrame:
        ...


@registry.register("clusterer", "mi-leiden")
class MiLeidenClusterer(Clusterer):
    """MI co-firing graph + Leiden; the behavior count emerges from ``resolution``."""

    def __init__(self, *, resolution: float = 1.0, knn: int = 0,
                 min_cluster_size: int = 1, seed: int = 0, **_) -> None:
        self.resolution, self.knn = resolution, knn
        self.min_cluster_size, self.seed = min_cluster_size, seed

    def cluster(self, z, *, features=None):
        return cluster_features(z, method="mi-leiden", features=features,
                                resolution=self.resolution, knn=(self.knn or None),
                                min_community_size=self.min_cluster_size, seed=self.seed)


@registry.register("clusterer", "spherical-kmeans")
class SphericalKmeansClusterer(Clusterer):
    """k-means on L2-normalized feature vectors; uses a preset ``n_clusters``."""

    def __init__(self, *, n_clusters: int = 10, seed: int = 0, **_) -> None:
        self.n_clusters, self.seed = n_clusters, seed

    def cluster(self, z, *, features=None):
        return cluster_features(z, method="spherical-kmeans", n_clusters=self.n_clusters,
                                features=features, seed=self.seed)


@registry.register("clusterer", "agglomerative")
class AgglomerativeClusterer(Clusterer):
    """Agglomerative clustering on the feature-correlation distance; preset ``n_clusters``."""

    def __init__(self, *, n_clusters: int = 10, seed: int = 0, **_) -> None:
        self.n_clusters, self.seed = n_clusters, seed

    def cluster(self, z, *, features=None):
        return cluster_features(z, method="agglomerative", n_clusters=self.n_clusters,
                                features=features, seed=self.seed)
