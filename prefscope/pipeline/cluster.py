"""Group SAE features into corpus-specific groups by co-activation.

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
- ``cofire-leiden`` — cluster interpreted feature poles using positive co-firing
  association. This is the preferred method for signed presence-style lenses:
  negative activations are the unnamed opposite pole, and mutually exclusive
  features do not become neighbors merely because they are statistically dependent.
- ``spherical-kmeans`` — cosine k-means on the unit-normalized feature activation
  columns; a fast, dependency-light approximation that takes a preset
  ``n_clusters``.
- ``agglomerative`` — average-linkage clustering on the co-activation distance
  ``1 - |corr(z_f, z_g)|``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from prefscope.analysis.presence import annotation_flag


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


def feature_cofire_affinity(
    z: np.ndarray,
    *,
    features=None,
    pole: str = "positive",
    metric: str = "phi",
    min_cooccur: int = 30,
    chunk_size: int = 8192,
    return_stats: bool = False,
):
    """Positive co-firing affinity between interpreted feature poles.

    Unlike :func:`feature_mi`, this represents *co-presence*, not arbitrary
    dependence. Signed lens concepts describe the positive pole, so the default
    uses ``z > 0`` and ignores the unnamed negative pole. ``metric`` can be
    positive ``phi`` (chance-corrected), binary ``cosine`` (overlap), or
    support-shrunk positive ``npmi``. Negative association is clipped to zero.

    Counts are accumulated in chunks so a large ``N x M`` firing matrix is never
    materialized. With ``return_stats=True``, ``stats["phi"]`` retains the signed
    positive-pole phi matrix for coherence diagnostics.
    """
    if pole not in {"positive", "negative", "nonzero"}:
        raise ValueError("pole must be 'positive', 'negative', or 'nonzero'")
    if metric not in {"phi", "cosine", "npmi"}:
        raise ValueError("metric must be 'phi', 'cosine', or 'npmi'")
    if min_cooccur < 0:
        raise ValueError("min_cooccur must be >= 0")

    shape = getattr(z, "shape", None)
    if shape is None or len(shape) != 2:
        raise ValueError("z must be a 2D code matrix")
    feats = np.arange(shape[1], dtype=int) if features is None \
        else np.asarray([int(f) for f in features], dtype=int)
    m = len(feats)
    if m == 0:
        empty = np.zeros((0, 0), dtype=float)
        stats = {"n": 0, "features": feats, "ones": np.zeros(0),
                 "cooccur": empty, "phi": empty}
        return (empty, stats) if return_stats else empty

    ones = np.zeros(m, dtype=np.float64)
    cooccur = np.zeros((m, m), dtype=np.float64)
    n = 0
    for start in range(0, shape[0], chunk_size):
        block = np.asarray(z[start:start + chunk_size])[:, feats]
        if pole == "positive":
            fired = block > 0
        elif pole == "negative":
            fired = block < 0
        else:
            fired = block != 0
        # Cast before @: bool @ bool does not produce numeric co-occurrence counts.
        fired = fired.astype(np.float32, copy=False)
        n += len(fired)
        ones += fired.sum(axis=0)
        cooccur += fired.T @ fired

    if n == 0:
        affinity = np.zeros((m, m), dtype=float)
        phi = affinity.copy()
    else:
        p = ones / n
        p11 = cooccur / n
        cov = p11 - p[:, None] * p[None, :]
        var = p * (1.0 - p)
        denom = np.sqrt(var[:, None] * var[None, :])
        phi = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 0)

        if metric == "phi":
            affinity = np.maximum(phi, 0.0)
        elif metric == "cosine":
            denom = np.sqrt(ones[:, None] * ones[None, :])
            affinity = np.divide(cooccur, denom, out=np.zeros_like(cooccur),
                                 where=denom > 0)
        else:
            expected = p[:, None] * p[None, :]
            ratio = np.divide(p11, expected, out=np.ones_like(p11), where=expected > 0)
            ppmi = np.maximum(np.log(np.maximum(ratio, 1e-30)), 0.0)
            affinity = np.divide(ppmi, -np.log(np.maximum(p11, 1e-30)),
                                 out=np.zeros_like(ppmi), where=p11 > 0)
            # Smooth the brittle rare-event tail rather than letting one coincidence
            # become the strongest graph edge.
            if min_cooccur:
                affinity *= np.minimum(1.0, cooccur / float(min_cooccur))

        if min_cooccur:
            affinity = np.where(cooccur >= min_cooccur, affinity, 0.0)
        affinity[~np.isfinite(affinity)] = 0.0
        phi[~np.isfinite(phi)] = 0.0
        np.fill_diagonal(affinity, 0.0)
        np.fill_diagonal(phi, 0.0)

    stats = {"n": int(n), "features": feats, "ones": ones,
             "cooccur": cooccur, "phi": phi}
    return (affinity, stats) if return_stats else affinity


def _sparsify_affinity(weights: np.ndarray, knn: int | None,
                       mode: str = "union") -> np.ndarray:
    """Return a symmetric kNN graph without inventing zero-weight edges."""
    if mode not in {"union", "mutual"}:
        raise ValueError("knn_mode must be 'union' or 'mutual'")
    W = np.array(weights, dtype=float, copy=True)
    np.fill_diagonal(W, 0.0)
    m = W.shape[0]
    if not knn or knn >= m - 1:
        return W
    directed = np.zeros((m, m), dtype=bool)
    top = np.argsort(-W, axis=1)[:, :knn]
    rows = np.repeat(np.arange(m), knn)
    cols = top.ravel()
    # Top-k ties at zero must not turn absent relationships into graph edges.
    positive = W[rows, cols] > 0
    directed[rows[positive], cols[positive]] = True
    keep = (directed | directed.T) if mode == "union" else (directed & directed.T)
    return np.where(keep, W, 0.0)


def _postprocess_small_communities(labels: np.ndarray, min_size: int,
                                   policy: str = "preserve") -> np.ndarray:
    """Compact IDs while never merging unrelated groups unless reproducing legacy output."""
    if policy not in {"preserve", "merge"}:
        raise ValueError("small_community_policy must be 'preserve' or 'merge'")
    labels = np.asarray(labels, dtype=int).copy()
    if policy == "merge" and min_size > 1 and len(labels):
        counts = np.bincount(labels)
        small = set(np.where(counts < min_size)[0].tolist())
        labels = np.array([-1 if c in small else c for c in labels], dtype=int)
    remap = {c: i for i, c in enumerate(sorted(set(labels.tolist())))}
    return np.array([remap[c] for c in labels], dtype=int)


def partition_stability(label_runs: list[np.ndarray]) -> dict:
    """Pairwise adjusted-Rand stability, invariant to numeric label permutations."""
    if len(label_runs) < 2:
        count = len(np.unique(label_runs[0])) if label_runs else None
        return {"seed_ari_mean": None, "seed_ari_min": None,
                "cluster_count_min": count, "cluster_count_max": count}
    from sklearn.metrics import adjusted_rand_score
    scores = [adjusted_rand_score(label_runs[i], label_runs[j])
              for i in range(len(label_runs)) for j in range(i + 1, len(label_runs))]
    counts = [len(np.unique(labels)) for labels in label_runs]
    return {"seed_ari_mean": float(np.mean(scores)), "seed_ari_min": float(np.min(scores)),
            "cluster_count_min": int(min(counts)), "cluster_count_max": int(max(counts))}


def _leiden_partition(mi: np.ndarray, *, resolution: float, seed: int,
                      knn: int | None = None, min_community_size: int = 1,
                      knn_mode: str = "union",
                      small_community_policy: str = "merge") -> np.ndarray:
    """Partition the MI feature graph with the Leiden algorithm (per-node labels).

    ``knn`` sparsifies the graph to each node's strongest edges before partitioning.
    ``knn_mode="union"`` reproduces legacy MI-Leiden; ``"mutual"`` removes one-sided
    weak bridges. New behavior clustering should use ``small_community_policy="preserve"``;
    ``"merge"`` exists only for reproducing legacy artifacts.
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
    W = _sparsify_affinity(mi, knn, mode=knn_mode)

    iu, ju = np.triu_indices(m, k=1)
    w = W[iu, ju]
    keep = w > 0
    graph = ig.Graph(n=m, edges=list(zip(iu[keep].tolist(), ju[keep].tolist())))
    graph.es["weight"] = w[keep].tolist()
    partition = leidenalg.find_partition(
        graph, leidenalg.RBConfigurationVertexPartition,
        weights="weight", resolution_parameter=resolution, seed=seed)
    labels = np.asarray(partition.membership, dtype=int)

    return _postprocess_small_communities(
        labels, min_community_size, policy=small_community_policy)


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
                     knn: int | None = None, min_community_size: int = 1,
                     affinity_metric: str = "phi", pole: str = "positive",
                     min_cooccur: int = 30, knn_mode: str = "mutual",
                     small_community_policy: str = "preserve",
                     stability_runs: int = 1,
                     super_resolution: float | None = None,
                     super_knn: int = 4) -> pd.DataFrame:
    """Group features into behaviors. Returns [feature_id, cluster_id].

    ``mi-leiden`` and ``cofire-leiden`` select the number of behaviors via ``resolution``
    (``n_clusters`` is ignored). ``spherical-kmeans`` and ``agglomerative`` use
    the preset ``n_clusters``; the latter can degenerate into one blob plus
    singletons when correlations are weak (sparse TopK codes).
    """
    feats = list(range(z.shape[1])) if features is None else [int(f) for f in features]
    if not feats:
        return pd.DataFrame(columns=["feature_id", "cluster_id"])

    if method == "mi-leiden":
        z = np.asarray(z, dtype=np.float32)
        sub = z[:, feats]
        labels = _leiden_partition(feature_mi(sub), resolution=resolution, seed=seed,
                                   knn=knn, min_community_size=min_community_size,
                                   knn_mode="union", small_community_policy="merge")
        result = pd.DataFrame({"feature_id": feats, "cluster_id": labels.astype(int)})
        result.attrs["run_diagnostics"] = {
            "method": method, "resolution": float(resolution), "knn": knn,
            "knn_mode": "union", "small_community_policy": "merge",
            "n_features": len(feats), "n_clusters": int(len(np.unique(labels))),
        }
        return result
    elif method == "cofire-leiden":
        affinity, stats = feature_cofire_affinity(
            z, features=feats, pole=pole, metric=affinity_metric,
            min_cooccur=min_cooccur, return_stats=True)
        runs = [_leiden_partition(
            affinity, resolution=resolution, seed=seed + i, knn=knn,
            min_community_size=min_community_size, knn_mode=knn_mode,
            small_community_policy=small_community_policy)
            for i in range(max(1, int(stability_runs)))]
        labels = runs[0]
        result = pd.DataFrame({"feature_id": feats, "cluster_id": labels.astype(int)})

        if super_resolution is not None:
            fine = sorted(np.unique(labels).tolist())
            group_affinity = np.zeros((len(fine), len(fine)), dtype=float)
            for i, ci in enumerate(fine):
                ii = np.where(labels == ci)[0]
                for j in range(i + 1, len(fine)):
                    jj = np.where(labels == fine[j])[0]
                    vals = affinity[np.ix_(ii, jj)]
                    group_affinity[i, j] = group_affinity[j, i] = (
                        float(vals.mean()) if vals.size else 0.0)
            super_labels = _leiden_partition(
                group_affinity, resolution=float(super_resolution), seed=seed,
                knn=min(int(super_knn), max(0, len(fine) - 1)),
                min_community_size=1, knn_mode="mutual",
                small_community_policy="preserve")
            fine_to_super = dict(zip(fine, super_labels.astype(int)))
            result["supercluster_id"] = result["cluster_id"].map(fine_to_super).astype(int)

        stability = partition_stability(runs)
        result.attrs["affinity"] = affinity
        result.attrs["phi"] = stats["phi"]
        result.attrs["local_features"] = np.asarray(feats, dtype=int)
        result.attrs["run_diagnostics"] = {
            "method": method, "affinity_metric": affinity_metric, "pole": pole,
            "resolution": float(resolution), "knn": knn, "knn_mode": knn_mode,
            "min_cooccur": int(min_cooccur),
            "small_community_policy": small_community_policy,
            "stability_runs": max(1, int(stability_runs)),
            "n_features": len(feats), "n_clusters": int(len(np.unique(labels))),
            **stability,
        }
        if "supercluster_id" in result:
            result.attrs["run_diagnostics"]["n_superclusters"] = int(
                result["supercluster_id"].nunique())
            result.attrs["run_diagnostics"]["super_resolution"] = float(super_resolution)
        return result
    elif method == "agglomerative":
        z = np.asarray(z, dtype=np.float32)
        sub = z[:, feats]
        from sklearn.cluster import AgglomerativeClustering
        k = max(1, min(n_clusters, len(feats)))
        labels = AgglomerativeClustering(
            n_clusters=k, metric="precomputed", linkage="average"
        ).fit_predict(feature_distance(sub))
    elif method == "spherical-kmeans":
        z = np.asarray(z, dtype=np.float32)
        sub = z[:, feats]
        k = max(1, min(n_clusters, len(feats)))
        labels = _spherical_kmeans(sub.T, k)   # cluster feature columns
    else:
        raise ValueError(
            "method must be 'cofire-leiden', 'mi-leiden', 'spherical-kmeans', "
            "or 'agglomerative', "
            f"got {method!r}")

    return pd.DataFrame({"feature_id": feats, "cluster_id": labels.astype(int)})


def cluster_run_diagnostics(clusters: pd.DataFrame) -> pd.DataFrame:
    """One-row run diagnostics stored separately from canonical memberships."""
    info = dict(clusters.attrs.get("run_diagnostics", {}))
    if clusters.empty:
        info.setdefault("n_features", 0)
        info.setdefault("n_clusters", 0)
    else:
        sizes = clusters.groupby("cluster_id").size()
        info.update({
            "n_features": int(len(clusters)),
            "n_clusters": int(len(sizes)),
            "largest_cluster": int(sizes.max()),
            "median_cluster_size": float(sizes.median()),
            "singleton_clusters": int((sizes == 1).sum()),
            "clusters_lt3": int((sizes < 3).sum()),
        })
    return pd.DataFrame([info])


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
    """One row per community with central/diverse members and coherence diagnostics.

    Multi-feature communities receive a neutral provisional label until an explicit
    naming step runs. This prevents an arbitrary high-fidelity member from masquerading
    as an umbrella behavior (the old source of labels such as ``written in Russian``).
    """
    if clusters.empty:
        return pd.DataFrame(columns=[
            "cluster_id", "n_features", "n_verified", "behavior",
            "feature_ids", "member_concepts"])
    affinity = clusters.attrs.get("affinity")
    phi = clusters.attrs.get("phi")
    local_features = clusters.attrs.get("local_features")
    local_index = ({int(f): i for i, f in enumerate(local_features)}
                   if local_features is not None else {})
    df = clusters.copy()
    if names is not None:
        keep = [c for c in ("feature_id", "concept", "correlation", "fidelity_pass")
                if c in names.columns]
        df = df.merge(names[keep], on="feature_id", how="left")

    rows = []
    for cid, g in df.groupby("cluster_id"):
        named = g
        if "fidelity_pass" in g.columns:
            named = g[g["fidelity_pass"].map(annotation_flag)]

        fids = [int(f) for f in g["feature_id"]]
        local = [local_index[f] for f in fids if f in local_index]
        within_affinity = within_phi = negative_fraction = external_affinity = None
        centrality = {f: 0.0 for f in fids}
        if affinity is not None and len(local) == len(fids):
            block = affinity[np.ix_(local, local)]
            if len(local) > 1:
                tri = block[np.triu_indices(len(local), 1)]
                within_affinity = float(tri.mean()) if len(tri) else None
                central = block.sum(axis=1) / max(1, len(local) - 1)
                centrality.update(dict(zip(fids, central.astype(float))))
            outside = np.setdiff1d(np.arange(affinity.shape[0]), np.asarray(local))
            if len(outside):
                external_affinity = float(affinity[np.ix_(local, outside)].mean())
        if phi is not None and len(local) == len(fids) and len(local) > 1:
            vals = phi[np.ix_(local, local)][np.triu_indices(len(local), 1)]
            within_phi = float(vals.mean()) if len(vals) else None
            negative_fraction = float((vals < 0).mean()) if len(vals) else None

        # Maximal-marginal-relevance ordering: start central, then retain strong
        # representatives that are not merely duplicates of what is already shown.
        remaining = list(fids)
        reps: list[int] = []
        while remaining and len(reps) < min(16, len(fids)):
            def _score(fid):
                score = centrality.get(fid, 0.0)
                if reps and affinity is not None and fid in local_index:
                    overlap = max(float(affinity[local_index[fid], local_index[r]])
                                  for r in reps if r in local_index)
                    score -= 0.35 * overlap
                return score
            chosen = max(remaining, key=lambda f: (_score(f), -f))
            reps.append(chosen)
            remaining.remove(chosen)

        concept_by = (dict(zip(named["feature_id"].astype(int), named["concept"]))
                      if "concept" in named.columns else {})
        representative_concepts = [str(concept_by[f]) for f in reps
                                   if f in concept_by and pd.notna(concept_by[f])]
        all_concepts = [str(concept_by[f]) for f in fids
                        if f in concept_by and pd.notna(concept_by[f])]
        n_verified = int(g["fidelity_pass"].map(annotation_flag).sum()) \
            if "fidelity_pass" in g.columns else None
        row = {
            "cluster_id": int(cid),
            "n_features": int(len(g)),
            "n_verified": n_verified,
            "behavior": (representative_concepts[0] if len(g) == 1
                         and representative_concepts else f"cluster {cid}"),
            "feature_ids": ",".join(str(f) for f in fids),
            "representative_feature_ids": ",".join(str(f) for f in reps),
            "representative_concepts": " | ".join(representative_concepts),
            "member_concepts": " | ".join(all_concepts),
            "within_affinity_mean": within_affinity,
            "external_affinity_mean": external_affinity,
            "affinity_separation": (within_affinity - external_affinity
                                    if within_affinity is not None
                                    and external_affinity is not None else None),
            "within_phi_mean": within_phi,
            "negative_pair_fraction": negative_fraction,
        }
        if "supercluster_id" in g.columns:
            vals = g["supercluster_id"].dropna().astype(int).unique()
            row["supercluster_id"] = int(vals[0]) if len(vals) == 1 else None
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n_features", ascending=False).reset_index(drop=True)


def name_clusters(summary: pd.DataFrame, client, *, concurrency: int = 1) -> dict:
    """LLM-name communities, explicitly allowing mixed/incoherent outcomes."""
    from prefscope.interpret._parallel import run as _run

    def _one(row) -> tuple:
        cid = int(row["cluster_id"])
        members = row.get("representative_concepts", "") or row.get("member_concepts", "")
        if not members:
            return cid, ""
        prompt = (
            "These SAE feature concepts were placed in one statistical co-firing "
            "community. Co-firing can reflect a genuine shared behavior, a topic/language "
            "mixture, or dataset confounding. Do not assume the community is coherent.\n\n"
            "Central and diverse member concepts:\n\n"
            f"{members}\n\n"
            "If there is a clear shared theme, give a SHORT 3-7 word umbrella label. "
            "If it is heterogeneous, reply `MIXED: theme A / theme B`. If no defensible "
            "summary exists, reply `ABSTAIN`. Reply with only that label.")
        try:
            out = client.raw([{"role": "user", "content": prompt}], max_tokens=50)
            label = out.strip().splitlines()[0].strip().strip('"').strip()
            if label.upper() == "MIXED":
                label = "Mixed / incoherent"
            elif label.upper() == "ABSTAIN":
                label = "Unlabeled mixed community"
            return cid, label
        except Exception:
            return cid, ""

    pairs = _run(_one, [r for _, r in summary.iterrows()], concurrency,
                 desc="naming clusters", usage=client)
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


@registry.register("clusterer", "cofire-leiden")
class CofireLeidenClusterer(Clusterer):
    """Positive-pole co-firing graph + Leiden with stability diagnostics."""

    def __init__(self, *, resolution: float = 5.0, knn: int = 8,
                 affinity_metric: str = "phi", pole: str = "positive",
                 min_cooccur: int = 30, knn_mode: str = "mutual",
                 min_cluster_size: int = 1,
                 small_community_policy: str = "preserve",
                 stability_runs: int = 5, super_resolution: float | None = None,
                 super_knn: int = 4, seed: int = 0, **_) -> None:
        self.resolution, self.knn = resolution, knn
        self.affinity_metric, self.pole = affinity_metric, pole
        self.min_cooccur, self.knn_mode = min_cooccur, knn_mode
        self.min_cluster_size = min_cluster_size
        self.small_community_policy = small_community_policy
        self.stability_runs = stability_runs
        self.super_resolution, self.super_knn = super_resolution, super_knn
        self.seed = seed

    def cluster(self, z, *, features=None):
        return cluster_features(
            z, method="cofire-leiden", features=features,
            resolution=self.resolution, knn=(self.knn or None),
            affinity_metric=self.affinity_metric, pole=self.pole,
            min_cooccur=self.min_cooccur, knn_mode=self.knn_mode,
            min_community_size=self.min_cluster_size,
            small_community_policy=self.small_community_policy,
            stability_runs=self.stability_runs,
            super_resolution=self.super_resolution, super_knn=self.super_knn,
            seed=self.seed)


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
