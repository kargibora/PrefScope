"""cluster_win_relevance: aggregate cluster members, reuse the length-controlled logistic."""
import numpy as np
import pandas as pd

from prefscope.pipeline.winrelevance import cluster_win_relevance


def test_cluster_win_relevance_aggregates_and_scores():
    rng = np.random.default_rng(0)
    n = 400
    y = np.array([1.0] * 200 + [0.0] * 200)   # decisive labels
    s = (y - 0.5) * 2.0                         # +1 on chosen-A, -1 on chosen-B
    z = np.zeros((n, 4))
    # cluster 0 tracks preference strongly but WITHOUT perfectly separating it —
    # a hard separator (tiny noise) is quasi-separable, so the unpenalized MLE would
    # (correctly) NaN its p-value; a strong-but-noisy signal is the realistic case.
    z[:, 0] = s + rng.normal(0, 1.0, n)
    z[:, 1] = s + rng.normal(0, 1.0, n)
    z[:, 2] = rng.normal(0, 1, n)               # cluster 1 is noise
    z[:, 3] = rng.normal(0, 1, n)
    length = np.zeros(n)
    clusters = pd.DataFrame({"feature_id": [0, 1, 2, 3], "cluster_id": [0, 0, 1, 1],
                             "behavior": ["detail", "detail", "noise", "noise"]})

    out = cluster_win_relevance(z, y, length, clusters)

    assert set(out["cluster_id"]) == {0, 1}
    assert (out["n_features"] == 2).all()
    assert "delta_win_rate" in out.columns and "delta_win_significant" in out.columns
    c0 = out[out["cluster_id"] == 0].iloc[0]
    c1 = out[out["cluster_id"] == 1].iloc[0]
    assert c0["behavior"] == "detail"
    assert abs(c0["delta_win_rate"]) > 0.2                       # strong, reliable
    assert abs(c1["delta_win_rate"]) < abs(c0["delta_win_rate"])  # noise weaker
    assert bool(c0["delta_win_significant"]) is True


def test_cluster_win_relevance_skips_out_of_range_members():
    # a clusters file may reference feature ids beyond the lens (defensive)
    z = np.random.default_rng(1).normal(0, 1, (100, 2))
    y = np.array([1.0] * 50 + [0.0] * 50)
    clusters = pd.DataFrame({"feature_id": [0, 1, 99], "cluster_id": [0, 0, 0]})
    out = cluster_win_relevance(z, y, np.zeros(100), clusters)
    assert len(out) == 1 and int(out.iloc[0]["n_features"]) == 2
