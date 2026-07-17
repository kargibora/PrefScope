"""The prompt-lens CLI paths folded from cluster_prompts.py / verify_prompts.py:
cluster-features / interpret verify with --lens-kind prompt read z_prompt.npy."""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from prefscope.__main__ import _cmd_cluster_features


def test_cluster_features_lens_kind_prompt(tmp_path):
    rng = np.random.default_rng(0)
    M = 12
    z = (rng.random((200, M)) * (rng.random((200, M)) < 0.15)).astype(np.float32)
    np.save(tmp_path / "z_prompt.npy", z)   # a prompt lens has only z_prompt.npy (no z_diff)
    pd.DataFrame({"feature_id": range(M), "concept": [f"c{i}" for i in range(M)]}).to_csv(
        tmp_path / "prompt_feature_names.csv", index=False)
    out = tmp_path / "prompt_feature_clusters.csv"
    args = SimpleNamespace(
        lens_dir=str(tmp_path), lens_kind="prompt", cluster_on="difference",
        names=str(tmp_path / "prompt_feature_names.csv"), fidelity_only=False,
        n_clusters=3, method="spherical-kmeans", resolution=1.0, knn=0,
        min_cluster_size=1, name_clusters=False, out=str(out), concurrency=1)

    assert _cmd_cluster_features(args) == 0
    df = pd.read_csv(out)
    assert "cluster_id" in df.columns and len(df) == M
    assert (tmp_path / "prompt_feature_clusters_summary.csv").exists()
