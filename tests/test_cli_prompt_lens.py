"""The prompt-lens CLI paths folded from cluster_prompts.py / verify_prompts.py:
cluster-features / interpret verify with --lens-kind prompt read z_prompt.npy."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from prefscope import __main__ as cli
from prefscope.cli import common as cli_common
from prefscope.cli.analysis import _cmd_cluster_features


def _prompt_lens_and_corpus(tmp_path):
    pd.DataFrame({"instruction_id": ["0", "1"]}).to_parquet(
        tmp_path / "battles.parquet")
    np.save(tmp_path / "z_prompt.npy",
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    (tmp_path / "manifest.json").write_text(json.dumps({
        "input_rep": "prompt", "lens_kind": "prompt",
        "sae_type": "batchtopk-relu", "activation_polarity": "nonnegative",
        "code_semantics": "presence",
    }))
    corpus = tmp_path / "corpus.parquet"
    pd.DataFrame({
        "battle_id": ["0", "1"], "source": ["test", "test"],
        "language": ["en", "en"], "prompt": ["prompt zero", "prompt one"],
        "model_a": ["A", "A"], "model_b": ["B", "B"],
        "completion_a": ["a0", "a1"], "completion_b": ["b0", "b1"],
    }).to_parquet(corpus, index=False)
    return corpus


class _FakeTrackedClient:
    """Small test double retaining the real UsageTracker supplied by the CLI."""

    def __init__(self, *, usage_tracker, usage_stage, model, backend, **kwargs):
        self.usage_tracker = usage_tracker
        self.usage_stage = usage_stage
        self.model = model
        self.backend = backend

    def record_test_request(self):
        self.usage_tracker.record_error(
            RuntimeError("synthetic request"), requested_model=self.model,
            backend=self.backend, stage=self.usage_stage, attempt=1)

    def write_usage(self, path):
        return self.usage_tracker.write_summary(path)

    def usage_progress(self):
        return self.usage_tracker.progress()


def test_build_prompt_lens_cli_defaults_to_auto_without_matryoshka(tmp_path, monkeypatch):
    seen = {}

    def fake_build_prompt_lens(emb, out, **kw):
        seen.update(kw)
        return {"input_rep": "prompt"}

    monkeypatch.setattr(
        "prefscope.pipeline.build_lens.build_prompt_lens", fake_build_prompt_lens)
    rc = cli.main(["build-prompt-lens", "--from-embeddings", str(tmp_path / "emb"),
                   "--out", str(tmp_path / "lens")])
    assert rc == 0
    assert seen["sae_type"] == "auto"
    assert seen["matryoshka_prefix"] == ()
    assert seen["sparsity_warmup_steps"] == 0


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


def test_feature_relations_lens_kind_prompt(tmp_path):
    z = np.zeros((40, 3), dtype=np.float32)
    z[:20, 0] = 1
    z[:10, 1] = 1  # specialization of feature 0
    z[30:, 2] = 1
    np.save(tmp_path / "z_prompt.npy", z)
    names = tmp_path / "prompt_feature_names.csv"
    pd.DataFrame({
        "feature_id": [0, 1, 2],
        "concept": ["written in Greek", "asks for code in Greek", "is written in Greek"],
    }).to_csv(names, index=False)
    out = tmp_path / "prompt_feature_relations.csv"

    rc = cli.main([
        "feature-relations", "--lens-dir", str(tmp_path), "--lens-kind", "prompt",
        "--names", str(names), "--min-cooccur", "1", "--no-decoder",
        "--out", str(out),
    ])
    assert rc == 0
    relations = pd.read_csv(out)
    assert ((relations.feature_a == 0) & (relations.feature_b == 1)).any()
    assert ((relations.feature_a == 0) & (relations.feature_b == 2)).any()
    assert (tmp_path / "prompt_feature_relations_summary.csv").exists()


def test_prompt_naming_checkpoint_usage_resume_alias_and_fresh(tmp_path, monkeypatch):
    corpus = _prompt_lens_and_corpus(tmp_path)
    out = tmp_path / "prompt_feature_names.csv"
    monkeypatch.setattr(cli_common, "LLMClient", _FakeTrackedClient)
    first_calls = []

    def interrupted_name(prompts, z_prompt, client, **kw):
        first_calls.append(list(kw["features"]))
        client.record_test_request()
        kw["on_result"]({"feature_id": 0, "concept": "zero", "status": "ok"})
        raise RuntimeError("connection lost")

    monkeypatch.setattr(
        "prefscope.interpret.prompt_name.name_prompt_features", interrupted_name)
    argv = [
        "interpret", "name", "--lens-dir", str(tmp_path),
        "--lens-kind", "prompt", "--corpus", str(corpus),
        "--out", str(out), "--model", "test-model",
    ]

    import pytest
    with pytest.raises(RuntimeError, match="connection lost"):
        cli.main(argv)

    assert first_calls == [[0, 1]]
    assert pd.read_csv(out)["feature_id"].tolist() == [0]
    assert (tmp_path / "prompt_feature_names.resume.json").exists()
    usage_path = tmp_path / "prompt_feature_names.usage.json"
    events_path = tmp_path / "prompt_feature_names.usage.jsonl"
    assert usage_path.exists() and events_path.exists()
    assert json.loads(usage_path.read_text())["total"]["attempted_requests"] == 1

    resumed_calls = []

    def finish_name(prompts, z_prompt, client, **kw):
        resumed_calls.append(list(kw["features"]))
        client.record_test_request()
        rows = [{"feature_id": int(f), "concept": f"prompt {f}", "status": "ok"}
                for f in kw["features"]]
        for row in rows:
            kw["on_result"](row)
        return pd.DataFrame(rows)

    monkeypatch.setattr(
        "prefscope.interpret.prompt_name.name_prompt_features", finish_name)
    assert cli.main(argv) == 0
    assert resumed_calls == [[1]]
    assert pd.read_csv(out)["feature_id"].tolist() == [0, 1]
    usage = json.loads(usage_path.read_text())
    assert usage["resumed_events"] == 1
    assert usage["total"]["attempted_requests"] == 2

    # The legacy spelling is a compatibility alias for the same completed run.
    alias = [
        "name-prompts", "--lens-dir", str(tmp_path), "--corpus", str(corpus),
        "--out", str(out), "--model", "test-model",
    ]
    assert cli.main(alias) == 0
    assert resumed_calls == [[1]]

    assert cli.main(argv + ["--fresh"]) == 0
    assert resumed_calls == [[1], [0, 1]]
    fresh_usage = json.loads(usage_path.read_text())
    assert fresh_usage["resumed_events"] == 0
    assert fresh_usage["total"]["attempted_requests"] == 1
