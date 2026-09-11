"""The prompt-lens CLI paths folded from cluster_prompts.py / verify_prompts.py:
cluster-features / interpret verify with --lens-kind prompt read z_prompt.npy."""
import json

import numpy as np
import pandas as pd

from prefscope import __main__ as cli
from prefscope.cli import common as cli_common


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


def test_prompt_naming_checkpoint_usage_resume_alias_and_fresh(tmp_path, monkeypatch):
    corpus = _prompt_lens_and_corpus(tmp_path)
    out = tmp_path / "results" / "prompt_feature_names.csv"
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
    assert (out.parent / "prompt_feature_names.resume.json").exists()
    usage_path = out.parent / "prompt_feature_names.usage.json"
    events_path = out.parent / "prompt_feature_names.usage.jsonl"
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

    assert cli.main(argv + ["--fresh"]) == 0
    assert resumed_calls == [[1], [0, 1]]
    fresh_usage = json.loads(usage_path.read_text())
    assert fresh_usage["resumed_events"] == 0
    assert fresh_usage["total"]["attempted_requests"] == 1
