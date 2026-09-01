from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prefscope.cli.parser import build_parser
from prefscope.pipeline.analyze import (
    AnalyzeConfig,
    apply_set_overrides,
    materialize_applied_lens,
    run_analysis,
)
from prefscope.api.loaded_lens import Lens
from prefscope.pipeline.concepts import export_concepts_from_codes
from prefscope.pipeline.encode_dataset import _manifest_digest


def _raw(tmp_path):
    data = tmp_path / "input.csv"
    data.write_text("prompt,response\nq,a\n")
    return {
        "lenses": {
            "repo": "owner/lenses",
            "revision": "v1",
            "completion_subfolder": "completion",
            "prompt_subfolder": "prompt",
        },
        "data": {
            "source": {"type": "local", "path": "input.csv"},
            "columns": {"prompt": "prompt", "response_a": "response"},
            "mode": "single",
        },
        "out_dir": "analysis",
        "viewer": {"enabled": False},
    }


def test_analyze_config_resolves_paths_and_hub_lenses(tmp_path):
    cfg = AnalyzeConfig.from_dict(_raw(tmp_path), base_dir=tmp_path)
    assert cfg.data["source"]["path"] == str(tmp_path / "input.csv")
    assert cfg.out_dir == str(tmp_path / "analysis")
    assert cfg.completion_lens.source == "hf://owner/lenses"
    assert cfg.completion_lens.subfolder == "completion"
    assert cfg.prompt_lens.subfolder == "prompt"
    assert cfg.completion_lens.revision == "v1"
    restored = AnalyzeConfig.from_dict(cfg.to_dict(), base_dir=tmp_path)
    assert restored.to_dict() == cfg.to_dict()


def test_direct_lens_override_can_replace_one_shared_repo_lens(tmp_path):
    raw = _raw(tmp_path)
    raw["lenses"]["completion"] = "local-completion"
    cfg = AnalyzeConfig.from_dict(raw, base_dir=tmp_path)
    assert cfg.completion_lens.source == str(tmp_path / "local-completion")
    assert cfg.completion_lens.subfolder is None
    assert cfg.completion_lens.revision is None
    assert cfg.prompt_lens.source == "hf://owner/lenses"


def test_dotted_overrides_parse_yaml_and_are_strict(tmp_path):
    raw = apply_set_overrides(
        _raw(tmp_path),
        ["data.source.limit=25", "viewer.enabled=false", "concepts.top_k=8"],
    )
    cfg = AnalyzeConfig.from_dict(raw, base_dir=tmp_path)
    assert cfg.data["source"]["limit"] == 25
    assert cfg.viewer["enabled"] is False
    assert cfg.concepts["top_k"] == 8
    bad = apply_set_overrides(raw, ["concepts.topk=3"])
    with pytest.raises(ValueError, match="unknown concepts key.*topk"):
        AnalyzeConfig.from_dict(bad, base_dir=tmp_path)


def test_bad_set_expression_is_rejected():
    with pytest.raises(ValueError, match="expected path.to.key=value"):
        apply_set_overrides({}, ["missing-equals"])


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ('viewer.enabled="yes"', "viewer.enabled must be a boolean"),
        ("concepts.chunk_size=0", "concepts.chunk_size must be a positive integer"),
        ("analysis.min_support=2.5", "analysis.min_support must be a positive integer"),
    ],
)
def test_config_rejects_wrong_option_types(tmp_path, override, message):
    raw = apply_set_overrides(_raw(tmp_path), [override])
    with pytest.raises(ValueError, match=message):
        AnalyzeConfig.from_dict(raw, base_dir=tmp_path)


def test_cli_shortcuts_compose_with_generic_overrides(monkeypatch, tmp_path):
    config = tmp_path / "analysis.yaml"
    raw = _raw(tmp_path)
    raw["lenses"] = {"completion": "old", "prompt": "old-prompt",
                     "completion_subfolder": "completion",
                     "prompt_subfolder": "prompt"}
    config.write_text(json.dumps(raw))
    captured = {}

    class CaptureConfig:
        @classmethod
        def from_dict(cls, value, *, base_dir):
            captured["raw"] = value
            return object()

    monkeypatch.setattr("prefscope.cli.workflow.AnalyzeConfig", CaptureConfig)
    monkeypatch.setattr("prefscope.cli.workflow.run_analysis", lambda *args, **kwargs: {})
    args = build_parser().parse_args([
        "analyze", "--config", str(config),
        "--set", "data.source.split=test",
        "--hf-dataset", "org/new-data",
        "--repo", "org/new-lenses",
    ])

    assert args.func(args) == 0
    assert captured["raw"]["data"]["source"] == {
        "type": "huggingface", "dataset_id": "org/new-data", "split": "test",
    }
    assert captured["raw"]["lenses"]["repo"] == "org/new-lenses"
    assert "completion" not in captured["raw"]["lenses"]
    assert "prompt" not in captured["raw"]["lenses"]


class _FakeLens:
    input_rep = "individual"
    activation_polarity = "nonnegative"

    def __init__(self):
        self.feature_table = pd.DataFrame({
            "feature_id": [0, 1, 2],
            "concept": ["alpha", "beta", "gamma"],
            "fidelity_pass": [True, False, True],
            "semantic_role": ["presentation", "topic_content", "response_policy"],
        })


def test_export_concepts_from_codes_uses_presence_filters_without_reembedding(tmp_path):
    bundle = tmp_path / "codes"
    bundle.mkdir()
    np.save(bundle / "z_a.npy", np.array([
        [2.0, 9.0, 1.0],
        [0.0, 5.0, 3.0],
    ], dtype=np.float32))
    pd.DataFrame({
        "row_id": [10, 11],
        "battle_id": ["x", "y"],
        "prompt": ["p0", "p1"],
        "completion_a": ["a0", "a1"],
    }).to_parquet(bundle / "meta.parquet", index=False)
    out = tmp_path / "concepts.parquet"

    result = export_concepts_from_codes(
        _FakeLens(), bundle, out, presence_policy="positive_nonzero",
        fidelity_only=True, top_k=1, include_text=True, log=lambda *_: None)
    table = pd.read_parquet(out)

    assert result["concept_rows"] == 2
    assert table[["row_id", "feature_id"]].values.tolist() == [[10, 0], [11, 2]]
    assert table["concept"].tolist() == ["alpha", "gamma"]
    assert table["semantic_role"].tolist() == ["presentation", "response_policy"]
    assert table["completion"].tolist() == ["a0", "a1"]


def test_materialize_applied_lens_combines_weights_annotations_and_new_codes(tmp_path):
    source = tmp_path / "source-lens"
    source.mkdir()
    (source / "sae_model.pt").write_bytes(b"weights")
    (source / "feature_names.csv").write_text("feature_id,concept\n0,alpha\n")
    (source / "manifest.json").write_text(json.dumps({
        "schema_version": 2,
        "lens_kind": "individual",
        "input_rep": "individual",
        "m_total": 2,
        "k": 1,
        "input_dim": 3,
        "matryoshka_prefix_lengths": [],
        "output_arrays": [],
        "embed_model_id": "example/embedder",
        "sae_type": "batchtopk",
        "activation_polarity": "signed",
        "code_semantics": "axis",
        "selection_rule": "batchtopk-absolute",
        "artifact_scope": "inference",
    }))
    encoded = tmp_path / "encoded"
    encoded.mkdir()
    np.save(encoded / "z_a.npy", np.ones((2, 2), np.float32))
    meta = pd.DataFrame({"battle_id": ["a", "b"], "prompt": ["p", "q"]})
    meta.to_parquet(encoded / "meta.parquet", index=False)
    meta.to_parquet(encoded / "battles.parquet", index=False)
    (encoded / "manifest.json").write_text(json.dumps({
        "output_arrays": ["z_a"], "n_rows": 2,
        "source_lens_manifest_sha256": _manifest_digest(
            json.loads((source / "manifest.json").read_text())),
    }))
    lens = Lens.__new__(Lens)
    lens.lens_dir = source
    lens.input_rep = "individual"

    out = materialize_applied_lens(lens, encoded, tmp_path / "applied")
    manifest = json.loads((out / "manifest.json").read_text())

    assert (out / "sae_model.pt").read_bytes() == b"weights"
    assert (out / "feature_names.csv").exists()
    assert np.load(out / "z_a.npy").shape == (2, 2)
    assert manifest["artifact_scope"] == "analysis"
    assert manifest["dataset_mode"] == "single"
    assert manifest["array_shapes"] == {"z_a": [2, 2]}


class _WorkflowLens:
    def __init__(self, lens_dir: Path, input_rep: str):
        self.lens_dir = lens_dir
        self.input_rep = input_rep
        self.embedder = object()
        self.feature_table = pd.DataFrame({
            "feature_id": [0], "concept": ["one"], "fidelity_pass": [True],
        })


def test_workflow_resumes_completed_stages(monkeypatch, tmp_path):
    raw = _raw(tmp_path)
    raw["lenses"] = {"completion": "completion-lens"}
    raw["analysis"] = {
        "relationships": False, "comparison": False, "preference": False,
    }
    cfg = AnalyzeConfig.from_dict(raw, base_dir=tmp_path)
    source = Path(cfg.data["source"]["path"])
    completion_dir = tmp_path / "completion-lens"
    completion_dir.mkdir()
    (completion_dir / "manifest.json").write_text("{}")
    lens = _WorkflowLens(completion_dir, "individual")
    calls = {"prepare": 0, "encode": 0, "concepts": 0}

    def fake_prepare(out, **kwargs):
        calls["prepare"] += 1
        pd.DataFrame({
            "prompt": ["q"], "completion_a": ["a"], "item_id": ["0"],
            "source": [str(source)],
        }).to_parquet(out, index=False)

    def fake_encode(lens_dir, data, out, **kwargs):
        calls["encode"] += 1
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "manifest.json").write_text(json.dumps({"output_arrays": ["z_a"]}))
        np.save(Path(out) / "z_a.npy", np.ones((1, 1), np.float32))
        pd.read_parquet(data).to_parquet(Path(out) / "meta.parquet", index=False)

    def fake_concepts(lens, codes_dir, out, **kwargs):
        calls["concepts"] += 1
        pd.DataFrame({"feature_id": [0], "concept": ["one"]}).to_parquet(out)

    monkeypatch.setattr("prefscope.pipeline.analyze.prepare_dataset", fake_prepare)
    monkeypatch.setattr("prefscope.pipeline.analyze._load_lens", lambda *_: lens)
    monkeypatch.setattr("prefscope.pipeline.analyze.run_encode_dataset", fake_encode)
    monkeypatch.setattr(
        "prefscope.pipeline.analyze.export_concepts_from_codes", fake_concepts)

    first = run_analysis(cfg, log=lambda *_: None)
    second = run_analysis(cfg, log=lambda *_: None)

    assert first == second
    assert calls == {"prepare": 1, "encode": 1, "concepts": 1}
    assert Path(first["response_concepts"]).exists()

    source.write_text("prompt,response\nchanged,new response\n")
    with pytest.raises(ValueError, match="different settings"):
        run_analysis(cfg, log=lambda *_: None)

    third = run_analysis(cfg, fresh=True, log=lambda *_: None)
    assert Path(third["response_concepts"]).exists()
    assert calls == {"prepare": 2, "encode": 2, "concepts": 2}


def test_fresh_refuses_to_delete_an_unrecognized_directory(tmp_path):
    raw = _raw(tmp_path)
    raw["out_dir"] = "not-an-analysis-directory"
    out = tmp_path / raw["out_dir"]
    out.mkdir()
    sentinel = out / "keep.txt"
    sentinel.write_text("user data")
    cfg = AnalyzeConfig.from_dict(raw, base_dir=tmp_path)

    with pytest.raises(ValueError, match="unrecognized non-empty directory"):
        run_analysis(cfg, fresh=True, log=lambda *_: None)

    assert sentinel.read_text() == "user data"


def test_input_fingerprints_bind_resolved_hub_revisions(monkeypatch, tmp_path):
    from prefscope.pipeline.analyze import _input_fingerprints

    raw = _raw(tmp_path)
    cfg = AnalyzeConfig.from_dict(raw, base_dir=tmp_path)
    calls = []

    def fake_resolve(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return "a" * 40

    monkeypatch.setattr("prefscope.api.hub.resolve_hf_revision", fake_resolve)
    fingerprints = _input_fingerprints(cfg)

    assert fingerprints["completion_lens"]["requested_revision"] == "v1"
    assert fingerprints["completion_lens"]["resolved_revision"] == "a" * 40
    assert fingerprints["prompt_lens"]["resolved_revision"] == "a" * 40
    # Both subfolders share one repository/ref resolution.
    assert len(calls) == 1


def test_workflow_runs_configured_continuous_outcome_analysis(monkeypatch, tmp_path):
    raw = _raw(tmp_path)
    raw["lenses"] = {"completion": "completion-lens"}
    raw["data"]["columns"]["metadata"] = ["rating"]
    raw["analysis"] = {
        "relationships": False, "comparison": False, "preference": False,
        "group_col": "prompt",
        "outcomes": {
            "columns": ["rating"], "kind": "continuous", "min_units": 3,
        },
    }
    cfg = AnalyzeConfig.from_dict(raw, base_dir=tmp_path)
    completion_dir = tmp_path / "completion-lens"
    completion_dir.mkdir()
    (completion_dir / "manifest.json").write_text("{}")
    lens = _WorkflowLens(completion_dir, "individual")

    def fake_prepare(out, **kwargs):
        pd.DataFrame({
            "prompt": ["repeat", "repeat", "other", "third"],
            "completion_a": ["a", "b", "c", "d"],
            "item_id": ["0", "1", "2", "3"],
            "rating": [1.0, 3.0, 2.0, 4.0],
        }).to_parquet(out, index=False)

    def fake_encode(lens_dir, data, out, **kwargs):
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "manifest.json").write_text(json.dumps({"output_arrays": ["z_a"]}))
        np.save(Path(out) / "z_a.npy", np.array([[0.0], [2.0], [1.0], [3.0]], np.float32))
        pd.read_parquet(data).to_parquet(Path(out) / "meta.parquet", index=False)

    def fake_concepts(lens, codes_dir, out, **kwargs):
        pd.DataFrame({"feature_id": [0], "concept": ["one"]}).to_parquet(out)

    monkeypatch.setattr("prefscope.pipeline.analyze.prepare_dataset", fake_prepare)
    monkeypatch.setattr("prefscope.pipeline.analyze._load_lens", lambda *_: lens)
    monkeypatch.setattr("prefscope.pipeline.analyze.run_encode_dataset", fake_encode)
    monkeypatch.setattr(
        "prefscope.pipeline.analyze.export_concepts_from_codes", fake_concepts)

    outputs = run_analysis(cfg, log=lambda *_: None)
    table = pd.read_csv(outputs["outcomes"])
    assert set(table["analysis_unit"]) == {"group"}
    assert set(table["n_units"]) == {3}
    assert (tmp_path / "analysis/outcome_associations_outcomes.json").exists()


def test_outcome_config_requires_source_metadata_retention(tmp_path):
    raw = _raw(tmp_path)
    raw["analysis"] = {
        "outcomes": {"columns": ["rating"], "kind": "continuous"},
    }
    with pytest.raises(ValueError, match="data.columns.metadata"):
        AnalyzeConfig.from_dict(raw, base_dir=tmp_path)


def test_outcome_config_rejects_managed_output_collision(tmp_path):
    raw = _raw(tmp_path)
    raw["data"]["columns"]["metadata"] = ["rating"]
    raw["analysis"] = {
        "outcomes": {
            "columns": ["rating"], "kind": "continuous",
            "output": "win_relevance.csv",
        },
    }
    with pytest.raises(ValueError, match="unreserved CSV filename"):
        AnalyzeConfig.from_dict(raw, base_dir=tmp_path)


def _materialize_sources(tmp_path):
    source = tmp_path / "materialize-source"
    source.mkdir()
    (source / "sae_model.pt").write_bytes(b"weights")
    (source / "feature_names.csv").write_text("feature_id,concept\n0,alpha\n")
    (source / "manifest.json").write_text(json.dumps({
        "schema_version": 2,
        "lens_kind": "individual",
        "input_rep": "individual",
        "m_total": 2,
        "k": 1,
        "input_dim": 3,
        "matryoshka_prefix_lengths": [],
        "output_arrays": [],
        "embed_model_id": "example/embedder",
        "sae_type": "batchtopk",
        "activation_polarity": "signed",
        "code_semantics": "axis",
        "selection_rule": "batchtopk-absolute",
        "dataset_hash": "a" * 64,
        "artifact_scope": "inference",
    }))
    encoded = tmp_path / "materialize-encoded"
    encoded.mkdir()
    np.save(encoded / "z_a.npy", np.ones((2, 2), np.float32))
    meta = pd.DataFrame({"battle_id": ["a", "b"], "prompt": ["p", "q"]})
    meta.to_parquet(encoded / "meta.parquet", index=False)
    meta.to_parquet(encoded / "battles.parquet", index=False)
    (encoded / "manifest.json").write_text(json.dumps({
        "output_arrays": ["z_a"], "n_rows": 2, "dataset_hash": "b" * 64,
        "source_lens_manifest_sha256": _manifest_digest(
            json.loads((source / "manifest.json").read_text())),
    }))
    lens = Lens.__new__(Lens)
    lens.lens_dir = source
    lens.input_rep = "individual"
    return lens, source, encoded


def test_materialize_rejects_unbound_encoded_manifest(tmp_path):
    lens, _, encoded = _materialize_sources(tmp_path)
    manifest_path = encoded / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("source_lens_manifest_sha256")
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="source_lens_manifest_sha256"):
        materialize_applied_lens(lens, encoded, tmp_path / "applied")


def test_materialize_rejects_source_output_overlap(tmp_path):
    lens, source, encoded = _materialize_sources(tmp_path)

    for output in (source, source / "nested", encoded, tmp_path):
        with pytest.raises(ValueError, match="overlaps"):
            materialize_applied_lens(lens, encoded, output)


def test_materialize_replaces_whole_destination_without_stale_files(tmp_path):
    lens, _, encoded = _materialize_sources(tmp_path)
    out = tmp_path / "applied"
    out.mkdir()
    (out / "stale.txt").write_text("old")
    np.save(out / "z_diff.npy", np.ones((1, 1), np.float32))

    materialize_applied_lens(lens, encoded, out, overwrite=True)

    assert not (out / "stale.txt").exists()
    assert not (out / "z_diff.npy").exists()
    assert (out / "z_a.npy").exists()
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["output_arrays"] == ["z_a"]
    assert manifest["dataset_hash"] == "b" * 64
    assert manifest["source_lens_dataset_hash"] == "a" * 64
    assert len(manifest["source_lens_manifest_sha256"]) == 64
    assert "applied_from" not in manifest
    assert str(tmp_path) not in json.dumps(manifest)


def test_materialize_validation_failure_preserves_destination(tmp_path):
    lens, _, encoded = _materialize_sources(tmp_path)
    np.save(encoded / "z_a.npy", np.ones((1, 2), np.float32))
    out = tmp_path / "applied"
    out.mkdir()
    (out / "keep.txt").write_text("old")

    with pytest.raises(ValueError, match="must have shape"):
        materialize_applied_lens(lens, encoded, out, overwrite=True)

    assert {path.name for path in out.iterdir()} == {"keep.txt"}
    assert (out / "keep.txt").read_text() == "old"
    assert not list(tmp_path.glob(".applied.tmp-*"))
    assert not list(tmp_path.glob(".applied.bak-*"))


def test_materialize_overwrite_false_refuses_nonempty_destination(tmp_path):
    lens, _, encoded = _materialize_sources(tmp_path)
    out = tmp_path / "applied"
    out.mkdir()
    (out / "keep.txt").write_text("old")

    with pytest.raises(FileExistsError, match="overwrite=True"):
        materialize_applied_lens(lens, encoded, out, overwrite=False)

    assert {path.name for path in out.iterdir()} == {"keep.txt"}
