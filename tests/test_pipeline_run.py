"""Phase 4: the config-driven pipeline runner (`prefscope run --config ...`)."""
import json

import numpy as np
import pandas as pd
import pytest

from prefscope.pipeline.run import (
    LLMConfig, PipelineConfig, StageConfig, run_pipeline)


# --- config parsing / validation (pure, no IO) -----------------------------------------

def test_from_dict_minimal_applies_defaults():
    cfg = PipelineConfig.from_dict({"lens_dir": "L", "out_dir": "O"})
    assert cfg.stages == ["name", "verify", "cluster", "win-relevance"]
    assert cfg.lens_kind == "completion"
    assert cfg.interpreter.component == "auto" and cfg.verifier.component == "auto"
    assert cfg.clusterer.component == "spherical-kmeans"   # the runner's clusterer default


def test_unknown_top_level_key_raises():
    with pytest.raises(ValueError, match="unknown config keys: lensdir"):
        PipelineConfig.from_dict({"lensdir": "L", "out_dir": "O"})


def test_missing_required_key_raises():
    with pytest.raises(ValueError, match="lens_dir"):
        PipelineConfig.from_dict({"out_dir": "O"})


def test_unknown_stage_raises():
    with pytest.raises(ValueError, match="unsupported stage"):
        PipelineConfig.from_dict({"lens_dir": "L", "out_dir": "O", "stages": ["name", "bogus"]})


def test_invalid_lens_kind_rejected():
    with pytest.raises(ValueError, match="lens_kind must be"):
        PipelineConfig.from_dict({"lens_dir": "L", "out_dir": "O", "lens_kind": "tokens"})


def test_prompt_lens_kind_accepted_with_default_stages():
    cfg = PipelineConfig.from_dict({"lens_dir": "L", "out_dir": "O", "lens_kind": "prompt"})
    assert cfg.lens_kind == "prompt"
    assert cfg.stages == ["name", "verify", "cluster"]   # win-relevance excluded for prompt


def test_win_relevance_rejected_for_prompt_lens():
    with pytest.raises(ValueError, match="win-relevance is completion-only"):
        PipelineConfig.from_dict({"lens_dir": "L", "out_dir": "O", "lens_kind": "prompt",
                                  "stages": ["name", "win-relevance"]})


def test_annotations_scalar_is_listified():
    cfg = PipelineConfig.from_dict({"lens_dir": "L", "out_dir": "O", "annotations": "a.json"})
    assert cfg.annotations == ["a.json"]


def test_stage_config_parse_forms():
    assert StageConfig.parse(None, default="x").component == "x"
    assert StageConfig.parse("pairwise") == StageConfig("pairwise", {})
    sc = StageConfig.parse({"name": "individual", "n_active": 20})
    assert sc.component == "individual" and sc.params == {"n_active": 20}
    sc2 = StageConfig.parse({"resolution": 1.5})            # no name -> default + params
    assert sc2.component == "auto" and sc2.params == {"resolution": 1.5}
    with pytest.raises(ValueError, match="name or a mapping"):
        StageConfig.parse(5)


def test_llm_config_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown llm keys: temperature"):
        LLMConfig.parse({"model": "m", "temperature": 0.5})


def test_stage_specific_llm_configs_are_parsed():
    cfg = PipelineConfig.from_dict({
        "lens_dir": "L", "out_dir": "O",
        "llm": {"model": "shared"},
        "name_llm": {"model": "namer"},
        "verify_llm": {"model": "verifier"}})
    assert cfg.llm.model == "shared"
    assert cfg.name_llm.model == "namer"
    assert cfg.verify_llm.model == "verifier"
    assert cfg.cluster_llm is None


def _cfg(**kw):
    base = {"lens_dir": "L", "out_dir": "O"}
    base.update(kw)
    return base


def test_misspelled_clusterer_param_rejected():
    with pytest.raises(ValueError, match="unknown clusterer param.*n_clustres"):
        PipelineConfig.from_dict(_cfg(clusterer={"name": "spherical-kmeans", "n_clustres": 5}))


def test_wrong_component_param_rejected():
    # n_clusters is meaningless for mi-leiden (only the k-means clusterers take it)
    with pytest.raises(ValueError, match="unknown clusterer param.*n_clusters"):
        PipelineConfig.from_dict(_cfg(clusterer={"name": "mi-leiden", "n_clusters": 5}))


def test_cluster_control_keys_allowed():
    cfg = PipelineConfig.from_dict(_cfg(
        clusterer={"name": "mi-leiden", "resolution": 1.5,
                   "cluster_on": "individual", "fidelity_only": True, "name_clusters": True}))
    assert cfg.clusterer.params["cluster_on"] == "individual"


def test_misspelled_interpreter_param_rejected():
    with pytest.raises(ValueError, match="unknown interpreter param.*n_activ"):
        PipelineConfig.from_dict(_cfg(interpreter={"name": "pairwise", "n_activ": 5}))


def test_misspelled_win_relevance_key_rejected():
    with pytest.raises(ValueError, match="unknown win_relevance key.*all_feature"):
        PipelineConfig.from_dict(_cfg(win_relevance={"all_feature": True}))


def test_unknown_component_name_is_deferred_to_make():
    # a bad component NAME isn't rejected by param validation (we can't know its params) —
    # registry.make raises the friendly "available: ..." error at run time instead
    cfg = PipelineConfig.from_dict(_cfg(clusterer={"name": "no-such-clusterer"}))
    assert cfg.clusterer.component == "no-such-clusterer"


def test_load_roundtrips_yaml_and_json(tmp_path):
    body = {"lens_dir": "L", "out_dir": "O", "stages": ["name"],
            "clusterer": {"name": "mi-leiden", "resolution": 1.2}}
    (tmp_path / "p.json").write_text(json.dumps(body))
    (tmp_path / "p.yaml").write_text(
        "lens_dir: L\nout_dir: O\nstages: [name]\n"
        "clusterer: {name: mi-leiden, resolution: 1.2}\n")
    for fn in ("p.json", "p.yaml"):
        cfg = PipelineConfig.load(tmp_path / fn)
        assert cfg.stages == ["name"]
        assert cfg.clusterer.component == "mi-leiden"
        assert cfg.clusterer.params == {"resolution": 1.2}


# --- end-to-end orchestration on a tiny on-disk lens -----------------------------------

class FakeClient:
    def raw(self, messages, **kw):
        return '- "uses code blocks"'


def _make_lens(tmp_path, *, n=12, m=4, with_human_pref=False):
    """A minimal difference lens dir + matching corpus parquet."""
    rng = np.random.default_rng(0)
    z = rng.standard_normal((n, m)).astype(np.float32)
    z[np.abs(z) < 0.3] = 0.0
    lens = tmp_path / "lens"
    lens.mkdir()
    (lens / "manifest.json").write_text(json.dumps({"input_rep": "difference"}))
    ids = [str(i) for i in range(n)]
    pd.DataFrame({"instruction_id": ids}).to_parquet(lens / "battles.parquet")
    np.save(lens / "z_diff.npy", z)

    corpus = pd.DataFrame({
        "battle_id": ids, "source": "t", "language": "en",
        "prompt": [f"p{i}" for i in range(n)],
        "model_a": "A", "model_b": "B",
        "completion_a": [f"a{i}" for i in range(n)],
        "completion_b": [f"b{i}" for i in range(n)],
    })
    if with_human_pref:
        corpus["human_pref"] = [i % 2 for i in range(n)]
    cpath = tmp_path / "corpus.parquet"
    corpus.to_parquet(cpath)
    return lens, cpath


def _make_single_lens(tmp_path, *, n=30, m=4):
    rng = np.random.default_rng(1)
    z = rng.standard_normal((n, m)).astype(np.float32)
    z[np.abs(z) < 0.4] = 0.0
    lens = tmp_path / "single_lens"
    lens.mkdir()
    (lens / "manifest.json").write_text(json.dumps({
        "input_rep": "individual", "dataset_mode": "single", "output_arrays": ["z_a"]}))
    pd.DataFrame({
        "instruction_id": [str(i) for i in range(n)],
        "prompt": [f"p{i}" for i in range(n)],
        "completion_a": [f"a{i}" for i in range(n)],
        "model_a": ["M"] * n,
    }).to_parquet(lens / "battles.parquet")
    np.save(lens / "z_a.npy", z)
    return lens


def test_run_name_then_cluster_threads_outputs(tmp_path):
    lens, corpus = _make_lens(tmp_path)
    out = tmp_path / "out"
    cfg = PipelineConfig.from_dict({
        "lens_dir": str(lens), "out_dir": str(out), "corpus": str(corpus),
        "stages": ["name", "cluster"],
        "clusterer": {"name": "spherical-kmeans", "n_clusters": 2}})
    outputs = run_pipeline(cfg, client=FakeClient(), verbose=False)

    assert set(outputs) == {"name", "cluster"}
    names = pd.read_csv(out / "feature_names.csv")
    assert (names["concept"] == "uses code blocks").all()
    clusters = pd.read_csv(out / "feature_clusters.csv")
    # cluster stage read the names it was threaded -> concept column carried through
    assert "concept" in clusters.columns
    assert clusters["cluster_id"].nunique() == 2           # n_clusters param flowed to make
    assert (out / "feature_clusters_summary.csv").exists()


def test_single_response_lens_names_and_clusters_without_external_corpus(tmp_path):
    lens = _make_single_lens(tmp_path)
    out = tmp_path / "out"
    cfg = PipelineConfig.from_dict({
        "lens_dir": str(lens), "out_dir": str(out),
        "stages": ["name", "cluster"],
        "interpreter": {"name": "auto", "n_active": 3, "n_zero": 3},
        "clusterer": {"name": "spherical-kmeans", "n_clusters": 2}})
    outputs = run_pipeline(cfg, client=FakeClient(), verbose=False)
    assert set(outputs) == {"name", "cluster"}
    assert (out / "feature_names.csv").exists()
    assert (out / "feature_clusters.csv").exists()


def test_single_response_lens_rejects_pairwise_win_relevance(tmp_path):
    lens = _make_single_lens(tmp_path)
    cfg = PipelineConfig.from_dict({
        "lens_dir": str(lens), "out_dir": str(tmp_path / "out"),
        "stages": ["win-relevance"]})
    with pytest.raises(ValueError, match="pairwise-only"):
        run_pipeline(cfg, client=FakeClient(), verbose=False)


def test_clusterer_swap_changes_partition(tmp_path):
    lens, corpus = _make_lens(tmp_path)
    out = tmp_path / "out"
    cfg = PipelineConfig.from_dict({
        "lens_dir": str(lens), "out_dir": str(out), "corpus": str(corpus),
        "stages": ["name", "cluster"],
        "clusterer": {"name": "spherical-kmeans", "n_clusters": 3}})
    run_pipeline(cfg, client=FakeClient(), verbose=False)
    assert pd.read_csv(out / "feature_clusters.csv")["cluster_id"].nunique() == 3


def test_verify_threads_names_and_forwards_opts(tmp_path, monkeypatch):
    lens, corpus = _make_lens(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame({"feature_id": [0, 1], "concept": ["x", "y"]}).to_csv(
        out / "feature_names.csv", index=False)

    seen = {}

    def spy(battles, z_diff, names, client, **kw):
        seen.update(kw)
        seen["n_names"] = len(names)
        return pd.DataFrame({"feature_id": [0, 1], "fidelity_pass": [True, False]})

    monkeypatch.setattr("prefscope.interpret.verify.verify_features", spy)
    cfg = PipelineConfig.from_dict({
        "lens_dir": str(lens), "out_dir": str(out), "corpus": str(corpus),
        "stages": ["verify"], "verifier": {"name": "pairwise", "n_per_bucket": 7}})
    outputs = run_pipeline(cfg, client=FakeClient(), verbose=False)

    assert seen["n_names"] == 2 and seen["n_per_bucket"] == 7
    fid = pd.read_csv(outputs["verify"])
    assert int(fid["fidelity_pass"].sum()) == 1


def test_verify_without_names_errors(tmp_path):
    lens, corpus = _make_lens(tmp_path)
    cfg = PipelineConfig.from_dict({
        "lens_dir": str(lens), "out_dir": str(tmp_path / "out"), "corpus": str(corpus),
        "stages": ["verify"]})
    with pytest.raises(FileNotFoundError, match="feature names"):
        run_pipeline(cfg, client=FakeClient(), verbose=False)


def test_win_relevance_stage_runs(tmp_path):
    lens, corpus = _make_lens(tmp_path, with_human_pref=True)
    out = tmp_path / "out"
    cfg = PipelineConfig.from_dict({
        "lens_dir": str(lens), "out_dir": str(out), "corpus": str(corpus),
        "stages": ["win-relevance"], "win_relevance": {"all_features": True}})
    outputs = run_pipeline(cfg, client=FakeClient(), verbose=False)
    wr = pd.read_csv(outputs["win-relevance"])
    assert {"feature_id", "win_assoc", "significant"} <= set(wr.columns)


def test_win_relevance_without_corpus_errors(tmp_path):
    lens, _ = _make_lens(tmp_path)
    cfg = PipelineConfig.from_dict({
        "lens_dir": str(lens), "out_dir": str(tmp_path / "out"),
        "stages": ["win-relevance"]})
    with pytest.raises(ValueError, match="corpus"):
        run_pipeline(cfg, client=FakeClient(), verbose=False)


def test_verify_then_cluster_threads_fidelity_and_filters(tmp_path, monkeypatch):
    lens, corpus = _make_lens(tmp_path, m=4)
    out = tmp_path / "out"

    def spy(battles, z_diff, names, client, **kw):       # 2 of 4 pass fidelity
        return pd.DataFrame({"feature_id": [0, 1, 2, 3],
                             "concept": ["a", "b", "c", "d"],
                             "fidelity_pass": [True, True, False, False]})

    monkeypatch.setattr("prefscope.interpret.verify.verify_features", spy)
    cfg = PipelineConfig.from_dict({
        "lens_dir": str(lens), "out_dir": str(out), "corpus": str(corpus),
        "stages": ["name", "verify", "cluster"], "verifier": {"name": "pairwise"},
        "clusterer": {"name": "spherical-kmeans", "n_clusters": 2, "fidelity_only": True}})
    run_pipeline(cfg, client=FakeClient(), verbose=False)

    clusters = pd.read_csv(out / "feature_clusters.csv")
    # cluster read the fidelity csv (not the raw names) and kept only passing features
    assert set(clusters["feature_id"]) == {0, 1}


def test_no_verified_features_writes_empty_cluster_and_reward_tables(tmp_path):
    lens, corpus = _make_lens(tmp_path, m=4, with_human_pref=True)
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame({
        "feature_id": [0, 1, 2, 3], "concept": ["a", "b", "c", "d"],
        "fidelity_pass": [False, False, False, False],
    }).to_csv(out / "feature_fidelity.csv", index=False)
    cfg = PipelineConfig.from_dict({
        "lens_dir": str(lens), "out_dir": str(out), "corpus": str(corpus),
        "stages": ["cluster", "win-relevance"],
        "clusterer": {"name": "spherical-kmeans", "n_clusters": 2,
                      "fidelity_only": True}})
    outputs = run_pipeline(cfg, client=FakeClient(), verbose=False)
    assert pd.read_csv(outputs["cluster"]).empty
    assert pd.read_csv(outputs["win-relevance"]).empty


def test_stages_run_in_canonical_order_regardless_of_listing(tmp_path):
    lens, corpus = _make_lens(tmp_path)
    out = tmp_path / "out"
    cfg = PipelineConfig.from_dict({
        "lens_dir": str(lens), "out_dir": str(out), "corpus": str(corpus),
        "stages": ["cluster", "name"],                   # listed out of order
        "clusterer": {"name": "spherical-kmeans", "n_clusters": 2}})
    run_pipeline(cfg, client=FakeClient(), verbose=False)
    # name ran before cluster -> concepts were available to thread into the cluster output
    assert "concept" in pd.read_csv(out / "feature_clusters.csv").columns


def test_unknown_component_raises_friendly_valueerror_at_run(tmp_path):
    lens, corpus = _make_lens(tmp_path)
    cfg = PipelineConfig.from_dict({
        "lens_dir": str(lens), "out_dir": str(tmp_path / "out"), "corpus": str(corpus),
        "stages": ["cluster"], "clusterer": {"name": "no-such-clusterer"}})
    with pytest.raises(ValueError, match="no clusterer named 'no-such-clusterer'"):
        run_pipeline(cfg, client=FakeClient(), verbose=False)


class FakePromptClient:
    def raw(self, messages, **kw):
        return '{"concept": "asks about finance"}'


def _make_prompt_lens(tmp_path, *, n=12, m=4):
    """A minimal prompt lens dir (z_prompt + manifest) + matching corpus parquet."""
    rng = np.random.default_rng(1)
    z = rng.standard_normal((n, m)).astype(np.float32)
    z[np.abs(z) < 0.3] = 0.0
    lens = tmp_path / "plens"
    lens.mkdir()
    (lens / "manifest.json").write_text(json.dumps({"input_rep": "prompt"}))
    ids = [str(i) for i in range(n)]
    pd.DataFrame({"instruction_id": ids}).to_parquet(lens / "battles.parquet")
    np.save(lens / "z_prompt.npy", z)
    corpus = pd.DataFrame({
        "battle_id": ids, "source": "t", "language": "en",
        "prompt": [f"finance question {i}" for i in range(n)],
        "model_a": "A", "model_b": "B",
        "completion_a": [f"a{i}" for i in range(n)],
        "completion_b": [f"b{i}" for i in range(n)]})
    cpath = tmp_path / "corpus.parquet"
    corpus.to_parquet(cpath)
    return lens, cpath


def test_prompt_lens_name_and_cluster_uses_prompt_artifacts(tmp_path):
    lens, corpus = _make_prompt_lens(tmp_path)
    out = tmp_path / "out"
    cfg = PipelineConfig.from_dict({
        "lens_dir": str(lens), "out_dir": str(out), "corpus": str(corpus),
        "lens_kind": "prompt", "stages": ["name", "cluster"],
        "clusterer": {"name": "spherical-kmeans", "n_clusters": 2}})
    outputs = run_pipeline(cfg, client=FakePromptClient(), verbose=False)

    # prompt lenses route the single-text namer and write the prompt_* filenames
    assert outputs["name"].name == "prompt_feature_names.csv"
    assert (out / "prompt_feature_clusters.csv").exists()
    names = pd.read_csv(out / "prompt_feature_names.csv")
    assert (names["concept"] == "asks about finance").all()
    assert pd.read_csv(out / "prompt_feature_clusters.csv")["cluster_id"].nunique() == 2


def test_prompt_lens_verify_routes_prompt_strategy(tmp_path, monkeypatch):
    lens, corpus = _make_prompt_lens(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame({"feature_id": [0, 1], "concept": ["x", "y"]}).to_csv(
        out / "prompt_feature_names.csv", index=False)

    seen = {}

    def spy(texts, z, names, client, **kw):
        seen["n_names"] = len(names)
        return pd.DataFrame({"feature_id": [0, 1], "fidelity_pass": [True, False]})

    monkeypatch.setattr("prefscope.interpret.verify.verify_single_text_features", spy)
    cfg = PipelineConfig.from_dict({
        "lens_dir": str(lens), "out_dir": str(out), "corpus": str(corpus),
        "lens_kind": "prompt", "stages": ["verify"]})   # auto verifier -> prompt strategy
    outputs = run_pipeline(cfg, client=FakePromptClient(), verbose=False)

    assert seen["n_names"] == 2                          # threaded the prompt names csv
    assert outputs["verify"].name == "prompt_feature_fidelity.csv"


def test_prompt_cluster_only_needs_no_corpus(tmp_path):
    # cluster reads z_prompt off disk; preflight must not demand a corpus for it
    lens, _ = _make_prompt_lens(tmp_path)
    cfg = PipelineConfig.from_dict({
        "lens_dir": str(lens), "out_dir": str(tmp_path / "out"),
        "lens_kind": "prompt", "stages": ["cluster"],
        "clusterer": {"name": "spherical-kmeans", "n_clusters": 2}})
    outputs = run_pipeline(cfg, client=FakePromptClient(), verbose=False)
    assert outputs["cluster"].name == "prompt_feature_clusters.csv"


def test_prompt_name_without_corpus_errors(tmp_path):
    lens, _ = _make_prompt_lens(tmp_path)
    cfg = PipelineConfig.from_dict({
        "lens_dir": str(lens), "out_dir": str(tmp_path / "out"),
        "lens_kind": "prompt", "stages": ["name"]})
    with pytest.raises(ValueError, match="corpus"):
        run_pipeline(cfg, client=FakePromptClient(), verbose=False)


def test_preflight_missing_lens(tmp_path):
    cfg = PipelineConfig.from_dict({
        "lens_dir": str(tmp_path / "nope"), "out_dir": str(tmp_path / "out"),
        "stages": ["cluster"]})
    with pytest.raises(FileNotFoundError, match="no lens"):
        run_pipeline(cfg, client=FakeClient(), verbose=False)


def test_preflight_requires_exactly_one_text_source(tmp_path):
    lens, corpus = _make_lens(tmp_path)
    # name needs a battle source, and giving BOTH is ambiguous
    cfg = PipelineConfig.from_dict({
        "lens_dir": str(lens), "out_dir": str(tmp_path / "out"), "stages": ["name"],
        "corpus": str(corpus), "annotations": "a.json"})
    with pytest.raises(ValueError, match="exactly one"):
        run_pipeline(cfg, client=FakeClient(), verbose=False)


def test_cluster_only_needs_no_text_source(tmp_path):
    # cluster reads codes off disk; it must run without a corpus/annotations
    lens, _ = _make_lens(tmp_path)
    cfg = PipelineConfig.from_dict({
        "lens_dir": str(lens), "out_dir": str(tmp_path / "out"), "stages": ["cluster"],
        "clusterer": {"name": "spherical-kmeans", "n_clusters": 2}})
    outputs = run_pipeline(cfg, client=FakeClient(), verbose=False)
    assert outputs["cluster"].exists()
