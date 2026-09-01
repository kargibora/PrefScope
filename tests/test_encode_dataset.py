"""Unit tests for `prefscope encode-dataset` (Phase 0).

The real path embeds with the 8B model on GPU; here we monkeypatch the embedder and the
SAEProjector so the orchestration, rep branch, provenance, and error handling are tested
without any model — mirroring how test_export_diagnosis stubs load_bank.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prefscope.pipeline import encode_dataset as ed
from prefscope.pipeline.encode_dataset import codes_from_embeddings, run_encode_dataset
from prefscope.pipeline.lens_rep import get_lens_rep

D, M = 8, 4  # embedding dim / lens m_total for the fakes


class FakeProj:
    """Duck-typed SAEProjector: first M dims are the code; enforces the dim contract."""
    def __init__(self, model_path, device="cpu"):
        self.input_dim = D
        self.m_total = M

    def project(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"embedding dim {x.shape} != input_dim {self.input_dim}")
        return x[:, :M].astype(np.float32)


class FakeEmbedder:
    """Deterministic, content-dependent embeddings (distinct strings -> distinct vectors)."""
    width = D
    model_id = "fake/embedder-1B"   # matches the fake lens manifest below

    def provenance(self, *, prompt=False):
        return {
            "embed_model_id": self.model_id,
            "embed_model_revision": "fake-rev",
            "max_tokens": 128,
            "embed_instruction": (
                "fake prompt instruction" if prompt else "fake completion instruction"),
            "pooling": "last-token",
            "normalization": "l2",
            "dtype": "float32",
            "backend": "fake",
        }

    def encode(self, prompts, completions):
        return np.array([[float(sum(ord(ch) for ch in c) + 1)] * self.width
                         for c in completions], dtype=np.float32)

    def encode_prompts(self, prompts):
        return np.array([[float(sum(ord(ch) for ch in p) + 1)] * self.width
                         for p in prompts], dtype=np.float32)


class WrongDimEmbedder(FakeEmbedder):
    width = D - 2  # deliberately mismatched to trip the projector's dim check


def _make_lens(tmp_path, input_rep="individual") -> Path:
    lens = tmp_path / "lens"
    lens.mkdir()
    (lens / "manifest.json").write_text(json.dumps({
        "input_rep": input_rep,
        "embed_model_id": "fake/embedder-1B",
        "embed_model_revision": "fake-rev",
        "max_tokens": 128,
        "embed_instruction": (
            "fake prompt instruction" if input_rep == "prompt"
            else "fake completion instruction"),
        "pooling": "last-token",
        "normalization": "l2",
        "dtype": "float32",
        "backend": "fake",
        "input_dim": D, "m_total": M, "k": 1, "whiten": "none"}))
    return lens


def _write(tmp_path, df, name="data.csv") -> Path:
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


@pytest.fixture(autouse=True)
def _patch_projector(monkeypatch):
    monkeypatch.setattr(ed, "SAEProjector", FakeProj)


def test_battle_individual_writes_all_arrays(tmp_path):
    lens = _make_lens(tmp_path)
    data = _write(tmp_path, pd.DataFrame({
        "prompt": ["p1", "p2"], "response": ["alpha", "bb"], "response_y": ["c", "delta"]}))
    out = tmp_path / "out"
    mf = run_encode_dataset(lens, data, out, embedder=FakeEmbedder(), response2_col="response_y")

    z_a = np.load(out / "z_a.npy")
    z_b = np.load(out / "z_b.npy")
    z_diff = np.load(out / "z_diff.npy")
    assert z_a.shape == (2, M)
    np.testing.assert_allclose(z_diff, z_a - z_b)   # the defining battle invariant
    assert set(mf["output_arrays"]) == {"z_a", "z_b", "z_diff"}
    assert mf["mode"] == "battle" and mf["n_rows"] == 2 and mf["n_dropped"] == 0
    assert len(pd.read_parquet(out / "meta.parquet")) == 2


def test_legacy_manifest_representation_is_inferred_from_declared_arrays(tmp_path):
    lens = _make_lens(tmp_path)
    manifest_path = lens / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("input_rep")
    manifest["output_arrays"] = ["z_a", "z_b", "z_diff"]
    manifest_path.write_text(json.dumps(manifest))
    data = _write(tmp_path, pd.DataFrame({
        "prompt": ["p"], "response": ["alpha"], "response_y": ["beta"],
    }))

    output = run_encode_dataset(
        lens, data, tmp_path / "out", embedder=FakeEmbedder(),
        response2_col="response_y",
    )

    assert output["lens_input_rep"] == "individual"
    assert set(output["output_arrays"]) == {"z_a", "z_b", "z_diff"}


def test_absolute_individual_writes_only_z_a(tmp_path):
    lens = _make_lens(tmp_path)
    data = _write(tmp_path, pd.DataFrame({"prompt": ["p1", "p2"], "response": ["aaa", "bb"]}))
    out = tmp_path / "out"
    mf = run_encode_dataset(lens, data, out, embedder=FakeEmbedder())

    assert (out / "z_a.npy").exists()
    assert not (out / "z_b.npy").exists() and not (out / "z_diff.npy").exists()
    assert mf["output_arrays"] == ["z_a"] and mf["mode"] == "absolute"


def test_prompt_lens_needs_only_prompt_and_writes_aligned_z_prompt(tmp_path):
    lens = _make_lens(tmp_path, input_rep="prompt")
    data = _write(
        tmp_path,
        pd.DataFrame({"prompt": ["p1", "p2"], "item_id": ["x", "y"]}),
    )
    out = tmp_path / "prompt-out"
    mf = run_encode_dataset(lens, data, out, embedder=FakeEmbedder())

    assert np.load(out / "z_prompt.npy").shape == (2, M)
    assert mf["output_arrays"] == ["z_prompt"] and mf["mode"] == "prompt"
    meta = pd.read_parquet(out / "meta.parquet")
    assert list(meta["battle_id"]) == ["x", "y"]
    assert (out / "battles.parquet").exists()


def test_difference_lens_refuses_absolute(tmp_path):
    lens = _make_lens(tmp_path, input_rep="difference")
    data = _write(tmp_path, pd.DataFrame({"prompt": ["p"], "response": ["r"]}))
    with pytest.raises(ValueError, match="individual|battle|A/B pair"):
        run_encode_dataset(lens, data, tmp_path / "out", embedder=FakeEmbedder())


def test_embedder_model_mismatch_refused_before_embedding(tmp_path):
    lens = _make_lens(tmp_path)
    data = _write(tmp_path, pd.DataFrame({"prompt": ["p"], "response": ["r"]}))

    class WrongModel(FakeEmbedder):
        model_id = "someone-elses/model"

    with pytest.raises(ValueError, match="embed_model_id"):
        run_encode_dataset(lens, data, tmp_path / "out", embedder=WrongModel())


def test_dim_mismatch_is_surfaced(tmp_path):
    lens = _make_lens(tmp_path)
    data = _write(tmp_path, pd.DataFrame({"prompt": ["p"], "response": ["r"]}))
    with pytest.raises(ValueError, match="input_dim|feature dim"):
        run_encode_dataset(lens, data, tmp_path / "out", embedder=WrongDimEmbedder())


def test_meta_passthrough_and_missing_optional_col(tmp_path):
    lens = _make_lens(tmp_path)
    df = pd.DataFrame({"prompt": ["p1", "p2"], "response": ["aa", "bb"],
                       "response_y": ["cc", "dd"], "who": ["X", "Y"], "winner": ["a", "b"]})
    data = _write(tmp_path, df)
    out = tmp_path / "out"
    mf = run_encode_dataset(lens, data, out, embedder=FakeEmbedder(), response2_col="response_y",
                            model_col="who", label_col="winner")
    meta = pd.read_parquet(out / "meta.parquet")
    # meta.parquet is emitted under the CANONICAL pair-schema names, whatever the source cols
    assert list(meta["model_a"]) == ["X", "Y"] and list(meta["human_pref"]) == ["a", "b"]
    assert {"prompt", "completion_a", "completion_b"} <= set(meta.columns)
    assert "model_b" not in meta.columns  # model2_col not requested
    assert mf["has_preference"] is True   # label_col given with non-null values

    with pytest.raises(ValueError, match="not in data"):
        run_encode_dataset(lens, data, tmp_path / "o2", embedder=FakeEmbedder(), model_col="nope")


def test_extra_metadata_columns_are_preserved_for_downstream_grouping(tmp_path):
    lens = _make_lens(tmp_path)
    data = _write(tmp_path, pd.DataFrame({
        "prompt": ["p1", "p1"], "response": ["aa", "bb"],
        "prompt_group": ["shared", "shared"],
    }))

    run_encode_dataset(
        lens, data, tmp_path / "out", embedder=FakeEmbedder(),
        metadata_cols=["prompt_group"])

    meta = pd.read_parquet(tmp_path / "out" / "meta.parquet")
    assert meta["prompt_group"].tolist() == ["shared", "shared"]
    with pytest.raises(ValueError, match="reserved"):
        run_encode_dataset(
            lens, data, tmp_path / "reserved", embedder=FakeEmbedder(),
            metadata_cols=["row_id"])


def test_no_label_col_means_no_preference(tmp_path):
    lens = _make_lens(tmp_path)
    data = _write(tmp_path, pd.DataFrame({"prompt": ["p1"], "response": ["aa"]}))
    mf = run_encode_dataset(lens, data, tmp_path / "out", embedder=FakeEmbedder())
    assert mf["has_preference"] is False


def test_canonical_input_columns_need_no_flags(tmp_path):
    # a dataset already in corpus form encodes with the DEFAULT column flags
    lens = _make_lens(tmp_path)
    df = pd.DataFrame({"prompt": ["p1"], "completion_a": ["aa"], "completion_b": ["bb"],
                       "model_a": ["X"], "model_b": ["Y"], "human_pref": [1.0]})
    out = tmp_path / "out"
    # genuinely NO column flags: battle mode + models + label are auto-detected from
    # the canonical column names themselves.
    mf = run_encode_dataset(lens, _write(tmp_path, df), out, embedder=FakeEmbedder())
    meta = pd.read_parquet(out / "meta.parquet")
    assert list(meta["completion_b"]) == ["bb"] and list(meta["model_b"]) == ["Y"]
    assert mf["has_preference"] is True


def test_drops_empty_rows_and_traces_row_id(tmp_path):
    lens = _make_lens(tmp_path)
    df = pd.DataFrame({"prompt": ["p1", "p2", "p3"], "response": ["aa", "", "cc"]})
    out = tmp_path / "out"
    mf = run_encode_dataset(lens, _write(tmp_path, df), out, embedder=FakeEmbedder())
    meta = pd.read_parquet(out / "meta.parquet")
    assert mf["n_rows"] == 2 and mf["n_dropped"] == 1
    assert list(meta["row_id"]) == [0, 2]                 # row 1 dropped; row_id traces originals
    assert np.load(out / "z_a.npy").shape[0] == 2


def test_codes_from_embeddings_battle_equals_rep_primitive(tmp_path):
    lens = _make_lens(tmp_path)
    e_a = np.arange(2 * D, dtype=np.float32).reshape(2, D)
    e_b = (e_a + 3).astype(np.float32)
    via = codes_from_embeddings(lens, e_a, e_b)
    direct = get_lens_rep("individual").output_arrays(FakeProj(lens), e_a, e_b)
    assert set(via) == set(direct)
    for k in direct:
        np.testing.assert_allclose(via[k], direct[k])


def test_provenance_copied_from_lens(tmp_path):
    lens = _make_lens(tmp_path)
    data = _write(tmp_path, pd.DataFrame({"prompt": ["p"], "response": ["r"]}))
    mf = run_encode_dataset(lens, data, tmp_path / "out", embedder=FakeEmbedder())
    assert mf["embed_model_id"] == "fake/embedder-1B"  # from the lens manifest, not hardcoded
    assert mf["input_dim"] == D and mf["lens_input_rep"] == "individual"
    assert mf["code_stats"]["n_rows"] == 1 and mf["code_stats"]["m_total"] == M
    assert "lens_dir" not in mf
    assert len(mf["source_lens_manifest_sha256"]) == 64
    assert str(tmp_path) not in json.dumps(mf)


def test_unsupported_format_errors(tmp_path):
    lens = _make_lens(tmp_path)
    bad = tmp_path / "data.txt"
    bad.write_text("nope")
    with pytest.raises(ValueError, match="unsupported data format"):
        run_encode_dataset(lens, bad, tmp_path / "out", embedder=FakeEmbedder())


@pytest.mark.parametrize(
    ("field", "wrong"),
    [
        ("embed_model_id", "other/model"),
        ("embed_model_revision", "other-rev"),
        ("max_tokens", 64),
        ("embed_instruction", "other instruction"),
        ("pooling", "mean"),
        ("normalization", "none"),
        ("dtype", "float16"),
        ("backend", "other"),
    ],
)
def test_full_embedder_provenance_contract_is_enforced(tmp_path, field, wrong):
    lens = _make_lens(tmp_path)
    data = _write(tmp_path, pd.DataFrame({"prompt": ["p"], "response": ["r"]}))

    class WrongProvenance(FakeEmbedder):
        def provenance(self, *, prompt=False):
            provenance = super().provenance(prompt=prompt)
            provenance[field] = wrong
            return provenance

    with pytest.raises(ValueError, match=field):
        run_encode_dataset(
            lens, data, tmp_path / "out", embedder=WrongProvenance())


def test_embedder_must_expose_provenance(tmp_path):
    lens = _make_lens(tmp_path)
    data = _write(tmp_path, pd.DataFrame({"prompt": ["p"], "response": ["r"]}))

    class NoProvenance:
        model_id = "fake/embedder-1B"

        def encode(self, prompts, completions):
            raise AssertionError("must fail before embedding")

    with pytest.raises(ValueError, match="must expose provenance"):
        run_encode_dataset(lens, data, tmp_path / "out", embedder=NoProvenance())


@pytest.mark.parametrize("bad", [np.ones(D), np.ones((2, D)),
                                 np.full((1, D), np.nan)])
def test_invalid_embedding_shape_rows_or_values_are_rejected(tmp_path, bad):
    lens = _make_lens(tmp_path)
    data = _write(tmp_path, pd.DataFrame({"prompt": ["p"], "response": ["r"]}))

    class BadEmbedding(FakeEmbedder):
        def encode(self, prompts, completions):
            return bad

    with pytest.raises(ValueError, match="2-D|rows|non-finite"):
        run_encode_dataset(lens, data, tmp_path / "out", embedder=BadEmbedding())


def test_misaligned_battle_embeddings_are_rejected(tmp_path):
    lens = _make_lens(tmp_path)
    e_a = np.ones((2, D), dtype=np.float32)
    e_b = np.ones((1, D), dtype=np.float32)
    with pytest.raises(ValueError, match="rows"):
        codes_from_embeddings(lens, e_a, e_b)


def test_invalid_prompt_projection_is_rejected(tmp_path, monkeypatch):
    lens = _make_lens(tmp_path, input_rep="prompt")
    data = _write(tmp_path, pd.DataFrame({"prompt": ["p1", "p2"]}))

    class BadProjection(FakeProj):
        def project(self, x):
            return np.full((len(x), M), np.inf, dtype=np.float32)

    monkeypatch.setattr(ed, "SAEProjector", BadProjection)
    with pytest.raises(ValueError, match="z_prompt contains non-finite"):
        run_encode_dataset(lens, data, tmp_path / "out", embedder=FakeEmbedder())


def test_encode_rejects_source_output_overlap_before_embedding(tmp_path):
    lens = _make_lens(tmp_path)
    data = _write(tmp_path, pd.DataFrame({"prompt": ["p"], "response": ["r"]}))

    class CountingEmbedder(FakeEmbedder):
        calls = 0

        def encode(self, prompts, completions):
            CountingEmbedder.calls += 1
            return super().encode(prompts, completions)

    for output in (lens, lens / "nested", data, tmp_path):
        with pytest.raises(ValueError, match="overlaps"):
            run_encode_dataset(lens, data, output, embedder=CountingEmbedder())
    assert CountingEmbedder.calls == 0


def test_encode_rebuild_replaces_whole_bundle_without_stale_arrays(tmp_path):
    lens = _make_lens(tmp_path)
    paired = _write(tmp_path, pd.DataFrame({
        "prompt": ["p"], "response": ["a"], "response_y": ["b"],
    }), name="paired.csv")
    out = tmp_path / "out"
    run_encode_dataset(
        lens, paired, out, embedder=FakeEmbedder(), response2_col="response_y")
    (out / "stale.txt").write_text("old")
    assert (out / "z_b.npy").exists() and (out / "z_diff.npy").exists()

    single = _write(tmp_path, pd.DataFrame({
        "prompt": ["p"], "response": ["a"],
    }), name="single.csv")
    manifest = run_encode_dataset(
        lens, single, out, embedder=FakeEmbedder(), overwrite=True)

    assert manifest["output_arrays"] == ["z_a"]
    assert manifest["array_shapes"] == {"z_a": [1, M]}
    assert {path.name for path in out.iterdir()} == {
        "manifest.json", "meta.parquet", "battles.parquet", "z_a.npy"}


def test_encode_overwrite_false_refuses_nonempty_destination(tmp_path):
    lens = _make_lens(tmp_path)
    data = _write(tmp_path, pd.DataFrame({"prompt": ["p"], "response": ["r"]}))
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("old")

    with pytest.raises(FileExistsError, match="overwrite=True"):
        run_encode_dataset(
            lens, data, out, embedder=FakeEmbedder(), overwrite=False)

    assert {path.name for path in out.iterdir()} == {"keep.txt"}


def test_encode_staging_validation_failure_preserves_destination(tmp_path, monkeypatch):
    lens = _make_lens(tmp_path)
    data = _write(tmp_path, pd.DataFrame({"prompt": ["p"], "response": ["r"]}))
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("old")

    def fail_validation(*args, **kwargs):
        raise ValueError("synthetic bundle validation failure")

    monkeypatch.setattr(ed, "_validate_encoded_bundle", fail_validation)
    with pytest.raises(ValueError, match="synthetic bundle validation failure"):
        run_encode_dataset(
            lens, data, out, embedder=FakeEmbedder(), overwrite=True)

    assert {path.name for path in out.iterdir()} == {"keep.txt"}
    assert not list(tmp_path.glob(".out.tmp-*"))
    assert not list(tmp_path.glob(".out.bak-*"))


def test_encode_commit_failure_rolls_back_destination(tmp_path, monkeypatch):
    lens = _make_lens(tmp_path)
    data = _write(tmp_path, pd.DataFrame({"prompt": ["p"], "response": ["r"]}))
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("old")
    real_replace = ed.os.replace

    def fail_commit(source, target):
        source, target = Path(source), Path(target)
        if source.name.startswith(".out.tmp-") and target == out:
            raise OSError("synthetic publish failure")
        return real_replace(source, target)

    monkeypatch.setattr(ed.os, "replace", fail_commit)
    with pytest.raises(OSError, match="synthetic publish failure"):
        run_encode_dataset(
            lens, data, out, embedder=FakeEmbedder(), overwrite=True)

    assert {path.name for path in out.iterdir()} == {"keep.txt"}
    assert (out / "keep.txt").read_text() == "old"
    assert not list(tmp_path.glob(".out.tmp-*"))
    assert not list(tmp_path.glob(".out.bak-*"))
