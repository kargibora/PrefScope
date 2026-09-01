"""Versioned lens manifest: strict production, safe loading, no silent defaults."""
from __future__ import annotations

import numpy as np
import pytest

from prefscope.core.manifest import SCHEMA_VERSION, LensManifest, infer_lens_kind


def _complete_dict():
    return {
        "schema_version": SCHEMA_VERSION,
        "lens_kind": "individual",
        "input_rep": "individual",
        "m_total": 1024,
        "k": 64,
        "input_dim": 4096,
        "matryoshka_prefix_lengths": [16, 32, 64, 1024],
        "output_arrays": ["z_a", "z_b", "z_diff"],
        "embed_model_id": "Qwen/Qwen3-Embedding-8B",
        "sae_type": "batchtopk-relu",
        "activation_polarity": "nonnegative",
        "code_semantics": "presence",
        "selection_rule": "batchtopk-relu",
    }


def test_roundtrip_complete_manifest():
    m = LensManifest.from_dict(_complete_dict(), strict=True)
    d = m.to_dict()
    assert d["schema_version"] == SCHEMA_VERSION
    assert d["embed_model_id"] == "Qwen/Qwen3-Embedding-8B"
    assert d["m_total"] == 1024
    assert d["activation_polarity"] == "nonnegative"
    # provenance fields are recorded (as None), not silently absent
    assert "pooling" in d and d["pooling"] is None


def test_strict_missing_required_field_raises():
    bad = _complete_dict()
    del bad["embed_model_id"]
    with pytest.raises(ValueError, match="missing required"):
        LensManifest.from_dict(bad, strict=True)


def test_manifest_rejects_contradictory_kind_and_future_schema():
    bad = _complete_dict()
    bad["lens_kind"] = "prompt"
    with pytest.raises(ValueError, match="contradictory"):
        LensManifest.from_dict(bad)
    future = _complete_dict()
    future["schema_version"] = SCHEMA_VERSION + 1
    with pytest.raises(ValueError, match="newer"):
        LensManifest.from_dict(future)


def test_legacy_manifest_migrates_and_warns(caplog):
    legacy = {"input_rep": "individual", "m_total": 1024, "k": 64,
              "output_arrays": ["z_a", "z_b"], "embed_model_id": "x"}  # no schema_version
    with caplog.at_level("WARNING"):
        m = LensManifest.from_dict(legacy)          # lenient load
    assert m.schema_version == SCHEMA_VERSION
    assert m.lens_kind == "individual"
    assert m.sae_type == "batchtopk"
    assert m.activation_polarity == "signed"
    assert m.code_semantics == "axis"
    assert any("legacy" in r.message for r in caplog.records)


@pytest.mark.parametrize("arrays,expected", [
    (["z_prompt"], "prompt"),
    (["z_a", "z_b", "z_diff"], "individual"),
    (["z_diff"], "difference"),
])
def test_infer_kind_from_arrays_when_input_rep_absent(arrays, expected):
    m = LensManifest.from_dict({"output_arrays": arrays, "m_total": 8})
    assert m.lens_kind == expected
    assert m.input_rep == expected


def test_refuses_to_guess_representation():
    # no input_rep, no recognizable arrays → must raise, never default to "difference"
    with pytest.raises(ValueError, match="refusing to guess"):
        LensManifest.from_dict({"m_total": 8, "k": 2})


def test_extra_keys_round_trip():
    d = _complete_dict()
    d["n_prompts"] = 123           # a legacy/producer-specific key not in the schema
    d["whiten"] = "standardize"
    m = LensManifest.from_dict(d)
    out = m.to_dict()
    assert out["n_prompts"] == 123
    assert out["whiten"] == "standardize"


def test_no_matryoshka_is_a_complete_v2_configuration():
    d = _complete_dict()
    d["matryoshka_prefix_lengths"] = []
    LensManifest.from_dict(d, strict=True)


def test_inference_artifact_may_omit_corpus_aligned_output_arrays():
    d = _complete_dict()
    d.update(output_arrays=[], artifact_scope="inference",
             source_output_arrays=["z_a", "z_b", "z_diff"])
    manifest = LensManifest.from_dict(d, strict=True)
    assert manifest.extra["artifact_scope"] == "inference"
    assert manifest.output_arrays == []

    d.pop("artifact_scope")
    with pytest.raises(ValueError, match="output_arrays"):
        LensManifest.from_dict(d, strict=True)


def test_v1_signed_semantics_are_migrated_without_guessing():
    d = _complete_dict()
    d.update(schema_version=1, sae_type="batchtopk")
    for key in ("activation_polarity", "code_semantics", "selection_rule"):
        d.pop(key)
    m = LensManifest.from_dict(d)
    assert m.schema_version == SCHEMA_VERSION
    assert (m.activation_polarity, m.code_semantics) == ("signed", "axis")


def test_v1_null_sae_type_is_the_legacy_signed_batchtopk():
    d = _complete_dict()
    d.update(schema_version=1, sae_type=None)
    for key in ("activation_polarity", "code_semantics", "selection_rule"):
        d.pop(key)
    m = LensManifest.from_dict(d)
    assert m.sae_type == "batchtopk"
    assert (m.activation_polarity, m.code_semantics) == ("signed", "axis")


def test_validate_arrays_shape_mismatch_raises(tmp_path):
    d = _complete_dict()
    d["output_arrays"] = ["z_a"]
    np.save(tmp_path / "z_a.npy", np.zeros((10, 512), dtype=np.float32))  # 512 != m_total 1024
    m = LensManifest.from_dict(d)
    with pytest.raises(ValueError, match="disagree"):
        m.validate_arrays(tmp_path)


@pytest.mark.parametrize("shape", [(10,), (2, 5, 1024)])
def test_validate_arrays_requires_exactly_two_dimensions(tmp_path, shape):
    d = _complete_dict()
    d["output_arrays"] = ["z_a"]
    np.save(tmp_path / "z_a.npy", np.zeros(shape, dtype=np.float32))
    manifest = LensManifest.from_dict(d)

    with pytest.raises(ValueError, match="must be 2-D"):
        manifest.validate_arrays(tmp_path)


def test_validate_arrays_requires_equal_row_counts(tmp_path):
    d = _complete_dict()
    d["output_arrays"] = ["z_a", "z_b"]
    np.save(tmp_path / "z_a.npy", np.zeros((10, 1024), dtype=np.float32))
    np.save(tmp_path / "z_b.npy", np.zeros((9, 1024), dtype=np.float32))
    manifest = LensManifest.from_dict(d)

    with pytest.raises(ValueError, match="row-aligned"):
        manifest.validate_arrays(tmp_path)


def test_validate_arrays_matches_recorded_shapes(tmp_path):
    d = _complete_dict()
    d.update(output_arrays=["z_a"], array_shapes={"z_a": [11, 1024]})
    np.save(tmp_path / "z_a.npy", np.zeros((10, 1024), dtype=np.float32))
    manifest = LensManifest.from_dict(d)

    with pytest.raises(ValueError, match="array_shapes records"):
        manifest.validate_arrays(tmp_path)


def test_validate_arrays_matches_n_battles(tmp_path):
    d = _complete_dict()
    d.update(output_arrays=["z_a"], n_battles=11)
    np.save(tmp_path / "z_a.npy", np.zeros((10, 1024), dtype=np.float32))
    manifest = LensManifest.from_dict(d)

    with pytest.raises(ValueError, match="n_battles=11"):
        manifest.validate_arrays(tmp_path)


def test_validate_arrays_matches_battles_parquet(tmp_path):
    import pandas as pd

    d = _complete_dict()
    d["output_arrays"] = ["z_a"]
    np.save(tmp_path / "z_a.npy", np.zeros((10, 1024), dtype=np.float32))
    pd.DataFrame({"battle_id": [f"b{i}" for i in range(9)]}).to_parquet(
        tmp_path / "battles.parquet", index=False)
    manifest = LensManifest.from_dict(d)

    with pytest.raises(ValueError, match="battles.parquet has 9"):
        manifest.validate_arrays(tmp_path)


def test_validate_arrays_ok_records_shapes(tmp_path):
    d = _complete_dict()
    d.update(output_arrays=["z_a"], n_battles=10,
             array_shapes={"z_a": [10, 1024]})
    np.save(tmp_path / "z_a.npy", np.zeros((10, 1024), dtype=np.float32))
    m = LensManifest.from_dict(d)
    m.validate_arrays(tmp_path)
    assert m.array_shapes == {"z_a": [10, 1024]}


def test_validate_arrays_missing_file_raises(tmp_path):
    d = _complete_dict()
    d["output_arrays"] = ["z_a"]
    m = LensManifest.from_dict(d)
    with pytest.raises(FileNotFoundError):
        m.validate_arrays(tmp_path)


def test_infer_helper_direct():
    assert infer_lens_kind("individual", None) == "individual"
    assert infer_lens_kind(None, ["z_prompt"]) == "prompt"
    assert infer_lens_kind(None, []) is None
