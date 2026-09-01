from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from prefscope import FeatureBatch, load_feature_batch, save_feature_batch
from prefscope.core.provenance import ordered_dataset_hash


def _bundle(tmp_path):
    root = tmp_path / "encoded"
    root.mkdir(parents=True)
    meta = pd.DataFrame({
        "battle_id": ["a", "b"],
        "prompt": ["same", "same"],
        "reward": [0.1, 0.9],
    })
    meta.to_parquet(root / "meta.parquet", index=False)
    meta.to_parquet(root / "battles.parquet", index=False)
    z_a = np.array([[1.0, 0.0], [0.0, 1.0]], np.float32)
    z_diff = np.array([[1.0, -1.0], [-1.0, 1.0]], np.float32)
    np.save(root / "z_a.npy", z_a)
    np.save(root / "z_diff.npy", z_diff)
    dataset_hash = ordered_dataset_hash(meta, {"z_a": z_a, "z_diff": z_diff})
    (root / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "n_rows": 2,
        "m_total": 2,
        "output_arrays": ["z_a", "z_diff"],
        "array_shapes": {"z_a": [2, 2], "z_diff": [2, 2]},
        "activation_polarity": "signed",
        "code_semantics": "axis",
        "dataset_hash": dataset_hash,
    }))
    return root


def test_load_feature_batch_preserves_views_metadata_and_orientation(tmp_path):
    batch = load_feature_batch(_bundle(tmp_path))
    assert set(batch.arrays) == {"z_a", "z_diff"}
    assert batch.matrix("z_a").orientation == "absolute_a"
    assert batch.matrix("z_diff").orientation == "a_minus_b"
    assert batch.metadata["prompt"] == ("same", "same")
    assert batch.activation_polarity == "signed"
    assert batch.provenance["encoded_bundle"]["dataset_hash"] == ordered_dataset_hash(
        pd.read_parquet(tmp_path / "encoded" / "meta.parquet"),
        {"z_a": batch.arrays["z_a"], "z_diff": batch.arrays["z_diff"]},
    )


def test_load_feature_batch_can_select_one_view(tmp_path):
    batch = load_feature_batch(_bundle(tmp_path), arrays=["z_diff"])
    assert set(batch.arrays) == {"z_diff"}


def test_load_feature_batch_rejects_duplicate_ids_and_shape_drift(tmp_path):
    root = _bundle(tmp_path)
    duplicate = pd.DataFrame({"battle_id": ["a", "a"]})
    duplicate.to_parquet(root / "meta.parquet", index=False)
    duplicate.to_parquet(root / "battles.parquet", index=False)
    with pytest.raises(ValueError, match="unique and nonmissing"):
        load_feature_batch(root)

    _bundle(tmp_path / "other")
    other = tmp_path / "other" / "encoded"
    np.save(other / "z_a.npy", np.ones((1, 2), np.float32))
    with pytest.raises(ValueError, match="expected"):
        load_feature_batch(other)


def test_load_feature_batch_rejects_dataset_hash_mismatch(tmp_path):
    root = _bundle(tmp_path)
    metadata = pd.read_parquet(root / "meta.parquet")
    metadata.loc[0, "reward"] = 0.7
    metadata.to_parquet(root / "meta.parquet", index=False)
    metadata.to_parquet(root / "battles.parquet", index=False)
    with pytest.raises(ValueError, match="dataset_hash"):
        load_feature_batch(root)


def test_load_feature_batch_rejects_extra_files_and_metadata_twin_drift(tmp_path):
    root = _bundle(tmp_path)
    (root / "stale.npy").write_bytes(b"stale")
    with pytest.raises(ValueError, match="undeclared artifacts"):
        load_feature_batch(root)

    root = _bundle(tmp_path / "other")
    metadata = pd.read_parquet(root / "meta.parquet")
    metadata.loc[0, "reward"] = 0.8
    metadata.to_parquet(root / "battles.parquet", index=False)
    with pytest.raises(ValueError, match="exactly identical"):
        load_feature_batch(root)


@pytest.mark.parametrize("schema", [True, 1.9, None, "1"])
def test_load_feature_batch_rejects_non_integer_schema_versions(tmp_path, schema):
    root = _bundle(tmp_path)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["schema_version"] = schema
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unsupported encoded bundle schema"):
        load_feature_batch(root)


@pytest.mark.parametrize("field", ["n_rows", "m_total"])
def test_load_feature_batch_rejects_boolean_integer_fields(tmp_path, field):
    root = _bundle(tmp_path)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest[field] = True
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=field):
        load_feature_batch(root)


def test_schema_two_roundtrip_preserves_selected_ids_and_view_semantics(tmp_path):
    batch = FeatureBatch(
        row_ids=("a", "b"),
        arrays={
            "z_a": np.array([[1, 2], [3, 4]], dtype=np.float32),
            "z_diff": np.array([[1, -1], [-1, 1]], dtype=np.float32),
        },
        roles={"z_a": "response_a", "z_diff": "response_difference"},
        orientations={"z_a": "absolute_a", "z_diff": "a_minus_b"},
        feature_ids=(41, 12),
        metadata={"pref": (1.0, 0.0), "group_id": ("g1", "g2")},
        activation_polarity="nonnegative",
        code_semantics="numerical_activity",
        provenance={
            "views": {
                "z_diff": {
                    "activation_polarity": "signed",
                    "code_semantics": "activity_difference",
                }
            }
        },
    )
    path = save_feature_batch(batch, tmp_path / "features")
    loaded = load_feature_batch(path)

    assert loaded.feature_ids == (41, 12)
    assert loaded.metadata["group_id"] == ("g1", "g2")
    assert loaded.matrix("z_diff").activation_polarity == "signed"
    assert loaded.matrix("z_diff").code_semantics == "activity_difference"
    np.testing.assert_array_equal(loaded.array("z_diff"), batch.array("z_diff"))
    with pytest.raises(FileExistsError):
        save_feature_batch(batch, path)
    save_feature_batch(batch, path, overwrite=True)


def test_feature_metadata_cells_must_be_parquet_safe_scalars():
    with pytest.raises(ValueError, match="scalar"):
        FeatureBatch(
            row_ids=("a",), arrays={"z_a": np.ones((1, 1), np.float32)},
            roles={"z_a": "response_a"},
            metadata={"nested": ((1, 2),)},
        )


def test_schema2_preserves_boolean_arrays_and_portable_missing_metadata(tmp_path):
    batch = FeatureBatch(
        row_ids=("a", "b"),
        arrays={"presence": np.array([[True, False], [False, True]])},
        roles={"presence": "custom"},
        feature_ids=(4, 9),
        metadata={"score": (1, None), "note": (None, "kept")},
        code_semantics="semantic_presence",
    )
    loaded = load_feature_batch(save_feature_batch(batch, tmp_path / "typed"))
    assert loaded.array("presence").dtype == bool
    np.testing.assert_array_equal(loaded.array("presence"), batch.array("presence"))
    assert loaded.metadata["score"] == (1, None)
    assert loaded.metadata["note"] == (None, "kept")


def test_feature_metadata_rejects_nonportable_scalar_objects():
    with pytest.raises(ValueError, match="portable scalar"):
        FeatureBatch(
            row_ids=("a",), arrays={"z_a": np.ones((1, 1), np.float32)},
            roles={"z_a": "response_a"}, metadata={"bad": (1 + 2j,)},
        )


def test_schema2_preserves_large_integer_metadata_and_rejects_type_tampering(tmp_path):
    large = 2 ** 60 + 1
    batch = FeatureBatch(
        row_ids=("a", "b"), arrays={"z_a": np.ones((2, 1), np.float32)},
        roles={"z_a": "response_a"}, metadata={"group_id": (large, None)},
    )
    root = save_feature_batch(batch, tmp_path / "large-int")
    loaded = load_feature_batch(root)
    assert loaded.metadata["group_id"] == (large, None)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["metadata_types"]["group_id"] = "bool"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="disagrees with metadata_types"):
        load_feature_batch(root)


def test_feature_metadata_normalizes_common_missing_scalars():
    batch = FeatureBatch(
        row_ids=("a", "b", "c"),
        arrays={"z_a": np.ones((3, 1), np.float32)},
        roles={"z_a": "response_a"},
        metadata={"missing": (np.nan, pd.NA, pd.NaT)},
    )
    assert batch.metadata["missing"] == (None, None, None)
