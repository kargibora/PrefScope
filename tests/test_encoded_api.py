import json

import numpy as np
import pytest

from prefscope import FeatureBatch, load_feature_batch, save_feature_batch


def _batch():
    return FeatureBatch(
        row_ids=("r0", "r1"),
        feature_ids=(3, 8),
        arrays={
            "z_b": np.array([[0.5, 1.0], [1.0, 0.0]], dtype=np.float32),
            "z_a": np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32),
        },
        roles={"z_a": "response", "z_b": "response"},
        orientations={"z_a": "absolute", "z_b": "absolute"},
        metadata={"group": ("x", "y")},
        activation_polarity="nonnegative",
        code_semantics="numerical_activity",
        provenance={
            "feature_space_id": "sha256:test",
            "views": {
                "z_b": {"code_semantics": "numerical_activity"},
                "z_a": {"code_semantics": "numerical_activity"},
            },
        },
    )


def test_feature_batch_directory_round_trip(tmp_path):
    path = save_feature_batch(_batch(), tmp_path / "features")
    restored = load_feature_batch(path)
    assert restored.row_ids == ("r0", "r1")
    assert restored.feature_ids == (3, 8)
    assert tuple(restored.arrays) == ("z_b", "z_a")
    np.testing.assert_array_equal(restored.array("z_a"), _batch().array("z_a"))
    assert restored.metadata["group"] == ("x", "y")
    assert restored.provenance["feature_space_id"] == "sha256:test"


def test_feature_batch_loader_selects_named_views(tmp_path):
    path = save_feature_batch(_batch(), tmp_path / "features")
    restored = load_feature_batch(path, arrays=("z_b",))
    assert tuple(restored.arrays) == ("z_b",)
    assert tuple(restored.provenance["views"]) == ("z_b",)
    with pytest.raises(KeyError, match="unknown"):
        load_feature_batch(path, arrays=("missing",))


def test_feature_batch_save_requires_explicit_overwrite(tmp_path):
    path = save_feature_batch(_batch(), tmp_path / "features")
    with pytest.raises(FileExistsError):
        save_feature_batch(_batch(), path)
    save_feature_batch(_batch(), path, overwrite=True)
    assert load_feature_batch(path).feature_ids == (3, 8)


def test_feature_batch_loader_rejects_unknown_schema(tmp_path):
    path = save_feature_batch(_batch(), tmp_path / "features")
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = 99
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="schema_version"):
        load_feature_batch(path)
