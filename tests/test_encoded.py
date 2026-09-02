from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pytest

from prefscope.api import encoded as encoded_module
from prefscope.api.encoded import load_feature_batch, save_feature_batch
from prefscope.core.features import FeatureBatch


def _batch(offset: float = 0.0) -> FeatureBatch:
    return FeatureBatch(
        row_ids=("a", "b"),
        arrays={
            "z_a": np.array(
                [[1.0 + offset, 2.0], [3.0, 4.0 + offset]],
                dtype=np.float32,
            )
        },
        roles={"z_a": "response_a"},
        orientations={"z_a": "absolute_a"},
        feature_ids=(5, 9),
        metadata={"group_id": ("g1", "g2")},
        activation_polarity="nonnegative",
        code_semantics="numerical_activity",
        provenance={"producer": "test"},
    )


@pytest.mark.parametrize("field", ["activation_polarity", "code_semantics"])
@pytest.mark.parametrize("value", [None, "", "   "])
def test_feature_batch_rejects_invalid_schema2_global_semantics(field, value):
    kwargs = {field: value}
    with pytest.raises(ValueError, match=field):
        FeatureBatch(
            row_ids=("r",),
            arrays={"z_a": np.ones((1, 1), dtype=np.float32)},
            roles={"z_a": "response_a"},
            **kwargs,
        )


@pytest.mark.parametrize(
    ("field", "message"),
    [("roles", "feature role"), ("orientations", "feature orientation")],
)
def test_feature_batch_rejects_whitespace_view_contract(field, message):
    kwargs = {
        "roles": {"z_a": "response_a"},
        "orientations": {"z_a": "absolute_a"},
    }
    kwargs[field] = {"z_a": "   "}

    with pytest.raises(ValueError, match=message):
        FeatureBatch(
            row_ids=("r",),
            arrays={"z_a": np.ones((1, 1), dtype=np.float32)},
            **kwargs,
        )


@pytest.mark.parametrize(
    ("views", "message"),
    [
        ("not-a-mapping", "provenance views must be a mapping"),
        ({"unknown": {}}, "unknown arrays"),
        ({"z_a": "not-a-mapping"}, "must be a mapping"),
        ({"z_a": {"activation_polarity": " "}}, "activation_polarity"),
        ({"z_a": {"code_semantics": None}}, "code_semantics"),
    ],
)
def test_feature_batch_rejects_invalid_schema2_view_semantics(views, message):
    with pytest.raises(ValueError, match=message):
        FeatureBatch(
            row_ids=("r",),
            arrays={"z_a": np.ones((1, 1), dtype=np.float32)},
            roles={"z_a": "response_a"},
            provenance={"views": views},
        )


def test_eager_loader_rejects_invalid_schema2_semantics(tmp_path):
    root = save_feature_batch(_batch(), tmp_path / "invalid-global")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["activation_polarity"] = None
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="activation_polarity"):
        load_feature_batch(root)

    root = save_feature_batch(_batch(), tmp_path / "invalid-view")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["provenance"]["views"] = {"unknown": {}}
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unknown arrays"):
        load_feature_batch(root)


def test_selective_eager_load_prunes_other_view_semantics(tmp_path):
    batch = FeatureBatch(
        row_ids=("r",),
        arrays={
            "z_a": np.ones((1, 1), dtype=np.float32),
            "z_diff": np.zeros((1, 1), dtype=np.float32),
        },
        roles={"z_a": "response_a", "z_diff": "response_difference"},
        provenance={
            "views": {
                "z_diff": {
                    "activation_polarity": "signed",
                    "code_semantics": "activity_difference",
                }
            }
        },
    )
    root = save_feature_batch(batch, tmp_path / "selected")

    selected = load_feature_batch(root, arrays=("z_a",))

    assert tuple(selected.arrays) == ("z_a",)
    assert selected.provenance["views"] == {}


@pytest.mark.parametrize("field", ["roles", "orientations"])
def test_selective_eager_load_validates_all_view_contracts(tmp_path, field):
    batch = FeatureBatch(
        row_ids=("r",),
        arrays={
            "z_a": np.ones((1, 1), dtype=np.float32),
            "z_diff": np.zeros((1, 1), dtype=np.float32),
        },
        roles={"z_a": "response_a", "z_diff": "response_difference"},
        orientations={"z_a": "absolute_a", "z_diff": "a_minus_b"},
    )
    root = save_feature_batch(batch, tmp_path / field)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[field]["z_diff"] = "   "
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="roles/orientations"):
        load_feature_batch(root, arrays=("z_a",))


def test_overwrite_rejects_unmanaged_directory_without_touching_it(tmp_path):
    root = tmp_path / "encoded"
    root.mkdir()
    marker = root / "user-data.txt"
    marker.write_text("keep")

    with pytest.raises(ValueError, match="valid managed encoded bundle"):
        save_feature_batch(_batch(), root, overwrite=True)
    assert marker.read_text() == "keep"
    assert {path.name for path in root.iterdir()} == {"user-data.txt"}


def test_save_and_load_reject_symlink_destination_and_members(tmp_path):
    managed = save_feature_batch(_batch(), tmp_path / "managed")
    destination = tmp_path / "destination"
    destination.symlink_to(managed, target_is_directory=True)
    with pytest.raises(ValueError, match="file/symlink"):
        save_feature_batch(_batch(1.0), destination, overwrite=True)
    with pytest.raises(ValueError, match="must not be a symlink"):
        load_feature_batch(destination)

    member = managed / "z_a.npy"
    external = tmp_path / "external.npy"
    external.write_bytes(member.read_bytes())
    member.unlink()
    member.symlink_to(external)
    with pytest.raises(ValueError, match="member.*regular file.*symlink"):
        load_feature_batch(managed)


def test_load_rejects_duplicate_json_keys_and_secret_provenance(tmp_path):
    root = save_feature_batch(_batch(), tmp_path / "duplicate")
    manifest_path = root / "manifest.json"
    text = manifest_path.read_text().rstrip()
    manifest_path.write_text(text[:-1] + ', "schema_version": 2}')
    with pytest.raises(ValueError, match="duplicate JSON key 'schema_version'"):
        load_feature_batch(root)

    root = save_feature_batch(_batch(), tmp_path / "secret")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["provenance"]["api_key"] = "sk-proj-not-safe-to-publish"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="credential-like field.*api_key"):
        load_feature_batch(root)


def test_no_overwrite_rechecks_destination_under_publication_lock(
    tmp_path, monkeypatch
):
    template = save_feature_batch(_batch(), tmp_path / "template")
    destination = tmp_path / "destination"

    @contextmanager
    def injecting_lock(root: Path):
        shutil.copytree(template, root)
        yield

    monkeypatch.setattr(encoded_module, "_publication_lock", injecting_lock)
    with pytest.raises(FileExistsError, match="already exists"):
        save_feature_batch(_batch(2.0), destination)
    loaded = load_feature_batch(destination)
    np.testing.assert_array_equal(loaded.array("z_a"), _batch().array("z_a"))


def test_concurrent_destination_after_backup_is_preserved(tmp_path, monkeypatch):
    root = save_feature_batch(_batch(), tmp_path / "encoded")
    old_values = load_feature_batch(root).array("z_a").copy()
    original_replace = os.replace
    injected = False

    def racing_replace(source, destination):
        nonlocal injected
        source_path = Path(source)
        destination_path = Path(destination)
        result = original_replace(source, destination)
        if (
            not injected
            and source_path == root
            and destination_path.name.startswith(f".{root.name}.bak-")
        ):
            injected = True
            root.mkdir()
            (root / "intruder.txt").write_text("concurrent")
        return result

    monkeypatch.setattr(encoded_module.os, "replace", racing_replace)
    with pytest.raises(FileExistsError, match="appeared during staging"):
        save_feature_batch(_batch(4.0), root, overwrite=True)

    assert (root / "intruder.txt").read_text() == "concurrent"
    backups = list(tmp_path.glob(f".{root.name}.bak-*"))
    assert len(backups) == 1
    np.testing.assert_array_equal(
        load_feature_batch(backups[0]).array("z_a"), old_values
    )
    assert list(tmp_path.glob(f".{root.name}.unexpected-*")) == []


def test_next_write_recovers_one_orphan_backup(tmp_path):
    root = save_feature_batch(_batch(), tmp_path / "encoded")
    backup = tmp_path / f".{root.name}.bak-orphan"
    os.replace(root, backup)

    save_feature_batch(_batch(3.0), root, overwrite=True)
    loaded = load_feature_batch(root)
    np.testing.assert_array_equal(loaded.array("z_a"), _batch(3.0).array("z_a"))
    assert not backup.exists()


def test_valid_destination_replaced_during_staging_fails_inode_check(
    tmp_path,
    monkeypatch,
):
    root = save_feature_batch(_batch(), tmp_path / "encoded")
    replacement = save_feature_batch(_batch(11.0), tmp_path / "replacement")
    stolen = tmp_path / "old-bundle-moved-by-actor"
    original_validate = encoded_module._validate_managed_destination
    validations = 0

    def replacing_validate(destination):
        nonlocal validations
        validations += 1
        if validations == 2:
            os.replace(destination, stolen)
            shutil.copytree(replacement, destination)
        return original_validate(destination)

    monkeypatch.setattr(
        encoded_module, "_validate_managed_destination", replacing_validate
    )
    with pytest.raises(RuntimeError, match="changed during staging"):
        save_feature_batch(_batch(7.0), root, overwrite=True)

    assert validations == 2
    np.testing.assert_array_equal(
        load_feature_batch(root).array("z_a"), _batch(11.0).array("z_a")
    )
    np.testing.assert_array_equal(
        load_feature_batch(stolen).array("z_a"), _batch().array("z_a")
    )
    assert list(tmp_path.glob(f".{root.name}.bak-*")) == []


def test_initially_absent_valid_concurrent_destination_is_not_touched(
    tmp_path,
    monkeypatch,
):
    template = save_feature_batch(_batch(13.0), tmp_path / "template")
    destination = tmp_path / "destination"
    original_load = encoded_module.load_feature_batch
    injected = False

    def injecting_load(path, *, arrays=None):
        nonlocal injected
        result = original_load(path, arrays=arrays)
        if not injected and Path(path).name.startswith(".destination.tmp-"):
            injected = True
            shutil.copytree(template, destination)
        return result

    monkeypatch.setattr(encoded_module, "load_feature_batch", injecting_load)
    with pytest.raises(FileExistsError, match="appeared during staging"):
        save_feature_batch(_batch(2.0), destination)

    assert injected
    np.testing.assert_array_equal(
        original_load(destination).array("z_a"), _batch(13.0).array("z_a")
    )
    assert list(tmp_path.glob(".destination.unexpected-*")) == []


def test_save_rejects_symlink_immediate_parent(tmp_path):
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(ValueError, match="parent must not be a symlink"):
        save_feature_batch(_batch(), linked_parent / "encoded")
    assert not (real_parent / "encoded").exists()


def test_load_rejects_root_replacement_after_descriptor_snapshot(
    tmp_path,
    monkeypatch,
):
    root = save_feature_batch(_batch(), tmp_path / "encoded")
    moved = tmp_path / "moved"
    original_read = encoded_module._read_parquet_member
    replaced = False

    def replacing_read(descriptor, name):
        nonlocal replaced
        if not replaced:
            replaced = True
            os.replace(root, moved)
            root.symlink_to(moved, target_is_directory=True)
        return original_read(descriptor, name)

    monkeypatch.setattr(encoded_module, "_read_parquet_member", replacing_read)
    with pytest.raises(ValueError, match="root changed"):
        load_feature_batch(root)


def test_manifest_byte_and_depth_limits_are_enforced(tmp_path, monkeypatch):
    root = save_feature_batch(_batch(), tmp_path / "encoded")
    manifest_path = root / "manifest.json"
    payload = manifest_path.read_bytes()
    monkeypatch.setattr(encoded_module, "_MAX_MANIFEST_BYTES", len(payload) - 1)
    with pytest.raises(ValueError, match="maximum size"):
        load_feature_batch(root)

    monkeypatch.setattr(encoded_module, "_MAX_MANIFEST_BYTES", 8 * 1024 * 1024)
    manifest = json.loads(payload)
    nested = "leaf"
    for _ in range(encoded_module._MAX_MANIFEST_DEPTH):
        nested = {"nested": nested}
    manifest["provenance"] = nested
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="nesting depth"):
        load_feature_batch(root)


@pytest.mark.parametrize(
    ("limit_name", "limit_value", "message"),
    [
        ("_MAX_METADATA_COMPRESSED_BYTES", 1, "compressed bytes"),
        ("_MAX_METADATA_COLUMNS", 1, "columns"),
        ("_MAX_METADATA_ROW_GROUPS", 0, "row groups"),
    ],
)
def test_parquet_limits_reject_before_dataframe_materialization(
    tmp_path,
    monkeypatch,
    limit_name,
    limit_value,
    message,
):
    root = save_feature_batch(_batch(), tmp_path / "encoded")
    monkeypatch.setattr(encoded_module, limit_name, limit_value)
    monkeypatch.setattr(
        encoded_module,
        "_read_parquet_member",
        lambda *_args, **_kwargs: pytest.fail(
            "Parquet was materialized before preflight"
        ),
    )
    with pytest.raises(ValueError, match=message):
        load_feature_batch(root)


def test_parquet_aggregate_uncompressed_limit_precedes_pandas(tmp_path, monkeypatch):
    import pyarrow.parquet as parquet

    root = save_feature_batch(_batch(), tmp_path / "encoded")
    sizes = []
    for name in ("meta.parquet", "battles.parquet"):
        parquet_file = parquet.ParquetFile(root / name)
        sizes.append(
            sum(
                parquet_file.metadata.row_group(index).total_byte_size
                for index in range(parquet_file.metadata.num_row_groups)
            )
        )
    monkeypatch.setattr(
        encoded_module, "_MAX_METADATA_UNCOMPRESSED_BYTES", sum(sizes) - 1
    )
    monkeypatch.setattr(
        encoded_module,
        "_read_parquet_member",
        lambda *_args, **_kwargs: pytest.fail(
            "Parquet was materialized before preflight"
        ),
    )
    with pytest.raises(ValueError, match="aggregate uncompressed bytes"):
        load_feature_batch(root)


def test_array_element_limit_precedes_parquet_and_trailing_bytes_rejected(
    tmp_path,
    monkeypatch,
):
    root = save_feature_batch(_batch(), tmp_path / "encoded")
    monkeypatch.setattr(encoded_module, "_MAX_ARRAY_ELEMENTS", 3)
    monkeypatch.setattr(
        encoded_module,
        "_preflight_parquet_member",
        lambda *_args, **_kwargs: pytest.fail(
            "Parquet preflight preceded array budget"
        ),
    )
    with pytest.raises(ValueError, match="element budget"):
        load_feature_batch(root)

    monkeypatch.undo()
    with (root / "z_a.npy").open("ab") as stream:
        stream.write(b"hidden trailing payload")
    with pytest.raises(ValueError, match="trailing bytes"):
        load_feature_batch(root)


def test_metadata_cell_and_save_row_limits(tmp_path, monkeypatch):
    root = save_feature_batch(_batch(), tmp_path / "encoded")
    monkeypatch.setattr(encoded_module, "_MAX_METADATA_CELLS", 3)
    monkeypatch.setattr(
        encoded_module,
        "_read_parquet_member",
        lambda *_args, **_kwargs: pytest.fail(
            "Parquet was materialized before cell cap"
        ),
    )
    with pytest.raises(ValueError, match="metadata cells"):
        load_feature_batch(root)

    monkeypatch.undo()
    monkeypatch.setattr(encoded_module, "_MAX_ROWS", 1)
    with pytest.raises(ValueError, match="batch rows exceed"):
        save_feature_batch(_batch(), tmp_path / "too-many-rows")
    assert not (tmp_path / "too-many-rows").exists()
