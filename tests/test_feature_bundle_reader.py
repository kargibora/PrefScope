from __future__ import annotations

import json
import os
from types import MappingProxyType

import numpy as np
import pytest

from prefscope.api.encoded import save_feature_batch
from prefscope.core.features import FeatureBatch
from prefscope.core import provenance as provenance_module
from prefscope.reporting import source as source_module
from prefscope.reporting.source import FeatureBundleReader, FeatureChunk, FeatureSource


def _bundle(tmp_path):
    batch = FeatureBatch(
        row_ids=("r0", "r1", "r2", "r3", "r4"),
        arrays={
            "z_a": np.arange(15, dtype=np.float32).reshape(5, 3),
            "presence": np.array([
                [True, False, False],
                [False, True, False],
                [False, False, True],
                [True, True, False],
                [False, True, True],
            ]),
        },
        roles={"z_a": "response_a", "presence": "response"},
        orientations={"z_a": "absolute_a", "presence": "none"},
        feature_ids=(8, 3, 21),
        metadata={
            "group_id": ("g0", "g0", "g1", "g1", None),
            "score": (1.0, 0.5, None, 0.25, 0.0),
        },
        activation_polarity="nonnegative",
        code_semantics="numerical_activity",
        provenance={
            "producer": "deterministic-test",
            "views": {
                "presence": {
                    "activation_polarity": "nonnegative",
                    "code_semantics": "semantic_presence",
                }
            },
        },
    )
    return save_feature_batch(batch, tmp_path / "features")


def _edit_manifest(root, edit):
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    edit(manifest)
    path.write_text(json.dumps(manifest))


def test_open_is_torch_free_protocol_source_with_read_only_mmaps(tmp_path):
    root = _bundle(tmp_path)
    source = FeatureBundleReader.open(root)

    assert isinstance(source, FeatureSource)
    assert source.root == root.resolve()
    assert source.row_ids == ("r0", "r1", "r2", "r3", "r4")
    assert source.feature_ids == (8, 3, 21)
    assert source.view_names == ("z_a", "presence")
    assert source.views == source.view_names
    assert source.roles == {"z_a": "response_a", "presence": "response"}
    assert source.orientations["z_a"] == "absolute_a"
    assert source.metadata["group_id"] == ("g0", "g0", "g1", "g1", None)
    assert source.n_rows == 5
    assert source.n_features == 3

    values = source.array("z_a")
    assert isinstance(values, np.memmap)
    assert not values.flags.writeable
    with pytest.raises(ValueError):
        values[0, 0] = 99
    assert source.activation_polarities["z_a"] == "nonnegative"
    assert source.code_semantics_by_view["z_a"] == "numerical_activity"
    assert source.code_semantics_by_view["presence"] == "semantic_presence"
    assert source.code_semantics is source.code_semantics_by_view
    with pytest.raises(TypeError):
        source.code_semantics["presence"] = "changed"


def test_provenance_manifest_and_metadata_are_immutable(tmp_path):
    source = FeatureBundleReader.open(_bundle(tmp_path))
    assert isinstance(source.provenance, MappingProxyType)
    assert source.provenance["producer"] == "deterministic-test"
    assert source.provenance["encoded_bundle"]["schema_version"] == 2
    with pytest.raises(TypeError):
        source.provenance["new"] = "value"
    with pytest.raises(TypeError):
        source.provenance["views"]["presence"]["code_semantics"] = "changed"
    with pytest.raises(TypeError):
        source.metadata["new"] = (1,)
    with pytest.raises(TypeError):
        source.manifest["n_rows"] = 6


def test_row_alignment_helpers_are_exact_and_explicit(tmp_path):
    source = FeatureBundleReader.open(_bundle(tmp_path))
    source.assert_row_ids(source.row_ids)
    assert source.row_positions(("r3", "r0")) == (3, 0)

    with pytest.raises(ValueError, match="expected 5 rows, got 2"):
        source.assert_row_ids(("r0", "r1"))
    with pytest.raises(ValueError, match="first mismatch at position 1"):
        source.assert_row_ids(("r0", "r2", "r1", "r3", "r4"))
    with pytest.raises(ValueError, match="must be unique"):
        source.row_positions(("r0", "r0"))
    with pytest.raises(ValueError, match="absent.*missing"):
        source.row_positions(("r0", "missing"))


def test_iter_chunks_preserves_alignment_order_and_lazy_views(tmp_path):
    source = FeatureBundleReader.open(_bundle(tmp_path))
    chunks = list(source.iter_chunks(2))

    assert [chunk.row_ids for chunk in chunks] == [
        ("r0", "r1"), ("r2", "r3"), ("r4",),
    ]
    assert [chunk.positions for chunk in chunks] == [range(0, 2), range(2, 4), range(4, 5)]
    assert [chunk.row_slice for chunk in chunks] == [slice(0, 2), slice(2, 4), slice(4, 5)]
    assert [tuple(chunk.arrays) for chunk in chunks] == [
        ("z_a", "presence"), ("z_a", "presence"), ("z_a", "presence"),
    ]
    assert all(isinstance(chunk.array("z_a"), np.memmap) for chunk in chunks)
    assert all(not chunk.array("z_a").flags.writeable for chunk in chunks)
    np.testing.assert_array_equal(
        np.concatenate([chunk.array("z_a") for chunk in chunks]),
        source.array("z_a"),
    )

    selected = list(source.iter_chunks(3, views=("presence",)))
    assert [tuple(chunk.arrays) for chunk in selected] == [("presence",), ("presence",)]
    with pytest.raises(ValueError, match="positive integer"):
        list(source.iter_chunks(0))
    with pytest.raises(ValueError, match="positive integer"):
        list(source.iter_chunks(True))
    with pytest.raises(ValueError, match="duplicates"):
        list(source.iter_chunks(2, views=("z_a", "z_a")))
    with pytest.raises(ValueError, match="unknown feature views"):
        list(source.iter_chunks(2, views=("unknown",)))


def test_open_rejects_schema_one_and_manifest_contract_corruption(tmp_path):
    root = _bundle(tmp_path)
    _edit_manifest(root, lambda manifest: manifest.__setitem__("schema_version", 1))
    with pytest.raises(ValueError, match="requires schema_version 2"):
        FeatureBundleReader.open(root)

    root = _bundle(tmp_path / "roles")
    _edit_manifest(root, lambda manifest: manifest["roles"].pop("presence"))
    with pytest.raises(ValueError, match="roles must name every"):
        FeatureBundleReader.open(root)

    root = _bundle(tmp_path / "semantics")
    _edit_manifest(root, lambda manifest: manifest.__setitem__("code_semantics", ""))
    with pytest.raises(ValueError, match="code_semantics"):
        FeatureBundleReader.open(root)

    root = _bundle(tmp_path / "feature-ids")
    _edit_manifest(root, lambda manifest: manifest.__setitem__("feature_ids", [8, 8, 21]))
    with pytest.raises(ValueError, match="feature_ids must be unique"):
        FeatureBundleReader.open(root)


def test_open_rejects_array_metadata_provenance_and_artifact_corruption(tmp_path):
    root = _bundle(tmp_path)
    array = np.load(root / "z_a.npy")
    array[0, 0] = 100
    np.save(root / "z_a.npy", array)
    with pytest.raises(ValueError, match="dataset_hash"):
        FeatureBundleReader.open(root)

    root = _bundle(tmp_path / "dtype")
    np.save(root / "z_a.npy", np.ones((5, 3), dtype=np.float64))
    with pytest.raises(ValueError, match="dtype.*expected canonical"):
        FeatureBundleReader.open(root)

    root = _bundle(tmp_path / "metadata")
    _edit_manifest(
        root,
        lambda manifest: manifest["metadata_types"].__setitem__("score", "bool"),
    )
    with pytest.raises(ValueError, match="disagrees with metadata_types"):
        FeatureBundleReader.open(root)

    root = _bundle(tmp_path / "provenance")
    _edit_manifest(
        root,
        lambda manifest: manifest["provenance"]["views"].__setitem__(
            "not_a_view", {"code_semantics": "custom"}),
    )
    with pytest.raises(ValueError, match="unknown views"):
        FeatureBundleReader.open(root)

    root = _bundle(tmp_path / "extra")
    (root / "stale.txt").write_text("stale")
    with pytest.raises(ValueError, match="undeclared artifacts"):
        FeatureBundleReader.open(root)


def test_open_rejects_duplicate_row_ids_even_when_twins_match(tmp_path):
    root = _bundle(tmp_path)
    import pandas as pd

    metadata = pd.read_parquet(root / "meta.parquet")
    metadata.loc[1, "row_id"] = "r0"
    metadata.to_parquet(root / "meta.parquet", index=False)
    metadata.to_parquet(root / "battles.parquet", index=False)
    with pytest.raises(ValueError, match="row_ids must be unique"):
        FeatureBundleReader.open(root)



def test_open_rejects_root_and_member_symlinks(tmp_path):
    root = _bundle(tmp_path)
    root_link = tmp_path / "root-link"
    root_link.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="root must not be a symlink"):
        FeatureBundleReader.open(root_link)

    root = _bundle(tmp_path / "member")
    array_path = root / "z_a.npy"
    target = tmp_path / "external.npy"
    target.write_bytes(array_path.read_bytes())
    array_path.unlink()
    array_path.symlink_to(target)
    with pytest.raises(ValueError, match="member.*regular file.*symlink"):
        FeatureBundleReader.open(root)


def test_open_rejects_duplicate_nonfinite_and_extra_manifest_fields(tmp_path):
    root = _bundle(tmp_path)
    manifest_path = root / "manifest.json"
    text = manifest_path.read_text().rstrip()
    manifest_path.write_text(text[:-1] + ', "schema_version": 2}')
    with pytest.raises(ValueError, match="duplicate JSON key 'schema_version'"):
        FeatureBundleReader.open(root)

    root = _bundle(tmp_path / "nan")
    manifest_path = root / "manifest.json"
    text = manifest_path.read_text().rstrip()
    manifest_path.write_text(text[:-1] + ', "poison": NaN}')
    with pytest.raises(ValueError, match="non-finite JSON value NaN"):
        FeatureBundleReader.open(root)

    root = _bundle(tmp_path / "extra")
    _edit_manifest(root, lambda manifest: manifest.__setitem__("future_field", True))
    with pytest.raises(ValueError, match=r"manifest keys.*extra=.*future_field"):
        FeatureBundleReader.open(root)


def test_open_preflights_configurable_dimension_and_product_limits(tmp_path):
    root = _bundle(tmp_path)
    with pytest.raises(ValueError, match="n_rows 5 exceeds configured max_rows 4"):
        FeatureBundleReader.open(root, max_rows=4)
    with pytest.raises(ValueError, match="feature_width 3 exceeds configured max_features 2"):
        FeatureBundleReader.open(root, max_features=2)
    with pytest.raises(ValueError, match=r"n_rows \* feature_width.*max_elements 14"):
        FeatureBundleReader.open(root, max_elements=14)

    source = FeatureBundleReader.open(
        root, max_rows=5, max_features=3, max_elements=15)
    assert source.n_rows == 5
    with pytest.raises(ValueError, match="max_rows must be a positive integer"):
        FeatureBundleReader.open(root, max_rows=True)


def test_hash_finiteness_scan_is_bounded_and_single_pass(tmp_path, monkeypatch):
    n_rows = 9_001
    batch = FeatureBatch(
        row_ids=tuple(f"r{index}" for index in range(n_rows)),
        arrays={"z": np.arange(n_rows * 2, dtype=np.float32).reshape(n_rows, 2)},
        roles={"z": "custom"},
        feature_ids=(0, 1),
    )
    root = save_feature_batch(batch, tmp_path / "large")
    observed_rows = []
    original_isfinite = provenance_module.np.isfinite

    def recording_isfinite(values, *args, **kwargs):
        array = np.asarray(values)
        if array.ndim == 2:
            observed_rows.append(array.shape[0])
        return original_isfinite(values, *args, **kwargs)

    monkeypatch.setattr(provenance_module.np, "isfinite", recording_isfinite)
    source = FeatureBundleReader.open(root)
    assert source.n_rows == n_rows
    assert observed_rows == [4_096, 4_096, 809]


def test_open_detects_array_member_mutation_during_validation(tmp_path, monkeypatch):
    root = _bundle(tmp_path)
    original_hash = source_module.ordered_dataset_hash

    def mutating_hash(metadata, arrays):
        result = original_hash(metadata, arrays)
        path = root / "z_a.npy"
        current = path.stat()
        os.utime(
            path,
            ns=(current.st_atime_ns, current.st_mtime_ns + 1_000_000_000),
        )
        return result

    monkeypatch.setattr(source_module, "ordered_dataset_hash", mutating_hash)
    with pytest.raises(ValueError, match="changed while opening: z_a.npy"):
        FeatureBundleReader.open(root)


def test_selected_row_chunks_are_bounded_and_metadata_aligned(tmp_path):
    source = FeatureBundleReader.open(_bundle(tmp_path))
    chunks = list(source.iter_chunks(
        2,
        views=("z_a",),
        row_ids=("r4", "r1", "r3"),
    ))
    assert [chunk.row_ids for chunk in chunks] == [("r4", "r1"), ("r3",)]
    assert [chunk.positions for chunk in chunks] == [(4, 1), (3,)]
    assert chunks[0].row_slice is None
    assert chunks[0].metadata["group_id"] == (None, "g0")
    assert chunks[1].metadata["score"] == (0.25,)
    assert chunks[0].array("z_a").shape == (2, 3)
    assert not chunks[0].array("z_a").flags.writeable
    np.testing.assert_array_equal(
        chunks[0].array("z_a"),
        source.array("z_a")[[4, 1]],
    )



def test_hash_chunking_honors_element_budget_for_very_wide_rows(monkeypatch):
    import pandas as pd

    metadata = pd.DataFrame({"row_id": ["a", "b"]})
    values = np.arange(40, dtype=np.float32).reshape(2, 20)
    expected = provenance_module.ordered_dataset_hash(metadata, {"wide": values})
    observed_sizes = []
    original_isfinite = provenance_module.np.isfinite

    def recording_isfinite(chunk, *args, **kwargs):
        array = np.asarray(chunk)
        if array.ndim == 2:
            observed_sizes.append(array.size)
        return original_isfinite(chunk, *args, **kwargs)

    monkeypatch.setattr(provenance_module.np, "isfinite", recording_isfinite)
    observed = provenance_module.ordered_dataset_hash(
        metadata,
        {"wide": values},
        chunk_rows=4_096,
        chunk_elements=7,
    )
    assert observed == expected
    assert observed_sizes
    assert max(observed_sizes) <= 7


def test_open_preflights_manifest_and_aggregate_parquet_byte_limits(tmp_path):
    root = _bundle(tmp_path)
    manifest_size = (root / "manifest.json").stat().st_size
    with pytest.raises(ValueError, match="max_manifest_bytes"):
        FeatureBundleReader.open(root, max_manifest_bytes=manifest_size - 1)

    metadata_file_bytes = (root / "meta.parquet").stat().st_size
    with pytest.raises(ValueError, match="max_metadata_file_bytes"):
        FeatureBundleReader.open(
            root, max_metadata_file_bytes=metadata_file_bytes - 1)

    compressed_bytes = sum(
        (root / name).stat().st_size
        for name in ("meta.parquet", "battles.parquet")
    )
    with pytest.raises(ValueError, match="max_metadata_compressed_bytes"):
        FeatureBundleReader.open(
            root, max_metadata_compressed_bytes=compressed_bytes - 1)


def test_open_preflights_aggregate_parquet_structure_before_pandas(
    tmp_path, monkeypatch,
):
    import pyarrow.parquet as parquet

    root = _bundle(tmp_path)
    parquet_metadata = parquet.ParquetFile(root / "meta.parquet").metadata
    columns = int(parquet_metadata.num_columns)
    metadata_cells = int(parquet_metadata.num_rows) * columns
    row_groups = int(parquet_metadata.num_row_groups)
    uncompressed = 2 * sum(
        int(parquet_metadata.row_group(group).column(column).total_uncompressed_size)
        for group in range(parquet_metadata.num_row_groups)
        for column in range(parquet_metadata.num_columns)
    )

    calls = 0
    original_read_parquet = source_module.pd.read_parquet

    def recording_read_parquet(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_read_parquet(*args, **kwargs)

    monkeypatch.setattr(source_module.pd, "read_parquet", recording_read_parquet)
    with pytest.raises(ValueError, match="max_metadata_columns"):
        FeatureBundleReader.open(root, max_metadata_columns=columns - 1)
    with pytest.raises(ValueError, match="max_metadata_cells"):
        FeatureBundleReader.open(root, max_metadata_cells=metadata_cells - 1)
    with pytest.raises(ValueError, match="max_metadata_row_groups"):
        FeatureBundleReader.open(root, max_metadata_row_groups=row_groups - 1)
    with pytest.raises(ValueError, match="max_metadata_uncompressed_bytes"):
        FeatureBundleReader.open(
            root, max_metadata_uncompressed_bytes=uncompressed - 1)
    assert calls == 0

    FeatureBundleReader.open(root)
    assert calls == 1


def test_open_requires_byte_identical_parquet_twins(tmp_path):
    import pandas as pd

    root = _bundle(tmp_path)
    metadata = pd.read_parquet(root / "battles.parquet")
    metadata.to_parquet(root / "battles.parquet", index=False, compression="gzip")
    with pytest.raises(ValueError, match="byte-for-byte identical"):
        FeatureBundleReader.open(root)


def _read_only(values) -> np.ndarray:
    array = np.asarray(values)
    array.setflags(write=False)
    return array


def test_direct_feature_chunk_validates_width_dtype_and_finiteness():
    valid = FeatureChunk(
        row_ids=("a",),
        positions=range(0, 1),
        arrays={
            "numeric": _read_only([[1.0, 2.0]]),
            "boolean": _read_only([[True, False]]),
        },
    )
    assert tuple(valid.arrays) == ("numeric", "boolean")

    with pytest.raises(ValueError, match="equal feature widths"):
        FeatureChunk(
            row_ids=("a",), positions=range(0, 1),
            arrays={
                "left": _read_only([[1.0]]),
                "right": _read_only([[1.0, 2.0]]),
            },
        )
    with pytest.raises(ValueError, match="real numeric/boolean"):
        FeatureChunk(
            row_ids=("a",), positions=range(0, 1),
            arrays={"text": _read_only([["unsafe"]])},
        )
    with pytest.raises(ValueError, match="real numeric/boolean"):
        FeatureChunk(
            row_ids=("a",), positions=range(0, 1),
            arrays={"complex": _read_only([[1.0 + 2.0j]])},
        )
    with pytest.raises(ValueError, match="finite"):
        FeatureChunk(
            row_ids=("a",), positions=range(0, 1),
            arrays={"nonfinite": _read_only([[np.inf]])},
        )


def test_open_detects_root_identity_mutation_during_validation(tmp_path, monkeypatch):
    root = _bundle(tmp_path)
    original_hash = source_module.ordered_dataset_hash

    def mutating_hash(metadata, arrays):
        result = original_hash(metadata, arrays)
        current = root.stat()
        os.utime(
            root,
            ns=(current.st_atime_ns, current.st_mtime_ns + 1_000_000_000),
        )
        return result

    monkeypatch.setattr(source_module, "ordered_dataset_hash", mutating_hash)
    with pytest.raises(ValueError, match="root changed while opening"):
        FeatureBundleReader.open(root)



def test_open_rejects_trailing_bytes_in_npy_member(tmp_path):
    root = _bundle(tmp_path)
    with (root / "z_a.npy").open("ab") as handle:
        handle.write(b"undeclared-trailing-data")
    with pytest.raises(ValueError, match="file size disagrees with its array header"):
        FeatureBundleReader.open(root)
