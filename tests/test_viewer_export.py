import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from prefscope import FeatureBatch, FeatureCatalog
from prefscope.viewer_export import export_viewer_bundle
from prefscope.viewer_export import export as export_module
from prefscope.viewer_export.cli import build_parser, main


def _features() -> FeatureBatch:
    return FeatureBatch(
        row_ids=("a", "b"),
        feature_ids=(3, 7),
        arrays={"z_a": np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)},
        roles={"z_a": "response"},
        orientations={"z_a": "absolute"},
        activation_polarity="nonnegative",
        code_semantics="numerical_activity",
    )


def _viewer_dist(tmp_path: Path, **build_overrides) -> Path:
    viewer_dist = tmp_path / "viewer-dist"
    (viewer_dist / "assets").mkdir(parents=True)
    (viewer_dist / "index.html").write_text("<main>PrefScope</main>\n")
    (viewer_dist / "assets/app.js").write_bytes(b"console.log('viewer');\n")
    build = {
        "schema": "prefscope.viewer_build",
        "schema_version": 1,
        "package": "@prefscope/viewer",
        "version": "0.4.2",
        "supported_data_schemas": [
            {"schema": "prefscope.viewer_data", "versions": [1, 2]}
        ],
    }
    build.update(build_overrides)
    (viewer_dist / "viewer-build.json").write_text(
        json.dumps(build, indent=2) + "\n", encoding="utf-8"
    )
    return viewer_dist


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_viewer_export_packages_exact_build_and_existing_data(tmp_path):
    features = _features()
    catalog = FeatureCatalog(
        pd.DataFrame({"feature_id": [3, 7], "name": ["alpha", None]})
    )
    table = pd.DataFrame(
        {4: [2], 9: [1]},
        index=pd.Index([3], name="feature_left"),
    )
    table.columns.name = "feature_right"
    viewer_dist = _viewer_dist(tmp_path)
    source_bytes = {
        path.relative_to(viewer_dist).as_posix(): path.read_bytes()
        for path in viewer_dist.rglob("*")
        if path.is_file()
    }

    bundle = export_viewer_bundle(
        features,
        tmp_path / "bundle",
        viewer_dist=viewer_dist,
        catalog=catalog,
        tables={"custom": table},
    )

    assert bundle == tmp_path / "bundle"
    for relative, content in source_bytes.items():
        assert (bundle / relative).read_bytes() == content
    data_path = bundle / "data/viewer-data.json"
    payload = json.loads(data_path.read_text())
    assert payload["schema"] == "prefscope.viewer_data"
    assert payload["schema_version"] == 2
    assert payload["prompt"] is None
    assert payload["feature_ids"] == [3, 7]
    assert payload["feature_space"] == {
        "feature_space_id": None,
        "feature_space_status": "unbound",
    }
    assert payload["views"]["z_a"]["values"] == [[1.0, 0.0], [0.0, 2.0]]
    assert payload["catalog"]["table"]["columns"] == ["feature_id", "name"]
    assert payload["catalog"]["feature_space"] == {
        "feature_space_id": None,
        "feature_space_status": "unbound",
    }
    assert payload["catalog"]["provenance"]["feature_space_status"] == "unbound"
    assert payload["tables"]["custom"] == {
        "index": [3],
        "index_names": ["feature_left"],
        "columns": [4, 9],
        "column_names": ["feature_right"],
        "data": [[2, 1]],
    }
    assert "coactivation" not in payload
    assert "map" not in payload

    manifest = json.loads((bundle / "viewer-bundle.json").read_text())
    assert manifest["schema"] == "prefscope.viewer_bundle"
    assert manifest["schema_version"] == 1
    from prefscope import __version__

    assert manifest["producer"] == {"package": "prefscope", "version": __version__}
    assert manifest["viewer"] == {
        "package": "@prefscope/viewer",
        "version": "0.4.2",
        "build": {
            "path": "viewer-build.json",
            "schema": "prefscope.viewer_build",
            "schema_version": 1,
            "supported_data_schemas": [
                {"schema": "prefscope.viewer_data", "versions": [1, 2]}
            ],
        },
        "build_sha256": manifest["viewer"]["build_sha256"],
    }
    assert manifest["data"] == {
        "path": "data/viewer-data.json",
        "schema": "prefscope.viewer_data",
        "schema_version": 2,
    }

    inventory = manifest["files"]
    assert [item["path"] for item in inventory] == [
        "assets/app.js",
        "data/viewer-data.json",
        "index.html",
        "viewer-build.json",
    ]
    for item in inventory:
        path = bundle / item["path"]
        assert item["size_bytes"] == path.stat().st_size
        assert item["sha256"] == _sha256(path)
    assert "viewer-bundle.json" not in {item["path"] for item in inventory}

    viewer_inventory = [
        item for item in inventory if item["path"] != "data/viewer-data.json"
    ]
    canonical = json.dumps(
        viewer_inventory,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    assert manifest["viewer"]["build_sha256"] == hashlib.sha256(canonical).hexdigest()


def test_viewer_export_build_identity_is_independent_of_destination(tmp_path):
    viewer_dist = _viewer_dist(tmp_path)
    first = export_viewer_bundle(
        _features(), tmp_path / "first", viewer_dist=viewer_dist
    )
    second = export_viewer_bundle(
        _features(), tmp_path / "second", viewer_dist=viewer_dist
    )
    first_manifest = json.loads((first / "viewer-bundle.json").read_text())
    second_manifest = json.loads((second / "viewer-bundle.json").read_text())
    assert (
        first_manifest["viewer"]["build_sha256"]
        == second_manifest["viewer"]["build_sha256"]
    )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"schema": "other"}, "schema must"),
        ({"schema_version": 2}, "schema_version must be 1"),
        ({"package": "other"}, "package must"),
        ({"version": 3}, "version must be a non-empty string"),
        ({"supported_data_schemas": []}, "must support prefscope.viewer_data v2"),
        (
            {
                "supported_data_schemas": [
                    {"schema": "prefscope.viewer_data", "versions": [1]}
                ]
            },
            "must support prefscope.viewer_data v2",
        ),
    ],
)
def test_viewer_export_rejects_incompatible_viewer_build(tmp_path, override, message):
    viewer_dist = _viewer_dist(tmp_path, **override)
    with pytest.raises(ValueError, match=message):
        export_viewer_bundle(_features(), tmp_path / "bundle", viewer_dist=viewer_dist)
    assert not (tmp_path / "bundle").exists()


def test_viewer_export_requires_root_index(tmp_path):
    viewer_dist = tmp_path / "viewer-dist"
    viewer_dist.mkdir()
    with pytest.raises(ValueError, match="must contain index.html at its root"):
        export_viewer_bundle(_features(), tmp_path / "bundle", viewer_dist=viewer_dist)


def test_viewer_export_requires_root_build_manifest(tmp_path):
    viewer_dist = tmp_path / "viewer-dist"
    viewer_dist.mkdir()
    (viewer_dist / "index.html").write_text("<main>PrefScope</main>\n")
    with pytest.raises(ValueError, match="must contain viewer-build.json at its root"):
        export_viewer_bundle(_features(), tmp_path / "bundle", viewer_dist=viewer_dist)


def test_viewer_export_rejects_malformed_build_manifest(tmp_path):
    viewer_dist = _viewer_dist(tmp_path)
    (viewer_dist / "viewer-build.json").write_text("not json")
    with pytest.raises(ValueError, match="invalid viewer-build.json"):
        export_viewer_bundle(_features(), tmp_path / "bundle", viewer_dist=viewer_dist)


def test_viewer_export_rejects_destination_inside_viewer_dist(tmp_path):
    viewer_dist = _viewer_dist(tmp_path)
    with pytest.raises(ValueError, match="must not be inside viewer_dist"):
        export_viewer_bundle(
            _features(), viewer_dist / "bundle", viewer_dist=viewer_dist
        )
    assert not list(viewer_dist.glob(".bundle.staging-*"))


def test_viewer_export_rejects_source_root_swap_during_copy(tmp_path, monkeypatch):
    viewer_dist = _viewer_dist(tmp_path)
    original_tree = tmp_path / "original-tree"
    outside_tree = tmp_path / "outside-tree"
    shutil.copytree(viewer_dist, outside_tree)
    (outside_tree / "secret.txt").write_text("must not be published")
    real_copytree = shutil.copytree

    def swap_then_copy(source, destination, *args, **kwargs):
        if Path(source) == viewer_dist:
            Path(source).rename(original_tree)
            Path(source).symlink_to(outside_tree, target_is_directory=True)
        return real_copytree(source, destination, *args, **kwargs)

    monkeypatch.setattr(export_module.shutil, "copytree", swap_then_copy)
    with pytest.raises(ValueError, match="changed while it was being copied"):
        export_viewer_bundle(
            _features(), tmp_path / "bundle", viewer_dist=viewer_dist
        )
    assert not (tmp_path / "bundle").exists()
    assert not list(tmp_path.glob(".bundle.staging-*"))


def test_viewer_export_exclusive_publish_does_not_replace_raced_destination(
    tmp_path, monkeypatch
):
    viewer_dist = _viewer_dist(tmp_path)
    real_publish = export_module._publish_directory_no_replace
    destination = tmp_path / "bundle"

    def race(source, target):
        destination.mkdir()
        (destination / "sentinel.txt").write_text("keep")
        return real_publish(source, target)

    monkeypatch.setattr(export_module, "_publish_directory_no_replace", race)
    with pytest.raises(FileExistsError):
        export_viewer_bundle(
            _features(), destination, viewer_dist=viewer_dist
        )
    assert (destination / "sentinel.txt").read_text() == "keep"
    assert not list(tmp_path.glob(".bundle.staging-*"))


def test_viewer_export_removes_staging_directory_after_failure(tmp_path, monkeypatch):
    viewer_dist = _viewer_dist(tmp_path)
    real_inventory = export_module._inventory
    calls = 0

    def fail_second_inventory(root, paths):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected failure")
        return real_inventory(root, paths)

    monkeypatch.setattr(export_module, "_inventory", fail_second_inventory)
    with pytest.raises(RuntimeError, match="injected failure"):
        export_viewer_bundle(_features(), tmp_path / "bundle", viewer_dist=viewer_dist)
    assert not (tmp_path / "bundle").exists()
    assert not list(tmp_path.glob(".bundle.staging-*"))


def test_viewer_export_does_not_overwrite_existing_destination(tmp_path):
    viewer_dist = _viewer_dist(tmp_path)
    out = tmp_path / "bundle"
    out.mkdir()
    sentinel = out / "keep.txt"
    sentinel.write_text("keep")

    with pytest.raises(FileExistsError, match="destination already exists"):
        export_viewer_bundle(_features(), out, viewer_dist=viewer_dist)

    assert sentinel.read_text() == "keep"
    assert not list(tmp_path.glob(".bundle.staging-*"))


def test_viewer_export_rejects_any_data_in_viewer_build(tmp_path):
    viewer_dist = _viewer_dist(tmp_path)
    (viewer_dist / "data").mkdir()
    (viewer_dist / "data/stale-v2-artifact.json").write_text("old")
    with pytest.raises(ValueError, match="reserved path data"):
        export_viewer_bundle(_features(), tmp_path / "bundle", viewer_dist=viewer_dist)


@pytest.mark.parametrize("link_kind", ["file", "directory"])
def test_viewer_export_rejects_symbolic_links(tmp_path, link_kind):
    viewer_dist = _viewer_dist(tmp_path)
    if link_kind == "file":
        outside = tmp_path / "outside-secret.txt"
        outside.write_text("must not be published")
        link = viewer_dist / "linked-secret.txt"
        link.symlink_to(outside)
    else:
        outside = tmp_path / "outside-directory"
        outside.mkdir()
        (outside / "secret.txt").write_text("must not be published")
        link = viewer_dist / "linked-directory"
        link.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="must not contain symbolic links"):
        export_viewer_bundle(_features(), tmp_path / "bundle", viewer_dist=viewer_dist)
    assert not (tmp_path / "bundle").exists()


def test_viewer_export_rejects_symlinked_root(tmp_path):
    viewer_dist = _viewer_dist(tmp_path)
    linked_root = tmp_path / "linked-viewer"
    linked_root.symlink_to(viewer_dist, target_is_directory=True)
    with pytest.raises(ValueError, match="viewer_dist must not be a symbolic link"):
        export_viewer_bundle(
            _features(), tmp_path / "bundle", viewer_dist=linked_root
        )


def test_viewer_export_rejects_special_files(tmp_path):
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO creation is unavailable")

    viewer_dist = _viewer_dist(tmp_path)
    os.mkfifo(viewer_dist / "unexpected.fifo")
    with pytest.raises(ValueError, match="only regular files and directories"):
        export_viewer_bundle(
            _features(), tmp_path / "bundle", viewer_dist=viewer_dist
        )


def test_viewer_export_accepts_read_only_viewer_root(tmp_path):
    viewer_dist = _viewer_dist(tmp_path)
    original_mode = viewer_dist.stat().st_mode
    viewer_dist.chmod(0o555)
    try:
        bundle = export_viewer_bundle(
            _features(), tmp_path / "bundle", viewer_dist=viewer_dist
        )
    finally:
        viewer_dist.chmod(original_mode)
    assert (bundle / "data/viewer-data.json").is_file()


def test_viewer_export_rejects_outer_manifest_in_viewer_build(tmp_path):
    viewer_dist = _viewer_dist(tmp_path)
    (viewer_dist / "viewer-bundle.json").write_text("old")
    with pytest.raises(ValueError, match="reserved path viewer-bundle.json"):
        export_viewer_bundle(_features(), tmp_path / "bundle", viewer_dist=viewer_dist)


def test_viewer_export_rejects_feature_ids_that_json_cannot_preserve(tmp_path):
    features = FeatureBatch(
        row_ids=("a",),
        feature_ids=(2**53,),
        arrays={"z_a": np.array([[1.0]], dtype=np.float32)},
        roles={"z_a": "response"},
    )
    with pytest.raises(ValueError, match="JavaScript safe integers"):
        export_viewer_bundle(
            features,
            tmp_path / "bundle",
            viewer_dist=_viewer_dist(tmp_path),
        )
    assert not (tmp_path / "bundle").exists()


def test_viewer_export_allows_partial_catalog_with_explicit_id_join(tmp_path):
    catalog = FeatureCatalog(pd.DataFrame({"feature_id": [7], "name": ["partial"]}))
    bundle = export_viewer_bundle(
        _features(),
        tmp_path / "bundle",
        viewer_dist=_viewer_dist(tmp_path),
        catalog=catalog,
    )
    payload = json.loads((bundle / "data/viewer-data.json").read_text())
    assert payload["catalog"]["table"]["data"] == [[7, "partial"]]


def test_viewer_export_rejects_catalog_ids_outside_batch(tmp_path):
    catalog = FeatureCatalog(
        pd.DataFrame({"feature_id": [99], "name": ["unrelated"]})
    )
    with pytest.raises(ValueError, match="outside the exported batch"):
        export_viewer_bundle(
            _features(),
            tmp_path / "bundle",
            viewer_dist=_viewer_dist(tmp_path),
            catalog=catalog,
        )


def test_viewer_export_rejects_catalog_from_another_feature_space(tmp_path):
    features = FeatureBatch(
        row_ids=("a",),
        feature_ids=(0,),
        arrays={"z_a": np.array([[1.0]], dtype=np.float32)},
        roles={"z_a": "response"},
        orientations={"z_a": "absolute"},
        provenance={
            "lens": {
                "feature_space_id": "sha256:space-a",
                "feature_space_status": "exact_weights",
            }
        },
    )
    catalog = FeatureCatalog(
        pd.DataFrame({"feature_id": [0], "name": ["other"]}),
        provenance={
            "feature_space_id": "sha256:space-b",
            "feature_space_status": "exact_weights",
        },
    )
    with pytest.raises(ValueError, match="different feature spaces"):
        export_viewer_bundle(
            features,
            tmp_path / "bundle",
            viewer_dist=_viewer_dist(tmp_path),
            catalog=catalog,
        )


def test_viewer_export_preserves_table_float_precision(tmp_path):
    values = [
        0.049999999999,
        0.050000000001,
        1.2345678901234568e-15,
        np.nextafter(1.0, 2.0),
        1e20,
    ]
    bundle = export_viewer_bundle(
        _features(),
        tmp_path / "bundle",
        viewer_dist=_viewer_dist(tmp_path),
        tables={"precise": pd.DataFrame({"value": values})},
    )
    payload = json.loads((bundle / "data/viewer-data.json").read_text())
    assert [row[0] for row in payload["tables"]["precise"]["data"]] == values


def test_viewer_export_distinguishes_missing_and_nonfinite_table_values(tmp_path):
    viewer_dist = _viewer_dist(tmp_path)
    bundle = export_viewer_bundle(
        _features(),
        tmp_path / "missing-bundle",
        viewer_dist=viewer_dist,
        tables={"missing": pd.DataFrame({"value": [np.nan]})},
    )
    payload = json.loads((bundle / "data/viewer-data.json").read_text())
    assert payload["tables"]["missing"]["data"] == [[None]]

    with pytest.raises(ValueError, match="must be finite or missing"):
        export_viewer_bundle(
            _features(),
            tmp_path / "infinite-bundle",
            viewer_dist=viewer_dist,
            tables={"infinite": pd.DataFrame({"value": [np.inf]})},
        )


def test_viewer_export_enforces_safe_integer_boundary_in_caller_data(tmp_path):
    viewer_dist = _viewer_dist(tmp_path)
    safe_values = [2**53 - 1, -(2**53 - 1)]
    safe = export_viewer_bundle(
        _features(),
        tmp_path / "safe-bundle",
        viewer_dist=viewer_dist,
        tables={"safe": pd.DataFrame({"value": safe_values})},
    )
    payload = json.loads((safe / "data/viewer-data.json").read_text())
    assert [row[0] for row in payload["tables"]["safe"]["data"]] == safe_values

    with pytest.raises(ValueError, match="JavaScript safe integer"):
        export_viewer_bundle(
            _features(),
            tmp_path / "unsafe-bundle",
            viewer_dist=viewer_dist,
            tables={"unsafe": pd.DataFrame({"value": [2**53]})},
        )


def test_viewer_export_preserves_duplicate_table_labels_and_time_scalars(tmp_path):
    table = pd.DataFrame(
        [[pd.Timestamp("2026-09-03T12:34:56Z"), pd.Timedelta("1 day 2 hours")]],
        columns=["value", "value"],
    )
    bundle = export_viewer_bundle(
        _features(),
        tmp_path / "bundle",
        viewer_dist=_viewer_dist(tmp_path),
        tables={"time_values": table},
    )
    payload = json.loads((bundle / "data/viewer-data.json").read_text())
    encoded = payload["tables"]["time_values"]
    assert encoded["columns"] == ["value", "value"]
    assert encoded["data"] == [["2026-09-03T12:34:56+00:00", "P1DT2H0M0S"]]


def test_viewer_export_preserves_multiindex_column_names(tmp_path):
    columns = pd.MultiIndex.from_tuples(
        [("left", 1), ("right", 2)], names=("side", "feature_id")
    )
    table = pd.DataFrame([[3, 4]], columns=columns)
    bundle = export_viewer_bundle(
        _features(),
        tmp_path / "bundle",
        viewer_dist=_viewer_dist(tmp_path),
        tables={"multi": table},
    )
    payload = json.loads((bundle / "data/viewer-data.json").read_text())
    assert payload["tables"]["multi"]["column_names"] == ["side", "feature_id"]


def test_viewer_cli_requires_viewer_dist_and_describes_out_as_directory():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--features", "features", "--out", "bundle"])
    args = parser.parse_args(
        [
            "--features",
            "features",
            "--viewer-dist",
            "viewer-dist",
            "--out",
            "bundle",
        ]
    )
    assert args.viewer_dist == "viewer-dist"
    assert args.out == "bundle"
    assert "directory" in parser.format_help()


def test_viewer_cli_forwards_viewer_dist_and_output_directory(tmp_path, monkeypatch):
    viewer_dist = _viewer_dist(tmp_path)
    features = _features()
    received = {}

    monkeypatch.setattr(
        "prefscope.viewer_export.cli.load_feature_batch", lambda path: features
    )

    def fake_export(value, out, **kwargs):
        received.update(features=value, out=out, **kwargs)
        return Path(out)

    monkeypatch.setattr("prefscope.viewer_export.cli.export_viewer_bundle", fake_export)
    assert (
        main(
            [
                "--features",
                "features",
                "--viewer-dist",
                str(viewer_dist),
                "--out",
                str(tmp_path / "bundle"),
            ]
        )
        == 0
    )
    assert received["features"] is features
    assert received["viewer_dist"] == str(viewer_dist)
    assert received["out"] == str(tmp_path / "bundle")


def _prompt_features(**overrides) -> FeatureBatch:
    kwargs = dict(
        row_ids=("a", "b"),
        feature_ids=(7,),
        arrays={"z_prompt": np.array([[0.5], [1.5]], dtype=np.float32)},
        roles={"z_prompt": "prompt"},
        orientations={"z_prompt": "absolute"},
        provenance={
            "lens": {
                "feature_space_id": "sha256:prompt-space",
                "feature_space_status": "exact_weights",
            }
        },
    )
    kwargs.update(overrides)
    return FeatureBatch(**kwargs)


@pytest.mark.parametrize(
    "roles", [("response_a", "response_b"), ("response_difference",), ("response_a",)]
)
def test_viewer_export_keeps_prompt_space_separate_for_response_modes(tmp_path, roles):
    features = FeatureBatch(
        row_ids=("a", "b"),
        feature_ids=(3, 7),
        arrays={role: _features().array("z_a") for role in roles},
        roles={role: role for role in roles},
        provenance={
            "lens": {
                "feature_space_id": "sha256:response-space",
                "feature_space_status": "exact_weights",
            }
        },
        metadata={
            "prompt": ("Question A", "Question B"),
            "response_a": ("First A", "First B"),
            "response_b": ("Second A", "Second B"),
            "model_a": ("supplied-a", "supplied-a"),
            "preference_probability": (0.25, 0.75),
        },
    )
    prompt = _prompt_features()
    catalog = FeatureCatalog(pd.DataFrame({"feature_id": [7], "name": ["response"]}))
    prompt_catalog = FeatureCatalog(
        pd.DataFrame({"feature_id": [7], "name": ["prompt"]}),
        provenance=dict(prompt.provenance["lens"]),
    )
    tables = {
        "feature_map": pd.DataFrame({"feature_id": [7], "x": [1.0], "y": [2.0]}),
        "prompt_feature_map": pd.DataFrame({"feature_id": [7], "x": [3.0], "y": [4.0]}),
    }
    bundle = export_viewer_bundle(
        features, tmp_path / "bundle", viewer_dist=_viewer_dist(tmp_path),
        catalog=catalog, prompt_features=prompt, prompt_catalog=prompt_catalog,
        tables=tables,
    )
    payload = json.loads((bundle / "data/viewer-data.json").read_text())
    assert set(payload) == {
        "schema", "schema_version", "row_ids", "feature_ids", "feature_space",
        "views", "row_metadata", "provenance", "catalog", "tables", "prompt",
    }
    assert payload["schema_version"] == 2
    assert payload["prompt"] == export_module._viewer_data(prompt, prompt_catalog, None)
    assert payload["prompt"]["schema_version"] == 1
    assert "prompt" not in payload["prompt"]
    assert payload["row_ids"] == payload["prompt"]["row_ids"] == ["a", "b"]
    assert payload["feature_ids"] == [3, 7]
    assert payload["prompt"]["feature_ids"] == [7]
    assert payload["feature_space"]["feature_space_id"] == "sha256:response-space"
    assert payload["prompt"]["feature_space"]["feature_space_id"] == "sha256:prompt-space"
    assert payload["catalog"]["table"]["data"] == [[7, "response"]]
    assert payload["prompt"]["catalog"]["table"]["data"] == [[7, "prompt"]]
    assert payload["tables"]["feature_map"]["data"] == [[7, 1.0, 2.0]]
    assert payload["tables"]["prompt_feature_map"]["data"] == [[7, 3.0, 4.0]]
    assert payload["row_metadata"]["preference_probability"] == [0.25, 0.75]
    assert "model_b" not in payload["row_metadata"]
    manifest = json.loads((bundle / "viewer-bundle.json").read_text())
    entry = next(item for item in manifest["files"] if item["path"] == "data/viewer-data.json")
    assert entry["sha256"] == _sha256(bundle / entry["path"])


def test_viewer_export_rejects_orphan_prompt_catalog(tmp_path):
    with pytest.raises(ValueError, match="prompt_catalog requires prompt_features"):
        export_viewer_bundle(
            _features(), tmp_path / "bundle", viewer_dist=_viewer_dist(tmp_path),
            prompt_catalog=FeatureCatalog(pd.DataFrame({"feature_id": [7]})),
        )
    assert not (tmp_path / "bundle").exists()
    assert not list(tmp_path.glob(".bundle.staging-*"))


@pytest.mark.parametrize("row_ids", [("b", "a"), ("a", "c")])
def test_viewer_export_rejects_unaligned_prompt_rows(tmp_path, row_ids):
    with pytest.raises(ValueError, match="row_ids must exactly match features in order"):
        export_viewer_bundle(
            _features(), tmp_path / "bundle", viewer_dist=_viewer_dist(tmp_path),
            prompt_features=_prompt_features(row_ids=row_ids),
        )
    assert not (tmp_path / "bundle").exists()
    assert not list(tmp_path.glob(".bundle.staging-*"))


def test_viewer_export_requires_every_prompt_view_to_have_prompt_role(tmp_path):
    prompt = _prompt_features(
        arrays={"z_prompt": np.ones((2, 1)), "other": np.zeros((2, 1))},
        roles={"z_prompt": "prompt", "other": "response"},
        orientations={"z_prompt": "absolute", "other": "absolute"},
    )
    with pytest.raises(ValueError, match="all prompt_features views must have role 'prompt'"):
        export_viewer_bundle(
            _features(), tmp_path / "bundle", viewer_dist=_viewer_dist(tmp_path),
            prompt_features=prompt,
        )
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize(
    "identity", [
        {"feature_space_id": 123, "feature_space_status": "exact_weights"},
        {"feature_space_id": " ", "feature_space_status": "exact_weights"},
        {"feature_space_id": None, "feature_space_status": "exact_weights"},
        {"feature_space_id": "sha256:prompt", "feature_space_status": "unbound"},
    ],
)
def test_viewer_export_rejects_invalid_prompt_identity(tmp_path, identity):
    with pytest.raises(ValueError, match="feature_space_id|feature space"):
        export_viewer_bundle(
            _features(), tmp_path / "bundle", viewer_dist=_viewer_dist(tmp_path),
            prompt_features=_prompt_features(provenance={"lens": identity}),
        )
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize(
    ("catalog", "message"), [
        (FeatureCatalog(pd.DataFrame({"feature_id": [99]})), "outside the exported batch"),
        (FeatureCatalog(
            pd.DataFrame({"feature_id": [7]}),
            provenance={"feature_space_id": "sha256:other", "feature_space_status": "exact_weights"},
        ), "different feature spaces"),
    ],
)
def test_viewer_export_validates_prompt_catalog_in_its_own_space(tmp_path, catalog, message):
    with pytest.raises(ValueError, match=message):
        export_viewer_bundle(
            _features(), tmp_path / "bundle", viewer_dist=_viewer_dist(tmp_path),
            prompt_features=_prompt_features(), prompt_catalog=catalog,
        )
    assert not (tmp_path / "bundle").exists()


def test_viewer_cli_loads_and_forwards_optional_prompt_inputs(tmp_path, monkeypatch):
    features, prompt = _features(), _prompt_features()
    catalog = FeatureCatalog(pd.DataFrame({"feature_id": [7]}))
    received = {}
    monkeypatch.setattr(
        "prefscope.viewer_export.cli.load_feature_batch",
        lambda path: {"features": features, "prompt-features": prompt}[path],
    )
    monkeypatch.setattr(
        "prefscope.viewer_export.cli.load_feature_catalog",
        lambda path: {"prompt-catalog.json": catalog}[path],
    )
    def fake_export(value, out, **kwargs):
        received.update(features=value, **kwargs)
        return Path(out)
    monkeypatch.setattr("prefscope.viewer_export.cli.export_viewer_bundle", fake_export)
    assert main([
        "--features", "features", "--prompt-features", "prompt-features",
        "--prompt-catalog", "prompt-catalog.json", "--viewer-dist", "viewer-dist",
        "--out", str(tmp_path / "bundle"),
    ]) == 0
    assert received["features"] is features
    assert received["prompt_features"] is prompt
    assert received["prompt_catalog"] is catalog


def test_viewer_export_preserves_missing_prompt_catalog_and_metadata(tmp_path):
    bundle = export_viewer_bundle(
        _features(), tmp_path / "bundle", viewer_dist=_viewer_dist(tmp_path),
        prompt_features=_prompt_features(provenance={}),
    )
    payload = json.loads((bundle / "data/viewer-data.json").read_text())
    assert payload["catalog"] is None
    assert payload["row_metadata"] == {}
    assert payload["prompt"]["catalog"] is None
    assert payload["prompt"]["row_metadata"] == {}
    assert payload["prompt"]["tables"] == {}
    assert payload["prompt"]["feature_space"] == {
        "feature_space_id": None, "feature_space_status": "unbound",
    }
