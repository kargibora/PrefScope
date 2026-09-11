import json
from pathlib import Path

import pandas as pd
import pytest

from prefscope import (
    FeatureCatalog,
    decode_feature_catalog,
    encode_feature_catalog,
    load_feature_catalog,
    save_feature_catalog,
)


def _catalog(*, bound=True):
    provenance = (
        {"feature_space_status": "exact_weights", "feature_space_id": "sha256:test"}
        if bound
        else {"feature_space_status": "unbound", "feature_space_id": None}
    )
    return FeatureCatalog(
        pd.DataFrame(
            {
                "feature_id": [0, 1, 2, 3],
                "name": ["NA", "N/A", "null", None],
            }
        ),
        provenance=provenance,
        column_sources={"name": {"kind": "test"}},
    )


@pytest.mark.parametrize("bound", [True, False])
def test_catalog_json_round_trip_preserves_annotations_and_identity(bound):
    catalog = _catalog(bound=bound)
    restored = decode_feature_catalog(encode_feature_catalog(catalog))
    pd.testing.assert_frame_equal(restored.to_frame(), catalog.to_frame())
    assert dict(restored.provenance) == dict(catalog.provenance)
    assert dict(restored.column_sources) == dict(catalog.column_sources)


def test_catalog_file_round_trip_and_explicit_overwrite(tmp_path):
    path = save_feature_catalog(_catalog(), tmp_path / "catalog.json")
    assert load_feature_catalog(path).feature_ids == (0, 1, 2, 3)
    with pytest.raises(FileExistsError):
        save_feature_catalog(_catalog(), path)
    save_feature_catalog(_catalog(bound=False), path, overwrite=True)
    assert load_feature_catalog(path).feature_space_status == "unbound"


def test_catalog_save_preserves_caller_temporary_file(tmp_path):
    caller_file = tmp_path / "catalog.json.tmp"
    caller_file.write_bytes(b"caller data")

    path = save_feature_catalog(_catalog(), tmp_path / "catalog.json")

    assert load_feature_catalog(path).feature_ids == (0, 1, 2, 3)
    assert caller_file.read_bytes() == b"caller data"
    assert set(tmp_path.iterdir()) == {path, caller_file}


def test_catalog_save_cleans_its_temporary_file_on_failure(tmp_path, monkeypatch):
    path = tmp_path / "catalog.json"
    path.write_bytes(b"existing catalog")

    def fail_replace(source, target):
        raise OSError("injected replacement failure")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="injected replacement failure"):
        save_feature_catalog(_catalog(), path, overwrite=True)

    assert path.read_bytes() == b"existing catalog"
    assert list(tmp_path.iterdir()) == [path]


def test_catalog_decoder_rejects_unknown_schema_version():
    payload = json.loads(encode_feature_catalog(_catalog()))
    payload["schema_version"] = 2
    with pytest.raises(ValueError, match="schema_version"):
        decode_feature_catalog(json.dumps(payload))
