from __future__ import annotations

import io
import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from prefscope import FeatureCatalog, FeatureMatrix, feature_activation_table
from prefscope.integrations import NeuronpediaProvider
from prefscope.api._feature_space import projector_feature_space_identity
from prefscope.presentation import FeatureTableRenderer


def _catalog(ids=(0, 1), **columns):
    return FeatureCatalog(pd.DataFrame({"feature_id": ids, **columns}))


def test_external_coordinate_is_pinned_only_by_explicit_status():
    projector = SimpleNamespace(
        m_total=2,
        projector_provenance={"coordinate_pin_status": "unknown", "coordinate": "x"},
    )
    identity = projector_feature_space_identity(
        projector, input_rep="individual", backend="external"
    )
    assert identity["feature_space_status"] == "declared_unpinned"
    projector.projector_provenance["coordinate_pin_status"] = "pinned"
    pinned = projector_feature_space_identity(
        projector, input_rep="individual", backend="external"
    )
    assert pinned["feature_space_status"] == "declared_pinned_coordinate"


def test_catalog_validates_identity_columns_and_feature_ids():
    with pytest.raises(ValueError, match="unique"):
        _catalog((1, 1))
    with pytest.raises(ValueError, match="boolean"):
        _catalog((True,))
    with pytest.raises(ValueError, match="integers"):
        _catalog((1.5,))
    with pytest.raises(ValueError, match="integers"):
        _catalog(("1",))
    with pytest.raises(ValueError, match="integers"):
        _catalog((1.0,))
    with pytest.raises(ValueError, match="display labels only"):
        _catalog((0,), semantic_threshold=[0.4])
    with pytest.raises(ValueError, match="strings or missing"):
        _catalog((0,), name=[3])
    with pytest.raises(ValueError, match="only be proposed_label"):
        _catalog((0,), evidence_layer=["semantic_presence"])
    with pytest.raises(ValueError, match="feature_space_id"):
        FeatureCatalog(
            pd.DataFrame({"feature_id": [0]}),
            provenance={"feature_space_status": "exact_weights"},
        )


def test_catalog_mapping_selection_and_labels_are_copy_safe():
    catalog = FeatureCatalog.from_mapping({7: "seven", 2: "two"}, column="name")
    assert dict(catalog.labels) == {7: "seven", 2: "two"}
    assert catalog.select((2, 7)).feature_ids == (2, 7)
    frame = catalog.to_frame()
    frame.loc[0, "name"] = "changed"
    assert catalog.labels[7] == "seven"


def test_catalog_merge_is_explicit_and_preserves_bound_identity():
    base = FeatureCatalog(
        pd.DataFrame({"feature_id": [0, 1], "name": ["zero", None]}),
        provenance={
            "feature_space_id": "space-a",
            "feature_space_status": "exact_weights",
        },
    )
    external = FeatureCatalog(pd.DataFrame({"feature_id": [1], "description": ["one"]}))
    merged = base.merge(external)
    assert merged.feature_space_id == "space-a"
    assert merged.feature_space_status == "exact_weights"
    assert dict(merged.labels) == {0: "zero", 1: "one"}
    incompatible = FeatureCatalog(
        pd.DataFrame({"feature_id": [0]}),
        provenance={
            "feature_space_id": "space-b",
            "feature_space_status": "exact_weights",
        },
    )
    with pytest.raises(ValueError, match="different feature spaces"):
        base.merge(incompatible)


def test_feature_activation_table_uses_matrix_feature_ids_not_column_positions():
    matrix = FeatureMatrix(
        np.array([[1.0, -3.0]], dtype=np.float32),
        ("row",),
        role="response_a",
        orientation="absolute_a",
        feature_ids=(7, 2),
        activation_polarity="signed",
        code_semantics="numerical_activity",
    )
    catalog = _catalog((2, 7), description=["two", "seven"])
    table = feature_activation_table(matrix, catalog=catalog, top_k=2)
    assert table["feature_id"].tolist() == [2, 7]
    assert table["activation"].tolist() == [-3.0, 1.0]
    assert table["description"].tolist() == ["two", "seven"]
    assert table["feature_orientation"].tolist() == ["absolute_a", "absolute_a"]


def test_feature_activation_table_allows_partial_catalog_annotations():
    matrix = FeatureMatrix(
        np.array([[1.0, 2.0]], dtype=np.float32),
        ("row",),
        feature_ids=(7, 2),
    )
    catalog = _catalog((2,), description=["two"])
    table = feature_activation_table(matrix, catalog=catalog)
    assert table["feature_id"].tolist() == [2, 7]
    assert table.loc[0, "description"] == "two"
    assert pd.isna(table.loc[1, "description"])


def test_feature_activation_table_handles_silent_rows_and_identity_mismatch():
    matrix = FeatureMatrix(
        np.zeros((1, 1), dtype=np.float32),
        ("row",),
        feature_ids=(4,),
        provenance={
            "lens": {
                "feature_space_id": "space-a",
                "feature_space_status": "exact_weights",
            }
        },
    )
    matching = FeatureCatalog(
        pd.DataFrame({"feature_id": [4], "name": ["four"]}),
        provenance={
            "feature_space_id": "space-a",
            "feature_space_status": "exact_weights",
        },
    )
    assert feature_activation_table(matrix, catalog=matching).empty
    mismatched = FeatureCatalog(
        pd.DataFrame({"feature_id": [4]}),
        provenance={
            "feature_space_id": "space-b",
            "feature_space_status": "exact_weights",
        },
    )
    with pytest.raises(ValueError, match="different feature spaces"):
        feature_activation_table(matrix, catalog=mismatched)


def test_catalog_from_lens_keeps_only_proposed_display_names():
    lens = SimpleNamespace(
        backend=SimpleNamespace(m_total=3),
        input_rep="individual",
        feature_table=pd.DataFrame(
            {
                "feature_id": [0, 1, 2],
                "concept": ["a", "b", "c"],
                "fidelity_pass": [True, False, True],
            }
        ),
        feature_space_identity={
            "feature_space_id": "space-a",
            "feature_space_status": "exact_weights",
        },
    )
    catalog = FeatureCatalog.from_lens(lens)
    assert list(catalog.to_frame()) == ["feature_id", "name"]
    assert dict(catalog.labels) == {0: "a", 1: "b", 2: "c"}


def test_renderer_includes_bounded_identity_for_multi_row_tables():
    table = pd.DataFrame(
        {
            "row_id": ["row-a\nsecret", "row-b-is-long"],
            "rank": [1, 1],
            "feature_id": [2, 3],
            "activation": [1.0, 0.5],
            "description": ["two", "three"],
        }
    )
    output = FeatureTableRenderer(style="plain", max_rows=2, max_row_id_chars=8).format(
        table
    )
    assert output.splitlines()[0].split()[:4] == [
        "row_id",
        "rank",
        "feature_id",
        "activation",
    ]
    assert "secret" not in output
    assert "row-b-i…" in output


def test_renderer_omits_empty_label_column():
    table = pd.DataFrame({"feature_id": [2], "activation": [1.0]})
    output = FeatureTableRenderer(style="plain").format(table)
    assert output.splitlines()[0].split() == ["feature_id", "activation"]


def test_renderer_uses_name_header_for_name_only_catalogs():
    table = pd.DataFrame(
        {"feature_id": [2], "activation": [1.0], "name": ["named feature"]}
    )
    output = FeatureTableRenderer(style="plain").format(table)
    assert "name" in output.splitlines()[0]
    assert "description" not in output.splitlines()[0]


def test_renderer_marks_external_evidence_as_proposed():
    table = pd.DataFrame(
        {
            "feature_id": [2],
            "activation": [1.0],
            "description": ["external label"],
            "evidence_layer": ["proposed_label"],
        }
    )
    output = FeatureTableRenderer(style="plain").format(table)
    assert "proposed_description" in output.splitlines()[0]


def test_plain_renderer_is_bounded_sanitized_and_does_not_import_rich(monkeypatch):
    table = pd.DataFrame(
        {
            "feature_id": [3, 4],
            "activation": [1.23456, -2.0],
            "description": ["safe" + chr(10) + "text", "x" * 200],
        }
    )
    renderer = FeatureTableRenderer(style="plain", max_rows=1, max_description_chars=20)
    before = set(sys.modules)
    text = renderer.format(table)
    assert "3" in text and "1.235" in text and "safe text" in text
    assert chr(10) + "text" not in text
    assert "rich" not in set(sys.modules) - before
    stream = io.StringIO()
    renderer.print(table, stream=stream)
    assert stream.getvalue().endswith(chr(10))


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, size):
        return self.payload


def test_neuronpedia_provider_rejects_unbounded_network_settings():
    with pytest.raises(ValueError, match="positive number"):
        NeuronpediaProvider("model/layer", timeout=float("nan"))
    with pytest.raises(ValueError, match="single-line"):
        NeuronpediaProvider("model/layer", user_agent="bad\nheader")


def test_neuronpedia_provider_returns_provenanced_catalog(monkeypatch):
    payload = json.dumps(
        {
            "explanations": [
                {"description": "unscored", "scores": []},
                {"description": "best", "scores": [{"value": 0.8}]},
            ]
        }
    ).encode()
    monkeypatch.setattr(
        "prefscope.integrations.neuronpedia.urlopen",
        lambda request, timeout: _Response(payload),
    )
    catalog = NeuronpediaProvider("model/layer").fetch((5,))
    frame = catalog.to_frame()
    assert catalog.feature_space_status == "unbound"
    assert frame.loc[0, "description"] == "best"
    assert frame.loc[0, "evidence_layer"] == "proposed_label"
    assert len(frame.loc[0, "content_sha256"]) == 64
    assert catalog.provenance["neuronpedia_id"] == "model/layer"


def test_neuronpedia_provider_derives_coordinate_and_identity_from_lens(monkeypatch):
    payload = json.dumps({"explanations": [{"description": "label"}]}).encode()
    monkeypatch.setattr(
        "prefscope.integrations.neuronpedia.urlopen",
        lambda request, timeout: _Response(payload),
    )
    metadata = SimpleNamespace(neuronpedia_id="model/layer")
    lens = SimpleNamespace(
        projector=SimpleNamespace(
            sae=SimpleNamespace(cfg=SimpleNamespace(metadata=metadata))
        ),
        feature_space_identity={
            "feature_space_id": "space-a",
            "feature_space_status": "declared_unpinned",
        },
    )
    provider = NeuronpediaProvider.from_lens(lens)
    assert provider is not None
    catalog = provider.fetch((3,))
    assert catalog.feature_space_id == "space-a"
    assert catalog.feature_space_status == "declared_unpinned"


def test_neuronpedia_provider_can_record_unavailable_without_error(monkeypatch):
    def fail(request, timeout):
        raise OSError("offline")

    monkeypatch.setattr("prefscope.integrations.neuronpedia.urlopen", fail)
    catalog = NeuronpediaProvider("model/layer").fetch((5,), strict=False)
    frame = catalog.to_frame()
    assert frame.loc[0, "retrieval_status"] == "unavailable"
    assert pd.isna(frame.loc[0, "description"])
