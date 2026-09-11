"""Numerical proof and table/export contracts for the optional UMAP example."""
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from prefscope import FeatureBatch
from prefscope.viewer_export import export_viewer_bundle
from examples.reporting.example_umap import example_umap_tables

umap = pytest.importorskip("umap")


def _batches():
    rng = np.random.default_rng(8)
    rows = tuple(f"row-{i}" for i in range(12))
    a = rng.random((12, 3), dtype=np.float32)
    b = rng.random((12, 3), dtype=np.float32)
    a[:2] = 0
    b[0] = 0
    b[-1] = a[3]  # Duplicate real vectors must retain both observation identities.
    prompt = rng.uniform(-1, 1, size=(12, 2)).astype(np.float32)
    prompt[4] = 0
    response = FeatureBatch(
        row_ids=rows,
        feature_ids=(71, 2, 19),
        arrays={"answer_a": a, "answer_b": b},
        roles={"answer_a": "response_a", "answer_b": "response_b"},
        orientations={"answer_a": "model_a", "answer_b": "model_b"},
        activation_polarity="nonnegative",
        code_semantics="presence",
        metadata={"prompt": tuple(f"Original prompt {i}" for i in range(12))},
        provenance={"lens": {"feature_space_id": "response-space", "feature_space_status": "declared_unpinned"}},
    )
    prompts = FeatureBatch(
        row_ids=rows,
        feature_ids=(2, 500),
        arrays={"prompt_axes": prompt},
        roles={"prompt_axes": "prompt"},
        orientations={"prompt_axes": "none"},
        activation_polarity="signed",
        code_semantics="axis",
        provenance={"lens": {"feature_space_id": "prompt-space", "feature_space_status": "declared_unpinned"}},
    )
    return response, prompts


def test_full_vectors_use_real_umap_and_shared_answer_geometry(monkeypatch):
    response, prompt = _batches()
    original_a = response.array("answer_a").copy()
    real_fit = umap.UMAP.fit_transform
    calls = []

    def observed_fit(model, values, *args, **kwargs):
        result = real_fit(model, values, *args, **kwargs)
        calls.append((values.copy(), result.copy(), model.graph_.nnz))
        return result

    monkeypatch.setattr(umap.UMAP, "fit_transform", observed_fit)
    tables = example_umap_tables(response, prompt)
    assert len(calls) == 2  # One shared answer fit, one separate prompt fit.
    expected = np.stack([response.array("answer_a"), response.array("answer_b")], axis=1).reshape(24, 3)
    np.testing.assert_array_equal(calls[0][0], expected)
    np.testing.assert_array_equal(calls[1][0], prompt.array("prompt_axes"))
    assert all(nnz > 0 for _, _, nnz in calls)  # Actual UMAP fuzzy neighbor graphs.
    np.testing.assert_array_equal(response.array("answer_a"), original_a)

    points = tables["example_umap_points"]
    answers = points.loc[points.projection_id.eq("answers")]
    prompts = points.loc[points.projection_id.eq("prompt")]
    assert list(zip(answers.row_id, answers.view)) == [
        (row, view) for row in response.row_ids for view in response.arrays
    ]
    assert not points.duplicated(["projection_id", "row_id", "view"]).any()
    assert len(answers) == 24 and len(prompts) == 12
    assert answers.zero_vector.sum() == 3 and prompts.zero_vector.sum() == 1
    np.testing.assert_array_equal(answers[["x", "y"]], calls[0][1])
    np.testing.assert_array_equal(prompts[["x", "y"]], calls[1][1])
    assert np.isfinite(points[["x", "y"]]).all().all()

    meta = tables["example_umap_meta"].set_index("projection_id")
    assert meta.at["answers", "feature_ids"] == [71, 2, 19]
    assert meta.at["prompt", "feature_ids"] == [2, 500]
    assert meta.at["answers", "feature_space_id"] == "response-space"
    assert meta.at["prompt", "feature_space_id"] == "prompt-space"
    assert meta.at["answers", "n_rows"] == 12
    assert meta.at["answers", "n_points"] == 24
    assert meta.at["answers", "n_zero_rows"] == 3
    assert [item["orientation"] for item in meta.at["answers", "view_descriptors"]] == ["model_a", "model_b"]
    assert set(meta.method) == {"umap"}
    assert set(meta.basis) == {"full_feature_activations"}
    assert set(meta.preprocessing) == {"none"}
    assert meta.at["answers", "versions"]["umap-learn"] == umap.__version__

    # Independent call with the EXPORTED settings must produce these coordinates.
    direct = umap.UMAP(**meta.at["answers", "parameters"]).fit_transform(expected)
    np.testing.assert_array_equal(answers[["x", "y"]], direct)
    repeated = example_umap_tables(response, prompt)
    pd.testing.assert_frame_equal(points, repeated["example_umap_points"])
    pd.testing.assert_frame_equal(tables["example_umap_meta"], repeated["example_umap_meta"])



def test_default_individual_lens_batch_uses_real_answers_not_derived_difference():
    from prefscope import Lens, PairItem

    class Projector:
        input_rep = "individual"
        activation_polarity = "nonnegative"
        code_semantics = "numerical_activity"
        m_total = 2
        input_dim = 2

        def project(self, values):
            return values

    class Reader:
        def encode(self, prompts, responses):
            return np.asarray(
                [[len(text), sum(map(ord, text)) % 17] for text in responses],
                dtype=np.float32,
            )

    lens = Lens(Projector(), Reader())
    items = [PairItem(str(i), "prompt", f"answer a{i}", f"answer b{i}") for i in range(4)]
    full = lens.featurize(items)
    answers = lens.featurize(items, views=("response_a", "response_b"))
    assert tuple(full.arrays) == ("z_a", "z_b", "z_diff")
    full_tables = example_umap_tables(full)
    answer_tables = example_umap_tables(answers)
    for name in full_tables:
        pd.testing.assert_frame_equal(full_tables[name], answer_tables[name])
    assert set(full_tables["example_umap_points"].view) == {"z_a", "z_b"}
    assert tuple(full.arrays) == ("z_a", "z_b", "z_diff")


def test_raw_pair_orientation_and_metadata_do_not_change_geometry():
    _, prompt = _batches()
    raw = prompt.array("prompt_axes")
    pair = replace(
        prompt,
        arrays={"pair_code": raw},
        roles={"pair_code": "response_difference"},
        orientations={"pair_code": "a_minus_b"},
        metadata={"preference_probability": (0.1,) * 12},
    )
    tables = example_umap_tables(pair)
    points = tables["example_umap_points"]
    meta = tables["example_umap_meta"].iloc[0]
    assert list(points.row_id) == list(pair.row_ids)
    assert set(points.projection_id) == {"pair"}
    assert points.zero_vector.sum() == 1
    assert meta.view_descriptors[0]["orientation"] == "a_minus_b"
    assert meta.view_descriptors[0]["activation_polarity"] == "signed"
    np.testing.assert_array_equal(pair.array("pair_code"), raw)
    direct = umap.UMAP(**meta.parameters).fit_transform(raw.copy())
    np.testing.assert_array_equal(points[["x", "y"]], direct)

    changed = replace(pair, metadata={"preference_probability": (0.9,) * 12, "name": ("Changed label",) * 12})
    renamed = example_umap_tables(changed)
    pd.testing.assert_frame_equal(points, renamed["example_umap_points"])
    assert meta.input_hash == renamed["example_umap_meta"].iloc[0].input_hash
    reordered = replace(pair, feature_ids=tuple(reversed(pair.feature_ids)))
    assert meta.input_hash != example_umap_tables(reordered)["example_umap_meta"].iloc[0].input_hash


def test_prompt_only_root_and_sft_answer():
    response, prompt = _batches()
    root_prompt = example_umap_tables(prompt)
    assert set(root_prompt["example_umap_points"].space) == {"main"}
    assert set(root_prompt["example_umap_points"].projection_id) == {"prompt"}
    single = replace(response, arrays={"answer_a": response.array("answer_a")}, roles={"answer_a": "response_a"}, orientations={"answer_a": "absolute_a"})
    tables = example_umap_tables(single, prompt)
    answers = tables["example_umap_points"].query("projection_id == 'answers'")
    assert len(answers) == len(response.row_ids)
    assert set(answers.view) == {"answer_a"}


@pytest.mark.parametrize("kind", ["tiny", "constant"])
def test_unavailable_inputs_do_not_fabricate_coordinates(kind):
    _, prompt = _batches()
    count = 3 if kind == "tiny" else 12
    values = prompt.array("prompt_axes")[:count] if kind == "tiny" else np.zeros((12, 2))
    batch = replace(prompt, row_ids=prompt.row_ids[:count], arrays={"prompt_axes": values})
    with pytest.raises(ValueError, match="UMAP unavailable"):
        example_umap_tables(batch)


def test_semantics_and_alignment_are_required():
    response, prompt = _batches()
    with pytest.raises(ValueError, match="exactly match"):
        example_umap_tables(response, replace(prompt, row_ids=tuple(reversed(prompt.row_ids))))
    incompatible = replace(response, provenance={"views": {"answer_b": {"code_semantics": "axis"}}})
    with pytest.raises(ValueError, match="compatible"):
        example_umap_tables(incompatible)
    mixed = replace(response, roles={"answer_a": "response", "answer_b": "response_difference"})
    with pytest.raises(ValueError, match="requires response"):
        example_umap_tables(mixed)


def test_projection_tables_round_trip_through_normal_export(tmp_path):
    response, prompt = _batches()
    tables = example_umap_tables(response, prompt)
    viewer = tmp_path / "viewer-dist"
    viewer.mkdir()
    (viewer / "index.html").write_text("<main>Example UMAP</main>")
    (viewer / "viewer-build.json").write_text(json.dumps({
        "schema": "prefscope.viewer_build", "schema_version": 1,
        "package": "@prefscope/viewer", "version": "0.1.0",
        "supported_data_schemas": [{"schema": "prefscope.viewer_data", "versions": [1, 2]}],
    }))
    target = export_viewer_bundle(response, tmp_path / "bundle", viewer_dist=viewer, prompt_features=prompt, tables=tables)
    payload = json.loads((target / "data/viewer-data.json").read_text())
    assert payload["schema_version"] == 2
    assert payload["row_ids"] == payload["prompt"]["row_ids"] == list(response.row_ids)
    for name, table in tables.items():
        saved = payload["tables"][name]
        restored = pd.DataFrame(saved["data"], columns=saved["columns"])
        assert restored.to_dict("records") == table.to_dict("records")
    assert payload["row_metadata"]["prompt"][0] == "Original prompt 0"
