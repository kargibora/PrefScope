from __future__ import annotations

import numpy as np
import pytest

from prefscope import (
    CallableRepresentationSource,
    EmbeddingRepresentationSource,
    FeatureMatrix,
    Lens,
    PairItem,
    PrecomputedRepresentationSource,
    RepresentationBatch,
)


class FakeEmbedder:
    model_id = "fake/embedder"

    def encode_prompts(self, prompts):
        return np.array([[len(value), 1.0, 0.0] for value in prompts], dtype=np.float32)

    def encode(self, prompts, completions):
        return np.array([
            [len(prompt), len(completion), 1.0]
            for prompt, completion in zip(prompts, completions, strict=True)
        ], dtype=np.float32)

    def provenance(self, *, prompt=False):
        return {"model_id": self.model_id, "mode": "prompt" if prompt else "response"}


class FakeProjector:
    m_total = 2
    input_dim = 3
    activation_polarity = "signed"
    code_semantics = "axis"

    def project(self, values):
        return np.asarray(values, dtype=np.float32)[:, :2]


def _pairs():
    return [
        PairItem("a", "short", "one", "three", pref=1.0),
        PairItem("b", "longer", "two", "four", pref=0.0),
    ]


def test_embedding_source_and_difference_lens_are_source_agnostic():
    source = EmbeddingRepresentationSource(FakeEmbedder())
    representations = source.encode(_pairs())
    lens = Lens(FakeProjector(), FakeEmbedder())

    features = lens.project_representations(representations)

    assert set(representations.arrays) == {"prompt", "response_a", "response_b"}
    assert set(features.arrays) == {"z_diff"}
    np.testing.assert_allclose(
        features.array("z_diff"),
        representations.array("response_a")[:, :2]
        - representations.array("response_b")[:, :2],
    )
    assert features.matrix("z_diff").role == "response_difference"
    assert features.provenance["representation_source"]["source_type"] == "text_embedding"


def test_individual_lens_projects_single_custom_representations():
    batch = RepresentationBatch(
        row_ids=("a", "b"),
        arrays={"response_a": np.array([[1, 2, 3], [4, 5, 6]], np.float32)},
        provenance={"source_type": "residual", "layer": 12},
    )
    manifest = {
        "schema_version": 2,
        "lens_kind": "individual",
        "input_rep": "individual",
        "m_total": 2,
        "input_dim": 3,
        "output_arrays": ["z_a"],
        "sae_type": "batchtopk",
        "activation_polarity": "signed",
        "code_semantics": "axis",
        "selection_rule": "batchtopk-absolute",
    }
    lens = Lens(FakeProjector(), None, manifest=manifest)

    features = lens.project_representations(batch)

    assert set(features.arrays) == {"z_a"}
    assert features.matrix("z_a").role == "response_a"
    assert features.provenance["representation_source"]["source_type"] == "residual"


def test_callable_source_requires_the_public_batch_contract():
    source = CallableRepresentationSource(
        lambda items: RepresentationBatch(
            row_ids=tuple(item.id for item in items),
            arrays={"custom": np.ones((len(items), 2))},
        ),
        name="my-source",
        provenance={"revision": "abc"},
    )
    result = source.encode(_pairs())
    assert result.provenance["source_type"] == "callable"
    assert result.provenance["source_name"] == "my-source"


def test_batches_fail_closed_on_alignment_width_and_credentials():
    with pytest.raises(ValueError, match="unique"):
        RepresentationBatch(
            row_ids=("x", "x"), arrays={"x": np.ones((2, 2))})
    with pytest.raises(ValueError, match="2, width"):
        RepresentationBatch(
            row_ids=("x", "y"), arrays={"x": np.ones((1, 2))})
    with pytest.raises(ValueError, match="credential"):
        RepresentationBatch(
            row_ids=("x",), arrays={"x": np.ones((1, 2))},
            provenance={"api_key": "secret"})


def test_embedding_source_rejects_mixed_pairing():
    rows = _pairs()
    rows[1] = PairItem("b", "longer", "two")
    with pytest.raises(ValueError, match="mix paired and single"):
        EmbeddingRepresentationSource(FakeEmbedder()).encode(rows)


def test_lens_item_encoding_uses_injected_custom_source():
    source = CallableRepresentationSource(
        lambda items: RepresentationBatch(
            row_ids=tuple(item.id for item in items),
            arrays={
                "response_a": np.array([[3.0, 2.0, 1.0]] * len(items)),
                "response_b": np.array([[1.0, 1.0, 1.0]] * len(items)),
            },
            metadata={"group_id": tuple(item.meta["group_id"] for item in items)},
            provenance={"source_type": "attempted-override"},
        ),
        name="residual-test",
    )
    lens = Lens(FakeProjector(), representation_source=source)
    items = [
        PairItem("a", "p", "a", "b", meta={"group_id": "g"}),
        PairItem("b", "q", "c", "d", meta={"group_id": "h"}),
    ]

    codes, metadata = lens.encode_pairs(items)
    typed = lens.project_representations(source.encode(items))

    np.testing.assert_allclose(codes, [[2.0, 1.0], [2.0, 1.0]])
    assert list(metadata.columns) == ["id", "pref", "model_a", "model_b"]
    assert typed.metadata["group_id"] == ("g", "h")
    assert typed.provenance["representation_source"]["source_type"] == "callable"


def test_pairs_to_battles_retains_custom_metadata_and_rejects_collisions():
    from prefscope.api.loaded_lens import pairs_to_battles

    frame = pairs_to_battles([
        PairItem("a", "p", "x", meta={"group_id": "g", "reward": 0.5})
    ])
    assert frame.loc[0, "group_id"] == "g"
    assert frame.loc[0, "reward"] == 0.5
    with pytest.raises(ValueError, match="collides"):
        pairs_to_battles([
            PairItem("a", "p", "x", meta={"prompt": "override"})
        ])


def test_portable_provenance_rejects_absolute_paths_and_nonfinite_json():
    with pytest.raises(ValueError, match="absolute local paths"):
        RepresentationBatch(
            row_ids=("a",), arrays={"x": np.ones((1, 2))},
            provenance={"cache": "/private/tmp/cache.npy"},
        )
    with pytest.raises(ValueError, match="JSON-serializable"):
        RepresentationBatch(
            row_ids=("a",), arrays={"x": np.ones((1, 2))},
            provenance={"metric": np.nan},
        )


def test_precomputed_representation_source_preserves_contract_and_requires_exact_ids():
    batch = RepresentationBatch(
        row_ids=("a", "b"),
        arrays={"response_a": np.ones((2, 3))},
        metadata={"group_id": ("g", "h")},
        provenance={"representation_family": "pooled_residual", "layer": 12},
    )
    source = PrecomputedRepresentationSource(batch, source_name="layer-12")
    rows = [PairItem("a", "p", "x"), PairItem("b", "q", "y")]
    result = source.encode(rows)
    assert result.row_ids == ("a", "b")
    assert result.metadata["group_id"] == ("g", "h")
    assert result.provenance["representation_family"] == "pooled_residual"
    assert result.provenance["source_type"] == "precomputed"
    with pytest.raises(ValueError, match="exactly match"):
        source.encode(list(reversed(rows)))


def test_callable_representation_source_rejects_reordered_output_ids():
    source = CallableRepresentationSource(
        lambda items: RepresentationBatch(
            row_ids=tuple(reversed([item.id for item in items])),
            arrays={"response_a": np.ones((len(items), 2))},
        )
    )
    with pytest.raises(ValueError, match="exactly match input item order"):
        source.encode([PairItem("a", "p", "x"), PairItem("b", "q", "y")])


def test_in_memory_projector_declares_prompt_projection_semantics():
    class PromptProjector(FakeProjector):
        input_rep = "prompt"
        activation_polarity = "nonnegative"
        code_semantics = "presence"

    batch = RepresentationBatch(
        row_ids=("a", "b"),
        arrays={"prompt": np.array([[1, 2, 3], [4, 5, 6]], np.float32)},
        provenance={"representation_family": "static_embedding"},
    )
    features = Lens(PromptProjector()).project_representations(batch)
    assert set(features.arrays) == {"z_prompt"}
    assert features.matrix("z_prompt").role == "prompt"
    assert features.matrix("z_prompt").orientation == "none"
    assert features.activation_polarity == "nonnegative"


def test_in_memory_projector_rejects_unknown_input_rep():
    class BadProjector(FakeProjector):
        input_rep = "token-ragged"

    with pytest.raises(ValueError, match="input_rep"):
        Lens(BadProjector())


def test_precomputed_source_is_registry_constructible():
    from prefscope.core import registry

    batch = RepresentationBatch(
        row_ids=("a",), arrays={"response_a": np.ones((1, 2))})
    source = registry.make(
        "representation_source", "precomputed", batch=batch, source_name="cached")
    assert isinstance(source, PrecomputedRepresentationSource)


def test_portable_provenance_is_deeply_immutable_and_json_serializable():
    import json

    original = {"config": {"layer": 3}, "tags": ["pooled"]}
    batch = RepresentationBatch(
        row_ids=("a",), arrays={"x": np.ones((1, 2))}, provenance=original)
    original["config"]["layer"] = 9
    original["tags"].append("changed")
    assert batch.provenance["config"]["layer"] == 3
    assert batch.provenance["tags"] == ("pooled",)
    with pytest.raises(TypeError, match="immutable"):
        batch.provenance["config"]["layer"] = 4
    json.dumps(dict(batch.provenance), allow_nan=False)


@pytest.mark.parametrize(
    "provenance",
    [
        {"hf_token": "x"},
        {"client-secret": "x"},
        {"service": {"api-key": "x"}},
        {"url": "https://user:pass@example.com/model"},
        {"url": "https://example.com/model?access_token=x"},
    ],
)
def test_portable_provenance_rejects_credential_variants(provenance):
    with pytest.raises(ValueError, match="credential"):
        RepresentationBatch(
            row_ids=("a",), arrays={"x": np.ones((1, 2))},
            provenance=provenance,
        )


def test_portable_provenance_rejects_unc_paths_and_path_keys():
    with pytest.raises(ValueError, match="absolute local paths"):
        RepresentationBatch(
            row_ids=("a",), arrays={"x": np.ones((1, 2))},
            provenance={"cache": r"\\server\share\vectors.npy"},
        )
    with pytest.raises(ValueError, match="absolute local paths"):
        RepresentationBatch(
            row_ids=("a",), arrays={"x": np.ones((1, 2))},
            provenance={"/private/cache": "logical"},
        )


def _embedding_manifest(model_id):
    return {
        "schema_version": 2,
        "lens_kind": "individual",
        "input_rep": "individual",
        "m_total": 2,
        "input_dim": 3,
        "output_arrays": ["z_a"],
        "sae_type": "batchtopk",
        "activation_polarity": "signed",
        "code_semantics": "axis",
        "selection_rule": "batchtopk-absolute",
        "embed_model_id": model_id,
    }


def test_lens_rejects_same_width_vectors_from_another_coordinate_system():
    lens = Lens(FakeProjector(), manifest=_embedding_manifest("expected/model"))
    batch = RepresentationBatch(
        row_ids=("a",),
        arrays={"response_a": np.ones((1, 3))},
        provenance={
            "representation_contract": {
                "representation_family": "text_embedding",
                "embed_model_id": "other/model",
            }
        },
    )
    with pytest.raises(ValueError, match="incompatible.*embed_model_id"):
        lens.project_representations(batch)
    projected = lens.project_representations(
        batch, allow_representation_mismatch=True)
    compatibility = projected.provenance["lens"]["representation_compatibility"]
    assert compatibility["status"] == "unsafe_override"
    assert compatibility["unsafe_override"] is True


def test_lens_records_matching_representation_fingerprints():
    lens = Lens(FakeProjector(), manifest=_embedding_manifest("expected/model"))
    batch = RepresentationBatch(
        row_ids=("a",),
        arrays={"response_a": np.ones((1, 3))},
        provenance={
            "representation_contract": {
                "representation_family": "text_embedding",
                "embed_model_id": "expected/model",
            }
        },
    )
    projected = lens.project_representations(batch)
    compatibility = projected.provenance["lens"]["representation_compatibility"]
    assert compatibility["status"] == "matched"
    assert compatibility["expected_fingerprint"] == compatibility["observed_fingerprint"]


def test_representation_and_feature_arrays_are_detached_float32_read_only():
    source = np.array([[1.0, 2.0]], dtype=np.float64)
    batch = RepresentationBatch(row_ids=("a",), arrays={"x": source})
    source[0, 0] = 9.0
    assert batch.arrays["x"].dtype == np.float32
    assert batch.arrays["x"][0, 0] == 1.0
    with pytest.raises(ValueError, match="read-only"):
        batch.arrays["x"][0, 0] = 3.0
    with pytest.raises(ValueError, match="cannot set WRITEABLE"):
        batch.arrays["x"].setflags(write=True)

    matrix_source = np.array([[1.0, 0.0]], dtype=np.float64)
    features = FeatureMatrix(matrix_source, row_ids=("a",), role="response")
    matrix_source[0, 0] = 7.0
    assert features.values.dtype == np.float32
    assert features.values[0, 0] == 1.0
    with pytest.raises(ValueError, match="read-only"):
        features.values[0, 0] = 2.0
    with pytest.raises(ValueError, match="cannot set WRITEABLE"):
        features.values.setflags(write=True)


def test_representation_contracts_reject_complex_arrays():
    with pytest.raises(ValueError, match="real numeric"):
        RepresentationBatch(
            row_ids=("a",), arrays={"x": np.array([[1.0 + 2.0j]])})
    with pytest.raises(ValueError, match="real numeric"):
        FeatureMatrix(
            np.array([[1.0 + 2.0j]]), row_ids=("a",), role="response")


def test_feature_ids_reject_lossy_float_and_boolean_coercion():
    with pytest.raises(ValueError, match="non-boolean integers"):
        FeatureMatrix(
            np.ones((1, 1)), row_ids=("a",), feature_ids=(1.9,))
    with pytest.raises(ValueError, match="non-boolean integers"):
        FeatureMatrix(
            np.ones((1, 1)), row_ids=("a",), feature_ids=(True,))


def test_numpy_bearing_contract_equality_is_identity_based_not_ambiguous():
    left = RepresentationBatch(row_ids=("a",), arrays={"x": np.ones((1, 2))})
    right = RepresentationBatch(row_ids=("a",), arrays={"x": np.ones((1, 2))})
    assert (left == right) is False
    assert left == left
