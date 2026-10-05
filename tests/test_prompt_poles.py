import numpy as np
import pandas as pd
import pytest

from prefscope import (
    FeatureBatch,
    FeatureCatalog,
    FeatureMatrix,
    Lens,
    LensBackend,
    LensCapabilities,
    PairItem,
    build_prompt_pole_catalog,
    expand_prompt_poles,
    load_prompt_pole_catalog,
)
from prefscope.api._lens_publication import save_lens
from prefscope.api.feature_catalog_io import encode_feature_catalog


def test_expand_prompt_poles_interleaves_values_and_preserves_native_input():
    matrix = FeatureMatrix(
        np.array([[-2.0, 3.0], [4.0, -5.0]], dtype=np.float32),
        ("a", "b"), role="prompt", feature_ids=(0, 1),
        provenance={"lens": {"feature_space_id": "prompt-v1", "feature_space_status": "declared_pinned_coordinate"}},
    )
    original = matrix.values.copy()
    expanded = expand_prompt_poles(matrix)

    np.testing.assert_array_equal(expanded.values, [[0, 2, 3, 0], [4, 0, 0, 5]])
    assert expanded.feature_ids == (0, 1, 2, 3)
    assert expanded.activation_polarity == "nonnegative"
    assert expanded.code_semantics == "derived_strength"
    assert expanded.provenance["coordinate_space"] == "derived_feature_view-v1"
    np.testing.assert_array_equal(matrix.values, original)


def test_expand_prompt_poles_requires_prompt_matrix():
    matrix = FeatureMatrix([[1.0]], ("a",), role="response_a")
    with pytest.raises(ValueError, match="prompt FeatureMatrix"):
        expand_prompt_poles(matrix)


def test_prompt_pole_catalog_round_trip_and_identity_validation(tmp_path):
    annotations = pd.DataFrame({
        "feature_id": [0, 0, 1],
        "pole": ["positive", "negative", "positive"],
        "concept_name": ["asks for setup", "not setup", "asks for code"],
        "reason": ["long explanation", "another explanation", "third explanation"],
        "naming_status": ["proposed", "mixed", "insufficient"],
    })
    catalog = build_prompt_pole_catalog(
        annotations, native_width=2, feature_space_id="prompt-v1")
    frame = catalog.to_frame()
    assert catalog.feature_ids == (0, 1, 2, 3)
    assert frame.loc[0, "name"] == "asks for setup"
    assert frame.loc[1, "description"] == "another explanation"
    assert frame.loc[1, "status"] == "mixed"
    assert catalog.provenance["coordinate_space"] == "derived_feature_view-v1"
    native = FeatureMatrix(
        np.zeros((1, 2)), ("row",), role="prompt", feature_ids=(0, 1),
        provenance={"lens": {"feature_space_id": "prompt-v1", "feature_space_status": "declared_pinned_coordinate"}},
    )
    with pytest.raises(ValueError, match="coordinate spaces"):
        catalog.validate_for(native)

    path = tmp_path / "prompt_pole_catalog.json"
    path.write_bytes(encode_feature_catalog(catalog))
    loaded = load_prompt_pole_catalog(
        path, native_width=2, feature_space_id="prompt-v1")
    standalone = load_prompt_pole_catalog(path, native_width=2)
    assert standalone.feature_space_id == "prompt-v1"
    pd.testing.assert_frame_equal(
        loaded.to_frame().fillna(""), frame.fillna(""), check_dtype=False)
    with pytest.raises(ValueError, match="does not match"):
        load_prompt_pole_catalog(path, native_width=2, feature_space_id="other")
    with pytest.raises(ValueError, match="do not match"):
        build_prompt_pole_catalog(
            annotations.assign(feature_space_id="other"),
            native_width=2,
            feature_space_id="prompt-v1",
        )


class _PromptBackend(LensBackend):
    input_rep = "prompt"
    activation_polarity = "signed"
    code_semantics = "axis"

    @property
    def capabilities(self):
        return LensCapabilities(("prompt",))

    @property
    def m_total(self):
        return 2

    @property
    def feature_space_identity(self):
        return {"feature_space_id": "prompt-v1", "feature_space_status": "declared_pinned_coordinate"}

    def featurize(self, items, *, views=None, feature_ids=None, batch_size=None):
        rows = list(items)
        values = np.array([[-2, 3], [4, -5]], dtype=np.float32)[:len(rows)]
        if feature_ids is not None:
            values = values[:, feature_ids]
        return FeatureBatch(
            row_ids=tuple(item.id for item in rows),
            arrays={"z_prompt": values},
            roles={"z_prompt": "prompt"},
            orientations={"z_prompt": "none"},
            feature_ids=tuple(range(2)) if feature_ids is None else tuple(feature_ids),
            provenance={"views": {"z_prompt": {"activation_polarity": "signed", "code_semantics": "signed_activity"}}},
        )


def test_declared_derived_view_is_generic_and_activation_driven():
    lens = Lens.from_backend(
        _PromptBackend(),
        manifest={"input_rep": "prompt", "activation_polarity": "signed"},
    )
    items = [PairItem("a", "prompt", "", y_b=None)]
    assert lens.derived_views["poles"]["transform"] == "signed_to_poles"
    derived = lens.featurize_derived(items, view="poles")
    assert derived.provenance["derived_transform"] == "signed_to_poles"



def test_lens_prompt_pole_convenience_keeps_native_lens_contract():
    lens = Lens.from_backend(_PromptBackend())
    items = [PairItem("a", "prompt", "", y_b=None), PairItem("b", "prompt2", "", y_b=None)]
    native = lens.featurize(items, views=("prompt",))
    poles = lens.featurize_prompt_poles(items)

    assert lens.backend.m_total == 2
    assert native.feature_ids == (0, 1)
    assert poles.feature_ids == (0, 1, 2, 3)
    np.testing.assert_array_equal(poles.values, [[0, 2, 3, 0], [4, 0, 0, 5]])
    assert poles.provenance["lens"]["feature_space_id"] == "prompt-v1"


class _PublicationBackend:
    m_total = 2


class _PublicationLens:
    lens_dir = None
    input_rep = "prompt"
    activation_polarity = "signed"
    backend = _PublicationBackend()
    projector = None
    _loaded_native_feature_space_identity = None

    @property
    def feature_space_identity(self):
        return {
            "feature_space_id": "prompt-v1",
            "feature_space_status": "declared_pinned_coordinate",
        }


def test_save_lens_bundles_prompt_pole_catalog_without_changing_native_width(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.json").write_text("{}")
    lens = _PublicationLens()
    lens.lens_dir = source
    annotations = tmp_path / "reviewed-poles.csv"
    pd.DataFrame({
        "feature_id": [0, 0, 1, 1],
        "pole": ["positive", "negative", "positive", "negative"],
        "concept_name": ["p+", "p-", "q+", "q-"],
        "reason": ["d+", "d-", "e+", "e-"],
        "naming_status": ["proposed"] * 4,
    }).to_csv(annotations, index=False)

    destination = save_lens(
        lens, tmp_path / "published", prompt_pole_catalog=annotations)
    loaded = load_prompt_pole_catalog(
        destination / "derived_feature_catalog.json",
        native_width=2,
        feature_space_id="prompt-v1",
    )
    assert len(loaded) == 4
    assert loaded.to_frame().loc[3, "name"] == "q-"
    assert (destination / "feature_catalog.json").is_file()


def test_selected_prompt_poles_keep_global_virtual_ids():
    matrix = FeatureMatrix(
        [[1.0, -2.0]], ("a",), role="prompt", feature_ids=(3, 7),
        provenance={"lens": {"feature_space_id": "prompt-v1"}},
    )
    expanded = expand_prompt_poles(matrix, native_width=8)
    assert expanded.feature_ids == (6, 7, 14, 15)
    np.testing.assert_array_equal(expanded.values, [[1, 0, 0, 2]])


def test_prompt_pole_catalog_rejects_wrong_encoding_or_identity():
    catalog = build_prompt_pole_catalog(
        pd.DataFrame({"feature_id": [0], "pole": ["positive"], "name": ["x"]}),
        native_width=1,
        feature_space_id="prompt-v1",
    )
    for change in (
        {"pole_encoding": "feature_id"},
        {"pole_order": ["negative", "positive"]},
        {"native_feature_space_id": "other"},
    ):
        provenance = dict(catalog.provenance)
        provenance.update(change)
        bad = type(catalog)(
            catalog.to_frame(),
            provenance=provenance,
            column_sources=dict(catalog.column_sources),
        )
        with pytest.raises(ValueError):
            from prefscope.api.prompt_poles import validate_prompt_pole_catalog
            validate_prompt_pole_catalog(bad, native_width=1, feature_space_id="prompt-v1")


def test_native_and_virtual_catalogs_cannot_merge():
    native = FeatureCatalog(
        pd.DataFrame({"feature_id": [0], "name": ["native"]}),
        provenance={"feature_space_id": "prompt-v1", "feature_space_status": "declared_pinned_coordinate"},
    )
    pole = build_prompt_pole_catalog(
        pd.DataFrame({"feature_id": [0], "pole": ["positive"], "name": ["pole"]}),
        native_width=1,
        feature_space_id="prompt-v1",
    )
    with pytest.raises(ValueError, match="coordinate spaces"):
        native.merge(pole)


@pytest.mark.parametrize(
    ("input_rep", "source_view", "array_name", "paired"),
    [
        ("individual", "response_a", "z_a", False),
        ("difference", "response_difference", "z_diff", True),
    ],
)
def test_signed_response_derived_views_use_canonical_sources(
    input_rep, source_view, array_name, paired
):
    class SignedBackend(LensBackend):
        activation_polarity = "signed"
        code_semantics = "axis"

        @property
        def capabilities(self):
            if paired:
                return LensCapabilities(
                    ("response_difference",), difference="direct_difference_projection"
                )
            return LensCapabilities(("response_a",))

        @property
        def m_total(self):
            return 2

        def featurize(self, items, *, views=None, feature_ids=None, batch_size=None):
            assert views == (source_view,)
            values = np.array([[-2, 3]], dtype=np.float32)
            if feature_ids is not None:
                values = values[:, feature_ids]
            return FeatureBatch(
                row_ids=tuple(item.id for item in items),
                arrays={array_name: values},
                roles={array_name: source_view},
                orientations={array_name: "a_minus_b" if paired else "absolute_a"},
                feature_ids=tuple(range(2)) if feature_ids is None else tuple(feature_ids),
                activation_polarity="signed",
                code_semantics="axis",
            )

    backend = SignedBackend()
    backend.input_rep = input_rep
    lens = Lens.from_backend(backend)
    assert lens.derived_views["poles"]["source_view"] == source_view
    items = [PairItem("row", "prompt", "answer A", "answer B" if paired else None)]
    native = lens.featurize(items, views=(source_view,))
    derived = lens.featurize_derived(items)
    selected = lens.featurize_derived(items, feature_ids=(1,))
    np.testing.assert_array_equal(native.array(array_name), [[-2, 3]])
    np.testing.assert_array_equal(derived.values, [[0, 2, 3, 0]])
    assert derived.role == source_view
    assert selected.feature_ids == (2, 3)
    np.testing.assert_array_equal(selected.values, [[3, 0]])
    assert lens.backend.m_total == 2

    legacy = Lens.from_backend(backend, manifest={
        "input_rep": input_rep,
        "activation_polarity": "signed",
        "code_semantics": "axis",
        "derived_views": {"poles": {
            "source_view": input_rep, "transform": "signed_to_poles"
        }},
    })
    np.testing.assert_array_equal(legacy.featurize_derived(items).values, derived.values)


@pytest.mark.parametrize("derived_view", ["poles", "foo"])
@pytest.mark.parametrize("inference_only", [False, True])
@pytest.mark.parametrize(
    ("input_rep", "source_view", "paired"),
    [
        ("individual", "response_a", False),
        ("difference", "response_difference", True),
    ],
)
def test_signed_response_derived_catalog_survives_save_and_reload(
    tmp_path, monkeypatch, input_rep, source_view, paired, inference_only, derived_view
):
    import json

    from prefscope.encode import embed, sae

    class FakeProjector:
        m_total = 2
        input_dim = 4

        def __init__(self, lens_dir, device="cpu"):
            pass

        def project(self, values):
            return np.stack((values[:, 0], -values[:, 0]), axis=1)

    class FakeEmbedder:
        model_id = "test/embedder"

        def __init__(self, *args, **kwargs):
            pass

        def encode(self, prompts, completions):
            return np.array([[len(text)] * 4 for text in completions], dtype=np.float32)

    monkeypatch.setattr(sae, "SAEProjector", FakeProjector)
    monkeypatch.setattr(embed, "Embedder", FakeEmbedder)
    source = tmp_path / "source"
    source.mkdir()
    (source / "sae_model.pt").write_bytes(b"fixed weights")
    (source / "manifest.json").write_text(json.dumps({
        "input_rep": input_rep, "embed_model_id": "test/embedder",
        "m_total": 2, "k": 1, "input_dim": 4,
        "matryoshka_prefix_lengths": [], "output_arrays": [],
    }))
    annotations = tmp_path / "derived.csv"
    pd.DataFrame({
        "feature_id": [0, 0, 1, 1],
        "pole": ["positive", "negative", "positive", "negative"],
        "name": ["first+", "first-", "second+", "second-"],
    }).to_csv(annotations, index=False)

    original = Lens.from_dir(source)
    native_id = original.feature_space_id
    published = original.save(
        tmp_path / "published", derived_catalog=annotations,
        derived_view=derived_view, inference_only=inference_only
    )
    assert json.loads((published / "manifest.json").read_text())["derived_views"] == {
        derived_view: {"source_view": source_view, "transform": "signed_to_poles"}
    }
    loaded = Lens.from_dir(published)
    repacked = loaded.save(tmp_path / "repacked", inference_only=inference_only)
    assert json.loads((repacked / "manifest.json").read_text())["derived_views"] == {
        derived_view: {"source_view": source_view, "transform": "signed_to_poles"}
    }
    assert Lens.from_dir(repacked).catalog_for(derived_view).feature_ids == (0, 1, 2, 3)
    if derived_view == "foo":
        replaced = loaded.save(tmp_path / "override", derived_catalog=annotations)
        assert Lens.from_dir(replaced).catalog_for("poles").feature_ids == (0, 1, 2, 3)
    assert loaded.feature_space_id == native_id
    assert loaded.backend.m_total == 2
    assert loaded.feature_catalog.feature_ids == (0, 1)
    assert loaded.catalog_for(derived_view).feature_ids == (0, 1, 2, 3)
    assert loaded.catalog_for(derived_view).to_frame().loc[1, "name"] == "first-"
    items = [PairItem("row", "prompt", "a", "bbb" if paired else None)]
    native = loaded.featurize(items, views=(source_view,))
    derived = loaded.featurize_derived(items, view=derived_view)
    assert derived.role == source_view
    assert derived.feature_ids == (0, 1, 2, 3)
    np.testing.assert_array_equal(
        derived.values[:, 0::2], np.maximum(native.matrix(
            "z_diff" if paired else "z_a").values, 0)
    )
    with pytest.raises(ValueError, match="coordinate spaces"):
        loaded.feature_catalog.merge(loaded.catalog_for(derived_view))



def test_save_rejects_nonnegative_lens_before_publication(tmp_path):
    import json

    source = tmp_path / "nonnegative"
    source.mkdir()
    (source / "manifest.json").write_text(json.dumps({
        "input_rep": "prompt", "sae_type": "batchtopk-relu",
        "activation_polarity": "nonnegative",
    }))
    (source / "sae_model.pt").write_bytes(b"weights")
    lens = _PublicationLens()
    lens.lens_dir = source
    lens.activation_polarity = "nonnegative"
    annotations = tmp_path / "poles.csv"
    pd.DataFrame({
        "feature_id": [0], "pole": ["positive"], "name": ["cannot split"]
    }).to_csv(annotations, index=False)
    destination = tmp_path / "published"
    with pytest.raises(ValueError, match="requires a signed native lens"):
        save_lens(lens, destination, derived_catalog=annotations)
    assert not destination.exists()
