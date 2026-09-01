from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prefscope import (
    FeatureBatch,
    Lens,
    LensBackend,
    LensCapabilities,
    OutcomeSpec,
    PairItem,
    PrecomputedRepresentationSource,
    RepresentationBatch,
    TableDataset,
    analyze_dataset,
    preference_relevance,
)
from prefscope.core.representation import validate_row_ids


class DemoBackend(LensBackend):
    input_rep = "individual"

    @property
    def capabilities(self):
        return LensCapabilities(
            ("prompt", "response_a", "response_b", "response_difference"),
            difference="a_minus_b_after_encoding",
        )

    @property
    def m_total(self):
        return 3

    @property
    def activation_polarity(self):
        return "nonnegative"

    @property
    def code_semantics(self):
        return "numerical_activity"

    def featurize(self, items, *, views=None, feature_ids=None, batch_size=None):
        del batch_size
        rows = list(items)
        selected = tuple(range(3)) if feature_ids is None else tuple(feature_ids)
        base = np.arange(len(rows) * 3, dtype=np.float32).reshape(len(rows), 3)
        base = base[:, selected]
        all_arrays = {
            "z_prompt": base + 1,
            "z_a": base + 2,
            "z_b": base,
            "z_diff": np.full_like(base, 2),
        }
        names = {
            "prompt": "z_prompt", "response_a": "z_a", "response_b": "z_b",
            "response_difference": "z_diff",
        }
        roles = {value: key for key, value in names.items()}
        requested = tuple(views or self.capabilities.views)
        arrays = {names[view]: all_arrays[names[view]] for view in requested}
        return FeatureBatch(
            row_ids=tuple(item.id for item in rows),
            arrays=arrays,
            roles={name: roles[name] for name in arrays},
            orientations={
                name: {
                    "z_prompt": "none", "z_a": "absolute_a",
                    "z_b": "absolute_b", "z_diff": "a_minus_b",
                }[name]
                for name in arrays
            },
            feature_ids=selected,
            metadata={},
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


def _items():
    return [
        PairItem("a", "p", "A", "B", 1.0, meta={"group_id": "g1"}),
        PairItem("b", "q", "C", "D", 0.0, meta={"group_id": "g2"}),
        PairItem("c", "r", "E", "F", 0.5, meta={"group_id": "g3"}),
    ]


def test_custom_backend_is_a_substitutable_lens_and_direct_analysis_input():
    lens = Lens.from_backend(DemoBackend())
    features = lens.featurize(
        _items(), views=("prompt", "response_difference"), feature_ids=(2, 0))

    assert lens.capabilities.supports("prompt", "response_difference")
    assert tuple(features.arrays) == ("z_prompt", "z_diff")
    assert features.feature_ids == (2, 0)
    assert features.matrix("z_diff").activation_polarity == "signed"
    assert features.matrix("z_diff").code_semantics == "activity_difference"

    outcome = OutcomeSpec.from_feature_batch(features)
    result = analyze_dataset(
        features.matrix("z_diff"),
        outcomes={"preference": outcome},
        group_ids=features.metadata["group_id"],
    )
    assert not result.outcome_associations.empty
    assert result.dataset.group_source == "explicit"

    legacy_codes, legacy_meta = lens.encode_pairs(_items())
    np.testing.assert_allclose(legacy_codes, np.full((3, 3), 2.0))
    assert list(legacy_meta.columns) == ["id", "pref", "model_a", "model_b"]
    assert lens.encode(["p"], ["A"]).shape == (1, 3)

    win_rates = preference_relevance(features)
    assert set(win_rates["feature_id"]) == {0, 2}
    assert set(win_rates["outcome_orientation"]) == {"p_a_preferred"}
    assert set(win_rates["causal_claim"]) == {
        "none_descriptive_dataset_specific"
    }


def test_custom_backend_internal_projector_does_not_change_facade_semantics():
    class WrappedBackend(DemoBackend):
        projector = object()

    lens = Lens.from_backend(WrappedBackend())
    assert lens.input_rep == "individual"
    assert lens.encode(["p"], ["A"]).shape == (1, 3)
    empty, _ = lens.encode_items([])
    assert empty.shape == (0, 3)


def test_featurize_rejects_mixed_modes_and_bad_backend_alignment():
    lens = Lens.from_backend(DemoBackend())
    mixed = _items()
    mixed[0] = PairItem("a", "p", "A")
    with pytest.raises(ValueError, match="cannot mix"):
        lens.featurize(mixed)

    class Misaligned(DemoBackend):
        def featurize(self, items, **kwargs):
            value = super().featurize(items, **kwargs)
            return FeatureBatch(
                row_ids=("wrong", *value.row_ids[1:]), arrays=value.arrays,
                roles=value.roles, orientations=value.orientations,
                feature_ids=value.feature_ids,
            )

    with pytest.raises(ValueError, match="row_ids must exactly match"):
        Lens.from_backend(Misaligned()).featurize(_items())


@pytest.mark.parametrize("value", [None, float("nan"), "  "])
def test_missing_like_row_ids_fail_closed(value):
    with pytest.raises(ValueError, match="row_ids"):
        validate_row_ids((value,))


def test_table_dataset_preserves_first_class_groups_and_metadata():
    frame = pd.DataFrame({
        "prompt": ["p"], "a": ["A"], "b": ["B"], "preference": [1.0],
        "pair_id": ["x"], "conversation": ["g"], "split": ["train"],
    })
    dataset = TableDataset(
        frame, prompt="prompt", a="a", b="b", pref="preference", id="pair_id",
        group_id="conversation", metadata=("split",),
    )
    item = next(iter(dataset))
    assert item.id == "x"
    assert item.meta == {"source_row_id": 0, "group_id": "g", "split": "train"}


def test_native_representation_lens_uses_the_same_featurize_contract():
    class Projector:
        input_rep = "individual"
        m_total = 2
        input_dim = 2
        activation_polarity = "nonnegative"
        code_semantics = "numerical_activity"

        @staticmethod
        def project(values):
            return np.asarray(values, dtype=np.float32)

    items = _items()
    source = PrecomputedRepresentationSource(RepresentationBatch(
        row_ids=("a", "b", "c"),
        arrays={
            "response_a": np.array([[3, 2], [2, 1], [1, 4]]),
            "response_b": np.array([[1, 1], [4, 0], [1, 2]]),
        },
    ))
    features = Lens(Projector(), representation_source=source).featurize(items)

    assert tuple(features.arrays) == ("z_a", "z_b", "z_diff")
    np.testing.assert_allclose(features.array("z_diff"), [[2, 1], [-2, 1], [0, 2]])
    assert features.matrix("z_diff").activation_polarity == "signed"
    assert features.matrix("z_diff").code_semantics == "activity_difference"


def test_lens_yaml_dispatches_to_saelens_factory(monkeypatch):
    sentinel = object()
    captured = {}

    def fake(cls, release, sae_id, **kwargs):
        captured.update({"release": release, "sae_id": sae_id, **kwargs})
        return sentinel

    monkeypatch.setattr(Lens, "from_saelens", classmethod(fake))
    loaded = Lens.from_config({
        "version": 1,
        "backend": "saelens",
        "release": "gpt2-small-res-jb",
        "sae_id": "blocks.8.hook_resid_pre",
        "device": "cpu",
        "text_batch_size": 4,
        "long_text_policy": "error",
    })

    assert loaded is sentinel
    assert captured["release"] == "gpt2-small-res-jb"
    assert captured["sae_id"] == "blocks.8.hook_resid_pre"
    assert captured["text_batch_size"] == 4
    assert captured["long_text_policy"] == "error"


def test_capability_and_backend_output_semantics_fail_closed():
    with pytest.raises(ValueError, match="explicit difference behavior"):
        LensCapabilities(("response_difference",))
    with pytest.raises(ValueError, match="not supported by FeatureBatch"):
        LensCapabilities(("prompt",), shared_feature_space=False)
    with pytest.raises(ValueError, match="input_kind must be 'pair_items'"):
        LensCapabilities(("prompt",), input_kind="tokens")

    class BadOrientation(DemoBackend):
        def featurize(self, items, **kwargs):
            value = super().featurize(items, **kwargs)
            return FeatureBatch(
                row_ids=value.row_ids, arrays=value.arrays, roles=value.roles,
                orientations={name: "none" for name in value.arrays},
                feature_ids=value.feature_ids,
            )

    with pytest.raises(ValueError, match="orientation must be 'a_minus_b'"):
        Lens.from_backend(BadOrientation()).featurize(
            _items(), views="response_difference")

    class Incomplete(DemoBackend):
        def featurize(self, items, **kwargs):
            kwargs["feature_ids"] = (0, 1)
            return super().featurize(items, **kwargs)

    with pytest.raises(ValueError, match="every feature ID"):
        Lens.from_backend(Incomplete()).featurize(_items())


def test_custom_config_passes_explicit_device_and_rejects_string_booleans(monkeypatch):
    from prefscope.core import registry

    captured = {}

    def make(kind, name, **options):
        captured.update({"kind": kind, "name": name, **options})
        return DemoBackend()

    monkeypatch.setattr(registry, "make", make)
    Lens.from_config({
        "version": 1, "backend": "demo", "device": "cuda",
        "options": {"scale": 0.1},
    })
    assert captured["device"] == "cuda"
    with pytest.raises(ValueError, match="unsupported lens config version"):
        Lens.from_config({"version": True, "backend": "demo"})
    with pytest.raises(ValueError, match="unsupported lens config version"):
        Lens.from_config({"version": 1.0, "backend": "demo"})
    with pytest.raises(ValueError, match="must be a boolean"):
        Lens.from_config({
            "version": 1, "backend": "saelens", "release": "r", "sae_id": "s",
            "allow_unregistered_release": "false",
        })


def test_preference_relevance_rejects_no_usable_labels():
    items = [
        PairItem("a", "p", "A", "B", meta={"group_id": "g1"}),
        PairItem("b", "q", "C", "D", meta={"group_id": "g2"}),
    ]
    features = Lens.from_backend(DemoBackend()).featurize(items)
    with pytest.raises(ValueError, match="at least one nonmissing"):
        preference_relevance(features)


def test_from_backend_rejects_manifest_semantics_that_conflict_with_backend():
    manifest = {
        "schema_version": 2, "lens_kind": "prompt", "input_rep": "prompt",
        "m_total": 3, "input_dim": 3, "output_arrays": ["z_prompt"],
        "sae_type": "batchtopk", "activation_polarity": "nonnegative",
        "code_semantics": "numerical_activity",
        "selection_rule": "batchtopk-relu",
    }
    with pytest.raises(ValueError, match="input_rep.*conflicts"):
        Lens.from_backend(DemoBackend(), manifest=manifest)


def test_from_backend_rejects_invalid_manifest_and_backend_widths():
    manifest = {
        "schema_version": 2, "lens_kind": "individual", "input_rep": "individual",
        "m_total": 3.9, "input_dim": 3, "output_arrays": ["z_a"],
        "sae_type": "batchtopk", "activation_polarity": "nonnegative",
        "code_semantics": "numerical_activity", "selection_rule": "batchtopk-relu",
    }
    with pytest.raises(ValueError, match="m_total must be a positive integer"):
        Lens.from_backend(DemoBackend(), manifest=manifest)

    class InvalidWidth(DemoBackend):
        m_total = True

    with pytest.raises(ValueError, match="backend m_total must be a positive integer"):
        Lens.from_backend(InvalidWidth())
