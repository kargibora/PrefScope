from __future__ import annotations

from contextlib import nullcontext
import os
import subprocess
import sys
import types

import numpy as np
import pytest

from prefscope import Lens, PairItem, PrecomputedRepresentationSource, RepresentationBatch
from prefscope.api._feature_space import projector_feature_space_identity
from prefscope.integrations import saelens as integration
from prefscope.integrations.saelens import SAELensProjector


class FakeMetadata:
    model_name = "reader/model"
    hook_name = "blocks.3.hook_resid_pre"
    hook_layer = 3
    context_size = 128
    prepend_bos = True
    model_from_pretrained_kwargs = {"fold_ln": False}


class FakeConfig:
    d_in = 3
    d_sae = 2
    dtype = "float32"
    device = "cpu"
    normalize_activations = "none"
    reshape_activations = "none"
    metadata = FakeMetadata()

    @staticmethod
    def architecture():
        return "jumprelu"


class FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)
        self.is_sparse = False

    def detach(self):
        return self

    def to(self, **_):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class FakeTorch:
    float32 = "float32"

    @staticmethod
    def inference_mode():
        return nullcontext()

    @staticmethod
    def as_tensor(values, **_):
        return FakeTensor(values)


class FakeSAE:
    cfg = FakeConfig()
    device = "cpu"
    dtype = "float32"

    def __init__(self):
        self.call_sizes = []

    def eval(self):
        self.evaluated = True
        return self

    def encode(self, tensor):
        values = tensor.values
        self.call_sizes.append(len(values))
        return FakeTensor(np.maximum(values[:, : self.cfg.d_sae], 0.0))


class StatefulFakeSAE(FakeSAE):
    def __init__(self, *, offset: float = 0.0, reverse: bool = False):
        super().__init__()
        entries = [
            ("W_dec", np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)),
            ("b_enc", np.array([offset, 0.0], dtype=np.float32)),
        ]
        self._state = dict(reversed(entries) if reverse else entries)

    def state_dict(self):
        return dict(self._state)


def _projector(monkeypatch, **kwargs):
    monkeypatch.setattr(integration, "_torch_module", lambda: FakeTorch)
    return SAELensProjector(
        FakeSAE(), release="test-release", sae_id="layer-3", **kwargs
    )


def _contract(*, layout="token"):
    contract = {
        "representation_family": "internal_activation",
        "model_id": "reader/model",
        "hook_name": "blocks.3.hook_resid_pre",
        "source_activation_preprocessing": "raw_hook_activation",
        "sae_input_normalization": "none",
        "activation_reshape": "none",
        "activation_layout": layout,
        "hook_layer": 3,
        "context_size": 128,
        "prepend_bos": True,
        "model_from_pretrained_kwargs": {"fold_ln": False},
        "exclude_special_tokens": False,
    }
    if layout == "one_token_per_item":
        contract["item_reduction"] = "single_token"
    return contract


def test_saelens_module_and_top_level_symbol_are_torch_free():
    code = (
        "from prefscope import SAELensProjector, SAELensTextBackend; "
        "from prefscope.api import SAELensProjector as ApiProjector; import sys; "
        "assert SAELensProjector is ApiProjector; "
        "assert SAELensTextBackend.__name__ == 'SAELensTextBackend'; "
        "assert 'torch' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_projector_wraps_saelens_encode_and_records_coordinate_contract(monkeypatch):
    projector = _projector(monkeypatch, batch_size=1)
    result = projector.project(np.array([[1.0, -2.0, 3.0], [4.0, 5.0, 6.0]]))

    np.testing.assert_allclose(result, [[1.0, 0.0], [4.0, 5.0]])
    assert projector.sae.call_sizes == [1, 1]
    assert projector.input_dim == 3
    assert projector.m_total == 2
    assert projector.activation_polarity == "nonnegative"
    assert projector.code_semantics == "numerical_activity"
    assert projector.representation_contract == _contract()
    assert projector.projector_provenance["backend"] == "saelens"
    assert projector.projector_provenance["coordinate_pin_status"] == (
        "reader_and_sae_unpinned"
    )
    assert len(projector.projector_provenance["sae_config_fingerprint"]) == 64
    assert projector.projector_provenance["sae_acquisition_status"] == (
        "unverified_loaded_state"
    )
    assert projector.projector_provenance["reader_identity_status"] == (
        "declared_unpinned"
    )


def test_loaded_saelens_state_has_exact_coordinate_only_identity():
    first = SAELensProjector(
        StatefulFakeSAE(),
        release="release-alias-a",
        sae_id="sae-alias-a",
        reader_model_revision="reader-a",
    )
    reordered = SAELensProjector(
        StatefulFakeSAE(reverse=True),
        release="release-alias-b",
        sae_id="sae-alias-b",
        reader_model_revision="reader-a",
    )
    changed_reader = SAELensProjector(
        StatefulFakeSAE(), reader_model_revision="reader-b"
    )
    changed = SAELensProjector(StatefulFakeSAE(offset=1.0))

    class AlternateStatefulFakeSAE(StatefulFakeSAE):
        pass

    changed_implementation = SAELensProjector(AlternateStatefulFakeSAE())
    changed_dtype_sae = StatefulFakeSAE()
    changed_dtype_sae._state["b_enc"] = changed_dtype_sae._state["b_enc"].astype(
        np.float64
    )
    changed_dtype = SAELensProjector(changed_dtype_sae)
    changed_shape_sae = StatefulFakeSAE()
    changed_shape_sae._state["b_enc"] = changed_shape_sae._state["b_enc"].reshape(2, 1)
    changed_shape = SAELensProjector(changed_shape_sae)

    first_identity = projector_feature_space_identity(
        first, input_rep="individual", backend="saelens"
    )
    reordered_identity = projector_feature_space_identity(
        reordered, input_rep="prompt", backend="saelens"
    )
    changed_reader_identity = projector_feature_space_identity(
        changed_reader, input_rep="individual", backend="saelens"
    )
    changed_identity = projector_feature_space_identity(
        changed, input_rep="individual", backend="saelens"
    )
    changed_implementation_identity = projector_feature_space_identity(
        changed_implementation, input_rep="individual", backend="saelens"
    )
    changed_dtype_identity = projector_feature_space_identity(
        changed_dtype, input_rep="individual", backend="saelens"
    )
    changed_shape_identity = projector_feature_space_identity(
        changed_shape, input_rep="individual", backend="saelens"
    )

    assert first.projector_provenance["sae_acquisition_status"] == (
        "observed_loaded_state"
    )
    assert first.projector_provenance["reader_identity_status"] == (
        "declared_unpinned"
    )
    assert first_identity["feature_space_status"] == "exact_weights"
    assert reordered_identity == first_identity
    assert changed_reader_identity["feature_space_id"] != first_identity["feature_space_id"]
    assert changed_identity["feature_space_id"] != first_identity["feature_space_id"]
    assert changed_implementation_identity["feature_space_id"] != (
        first_identity["feature_space_id"]
    )
    assert changed_dtype_identity["feature_space_id"] != first_identity["feature_space_id"]
    assert changed_shape_identity["feature_space_id"] != first_identity["feature_space_id"]


def test_saelens_feature_identity_includes_declared_model_and_hook(monkeypatch):
    first = SAELensProjector(StatefulFakeSAE())
    monkeypatch.setattr(FakeMetadata, "hook_name", "blocks.4.hook_resid_pre")
    second = SAELensProjector(StatefulFakeSAE())

    first_identity = projector_feature_space_identity(
        first, input_rep="individual", backend="saelens"
    )
    second_identity = projector_feature_space_identity(
        second, input_rep="individual", backend="saelens"
    )
    assert first.projector_provenance["sae_weights_sha256"] == (
        second.projector_provenance["sae_weights_sha256"]
    )
    assert first_identity["feature_space_id"] != second_identity["feature_space_id"]


def test_expected_saelens_weights_digest_verifies_or_fails_closed():
    observed = SAELensProjector(StatefulFakeSAE())
    expected = observed.projector_provenance["sae_weights_sha256"]
    lens = Lens.__new__(Lens)
    lens.projector = observed
    assert lens.sae_weights_sha256 == expected
    assert lens.sae_acquisition_status == "observed_loaded_state"

    verified = SAELensProjector(
        StatefulFakeSAE(), expected_sae_weights_sha256=f"sha256:{expected.upper()}"
    )
    assert verified.projector_provenance["sae_acquisition_status"] == (
        "expected_digest_verified"
    )
    with pytest.raises(ValueError, match="does not match.*expected.*observed"):
        SAELensProjector(
            StatefulFakeSAE(), expected_sae_weights_sha256="0" * 64
        )
    with pytest.raises(ValueError, match="64-character"):
        SAELensProjector(StatefulFakeSAE(), expected_sae_weights_sha256="main")
    with pytest.raises(ValueError, match="no digestible"):
        SAELensProjector(FakeSAE(), expected_sae_weights_sha256=expected)


def test_saelens_digest_supports_real_scalar_tensor_buffers():
    torch = pytest.importorskip("torch")
    first_sae = StatefulFakeSAE()
    first_sae._state["threshold"] = torch.tensor(0.5)
    second_sae = StatefulFakeSAE()
    second_sae._state["threshold"] = torch.tensor(0.75)

    first = SAELensProjector(first_sae)
    second = SAELensProjector(second_sae)

    assert len(first.projector_provenance["sae_weights_sha256"]) == 64
    assert first.projector_provenance["sae_weights_sha256"] != (
        second.projector_provenance["sae_weights_sha256"]
    )


def test_lens_from_saelens_forwards_expected_weights_digest(monkeypatch):
    projector = _projector(monkeypatch)
    captured = {}

    def fake(cls, release, sae_id, **kwargs):
        captured.update({"release": release, "sae_id": sae_id, **kwargs})
        return projector

    monkeypatch.setattr(
        integration.SAELensProjector, "from_pretrained", classmethod(fake)
    )
    Lens.from_saelens(
        "test-release", "layer-3", expected_sae_weights_sha256="a" * 64
    )

    assert captured["expected_sae_weights_sha256"] == "a" * 64


def test_saelens_token_projection_is_a_first_class_lens_backend(monkeypatch):
    projector = _projector(monkeypatch, batch_size=2)
    monkeypatch.setattr(
        integration.SAELensProjector,
        "from_pretrained",
        classmethod(lambda cls, release, sae_id, **_: projector),
    )
    lens = Lens.from_saelens("test-release", "layer-3")

    features = lens.project_saelens_tokens(
        row_ids=("a", "b"),
        token_activations={
            "response_a": np.array([[1.0, 0.0, 2.0], [3.0, 2.0, 1.0], [4.0, 1.0, 0.0]]),
            "response_b": np.array([[1.0, 1.0, 0.0], [0.0, 4.0, 1.0], [2.0, 3.0, 0.0]]),
        },
        token_row_ids={
            "response_a": ("a", "a", "b"),
            "response_b": ("a", "a", "b"),
        },
        representation_contract=_contract(),
        metadata={"group_id": ("g", "h")},
    )

    assert lens.pretrained_backend == "saelens"
    assert set(features.arrays) == {"z_a", "z_b", "z_diff"}
    np.testing.assert_allclose(features.array("z_a"), [[3.0, 2.0], [4.0, 1.0]])
    np.testing.assert_allclose(features.array("z_b"), [[1.0, 4.0], [2.0, 3.0]])
    np.testing.assert_allclose(features.array("z_diff"), [[2.0, -2.0], [2.0, -2.0]])
    assert features.provenance["token_reduction"] == "post_sae_max"
    assert features.provenance["lens"]["representation_compatibility"]["status"] == (
        "matched_declared_unpinned"
    )


def test_token_path_proves_encoding_happens_before_pooling(monkeypatch):
    class MixingSAE(FakeSAE):
        def encode(self, tensor):
            values = tensor.values
            self.call_sizes.append(len(values))
            return FakeTensor(
                np.column_stack(
                    (
                        np.maximum(values[:, 0] - values[:, 1], 0.0),
                        np.maximum(values[:, 1] - values[:, 0], 0.0),
                    )
                )
            )

    monkeypatch.setattr(integration, "_torch_module", lambda: FakeTorch)
    projector = SAELensProjector(MixingSAE(), batch_size=1)
    token_values = np.array([[3.0, 2.0, 0.0], [2.0, 3.0, 0.0]])

    post_sae_pool = projector.project_grouped(token_values, ("a", "a"), row_ids=("a",))
    pre_sae_pool = projector.project(token_values.max(axis=0, keepdims=True))

    np.testing.assert_allclose(post_sae_pool, [[1.0, 1.0]])
    np.testing.assert_allclose(pre_sae_pool, [[0.0, 0.0]])
    assert not np.array_equal(post_sae_pool, pre_sae_pool)


def test_default_saelens_lens_rejects_pre_sae_item_pooling(monkeypatch):
    representations = RepresentationBatch(
        row_ids=("a",),
        arrays={"response_a": np.ones((1, 3))},
        provenance={"representation_contract": _contract()},
    )
    lens = Lens(
        _projector(monkeypatch),
        representation_source=PrecomputedRepresentationSource(representations),
    )
    with pytest.raises(ValueError, match="encoding happens before pooling"):
        lens.featurize([PairItem("a", "p", "x")], views=("response_a",))

def test_explicit_single_token_item_policy_uses_standard_lens_path(monkeypatch):
    projector = _projector(monkeypatch, item_projection_policy="single_token")
    representations = RepresentationBatch(
        row_ids=("a",),
        arrays={"response_a": np.array([[2.0, 1.0, 0.0]])},
        provenance={"representation_contract": _contract(layout="one_token_per_item")},
    )
    lens = Lens(
        projector,
        representation_source=PrecomputedRepresentationSource(representations),
    )
    features = lens.featurize(
        [PairItem("a", "p", "x")], views=("response_a",)
    )
    np.testing.assert_allclose(features.array("z_a"), [[2.0, 1.0]])
    assert features.provenance["lens"]["representation_compatibility"]["status"] == (
        "matched_declared_unpinned"
    )

def test_saelens_token_source_rejects_wrong_activation_coordinates(monkeypatch):
    lens = Lens(_projector(monkeypatch))
    wrong = {**_contract(), "hook_name": "blocks.4.hook_resid_pre"}
    with pytest.raises(ValueError, match="incompatible.*hook_name"):
        lens.project_saelens_tokens(
            row_ids=("a",),
            token_activations={"response_a": np.ones((1, 3))},
            token_row_ids={"response_a": ("a",)},
            representation_contract=wrong,
        )


@pytest.mark.parametrize("input_rep", ["difference", "token"])
def test_saelens_rejects_unsupported_representation_roles(input_rep):
    with pytest.raises(ValueError, match="individual or prompt"):
        SAELensProjector(FakeSAE(), input_rep=input_rep)


def test_saelens_rejects_structured_and_temporal_architectures():
    class StructuredConfig(FakeConfig):
        reshape_activations = "hook_z"

    class StructuredSAE(FakeSAE):
        cfg = StructuredConfig()

    with pytest.raises(ValueError, match="structured hook adapter"):
        SAELensProjector(StructuredSAE())

    class TemporalConfig(FakeConfig):
        @staticmethod
        def architecture():
            return "temporal"

    class TemporalSAE(FakeSAE):
        cfg = TemporalConfig()

    with pytest.raises(ValueError, match="sequence-aware"):
        SAELensProjector(TemporalSAE())


def test_explicit_coordinate_contract_must_be_complete_and_consistent():
    with pytest.raises(ValueError, match="incomplete"):
        SAELensProjector(
            FakeSAE(),
            representation_contract={"representation_family": "internal_activation"},
        )
    with pytest.raises(ValueError, match="contradicts.*hook_name"):
        SAELensProjector(
            FakeSAE(),
            representation_contract={
                **_contract(),
                "hook_name": "blocks.4.hook_resid_pre",
            },
        )

    class BareMetadata:
        model_name = None
        hook_name = None

    class BareConfig(FakeConfig):
        metadata = BareMetadata()

    class BareSAE(FakeSAE):
        cfg = BareConfig()

    for change, message in (
        ({"model_id": None}, "model_id"),
        ({"hook_name": ""}, "hook_name"),
        ({"source_activation_preprocessing": "mean_pool"}, "raw_hook_activation"),
        ({"item_reduction": "mean"}, "must not declare"),
    ):
        with pytest.raises(ValueError, match=message):
            SAELensProjector(
                BareSAE(), representation_contract={**_contract(), **change}
            )


def test_registered_loader_uses_saelens_v650_api_and_blocks_unknown_repos(monkeypatch):
    captured = {}

    class Loader:
        @classmethod
        def from_pretrained(cls, **kwargs):
            captured.update(kwargs)
            return StatefulFakeSAE()

    root = types.ModuleType("sae_lens")
    root.SAE = Loader
    loading = types.ModuleType("sae_lens.loading")
    directory = types.ModuleType("sae_lens.loading.pretrained_saes_directory")
    directory.get_pretrained_saes_directory = lambda: {"release": object()}
    monkeypatch.setitem(sys.modules, "sae_lens", root)
    monkeypatch.setitem(sys.modules, "sae_lens.loading", loading)
    monkeypatch.setitem(
        sys.modules, "sae_lens.loading.pretrained_saes_directory", directory
    )

    projector = SAELensProjector.from_pretrained(
        "release", "sae-id", device="cpu", dtype="float32", force_download=True
    )
    assert projector.release == "release"
    assert captured == {
        "release": "release",
        "sae_id": "sae-id",
        "device": "cpu",
        "dtype": "float32",
        "force_download": True,
    }
    expected = projector.projector_provenance["sae_weights_sha256"]
    verified = SAELensProjector.from_pretrained(
        "release", "sae-id", expected_sae_weights_sha256=expected
    )
    assert verified.projector_provenance["sae_acquisition_status"] == (
        "expected_digest_verified"
    )
    captured_before_invalid = dict(captured)
    with pytest.raises(ValueError, match="64-character"):
        SAELensProjector.from_pretrained(
            "release", "sae-id", expected_sae_weights_sha256="main"
        )
    assert captured == captured_before_invalid
    with pytest.raises(ValueError, match="trusted registry"):
        SAELensProjector.from_pretrained("owner/repo", "id")


def test_dense_memory_budget_fails_before_allocation_but_selected_grouping_works(
    monkeypatch,
):
    class WideConfig(FakeConfig):
        d_sae = 100

    class WideSAE(FakeSAE):
        cfg = WideConfig()

        def encode(self, tensor):
            self.call_sizes.append(len(tensor.values))
            base = np.maximum(tensor.values[:, :1], 0.0)
            offsets = np.arange(self.cfg.d_sae, dtype=np.float32)[None, :]
            return FakeTensor(base + offsets)

    monkeypatch.setattr(integration, "_torch_module", lambda: FakeTorch)
    projector = SAELensProjector(WideSAE(), max_output_bytes=800, batch_size=10)
    with pytest.raises(ValueError, match="dense SAELens output"):
        projector.project(np.ones((3, 3)))
    lens = Lens(projector)
    features = lens.project_saelens_tokens(
        row_ids=("a", "b"),
        token_activations={
            "response_a": np.array(
                [
                    [1.0, 0.0, 0.0],
                    [10.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [8.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0],
                ]
            )
        },
        token_row_ids={"response_a": ("a", "b", "a", "b", "a")},
        representation_contract=dict(projector.representation_contract),
        feature_ids=(5, 2),
    )
    np.testing.assert_allclose(features.array("z_a"), [[10.0, 7.0], [15.0, 12.0]])
    assert features.feature_ids == (5, 2)
    assert projector.sae.call_sizes == [2, 2, 1]


def test_saelens_input_validation_is_fail_closed(monkeypatch):
    projector = _projector(monkeypatch)
    with pytest.raises(ValueError, match="shape"):
        projector.project(np.ones((2, 4)))
    with pytest.raises(ValueError, match="finite"):
        projector.project(np.array([[1.0, np.nan, 2.0]]))
    with pytest.raises(ValueError, match="positive integer"):
        projector.project(np.ones((1, 3)), batch=0)
    with pytest.raises(ValueError, match="activation_polarity"):
        SAELensProjector(FakeSAE(), activation_polarity="")


def test_real_saelens_v650_tiny_projector_matches_direct_encode_when_installed():
    torch = pytest.importorskip("torch")
    pytest.importorskip("sae_lens")
    from sae_lens.saes.jumprelu_sae import JumpReLUSAE, JumpReLUSAEConfig
    from sae_lens.saes.sae import SAEMetadata

    values = np.array(
        [
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [1.0, 1.0, 3.0],
        ],
        dtype=np.float32,
    )
    for normalization in ("none", "layer_norm"):
        sae = JumpReLUSAE(
            JumpReLUSAEConfig(
                d_in=3,
                d_sae=4,
                device="cpu",
                dtype="float32",
                normalize_activations=normalization,
                metadata=SAEMetadata(
                    model_name="test/reader",
                    hook_name="blocks.0.hook_resid_pre",
                    context_size=16,
                    prepend_bos=True,
                ),
            )
        )
        with torch.no_grad():
            sae.W_enc.zero_()
            sae.W_enc[:3, :3] = torch.eye(3)
            sae.b_enc.zero_()
            sae.threshold.fill_(0.1)
        projector = SAELensProjector(sae, input_rep="individual", batch_size=2)
        result = projector.project(values)
        with torch.inference_mode():
            direct = sae.encode(torch.as_tensor(values)).detach().cpu().numpy()

        assert np.any(direct != 0)
        np.testing.assert_allclose(result, direct)
        assert projector.representation_contract["sae_input_normalization"] == (
            normalization
        )


def test_real_saelens_v650_flat_transcoder_variants_when_installed():
    torch = pytest.importorskip("torch")
    pytest.importorskip("sae_lens")
    from sae_lens import (
        JumpReLUSkipTranscoder,
        JumpReLUSkipTranscoderConfig,
        JumpReLUTranscoder,
        JumpReLUTranscoderConfig,
        SkipTranscoder,
        SkipTranscoderConfig,
        Transcoder,
        TranscoderConfig,
    )
    from sae_lens.saes.sae import SAEMetadata

    variants = (
        (Transcoder, TranscoderConfig),
        (SkipTranscoder, SkipTranscoderConfig),
        (JumpReLUTranscoder, JumpReLUTranscoderConfig),
        (JumpReLUSkipTranscoder, JumpReLUSkipTranscoderConfig),
    )
    values = torch.tensor([[1.0, 0.0, -1.0]])
    for sae_class, cfg_class in variants:
        sae = sae_class(
            cfg_class(
                d_in=3,
                d_sae=4,
                d_out=3,
                apply_b_dec_to_input=False,
                metadata=SAEMetadata(
                    model_name="test/reader",
                    hook_name="blocks.0.hook_mlp_in",
                    hook_name_out="blocks.0.hook_mlp_out",
                ),
            )
        )
        projector = SAELensProjector(sae)
        result = projector.project(values.numpy())
        with torch.inference_mode():
            direct = sae.encode(values).detach().cpu().numpy()
        np.testing.assert_allclose(result, direct)
        assert result.shape == (1, 4)
        assert projector.activation_polarity == "nonnegative"
        grouped = projector.project_grouped(
            values.numpy(), ("a",), row_ids=("a",), feature_ids=(0, 1)
        )
        np.testing.assert_allclose(grouped, direct[:, :2])


@pytest.mark.slow
def test_registered_pretrained_saelens_checkpoint_when_enabled():
    if os.environ.get("PREFSCOPE_RUN_SAELENS_SMOKE") != "1":
        pytest.skip("set PREFSCOPE_RUN_SAELENS_SMOKE=1 for the network smoke")
    lens = Lens.from_saelens(
        "gpt2-small-res-jb", "blocks.8.hook_resid_pre", device="cpu"
    )
    result = lens.projector.project(np.zeros((1, 768), dtype=np.float32))

    assert result.shape == (1, 24576)
    assert np.isfinite(result).all()
    assert lens.projector.representation_contract["model_id"] == "gpt2-small"
    assert lens.projector.representation_contract["hook_name"] == (
        "blocks.8.hook_resid_pre"
    )


def test_text_backend_featurizes_prompt_and_both_responses_with_one_reader(monkeypatch):
    torch = pytest.importorskip("torch")

    class TorchSAE(FakeSAE):
        def encode(self, tensor):
            self.call_sizes.append(len(tensor))
            return torch.relu(tensor[:, :2])

    projector = SAELensProjector(TorchSAE(), release="test-release", sae_id="layer-3")
    monkeypatch.setattr(
        integration.SAELensProjector,
        "from_pretrained",
        classmethod(lambda cls, release, sae_id, **_: projector),
    )
    calls = []

    class Reader:
        def eval(self):
            return self

        def to_tokens(self, text, *, truncate, prepend_bos):
            assert truncate is False
            values = {
                "prompt one": [1, 2],
                "prompt two": [2, 1],
                "three": [3],
                "two": [2],
                "one": [1],
                "four": [4],
            }[text]
            return torch.tensor([[0, *values]] if prepend_bos else [values])

        def run_with_cache(
            self,
            tokens,
            *,
            names_filter,
            prepend_bos,
            return_type=None,
            stop_at_layer=None,
        ):
            assert prepend_bos is False
            assert return_type is None
            assert stop_at_layer == 4
            assert names_filter == [FakeMetadata.hook_name]
            values = tokens.to(torch.float32)
            hook = torch.stack((values, values + 1, values * 0), dim=-1)
            return None, {FakeMetadata.hook_name: hook}

    def reader_factory(**kwargs):
        calls.append(kwargs)
        return Reader()

    lens = Lens.from_saelens("test-release", "layer-3")
    lens.backend.reader_factory = reader_factory
    items = [
        PairItem("a", "prompt one", "three", "one", 1.0, meta={"group_id": "g1"}),
        PairItem("b", "prompt two", "two", "four", 0.0, meta={"group_id": "g2"}),
    ]
    features = lens.featurize(items, feature_ids=(1, 0), batch_size=2)

    assert tuple(features.arrays) == ("z_prompt", "z_a", "z_b", "z_diff")
    np.testing.assert_allclose(features.array("z_prompt"), [[3, 2], [3, 2]])
    np.testing.assert_allclose(features.array("z_a"), [[4, 3], [3, 2]])
    np.testing.assert_allclose(features.array("z_b"), [[2, 1], [5, 4]])
    np.testing.assert_allclose(features.array("z_diff"), [[2, 2], [-2, -2]])
    assert features.matrix("z_diff").activation_polarity == "signed"
    assert features.matrix("z_diff").code_semantics == "activity_difference"
    assert features.metadata["group_id"] == ("g1", "g2")
    assert features.provenance["text_context"] == "independent_documents"
    assert features.provenance["lens"]["feature_space_id"] == lens.feature_space_id
    assert features.provenance["lens"]["feature_space_status"] == "declared_unpinned"
    assert lens.backend.feature_space_identity == lens.feature_space_identity
    catalog = lens.feature_catalog.select(features.feature_ids)
    catalog.validate_for(features.matrix("z_a"))
    assert catalog.feature_ids == (1, 0)
    assert len(calls) == 1
    assert calls[0]["hook_names"] == [FakeMetadata.hook_name]


def test_text_backend_supports_proxy_tokenization_and_special_token_exclusion():
    torch = pytest.importorskip("torch")

    class ProxyMetadata(FakeMetadata):
        model_class_name = "AutoModelForCausalLM"
        prepend_bos = True
        exclude_special_tokens = True

    class ProxyConfig(FakeConfig):
        metadata = ProxyMetadata()

    class ProxySAE(FakeSAE):
        cfg = ProxyConfig()

        def encode(self, tensor):
            return torch.relu(tensor[:, :2])

    projector = SAELensProjector(ProxySAE())

    class Tokenizer:
        bos_token_id = 0
        eos_token_id = 99
        pad_token_id = None
        sep_token_id = None
        decoder_start_token_id = None
        additional_special_tokens_ids = [77]
        all_special_ids = [0, 99, 77]

    class HookedProxyLM:
        tokenizer = Tokenizer()

        def eval(self):
            return self

        def to_tokens(
            self,
            text,
            *,
            truncate,
            prepend_bos,
            move_to_device,
        ):
            assert text == "text"
            assert truncate is False
            assert prepend_bos is False
            assert move_to_device is False
            return torch.tensor([[99, 77, 2]])

        def run_with_cache(
            self,
            tokens,
            *,
            names_filter,
            prepend_bos,
            stop_at_layer,
            **kwargs,
        ):
            assert "return_type" not in kwargs
            assert tokens.tolist() == [[0, 99, 77, 2]]
            values = tokens.to(torch.float32)
            hook = torch.stack((values, values + 1, values * 0), dim=-1)
            return None, {FakeMetadata.hook_name: hook}

    backend = integration.SAELensTextBackend(
        projector,
        reader_factory=lambda **_: HookedProxyLM(),
        include_bos=False,
    )
    lens = Lens(projector, backend=backend)
    features = lens.featurize(
        [PairItem("a", "text", "text")],
        views=("prompt",),
    )
    np.testing.assert_allclose(features.array("z_prompt"), [[77.0, 78.0]])
    assert projector.representation_contract["exclude_special_tokens"] is True


def test_missing_saelens_metadata_exclusion_policy_resolves_to_false():
    class MetadataLikeSAELens(FakeMetadata):
        def __getattr__(self, name):
            return None

    class ConfigLikeSAELens(FakeConfig):
        metadata = MetadataLikeSAELens()

    class SAELike(FakeSAE):
        cfg = ConfigLikeSAELens()

    projector = SAELensProjector(SAELike())
    assert projector.representation_contract["exclude_special_tokens"] is False
