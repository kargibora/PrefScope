from __future__ import annotations

import json

import numpy as np
import pytest

from prefscope.api import loaded_lens as loaded_lens_module
from prefscope.api.loaded_lens import Lens
from prefscope.core.features import FeatureBatch
from prefscope.core.lens_backend import LensBackend, LensCapabilities
from prefscope.core.types import PairItem
from prefscope.observability import observe_run
from prefscope.observability import runtime as runtime_module


class FakeBackend(LensBackend):
    input_rep = "individual"

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    @property
    def capabilities(self):
        return LensCapabilities(
            ("response_a", "response_b", "response_difference"),
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
        self.calls += 1
        rows = list(items)
        if self.fail:
            raise RuntimeError(
                f"private prompt: {rows[0].x}; /private/secret/lens; api_key=hidden"
            )
        selected = tuple(range(3)) if feature_ids is None else tuple(feature_ids)
        base = np.arange(len(rows) * 3, dtype=np.float32).reshape(len(rows), 3)
        all_arrays = {
            "z_a": base[:, selected] + 2,
            "z_b": base[:, selected],
            "z_diff": np.full((len(rows), len(selected)), 2, dtype=np.float32),
        }
        names = {
            "response_a": "z_a",
            "response_b": "z_b",
            "response_difference": "z_diff",
        }
        requested = tuple(views or self.capabilities.views)
        arrays = {names[view]: all_arrays[names[view]] for view in requested}
        return FeatureBatch(
            row_ids=tuple(item.id for item in rows),
            arrays=arrays,
            roles={name: view for view, name in names.items() if name in arrays},
            orientations={
                name: {
                    "z_a": "absolute_a",
                    "z_b": "absolute_b",
                    "z_diff": "a_minus_b",
                }[name]
                for name in arrays
            },
            feature_ids=selected,
            activation_polarity="nonnegative",
            code_semantics="numerical_activity",
        )


class OneShot:
    def __init__(self, rows) -> None:
        self.rows = rows
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        if self.iterations > 1:
            raise AssertionError("input iterator was consumed twice")
        yield from self.rows


def _items():
    return [
        PairItem(
            "private-row-1", "raw secret prompt", "raw answer A", "raw answer B",
            1.0, meta={"group_id": "private-group"},
        ),
        PairItem(
            "private-row-2", "another raw prompt", "answer C", "answer D",
            0.0, meta={"group_id": "another-private-group"},
        ),
    ]


def _read_events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture(autouse=True)
def _clean_environment_recorder(monkeypatch):
    runtime_module._close_environment_recorder()
    monkeypatch.delenv("PREFSCOPE_EVENTS_PATH", raising=False)
    yield
    runtime_module._close_environment_recorder()


def test_featurize_records_structural_outputs_without_consuming_input_twice(
    tmp_path,
):
    backend = FakeBackend()
    lens = Lens.from_backend(backend)
    rows = OneShot(_items())
    path = tmp_path / "events.jsonl"

    with observe_run(path, durable=False):
        features = lens.featurize(
            rows, views=("response_a", "response_difference"), feature_ids=(2, 0)
        )

    assert rows.iterations == 1
    assert backend.calls == 1
    assert features.array("z_diff").shape == (2, 2)
    events = _read_events(path)
    assert [(event["stage"], event["status"]) for event in events] == [
        ("featurize", "started"),
        ("featurize", "completed"),
    ]
    assert events[0]["data"]["input_rep"] == "individual"
    assert events[1]["data"]["n_rows"] == 2
    assert events[1]["data"]["n_features"] == 2
    assert events[1]["data"]["views"] == ["z_a", "z_diff"]
    assert events[1]["data"]["shapes"] == {"z_a": [2, 2], "z_diff": [2, 2]}
    text = path.read_text()
    for private in (
        "private-row", "raw secret prompt", "raw answer", "private-group"
    ):
        assert private not in text


def test_legacy_encode_pairs_records_nested_stage_and_preserves_return(tmp_path):
    lens = Lens.from_backend(FakeBackend())
    path = tmp_path / "events.jsonl"

    with observe_run(path, durable=False):
        codes, metadata = lens.encode_pairs(iter(_items()))

    np.testing.assert_allclose(codes, np.full((2, 3), 2, dtype=np.float32))
    assert list(metadata["id"]) == ["private-row-1", "private-row-2"]
    events = _read_events(path)
    assert [(event["stage"], event["status"]) for event in events] == [
        ("encode_pairs", "started"),
        ("encode_pairs", "completed"),
    ]
    assert events[-1]["data"]["shape"] == [2, 3]
    assert events[-1]["data"]["n_rows"] == 2
    assert events[-1]["data"]["n_features"] == 3


def test_failure_is_recorded_without_exception_or_input_leakage(tmp_path):
    lens = Lens.from_backend(FakeBackend(fail=True))
    path = tmp_path / "events.jsonl"

    with pytest.raises(RuntimeError, match="raw secret prompt"):
        with observe_run(path, durable=False):
            lens.featurize(_items(), views=("response_difference",))

    events = _read_events(path)
    assert [event["status"] for event in events] == ["started", "failed"]
    assert events[-1]["stage"] == "featurize"
    assert events[-1]["data"]["error_type"] == "RuntimeError"
    text = path.read_text()
    assert "raw secret prompt" not in text
    assert "/private/secret/lens" not in text
    assert "hidden" not in text
    assert "Traceback" not in text


def test_fetch_lens_nests_load_lens_without_identifier_or_path_leakage(
    tmp_path, monkeypatch,
):
    class FakeDownloadedLens(Lens):
        @classmethod
        @loaded_lens_module._observe_lens_result(
            "load_lens", source_kind="directory"
        )
        def from_dir(cls, lens_dir, **kwargs):
            del lens_dir, kwargs
            return cls.from_backend(FakeBackend())

    import prefscope.api.hub as hub

    monkeypatch.setattr(
        hub, "resolve_hf_revision", lambda *args, **kwargs: "private-revision"
    )
    monkeypatch.setattr(
        hub, "download_lens", lambda *args, **kwargs: "/private/secret/lens"
    )
    path = tmp_path / "events.jsonl"
    with observe_run(path, durable=False):
        lens = FakeDownloadedLens.from_pretrained(
            "private/repository", revision="private-tag", token="hf_private_token"
        )

    assert isinstance(lens, FakeDownloadedLens)
    events = _read_events(path)
    assert [(event["stage"], event["status"]) for event in events] == [
        ("fetch_lens", "started"),
        ("load_lens", "started"),
        ("load_lens", "completed"),
        ("fetch_lens", "completed"),
    ]
    assert events[1]["data"]["parent_operation_id"] == events[0]["data"][
        "operation_id"
    ]
    assert events[-1]["data"]["n_features"] == 3
    text = path.read_text()
    for private in (
        "private/repository", "private-revision", "private-tag",
        "/private/secret/lens", "hf_private_token",
    ):
        assert private not in text


def test_no_recorder_is_a_noop_and_alias_identity_is_preserved(tmp_path):
    backend = FakeBackend()
    lens = Lens.from_backend(backend)
    expected_path = tmp_path / "not-created.jsonl"

    codes, metadata = lens.project(_items())

    assert codes.shape == (2, 3)
    assert len(metadata) == 2
    assert backend.calls == 1
    assert not expected_path.exists()
    assert Lens.load.__func__ is Lens.from_dir.__func__
    assert Lens.project is Lens.encode_pairs


def test_custom_array_name_is_not_recorded_when_it_is_not_a_safe_identifier(
    tmp_path, monkeypatch,
):
    private_name = "private_customer_prompt_and_response"
    custom = FeatureBatch(
        row_ids=("row",),
        arrays={private_name: np.ones((1, 3), dtype=np.float32)},
        roles={private_name: "custom"},
        orientations={private_name: "unspecified"},
    )
    monkeypatch.setattr(
        loaded_lens_module, "project_representations",
        lambda *args, **kwargs: custom,
    )
    lens = Lens.from_backend(FakeBackend())
    path = tmp_path / "events.jsonl"

    with observe_run(path, durable=False):
        result = lens.project_representations(object())

    assert result is custom
    completed = _read_events(path)[-1]
    assert completed["data"]["n_views"] == 1
    assert completed["data"]["shapes"] == [[1, 3]]
    assert "views" not in completed["data"]
    assert private_name not in path.read_text()


def test_constructor_metadata_swallows_hostile_keyboard_interrupt_only_in_observer(
    tmp_path, monkeypatch,
):
    lens = Lens.from_backend(FakeBackend())

    class HostileBackend:
        def __init__(self):
            self.accesses = 0

        @property
        def m_total(self):
            self.accesses += 1
            raise KeyboardInterrupt(
                "/private/result/path api_key=must-not-change-success"
            )

    hostile = HostileBackend()
    lens.backend = hostile
    import prefscope.api.lens_config as lens_config

    monkeypatch.setattr(lens_config, "load_lens_config", lambda *args, **kwargs: lens)

    assert Lens.from_config({"private": "raw-config"}) is lens
    assert hostile.accesses == 0

    path = tmp_path / "events.jsonl"
    with observe_run(path, durable=False):
        result = Lens.from_config({"private": "raw-config"})

    assert result is lens
    assert hostile.accesses == 1
    events = _read_events(path)
    assert [(event["stage"], event["status"]) for event in events] == [
        ("load_lens", "started"),
        ("load_lens", "completed"),
    ]
    assert events[-1]["data"]["source_kind"] == "config"
    assert "n_features" not in events[-1]["data"]
    text = path.read_text()
    assert "raw-config" not in text
    assert "/private/result/path" not in text
    assert "must-not-change-success" not in text


@pytest.mark.parametrize(
    ("config", "expected_stages"),
    [
        (
            {"version": 1, "backend": "prefscope", "source": "native-lens"},
            ["load_lens", "load_lens"],
        ),
        (
            {
                "version": 1,
                "backend": "saelens",
                "release": "public-release",
                "sae_id": "public-sae",
            },
            ["load_lens", "load_lens"],
        ),
        (
            {
                "version": 1,
                "backend": "prefscope",
                "source": "hf://private-owner/private-repository",
            },
            [
                "load_lens", "fetch_lens", "load_lens", "load_lens",
                "fetch_lens", "load_lens",
            ],
        ),
    ],
)
def test_from_config_coalesces_load_delegates_but_preserves_fetch_load(
    tmp_path, monkeypatch, config, expected_stages,
):
    class ConfigLens(Lens):
        @classmethod
        @loaded_lens_module._observe_lens_result(
            "load_lens", source_kind="directory"
        )
        def from_dir(cls, lens_dir, **kwargs):
            del lens_dir, kwargs
            return cls.from_backend(FakeBackend())

        load = from_dir

        @classmethod
        @loaded_lens_module._observe_lens_result(
            "load_lens", source_kind="saelens"
        )
        def from_saelens(cls, release, sae_id, **kwargs):
            del release, sae_id, kwargs
            return cls.from_backend(FakeBackend())

    import prefscope
    import prefscope.api.hub as hub

    monkeypatch.setattr(loaded_lens_module, "Lens", ConfigLens)
    monkeypatch.setattr(prefscope, "Lens", ConfigLens)
    monkeypatch.setattr(hub, "resolve_hf_revision", lambda *args, **kwargs: "sha")
    monkeypatch.setattr(hub, "download_lens", lambda *args, **kwargs: "downloaded")

    path = tmp_path / "events.jsonl"
    with observe_run(path, durable=False):
        lens = ConfigLens.from_config(config)

    assert isinstance(lens, ConfigLens)
    events = _read_events(path)
    assert [event["stage"] for event in events] == expected_stages
    assert events[0]["status"] == "started"
    assert events[-1]["status"] == "completed"
    assert Lens.load.__func__ is Lens.from_dir.__func__
    assert Lens.project is Lens.encode_pairs
    assert "private-owner" not in path.read_text()
    assert "private-repository" not in path.read_text()


@pytest.mark.parametrize("operation", ["array", "feature_batch"])
def test_result_accessors_are_not_traversed_without_recorder_and_are_best_effort(
    tmp_path, monkeypatch, operation,
):
    class HostileResult:
        def __init__(self):
            self.accesses = 0

        @property
        def shape(self):
            self.accesses += 1
            raise RuntimeError("private result shape")

        @property
        def arrays(self):
            self.accesses += 1
            raise RuntimeError("private result arrays")

    hostile = HostileResult()
    lens = Lens.from_backend(FakeBackend())
    if operation == "array":
        monkeypatch.setattr(
            loaded_lens_module, "encode", lambda *args, **kwargs: hostile
        )
        def call():
            return lens.encode(["private prompt"], ["private response"])
    else:
        monkeypatch.setattr(
            loaded_lens_module, "project_representations",
            lambda *args, **kwargs: hostile,
        )
        def call():
            return lens.project_representations(object())

    assert call() is hostile
    assert hostile.accesses == 0

    path = tmp_path / f"{operation}.jsonl"
    with observe_run(path, durable=False):
        result = call()

    assert result is hostile
    assert hostile.accesses == 1
    assert [event["status"] for event in _read_events(path)] == [
        "started", "completed"
    ]


def test_preference_result_metadata_is_inactive_or_best_effort(
    tmp_path, monkeypatch,
):
    class HostileTable:
        def __init__(self):
            self.accesses = 0

        @property
        def shape(self):
            self.accesses += 1
            raise RuntimeError("private preference result")

    hostile = HostileTable()
    import prefscope.api.preference as preference_module

    monkeypatch.setattr(
        preference_module, "preference_relevance", lambda *args, **kwargs: hostile
    )
    lens = Lens.from_backend(FakeBackend())

    assert lens.preference_relevance(object()) is hostile
    assert hostile.accesses == 0

    path = tmp_path / "preference.jsonl"
    with observe_run(path, durable=False):
        result = lens.preference_relevance(object())

    assert result is hostile
    assert hostile.accesses == 1
    events = _read_events(path)
    assert [event["status"] for event in events] == ["started", "completed"]
    assert "output_rows" not in events[-1]["data"]


def test_keyboard_interrupt_from_core_operation_still_propagates(tmp_path):
    class InterruptingBackend(FakeBackend):
        def featurize(self, items, **kwargs):
            del items, kwargs
            raise KeyboardInterrupt("core operation interrupted")

    lens = Lens.from_backend(InterruptingBackend())
    path = tmp_path / "events.jsonl"

    with pytest.raises(KeyboardInterrupt, match="core operation interrupted"):
        with observe_run(path, durable=False):
            lens.featurize(_items(), views=("response_difference",))

    events = _read_events(path)
    assert [event["status"] for event in events] == ["started", "failed"]
    assert events[-1]["data"]["error_type"] == "KeyboardInterrupt"
