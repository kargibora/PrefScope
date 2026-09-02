from __future__ import annotations

import builtins
import io
import runpy
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture()
def example():
    return SimpleNamespace(
        **runpy.run_path("examples/advanced/presentations/compare_completions.py")
    )


def test_rank_helpers_are_finite_deterministic_and_bounded(example):
    activity = example._rank_activity(
        np.array([np.nan, 2.0, 2.0, np.inf, -1.0, 0.0]),
        (9, 2, 1, 3, 4, 6),
        50,
    )
    assert [(row.feature_id, row.value) for row in activity] == [
        (1, 2.0),
        (2, 2.0),
        (4, -1.0),
    ]

    differences = example._rank_differences(
        np.array([1.0, 5.0, -2.0, np.nan]),
        np.array([4.0, 3.0, 2.0, 0.0]),
        (7, 5, 6, 8),
        2,
    )
    assert [(row.feature_id, row.value) for row in differences] == [
        (6, -4.0),
        (7, -3.0),
    ]
    assert example._rank_differences([1.0, 2.0], [1.0, 0.0], (0, 1), 2) == (
        example._RankedValue(1, 2.0),
    )
    assert example._rank_signed_activity([0.0, -0.0, -3.0, 2.0], (0, 1, 2, 3), 4) == (
        example._RankedValue(2, -3.0),
        example._RankedValue(3, 2.0),
    )


def test_rank_helpers_reject_inexact_inputs(example):
    with pytest.raises(ValueError, match="one-dimensional"):
        example._rank_activity(np.ones((1, 2)), (0, 1), 1)
    with pytest.raises(ValueError, match="exactly match"):
        example._rank_activity([1.0, 2.0], (0,), 1)
    with pytest.raises(ValueError, match="unique"):
        example._rank_activity([1.0, 2.0], (0, 0), 1)
    with pytest.raises(ValueError, match="positive integer"):
        example._rank_activity([1.0], (0,), True)
    with pytest.raises(ValueError, match="same width"):
        example._rank_differences([1.0], [1.0, 2.0], (0,), 1)
    with pytest.raises(ValueError, match="must not exceed 50"):
        example._rank_activity([1.0], (0,), 51)


def test_optional_names_are_only_presented_as_proposed_labels(example):
    unnamed = SimpleNamespace(concept_names=None)
    named = SimpleNamespace(concept_names={0: "helpful tone", 1: "", 99: "outside"})

    assert example._proposed_labels(unnamed, (0, 1)) == {}
    assert example._proposed_labels(named, (0, 1)) == {0: "helpful tone"}
    table = example._plain_table(
        "Completion A",
        "Raw activity",
        (example._RankedValue(0, 1.25),),
        {0: "helpful tone"},
    )
    assert "Proposed label" in table
    assert "concept" not in table.lower()


def test_proposed_labels_are_normalized_bounded_and_not_rich_markup(
    example, monkeypatch
):
    hostile = "[red]do not style[/red]\nnext\x1b[31m" + ("x" * 200)
    labels = example._proposed_labels(SimpleNamespace(concept_names={2: hostile}), (2,))
    assert len(labels[2]) == 120
    assert "\n" not in labels[2]
    assert "\x1b" not in labels[2]

    captured = {}

    class FakeText:
        def __init__(self, value, style=None):
            self.value = value
            self.style = style

    class FakeTable:
        def __init__(self, **kwargs):
            self.rows = []

        def add_column(self, *args, **kwargs):
            pass

        def add_row(self, *cells):
            self.rows.append(cells)
            if len(cells) == 3:
                assert isinstance(cells[2], FakeText)
                assert cells[2].value == labels[2]

    class FakeConsole:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def print(self, value):
            pass

    monkeypatch.setitem(
        __import__("sys").modules, "rich.console", SimpleNamespace(Console=FakeConsole)
    )
    monkeypatch.setitem(
        __import__("sys").modules, "rich.table", SimpleNamespace(Table=FakeTable)
    )
    monkeypatch.setitem(
        __import__("sys").modules, "rich.text", SimpleNamespace(Text=FakeText)
    )
    rows = (example._RankedValue(2, 1.0),)
    example._render_comparison(rows, rows, rows, labels, stream=io.StringIO())
    assert captured["markup"] is False
    assert captured["highlight"] is False


def test_rich_uses_neutral_distinct_styles_and_explicit_caveat(example, monkeypatch):
    rendered_text = []
    title_styles = []

    class FakeText:
        def __init__(self, value, style=None):
            self.value = value
            self.style = style

    class FakeTable:
        def __init__(self, **kwargs):
            title_styles.append(kwargs.get("title_style"))

        def add_column(self, *args, **kwargs):
            pass

        def add_row(self, *cells):
            rendered_text.extend(cell for cell in cells if isinstance(cell, FakeText))

    class FakeConsole:
        def __init__(self, **kwargs):
            pass

        def print(self, value):
            if isinstance(value, FakeText):
                rendered_text.append(value)

    monkeypatch.setitem(
        __import__("sys").modules, "rich.console", SimpleNamespace(Console=FakeConsole)
    )
    monkeypatch.setitem(
        __import__("sys").modules, "rich.table", SimpleNamespace(Table=FakeTable)
    )
    monkeypatch.setitem(
        __import__("sys").modules, "rich.text", SimpleNamespace(Text=FakeText)
    )

    rows = (
        example._RankedValue(1, 2.0),
        example._RankedValue(2, -1.0),
    )
    example._render_comparison(rows, rows, rows, {}, stream=io.StringIO())

    value_styles = {text.style for text in rendered_text if text.value in {"+2", "-1"}}
    assert value_styles == {"cyan", "magenta"}
    assert "green" not in value_styles
    assert "red" not in value_styles
    assert title_styles[:2] == ["bold cyan", "bold magenta"]
    note = next(text.value for text in rendered_text if text.style == "dim")
    assert "reward, a winner, or semantic presence" in note


def test_rich_absent_uses_clean_plain_text_fallback(example, monkeypatch):
    real_import = builtins.__import__

    def without_rich(name, *args, **kwargs):
        if name == "rich" or name.startswith("rich."):
            raise ImportError("Rich intentionally unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_rich)
    output = io.StringIO()
    rows = (example._RankedValue(4, 2.5),)
    example._render_comparison(rows, rows, rows, {}, stream=output)

    rendered = output.getvalue()
    assert "Completion A" in rendered
    assert "Completion B" in rendered
    assert "Strongest signed A-minus-B differences" in rendered
    assert "Feature ID | Raw activity" in rendered
    assert "Proposed label" not in rendered
    assert "raw completion-lens SAE activity" in rendered
    assert "does not indicate reward, a winner, or semantic presence" in rendered


def test_main_uses_one_pair_native_lens_and_automatic_wrapper(
    example, monkeypatch, capsys
):
    import prefscope
    import prefscope.observability

    calls = []

    class FakeFeatures:
        feature_ids = (3, 8, 12)

        def array(self, view):
            calls.append(("array", view))
            return {
                "z_a": np.array([[2.0, 0.0, 1.0]]),
                "z_b": np.array([[0.5, 3.0, 1.0]]),
            }[view]

    class FakeLensInstance:
        concept_names = None
        input_rep = "individual"

        def featurize(self, items, *, views):
            calls.append(("featurize", items, views))
            return FakeFeatures()

    class FakeLens:
        @classmethod
        def from_pretrained(cls, repo, **kwargs):
            calls.append(("from_pretrained", repo, kwargs))
            return FakeLensInstance()

    @contextmanager
    def fake_observe_run(path, *, pretty):
        calls.append(("observe_enter", path, pretty))
        try:
            yield object()
        finally:
            calls.append(("observe_exit",))

    monkeypatch.setattr(prefscope, "Lens", FakeLens)
    monkeypatch.setattr(prefscope.observability, "observe_run", fake_observe_run)

    example.main(
        [
            "--lens-repo",
            "user-repo",
            "--subfolder",
            "completion-lens",
            "--prompt",
            "Prompt text",
            "--completion-a",
            "First answer",
            "--completion-b",
            "Second answer",
            "--device",
            "cpu",
            "--top-k",
            "2",
            "--events",
            "trace.jsonl",
            "--no-pretty",
        ]
    )

    assert calls[0] == ("observe_enter", "trace.jsonl", False)
    assert calls[1] == (
        "from_pretrained",
        "user-repo",
        {
            "subfolder": "completion-lens",
            "revision": None,
            "device": "cpu",
            "local_files_only": False,
        },
    )
    assert calls[2][0] == "featurize"
    item = calls[2][1][0]
    assert len(calls[2][1]) == 1
    assert (item.x, item.y_a, item.y_b) == (
        "Prompt text",
        "First answer",
        "Second answer",
    )
    assert calls[2][2] == ("response_a", "response_b")
    assert calls[3] == ("observe_exit",)
    assert ("array", "z_a") in calls
    assert ("array", "z_b") in calls
    output = capsys.readouterr().out
    assert "Proposed label" not in output
    assert "Prompt text" not in output
    assert "First answer" not in output
    assert "Second answer" not in output


def test_main_difference_lens_reports_only_signed_direct_contrast_without_events(
    example, monkeypatch, capsys, tmp_path
):
    import prefscope
    import prefscope.observability

    calls = []

    class FakeFeatures:
        feature_ids = (4, 9, 12)

        def array(self, view):
            calls.append(("array", view))
            assert view == "z_diff"
            return np.array([[1.0, -3.0, 0.5]])

    class FakeDifferenceLens:
        input_rep = "difference"
        concept_names = {4: "positive pole", 9: "opposite pole"}

        def featurize(self, items, *, views):
            calls.append(("featurize", items, views))
            return FakeFeatures()

    class FakeLens:
        @classmethod
        def from_dir(cls, path, **kwargs):
            calls.append(("from_dir", path, kwargs))
            return FakeDifferenceLens()

    def fake_observe_run(path, *, pretty):
        raise AssertionError("observe_run must not be called without --events")

    monkeypatch.setattr(prefscope, "Lens", FakeLens)
    monkeypatch.setattr(prefscope.observability, "observe_run", fake_observe_run)
    monkeypatch.chdir(tmp_path)
    example.main(["--lens-dir", "direct-lens", "--top-k", "2"])

    featurize = next(call for call in calls if call[0] == "featurize")
    assert featurize[2] == ("response_difference",)
    assert ("array", "z_diff") in calls
    output = capsys.readouterr().out
    assert "Direct difference lens" in output
    assert "Positive-pole proposed label" in output
    assert "Completion A minus Completion B" in output
    assert "f(e_A) - f(e_B)" in output
    assert "Completion A" not in output.replace("Completion A minus Completion B", "")
    assert "reward" in output
    assert "winner" in output
    assert "semantic presence" in output
    assert list(tmp_path.iterdir()) == []
    item = featurize[1][0]
    assert (item.x, item.y_a, item.y_b) == (
        example._DEFAULT_PROMPT,
        example._DEFAULT_COMPLETION_A,
        example._DEFAULT_COMPLETION_B,
    )


@pytest.mark.parametrize(
    "custom_args",
    [
        ["--prompt", "prompt"],
        ["--completion-a", "answer a"],
        ["--completion-b", "answer b"],
        ["--prompt", "prompt", "--completion-a", "answer a"],
        ["--prompt", "prompt", "--completion-b", "answer b"],
        ["--completion-a", "answer a", "--completion-b", "answer b"],
    ],
)
def test_comparison_text_rejects_every_partial_custom_triplet(example, custom_args):
    args = example._parser().parse_args(["--lens-dir", "lens", *custom_args])
    with pytest.raises(ValueError, match="must be provided together"):
        example._comparison_text(args)


def test_comparison_text_uses_atomic_demo_or_complete_custom_triplet(example):
    demo = example._parser().parse_args(["--lens-dir", "lens"])
    assert example._comparison_text(demo) == (
        example._DEFAULT_PROMPT,
        example._DEFAULT_COMPLETION_A,
        example._DEFAULT_COMPLETION_B,
    )

    custom = example._parser().parse_args(
        [
            "--lens-dir",
            "lens",
            "--prompt",
            "custom prompt",
            "--completion-a",
            "custom a",
            "--completion-b",
            "custom b",
        ]
    )
    assert example._comparison_text(custom) == (
        "custom prompt",
        "custom a",
        "custom b",
    )


def test_partial_custom_text_fails_before_explicit_event_file(
    example, monkeypatch, tmp_path
):
    import prefscope.observability

    def fake_observe_run(path, *, pretty):
        raise AssertionError("validation must finish before event recording")

    monkeypatch.setattr(prefscope.observability, "observe_run", fake_observe_run)
    event_path = tmp_path / "events.jsonl"
    with pytest.raises(SystemExit, match="must be provided together") as error:
        example.main(
            [
                "--lens-dir",
                "lens",
                "--prompt",
                "private prompt",
                "--events",
                str(event_path),
            ]
        )
    assert "private prompt" not in str(error.value)
    assert not event_path.exists()


def test_parser_requires_one_source_and_pretty_defaults_true(example):
    with pytest.raises(SystemExit):
        example._parser().parse_args([])
    args = example._parser().parse_args(
        ["--lens-repo", "owner/lens", "--subfolder", "completion"]
    )
    assert args.pretty is True
    assert args.events is None
    assert args.local_files_only is False
    with pytest.raises(SystemExit):
        example._parser().parse_args(["--lens-dir", "lens", "--top-k", "51"])


def test_loader_supports_local_and_explicit_saelens_sources(example):
    calls = []

    class FakeLens:
        @classmethod
        def from_dir(cls, path, **kwargs):
            calls.append(("dir", path, kwargs))
            return object()

        @classmethod
        def from_saelens(cls, release, sae_id, **kwargs):
            calls.append(("saelens", release, sae_id, kwargs))
            return object()

    local = example._parser().parse_args(["--lens-dir", "native-lens"])
    example._load_lens(local, FakeLens)
    sae = example._parser().parse_args(
        [
            "--saelens-release",
            "trusted-release",
            "--sae-id",
            "hook-id",
            "--allow-unregistered-release",
        ]
    )
    example._load_lens(sae, FakeLens)

    assert calls[0] == ("dir", "native-lens", {"device": "cpu"})
    assert calls[1] == (
        "saelens",
        "trusted-release",
        "hook-id",
        {
            "input_rep": "individual",
            "device": "cpu",
            "long_text_policy": "truncate",
            "include_bos": False,
            "allow_unregistered_release": True,
        },
    )
    invalid = example._parser().parse_args(["--saelens-release", "release"])
    with pytest.raises(ValueError, match="--sae-id is required"):
        example._validate_lens_options(invalid)
