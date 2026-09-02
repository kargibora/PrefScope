from __future__ import annotations

import runpy
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from prefscope import FeatureBatch, FeatureCatalog, PairItem


BASIC_EXAMPLES = (
    "examples/inference/single_item.py",
    "examples/inference/local_dataset.py",
    "examples/inference/huggingface_dataset.py",
    "examples/training/train_completion_lens.py",
    "examples/analysis/outcome_association.py",
    "examples/analysis/preference_relevance.py",
    "examples/analysis/inspect_local_features.py",
)


def _load(path):
    loaded = runpy.run_path(path)
    return loaded["main"].__globals__


def _batch(rows, width=4):
    return FeatureBatch(
        row_ids=tuple(row.id for row in rows),
        arrays={"z_diff": np.ones((len(rows), width), dtype=np.float32)},
        roles={"z_diff": "response_difference"},
        orientations={"z_diff": "a_minus_b"},
        activation_polarity="signed",
        code_semantics="numerical_activity",
    )


class _Lens:
    input_rep = "individual"
    m_total = 4
    concept_names = None

    def __init__(self):
        self.rows = []
        descriptions = [None] * 100
        descriptions[0] = "name 0"
        self.feature_catalog = FeatureCatalog(
            pd.DataFrame({"feature_id": range(100), "description": descriptions})
        )

    def featurize(self, rows, **kwargs):
        self.rows = list(rows)
        if kwargs.get("views") == ("response_a",):
            return FeatureBatch(
                row_ids=tuple(row.id for row in self.rows),
                arrays={"z_a": np.ones((len(self.rows), 100), dtype=np.float32)},
                roles={"z_a": "response_a"},
                orientations={"z_a": "absolute_a"},
                activation_polarity="nonnegative",
                code_semantics="numerical_activity",
            )
        return _batch(self.rows, 100)


def _patch_run(ns, monkeypatch, tmp_path):
    lens = _Lens()
    monkeypatch.setitem(ns, "Lens", SimpleNamespace(from_config=lambda path: lens))
    monkeypatch.setitem(ns, "observe_run", lambda *args, **kwargs: nullcontext())
    monkeypatch.setitem(ns, "OUTPUT", tmp_path / "features")
    monkeypatch.setitem(ns, "EVENTS", tmp_path / "events.jsonl")
    monkeypatch.setitem(ns, "save_feature_batch", lambda *args, **kwargs: None)
    return lens


def test_basic_gallery_cards_are_small_self_contained_scripts():
    for path in BASIC_EXAMPLES:
        text = Path(path).read_text(encoding="utf-8")
        assert len(text.splitlines()) <= 80
        assert "def main() -> None:" in text
        assert "argparse" not in text
        assert "observe_run" in text


def test_single_item_card_prints_activity(monkeypatch, tmp_path, capsys):
    ns = _load(BASIC_EXAMPLES[0])
    lens = _patch_run(ns, monkeypatch, tmp_path)

    ns["main"]()

    assert lens.rows == [PairItem("example-0", ns["PROMPT"], ns["RESPONSE"])]
    output = capsys.readouterr().out
    assert "feature_id activation description" in output
    assert "name 0" in output


def test_local_table_card_uses_bundled_sample(monkeypatch, tmp_path, capsys):
    ns = _load(BASIC_EXAMPLES[1])
    lens = _patch_run(ns, monkeypatch, tmp_path)

    ns["main"]()

    assert len(lens.rows) == ns["LIMIT"]
    assert lens.rows[0].pref in (0.0, 0.5, 1.0)
    assert "mean active=" in capsys.readouterr().out


def test_hf_revision_is_optional_and_forwarded(monkeypatch, tmp_path, capsys):
    ns = _load(BASIC_EXAMPLES[2])
    lens = _patch_run(ns, monkeypatch, tmp_path)
    captured = {}

    class Dataset:
        def __iter__(self):
            yield PairItem("row-0", "prompt", "response A", "response B")

    def make(dataset, **kwargs):
        captured["dataset"] = dataset
        captured.update(kwargs)
        return Dataset()

    monkeypatch.setitem(ns, "HuggingFaceDataset", make)
    assert ns["REVISION"] is None

    ns["main"]()

    assert captured["revision"] is None
    assert captured["streaming"] is True
    assert captured["limit"] == ns["LIMIT"]
    assert len(lens.rows) == 1
    assert "streamed rows" in capsys.readouterr().out


def test_training_card_builds_balanced_toy_pairs(monkeypatch, tmp_path, capsys):
    ns = _load(BASIC_EXAMPLES[3])
    output = tmp_path / "lens"
    monkeypatch.setitem(ns, "OUTPUT", output)
    monkeypatch.setitem(ns, "observe_run", lambda *args, **kwargs: nullcontext())
    captured = {}

    def train(rows, *, config, out):
        captured.update(rows=list(rows), config=config, out=out)
        return SimpleNamespace(
            input_rep="individual", feature_table=pd.DataFrame(index=range(16))
        )

    monkeypatch.setitem(ns, "Lens", SimpleNamespace(train=train))

    ns["main"]()

    assert len(captured["rows"]) == 2 * len(ns["FACTS"])
    assert {row.pref for row in captured["rows"]} == {0.0, 1.0}
    assert captured["out"] == output
    assert "Trained individual lens" in capsys.readouterr().out


def test_outcome_association_card_runs_and_prints_estimand(
    monkeypatch, tmp_path, capsys
):
    ns = _load(BASIC_EXAMPLES[4])
    monkeypatch.setitem(ns, "EVENTS", tmp_path / "events.jsonl")
    monkeypatch.setitem(ns, "observe_run", lambda *args, **kwargs: nullcontext())

    ns["main"]()

    output = capsys.readouterr().out
    assert "Outcome associations:" in output
    assert "feature_id" in output


def test_preference_card_prints_descriptive_result(monkeypatch, tmp_path, capsys):
    ns = _load(BASIC_EXAMPLES[5])
    monkeypatch.setitem(ns, "EVENTS", tmp_path / "events.jsonl")
    monkeypatch.setitem(ns, "observe_run", lambda *args, **kwargs: nullcontext())
    lens = _Lens()

    def relevance(features):
        return pd.DataFrame(
            {
                "feature_id": [2],
                "n_fire": [12],
                "correlation": [0.25],
                "p_value": [0.2],
                "n_independent_groups": [12],
                "estimand": ["equal-group descriptive association"],
            }
        )

    lens.preference_relevance = relevance
    monkeypatch.setitem(ns, "Lens", SimpleNamespace(from_config=lambda path: lens))

    ns["main"]()

    output = capsys.readouterr().out
    assert "Preference relevance:" in output


def test_multi_row_inspection_card_keeps_row_identity(monkeypatch, tmp_path, capsys):
    ns = _load(BASIC_EXAMPLES[-1])
    lens = _patch_run(ns, monkeypatch, tmp_path)

    ns["main"]()

    output = capsys.readouterr().out
    assert len(lens.rows) == ns["LIMIT"]
    assert "row_id" in output.splitlines()[0]
    assert str(lens.rows[0].id) in output
    assert "rank" in output.splitlines()[0]
