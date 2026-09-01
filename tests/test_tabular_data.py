import numpy as np
import pandas as pd
import pytest

from prefscope.data.tabular import (
    ColumnMapping,
    canonical_table_hash,
    canonicalize_table,
    load_hf_table,
    normalize_preference_labels,
)


@pytest.mark.parametrize("columns", [{"response": ["a"]}, {"prompt": ["q"]}])
def test_missing_required_semantic_column_is_actionable(columns):
    with pytest.raises(ValueError, match="could not resolve required field"):
        canonicalize_table(pd.DataFrame(columns), ColumnMapping())


def test_probability_pair_maps_to_canonical_schema():
    raw = pd.DataFrame({
        "question": ["q1", "q2"],
        "left": ["a1", "a2"],
        "right": ["b1", "b2"],
        "score": [1.0, 0.25],
    })
    out, summary = canonicalize_table(
        raw,
        ColumnMapping(
            prompt="question",
            response_a="left",
            response_b="right",
            label="score",
            label_mode="probability",
        ),
        source="unit",
    )
    assert list(out["human_pref"]) == [1.0, 0.25]
    assert list(out["completion_a"]) == ["a1", "a2"]
    assert summary["mode"] == "paired" and summary["has_preference"] is True


def test_winner_tokens_are_explicit_and_orientation_is_correct():
    raw = pd.DataFrame({
        "prompt": ["q1", "q2", "q3"],
        "a": ["a1", "a2", "a3"],
        "b": ["b1", "b2", "b3"],
        "winner": ["left", "right", "draw"],
    })
    mapping = ColumnMapping(
        response_a="a",
        response_b="b",
        label="winner",
        label_mode="winner",
        a_values=("left",),
        b_values=("right",),
        tie_values=("draw",),
    )
    out, _ = canonicalize_table(raw, mapping)
    assert list(out["human_pref"]) == [1.0, 0.0, 0.5]


def test_ambiguous_winner_values_are_never_guessed():
    labels = pd.Series([0, 1])
    with pytest.raises(ValueError, match="explicit a_values and b_values"):
        normalize_preference_labels(labels, mode="winner", n_rows=2)
    with pytest.raises(ValueError, match="unmapped winner"):
        normalize_preference_labels(
            labels,
            mode="winner",
            n_rows=2,
            a_values=("A",),
            b_values=("B",),
        )


def test_chosen_rejected_layout_sets_a_as_winner_without_label_column():
    raw = pd.DataFrame({
        "prompt": ["q"],
        "chosen": ["good"],
        "rejected": ["bad"],
    })
    out, _ = canonicalize_table(
        raw,
        ColumnMapping(
            response_a="chosen",
            response_b="rejected",
            label_mode="a-wins",
        ),
    )
    assert out.loc[0, "human_pref"] == 1.0


def test_structured_conversations_can_supply_prompt_and_both_responses():
    chosen = [
        {"role": "user", "content": "Explain entropy"},
        {"role": "assistant", "content": [{"type": "text", "text": "Clear answer"}]},
    ]
    rejected = [
        {"role": "user", "content": "Explain entropy"},
        {"role": "assistant", "content": "Wrong answer"},
    ]
    raw = pd.DataFrame({"chosen": [chosen], "rejected": [rejected]})
    out, _ = canonicalize_table(
        raw,
        ColumnMapping(
            prompt="chosen",
            response_a="chosen",
            response_b="rejected",
            prompt_role="user:first",
            response_a_role="assistant:last",
            response_b_role="assistant:last",
            label_mode="a-wins",
        ),
    )
    assert out.loc[0, "prompt"] == "Explain entropy"
    assert out.loc[0, "completion_a"] == "Clear answer"
    assert out.loc[0, "completion_b"] == "Wrong answer"


def test_single_response_does_not_treat_classification_label_as_preference():
    raw = pd.DataFrame({
        "prompt": ["q"],
        "response": ["r"],
        "label": ["topic-7"],
    })
    out, summary = canonicalize_table(
        raw, ColumnMapping(auto_pair=False))
    assert "human_pref" not in out.columns
    assert summary["mode"] == "single"


def test_empty_required_text_is_dropped_and_original_row_is_traced():
    raw = pd.DataFrame({
        "prompt": ["q1", ""],
        "response": ["r1", "r2"],
    })
    out, summary = canonicalize_table(
        raw, ColumnMapping(auto_pair=False))
    assert list(out["row_id"]) == [0]
    assert summary["dropped_empty_rows"] == 1


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("item_id", "changed-id"),
        ("prompt", "changed prompt"),
        ("completion_a", "changed response"),
        ("human_pref", 0.25),
    ],
)
def test_canonical_table_hash_binds_ids_text_and_labels(column, value):
    frame = pd.DataFrame({
        "row_id": [0, 1],
        "item_id": ["x", "y"],
        "prompt": ["q1", "q2"],
        "completion_a": ["a1", "a2"],
        "completion_b": ["b1", "b2"],
        "human_pref": [1.0, 0.0],
        "source": ["first/path", "first/path"],
    })
    original = canonical_table_hash(frame)
    changed = frame.copy()
    changed.loc[0, column] = value

    assert original.startswith("sha256:")
    assert canonical_table_hash(changed) != original
    assert canonical_table_hash(frame.iloc[::-1].reset_index(drop=True)) != original
    relocated = frame.copy()
    relocated["source"] = "other/path"
    assert canonical_table_hash(relocated) == original


def test_load_hf_table_pins_resolved_revision_and_attaches_provenance(monkeypatch):
    datasets = pytest.importorskip("datasets")
    from prefscope.data import tabular

    resolved = "a" * 40
    calls = {}

    class FakeDataset:
        def to_pandas(self):
            return pd.DataFrame({"prompt": ["q"], "response": ["r"]})

    def fake_resolve(dataset_id, revision, token=None):
        calls["resolve"] = (dataset_id, revision, token)
        return resolved

    def fake_load(dataset_id, **kwargs):
        calls["load"] = (dataset_id, kwargs)
        return FakeDataset()

    monkeypatch.setattr(tabular, "_resolve_hf_revision", fake_resolve)
    monkeypatch.setattr(datasets, "load_dataset", fake_load)
    frame = load_hf_table(
        "owner/data", revision="main", token="secret-token")

    assert calls["resolve"] == ("owner/data", "main", "secret-token")
    assert calls["load"][1]["revision"] == resolved
    assert frame.attrs["prefscope_hf_source"] == {
        "requested_revision": "main",
        "resolved_revision": resolved,
    }
    assert "secret-token" not in repr(frame.attrs)



def test_hf_revision_resolution_uses_hub_commit_sha(monkeypatch):
    import huggingface_hub
    from prefscope.data import tabular

    resolved = "d" * 40
    calls = {}

    class FakeApi:
        def dataset_info(self, **kwargs):
            calls.update(kwargs)
            return type("Info", (), {"sha": resolved})()

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    assert tabular._resolve_hf_revision(
        "owner/data", "release", token="secret-token") == resolved
    assert calls == {
        "repo_id": "owner/data",
        "revision": "release",
        "token": "secret-token",
    }


def test_canonicalize_table_retains_requested_scalar_metadata():
    raw = pd.DataFrame({
        "prompt": ["q1", "q2"], "answer": ["a1", "a2"],
        "helpfulness": [1.0, np.nan], "conversation_id": ["g", "h"],
    })
    mapped, _ = canonicalize_table(
        raw,
        ColumnMapping(
            response_a="answer",
            metadata=("helpfulness", "conversation_id"),
            auto_pair=False,
        ),
    )

    assert mapped["helpfulness"].iloc[0] == 1.0
    assert np.isnan(mapped["helpfulness"].iloc[1])
    assert mapped["conversation_id"].tolist() == ["g", "h"]


def test_canonicalize_table_rejects_missing_metadata_column():
    with pytest.raises(ValueError, match="metadata column"):
        canonicalize_table(
            pd.DataFrame({"prompt": ["q"], "response": ["a"]}),
            ColumnMapping(metadata=("rating",)),
        )


def test_canonical_hash_supports_retained_timestamp_metadata():
    raw = pd.DataFrame({
        "prompt": ["q"], "response": ["a"],
        "created_at": [pd.Timestamp("2025-01-01T12:00:00Z")],
    })
    mapped, summary = canonicalize_table(
        raw, ColumnMapping(metadata=("created_at",)))
    assert mapped.loc[0, "created_at"] == pd.Timestamp("2025-01-01T12:00:00Z")
    assert summary["canonical_table_hash"].startswith("sha256:")
