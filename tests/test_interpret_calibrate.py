import json
import re

import numpy as np
import pandas as pd
import pytest

from prefscope.interpret.calibrate import (
    _deterministic_group_split,
    calibrate_single_text_features,
    sample_calibration_rows,
    sample_confirmation_rows,
    select_semantic_threshold,
    wilson_lower_bound,
)


def test_semantic_threshold_uses_precision_lower_bound_not_point_estimate():
    # Lowest threshold is only 75% precise; the upper 20/20 cases support threshold 2.
    samples = [
        {"kind": "active", "activation": 1.0 + i / 100, "threshold": 1.0,
         "present": i >= 10}
        for i in range(40)
    ]
    for sample in samples[20:]:
        sample["activation"] = 2.0 + sample["activation"]
        sample["threshold"] = 2.0
        sample["present"] = True
    result = select_semantic_threshold(samples, target_precision=0.8, min_above=20)
    assert result["chosen"] is not None
    assert result["chosen"]["threshold"] == 2.0
    assert result["chosen"]["precision"] == 1.0
    assert result["chosen"]["precision_lcb"] > 0.8
    assert wilson_lower_bound(15, 20) < 0.8


def test_calibration_sampling_spans_activation_range_and_deduplicates_prompts():
    # Two response rows per instruction; some prompts have one active and one silent side.
    ids = [f"p{i // 2}" for i in range(600)]
    z = np.zeros(600, dtype=np.float32)
    z[:400:2] = np.linspace(0.01, 10.0, 200)
    rows = sample_calibration_rows(
        z, ids, seed=0, feature_id=7, n_per_bin=3, n_top=5, n_zero=10)
    active = [r for r in rows if r["kind"] == "active"]
    # Every nonempty rank stratum contributes, including weak and top activations.
    assert {r["bin"] for r in active} == set(range(6))
    assert min(r["activation"] for r in active) < 1.0
    assert max(r["activation"] for r in active) > 9.0
    # A prompt with any active response cannot reappear as a silent control.
    active_ids = {ids[r["row_index"]] for r in active}
    silent_ids = {ids[r["row_index"]] for r in rows if r["kind"] == "silent"}
    assert len(silent_ids) == 10
    assert active_ids.isdisjoint(silent_ids)
    assert len([ids[r["row_index"]] for r in rows]) == len(
        set(ids[r["row_index"]] for r in rows))


def test_group_split_is_deterministic_and_keeps_paired_rows_together():
    ids = [f"prompt-{index // 2}" for index in range(40)]
    selection, confirmation = _deterministic_group_split(ids, seed=17)
    repeated_selection, repeated_confirmation = _deterministic_group_split(ids, seed=17)
    np.testing.assert_array_equal(selection, repeated_selection)
    np.testing.assert_array_equal(confirmation, repeated_confirmation)
    assert not np.any(selection & confirmation)
    assert np.all(selection | confirmation)
    for index in range(0, len(ids), 2):
        assert selection[index] == selection[index + 1]
        assert confirmation[index] == confirmation[index + 1]


def test_confirmation_sampling_is_uniform_conditional_and_group_disjoint():
    ids = [f"prompt-{index // 2}" for index in range(200)]
    z = np.zeros(200, dtype=np.float32)
    z[:120] = np.linspace(0.01, 2.0, 120)
    rows = sample_confirmation_rows(
        z, ids, threshold=1.0, pool=np.ones(200, dtype=bool),
        seed=3, feature_id=9, n_active=12, n_zero=8)
    repeated = sample_confirmation_rows(
        z, ids, threshold=1.0, pool=np.ones(200, dtype=bool),
        seed=3, feature_id=9, n_active=12, n_zero=8)
    assert [
        (row["row_index"], row["stage"], row["kind"], row["activation"])
        for row in rows
    ] == [
        (row["row_index"], row["stage"], row["kind"], row["activation"])
        for row in repeated
    ]
    active = [row for row in rows if row["kind"] == "active"]
    silent = [row for row in rows if row["kind"] == "silent"]
    assert len(active) == 12
    assert len(silent) == 8
    assert all(z[row["row_index"]] >= 1.0 for row in active)
    active_groups = {ids[row["row_index"]] for row in active}
    silent_groups = {ids[row["row_index"]] for row in silent}
    assert len(active_groups) == len(active)
    assert len(silent_groups) == len(silent)
    assert active_groups.isdisjoint(silent_groups)
    assert all(
        np.all(z[np.asarray(ids) == group] == 0) for group in silent_groups
    )


def test_wilson_requires_enough_perfect_labels():
    assert wilson_lower_bound(20, 20) == pytest.approx(0.8389, abs=1e-3)
    assert wilson_lower_bound(8, 8) < 0.8


class _SemanticClient:
    def raw(self, messages, **kwargs):
        prompt = messages[-1]["content"]
        labels = []
        for sid, body in re.findall(r"<sample id=(\d+)>(.*?)</sample>", prompt, re.S):
            present = "ZQX" in body
            labels.append({
                "id": int(sid), "concept_present": present,
                "explicitly_requested": "no",
                "role": "response_policy" if present else "not_present",
            })
        return json.dumps({"labels": labels})


class _DroppingClient(_SemanticClient):
    def raw(self, messages, **kwargs):
        prompt = messages[-1]["content"]
        labels = []
        for sid, body in re.findall(r"<sample id=(\d+)>(.*?)</sample>", prompt, re.S):
            if "DROP" in body:
                continue
            present = "ZQX" in body
            labels.append({
                "id": int(sid), "concept_present": present,
                "explicitly_requested": "no",
                "role": "response_policy" if present else "not_present",
            })
        return json.dumps({"labels": labels})


def test_full_presence_calibration_selects_then_confirms_on_disjoint_groups():
    z = np.zeros((5000, 1), dtype=np.float32)
    z[:4000, 0] = np.linspace(0.01, 10.0, 4000)
    texts = [f"{'ZQX ' if value > 0 else ''}response {i}"
             for i, value in enumerate(z[:, 0])]
    names = pd.DataFrame({"feature_id": [0], "concept": ["declines a request"]})
    out = calibrate_single_text_features(
        texts, z, names, _SemanticClient(), instruction_ids=list(range(5000)),
        n_per_bin=4, n_top=20, n_zero=10, batch_size=8,
        target_precision=0.5, min_above=5, max_silent_rate=0.2)
    row = out.iloc[0]
    assert row["selection_status"] == "threshold_selected"
    assert row["confirmation_status"] == "confirmed"
    assert row["calibration_status"] == "calibrated"
    assert bool(row["presence_pass"]) is True
    assert row["semantic_threshold"] > 0
    assert row["precision"] == row["confirmation_precision"] == 1.0
    assert row["precision_lcb"] == row["confirmation_precision_lcb"]
    assert row["semantic_role"] == "response_policy"

    samples = json.loads(row["samples_json"])
    selection_ids = {
        sample["row_index"] for sample in samples if sample["stage"] == "selection"
    }
    confirmation_ids = {
        sample["row_index"] for sample in samples if sample["stage"] == "confirmation"
    }
    assert selection_ids.isdisjoint(confirmation_ids)
    assert row["selection_group_count"] + row["confirmation_group_count"] == 5000
    assert row["confirmation_n"] >= 5
    assert row["confirmation_silent_n"] == row["confirmation_silent_required_n"]


def test_selection_success_cannot_substitute_for_confirmation_failure():
    ids = list(range(2000))
    selection, _ = _deterministic_group_split(ids, seed=0)
    z = np.zeros((2000, 1), dtype=np.float32)
    z[:1600, 0] = np.linspace(0.01, 10.0, 1600)
    texts = []
    for index, value in enumerate(z[:, 0]):
        marker = "ZQX " if selection[index] and value >= 9.8 else ""
        texts.append(f"{marker}response {index}")
    names = pd.DataFrame({"feature_id": [0], "concept": ["declines a request"]})
    row = calibrate_single_text_features(
        texts, z, names, _SemanticClient(), instruction_ids=ids,
        n_per_bin=4, n_top=20, n_zero=10, batch_size=8,
        target_precision=0.4, min_above=3, max_silent_rate=0.2,
        seed=0,
    ).iloc[0]

    assert row["selection_status"] == "threshold_selected"
    assert row["selection_precision_lcb"] >= 0.4
    assert row["confirmation_status"] == "failed_precision"
    assert row["calibration_status"] == "confirmation_failed"
    assert row["confirmation_precision"] == 0.0
    assert row["precision"] == row["confirmation_precision"]
    assert bool(row["presence_pass"]) is False
    assert np.isfinite(row["semantic_threshold"])


def test_confirmation_silent_leakage_blocks_presence_pass():
    ids = list(range(5000))
    _, confirmation = _deterministic_group_split(ids, seed=11)
    z = np.zeros((5000, 1), dtype=np.float32)
    z[:4000, 0] = np.linspace(0.01, 10.0, 4000)
    texts = []
    for index, value in enumerate(z[:, 0]):
        active_marker = value > 0
        leaked_confirmation_control = confirmation[index] and value == 0
        marker = "ZQX " if active_marker or leaked_confirmation_control else ""
        texts.append(f"{marker}response {index}")
    names = pd.DataFrame({"feature_id": [0], "concept": ["declines a request"]})
    row = calibrate_single_text_features(
        texts, z, names, _SemanticClient(), instruction_ids=ids,
        n_per_bin=4, n_top=20, n_zero=10, batch_size=8,
        target_precision=0.5, min_above=5, max_silent_rate=0.2,
        seed=11,
    ).iloc[0]

    assert row["selection_status"] == "threshold_selected"
    assert row["confirmation_precision_lcb"] >= 0.5
    assert row["confirmation_silent_n"] == row["confirmation_silent_required_n"]
    assert row["confirmation_silent_concept_rate"] == 1.0
    assert row["confirmation_status"] == "silent_leakage"
    assert row["calibration_status"] == "silent_leakage"
    assert bool(row["presence_pass"]) is False


def test_selected_threshold_requires_enough_confirmation_labels():
    ids = list(range(2000))
    _, confirmation = _deterministic_group_split(ids, seed=23)
    z = np.zeros((2000, 1), dtype=np.float32)
    z[:1600, 0] = np.linspace(0.01, 10.0, 1600)
    texts = []
    for index, value in enumerate(z[:, 0]):
        marker = "ZQX " if value > 0 else ""
        drop = "DROP " if confirmation[index] and value > 0 else ""
        texts.append(f"{marker}{drop}response {index}")
    names = pd.DataFrame({"feature_id": [0], "concept": ["declines a request"]})
    row = calibrate_single_text_features(
        texts, z, names, _DroppingClient(), instruction_ids=ids,
        n_per_bin=4, n_top=20, n_zero=10, batch_size=8,
        target_precision=0.5, min_above=5, max_silent_rate=0.2,
        seed=23,
    ).iloc[0]

    assert row["selection_status"] == "threshold_selected"
    assert row["confirmation_n"] == 0
    assert row["confirmation_status"] == "insufficient"
    assert row["calibration_status"] == "confirmation_insufficient"
    assert bool(row["confirmation_precision_pass"]) is False
    assert bool(row["presence_pass"]) is False
