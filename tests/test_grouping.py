from __future__ import annotations

import pandas as pd
import pytest

from prefscope.analysis.grouping import factorize_group_ids, resolve_group_ids


def test_resolve_group_ids_prefers_explicit_and_canonical_columns():
    frame = pd.DataFrame({"prompt": ["same", "same"], "group_id": ["a", "b"], "fold": [1, 1]})
    assert resolve_group_ids(frame).tolist() == ["a", "b"]
    assert resolve_group_ids(frame, group_col="fold").tolist() == [1, 1]


def test_resolve_group_ids_hashes_normalized_prompts():
    frame = pd.DataFrame({"prompt": [" question\r\n", "question\n", "other"]})
    groups = resolve_group_ids(frame)
    assert groups[0] == groups[1]
    assert groups[0] != groups[2]


def test_resolve_group_ids_fails_closed_for_bad_explicit_groups():
    with pytest.raises(ValueError, match="absent"):
        resolve_group_ids(pd.DataFrame({"prompt": ["p"]}), group_col="missing")
    with pytest.raises(ValueError, match="missing values"):
        resolve_group_ids(pd.DataFrame({"group_id": [None]}))


def test_resolve_group_ids_returns_none_without_grouping_information():
    assert resolve_group_ids(pd.DataFrame({"response": ["x"]})) is None


def test_explicit_group_ids_preserve_type_identity():
    groups = resolve_group_ids(
        pd.DataFrame({"group": [1, "1"]}), group_col="group")
    assert groups.tolist() == [1, "1"]


def test_factorization_preserves_mixed_type_identity():
    codes, n_groups = factorize_group_ids([1, True, 1.0, "1"])
    assert n_groups == 4
    assert len(set(codes.tolist())) == 4


def test_group_validation_preserves_hashable_tuple_ids_as_scalars():
    values = [("prompt", 1), ("prompt", 1), ("prompt", 2)]
    codes, n_groups = factorize_group_ids(values)
    assert n_groups == 2
    assert codes.tolist() == [0, 0, 1]
