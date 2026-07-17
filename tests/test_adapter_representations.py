# tests/test_adapter_representations.py
import numpy as np
import pytest

from prefscope.core import registry
from prefscope.core.types import SideVectors
import prefscope.adapters  # noqa: F401  (triggers registration)


def _sv(a, b):
    return SideVectors(a=np.asarray(a, float), b=None if b is None else np.asarray(b, float), item_id="1")


def test_identity_returns_a_for_response_and_token():
    rep = registry.get("representation", "identity")()
    assert rep.compatible == frozenset({"response", "token"})
    out = rep.combine(_sv([[1, 2, 3]], [[9, 9, 9]]))
    np.testing.assert_array_equal(out, [[1, 2, 3]])
    tok = rep.combine(_sv([[1, 1], [2, 2]], None))
    assert tok.shape == (2, 2)


def test_diff_is_response_only_and_subtracts():
    rep = registry.get("representation", "diff")()
    assert rep.compatible == frozenset({"response"})
    out = rep.combine(_sv([[5, 5]], [[2, 1]]))
    np.testing.assert_array_equal(out, [[3, 4]])


def test_diff_requires_b():
    rep = registry.get("representation", "diff")()
    with pytest.raises(ValueError):
        rep.combine(_sv([[5, 5]], None))


def test_concat_joins_sides_on_feature_axis():
    rep = registry.get("representation", "concat")()
    out = rep.combine(_sv([[1, 2]], [[3, 4]]))
    np.testing.assert_array_equal(out, [[1, 2, 3, 4]])


def test_bothsides_stacks_rows():
    rep = registry.get("representation", "both")()
    out = rep.combine(_sv([[1, 2]], [[3, 4]]))
    np.testing.assert_array_equal(out, [[1, 2], [3, 4]])
    single = rep.combine(_sv([[1, 2]], None))
    np.testing.assert_array_equal(single, [[1, 2]])
