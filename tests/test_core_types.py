import numpy as np
from prefscope.core.types import PairItem, SideVectors


def test_pairitem_single_vs_pair():
    pair = PairItem(id="1", x="q", y_a="a", y_b="b", pref=0.7)
    single = PairItem(id="2", x="q", y_a="a")
    assert pair.is_single is False
    assert single.is_single is True
    assert single.y_b is None and single.pref is None
    assert pair.meta == {} and single.meta == {}


def test_pairitem_is_frozen():
    p = PairItem(id="1", x="q", y_a="a")
    try:
        p.x = "z"  # type: ignore[misc]
    except Exception as e:
        assert "frozen" in str(type(e)).lower() or "attribute" in str(e).lower()
    else:
        raise AssertionError("PairItem must be frozen")


def test_sidevectors_shapes():
    sv = SideVectors(a=np.zeros((1, 4)), b=np.zeros((1, 4)), item_id="1")
    assert sv.a.shape == (1, 4) and sv.b.shape == (1, 4)
    assert sv.meta == {}
