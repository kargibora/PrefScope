import pytest
from prefscope.core.types import PairItem


def test_model_identity_is_first_class():
    p = PairItem(id="1", x="q", y_a="a", y_b="b", pref=0.7,
                 model_a="gpt", model_b="llama")
    assert p.model_a == "gpt" and p.model_b == "llama"
    assert p.meta == {}                      # identity is NOT smuggled into meta


def test_model_identity_defaults_none():
    p = PairItem(id="1", x="q", y_a="a")
    assert p.model_a is None and p.model_b is None


def test_pref_must_be_probability():
    PairItem(id="1", x="q", y_a="a", y_b="b", pref=0.0)   # ok
    PairItem(id="1", x="q", y_a="a", y_b="b", pref=1.0)   # ok
    PairItem(id="1", x="q", y_a="a", y_b="b", pref=0.5)   # ok (tie)
    for bad in (-0.1, 1.5, 2.0):
        with pytest.raises(ValueError, match="P\\(A preferred\\)"):
            PairItem(id="1", x="q", y_a="a", y_b="b", pref=bad)


def test_pref_none_allowed():
    assert PairItem(id="1", x="q", y_a="a").pref is None
