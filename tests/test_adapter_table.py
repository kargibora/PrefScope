import pandas as pd
import pytest

from prefscope.core import registry
from prefscope.core.types import PairItem
import prefscope.adapters  # noqa: F401


def _df():
    return pd.DataFrame({
        "q": ["p1", "p2"],
        "ra": ["a1", "a2"],
        "rb": ["b1", "b2"],
        "winner": [1.0, 0.0],
        "ma": ["m1", "m1"],
        "mb": ["m2", "m2"],
    })


def test_table_from_dataframe_full_mapping():
    ds = registry.get("dataset", "table")(
        _df(), prompt="q", a="ra", b="rb", pref="winner", model_a="ma", model_b="mb")
    items = list(ds)
    assert len(ds) == 2 and all(isinstance(i, PairItem) for i in items)
    assert items[0].x == "p1" and items[0].y_a == "a1" and items[0].y_b == "b1"
    assert items[0].pref == 1.0 and items[0].model_a == "m1"
    assert items[0].id == "0"                     # falls back to row index


def test_table_minimal_single_response():
    ds = registry.get("dataset", "table")(_df(), prompt="q", a="ra")
    items = list(ds)
    assert items[0].y_b is None and items[0].pref is None
    assert items[0].is_single is True


def test_table_missing_column_raises():
    with pytest.raises(ValueError, match="nope"):
        registry.get("dataset", "table")(_df(), prompt="q", a="ra", pref="nope")


def test_table_from_csv_file(tmp_path):
    p = tmp_path / "b.csv"
    _df().to_csv(p, index=False)
    ds = registry.get("dataset", "table")(str(p), prompt="q", a="ra", b="rb", pref="winner")
    items = list(ds)
    assert items[1].pref == 0.0 and items[1].y_b == "b2"


def test_table_nan_pref_becomes_none():
    df = _df()
    df.loc[0, "winner"] = float("nan")
    ds = registry.get("dataset", "table")(df, prompt="q", a="ra", pref="winner")
    assert list(ds)[0].pref is None


def test_table_non_identifier_column_names():
    """Columns like 'user prompt' / 'win-rate' aren't valid Python identifiers; itertuples
    renames them, which used to KeyError. Index by position instead (#7)."""
    df = pd.DataFrame({
        "user prompt": ["p1", "p2"], "response a": ["a1", "a2"],
        "response b": ["b1", "b2"], "win-rate": [1.0, 0.0],
    })
    ds = registry.get("dataset", "table")(
        df, prompt="user prompt", a="response a", b="response b", pref="win-rate")
    items = list(ds)
    assert items[0].x == "p1" and items[0].y_a == "a1" and items[0].y_b == "b1"
    assert items[0].pref == 1.0 and items[1].pref == 0.0
