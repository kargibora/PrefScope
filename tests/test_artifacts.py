import numpy as np
import pandas as pd

from prefscope import artifacts


def test_battle_id_col_prefers_battle_id():
    assert artifacts.battle_id_col(pd.DataFrame({"battle_id": [1], "instruction_id": [2]})) == "battle_id"
    assert artifacts.battle_id_col(pd.DataFrame({"instruction_id": [2]})) == "instruction_id"


def test_lens_battle_ids_from_frame_and_dir(tmp_path):
    df = pd.DataFrame({"instruction_id": [10, 11], "x": ["a", "b"]})
    # numeric battle ids are coerced to strings
    np.testing.assert_array_equal(artifacts.lens_battle_ids(df), np.array(["10", "11"]))
    # from a lens dir (reads battles.parquet, prefers battle_id when present)
    pd.DataFrame({"battle_id": ["z", "y"]}).to_parquet(tmp_path / artifacts.BATTLES)
    np.testing.assert_array_equal(artifacts.lens_battle_ids(tmp_path), np.array(["z", "y"]))


def test_lens_battle_ids_dir_falls_back_to_instruction_id(tmp_path):
    pd.DataFrame({"instruction_id": [1, 2]}).to_parquet(tmp_path / artifacts.BATTLES)
    np.testing.assert_array_equal(artifacts.lens_battle_ids(tmp_path), np.array(["1", "2"]))


def test_battle_id_col_raises_when_neither_present():
    import pytest
    with pytest.raises(KeyError, match="battle_id"):
        artifacts.battle_id_col(pd.DataFrame({"x": [1]}))
