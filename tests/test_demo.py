import pandas as pd
import pytest

from prefscope.data.demo import create_demo, make_demo_corpus


def test_demo_corpus_is_deterministic_and_canonical():
    first = make_demo_corpus()
    second = make_demo_corpus()
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 60
    assert {"battle_id", "prompt", "completion_a", "completion_b", "human_pref"} <= set(
        first
    )
    assert first["battle_id"].is_unique


def test_create_demo_writes_corpus_and_refuses_stale_directory(tmp_path):
    paths = create_demo(tmp_path / "demo")
    assert pd.read_parquet(paths["corpus"]).shape[0] == 60
    assert not (paths["root"] / "quickstart.yaml").exists()
    assert paths["lens"] == paths["root"] / "lens"
    with pytest.raises(FileExistsError):
        create_demo(tmp_path / "demo")


def test_create_demo_force_rewrites_generated_files(tmp_path):
    paths = create_demo(tmp_path / "demo")
    paths["corpus"].write_bytes(b"broken")
    create_demo(tmp_path / "demo", force=True)
    assert len(pd.read_parquet(paths["corpus"])) == 60
