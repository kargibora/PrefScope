import pandas as pd
import pytest
import threading
import time

from prefscope.interpret._parallel import run
from prefscope.interpret.checkpoint import FeatureCheckpoint


def test_feature_checkpoint_persists_rows_and_validates_signature(tmp_path):
    out = tmp_path / "names.csv"
    checkpoint = FeatureCheckpoint(out, {"model": "a"})
    checkpoint.record({"feature_id": 2, "concept": "two"})
    checkpoint.record({"feature_id": 0, "concept": "zero"})

    resumed = FeatureCheckpoint(out, {"model": "a"})
    assert resumed.completed_ids == {0, 2}
    assert pd.read_csv(out)["feature_id"].tolist() == [0, 2]
    with pytest.raises(ValueError, match="--fresh"):
        FeatureCheckpoint(out, {"model": "b"})

    fresh = FeatureCheckpoint(out, {"model": "b"}, fresh=True)
    assert fresh.completed_ids == set()
    assert not out.exists()


def test_parallel_checkpoints_other_successes_when_one_item_fails():
    saved = []

    def work(i):
        if i == 1:
            raise RuntimeError("failed")
        return {"feature_id": i}

    with pytest.raises(RuntimeError, match="failed"):
        run(work, [0, 1, 2], concurrency=3, on_result=saved.append)
    assert sorted(row["feature_id"] for row in saved) == [0, 2]


def test_parallel_progress_does_not_count_canceled_items(monkeypatch):
    from prefscope.interpret import _parallel

    class Bar:
        def __init__(self):
            self.n = 0

        def update(self, k=1):
            self.n += k

        def close(self):
            pass

    bar = Bar()
    monkeypatch.setattr(_parallel, "_make_bar", lambda total, desc: bar)
    release = threading.Event()

    def work(i):
        if i == 0:
            release.set()
            raise RuntimeError("failed early")
        release.wait()
        time.sleep(0.05)
        return i

    with pytest.raises(RuntimeError, match="failed early"):
        run(work, range(100), concurrency=2)
    assert bar.n < 100
