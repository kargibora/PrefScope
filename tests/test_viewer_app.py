"""End-to-end smoke test of the Streamlit viewer via AppTest."""
import json
import sys

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def _make_lens(tmp_path):
    iids = ["0", "1", "2", "3"]
    pd.DataFrame({"instruction_id": iids,
                  "model_a": ["X", "Y", "X", "Y"],
                  "model_b": ["Y", "X", "Y", "X"]}
                 ).to_parquet(tmp_path / "battles.parquet")
    z = np.array([[2.0, 0.0], [-1.0, 0.0], [0.0, 3.0], [4.0, -5.0]], np.float32)
    np.save(tmp_path / "z_diff.npy", z)
    (tmp_path / "manifest.json").write_text(json.dumps({"input_rep": "difference"}))
    pd.DataFrame({"feature_id": [0, 1], "concept": ["direct answer", "humor"],
                  "concept_abbrev": ["direct", "humor"]}
                 ).to_csv(tmp_path / "feature_names.csv", index=False)
    pd.DataFrame({"feature_id": [0, 1], "correlation": [0.7, 0.2],
                  "p_bonferroni": [0.01, 0.4], "fidelity_pass": [True, False]}
                 ).to_csv(tmp_path / "feature_fidelity.csv", index=False)
    ann = {"per_sample": [
        {"instruction_id": i, "model_a": "X", "model_b": "Y", "instruction": f"p{i}",
         "completion_a": f"a{i}", "completion_b": f"b{i}", "judge_pref": 1.0}
        for i in iids]}
    apath = tmp_path / "ann.json"
    apath.write_text(json.dumps(ann))
    return apath


def test_viewer_app_renders(tmp_path, monkeypatch):
    ann = _make_lens(tmp_path)
    monkeypatch.setattr(
        sys, "argv",
        ["app.py", "--", "--lens-dir", str(tmp_path), "--annotations", str(ann)])
    at = AppTest.from_file("prefscope/viewer/app.py", default_timeout=30).run()
    assert not at.exception
    # title rendered and the corpus caption mentions the axes
    assert any("PrefScope viewer" in str(t.value) for t in at.title)
    assert at.tabs  # Features / Feature detail / Model diagnosis
