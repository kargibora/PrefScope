import importlib
import json

import pandas as pd

from prefscope.pipeline.prepare_dataset import (
    load_dataset_spec, mapping_from_spec, prepare_dataset,
)


def test_prepare_local_dataset_writes_canonical_table_and_manifest(tmp_path):
    source = tmp_path / "raw.csv"
    pd.DataFrame({
        "q": ["q1", "q2"],
        "chosen": ["a1", "a2"],
        "rejected": ["b1", "b2"],
    }).to_csv(source, index=False)
    spec_path = tmp_path / "dataset.yaml"
    spec_path.write_text(
        """
mode: paired
columns:
  prompt: q
  response_a: chosen
  response_b: rejected
label:
  mode: a-wins
""")
    mapping = mapping_from_spec(load_dataset_spec(spec_path))
    out = tmp_path / "canonical.parquet"
    result = prepare_dataset(out, data=source, mapping=mapping)

    frame = pd.read_parquet(out)
    assert list(frame["human_pref"]) == [1.0, 1.0]
    manifest = json.loads((tmp_path / "canonical.prefscope.json").read_text())
    assert manifest["source_spec"]["type"] == "local"
    assert manifest["canonical_table_hash"].startswith("sha256:")
    assert manifest["canonical_table_hash"] == result["canonical_table_hash"]
    assert result["output_rows"] == 2



def test_prepare_hf_records_requested_and_resolved_revision_without_token(
    tmp_path, monkeypatch,
):
    module = importlib.import_module("prefscope.pipeline.prepare_dataset")
    resolved = "b" * 40

    def fake_load_hf_table(dataset_id, **kwargs):
        assert dataset_id == "owner/data"
        assert kwargs["revision"] == "release"
        assert kwargs["token"] == "top-secret"
        frame = pd.DataFrame({"prompt": ["q"], "response": ["r"]})
        frame.attrs["prefscope_hf_source"] = {
            "requested_revision": "release",
            "resolved_revision": resolved,
        }
        return frame

    monkeypatch.setattr(module, "load_hf_table", fake_load_hf_table)
    out = tmp_path / "canonical.parquet"
    result = prepare_dataset(
        out,
        hf_dataset="owner/data",
        revision="release",
        token="top-secret",
        mapping=mapping_from_spec({"mode": "single"}),
    )

    sidecar_text = (tmp_path / "canonical.prefscope.json").read_text()
    manifest = json.loads(sidecar_text)
    source = manifest["source_spec"]
    assert source["revision"] == "release"
    assert source["requested_revision"] == "release"
    assert source["resolved_revision"] == resolved
    assert manifest["canonical_table_hash"].startswith("sha256:")
    assert manifest["canonical_table_hash"] == result["canonical_table_hash"]
    assert "top-secret" not in sidecar_text


def test_prepare_hf_offline_fake_accepts_explicit_commit_revision(
    tmp_path, monkeypatch,
):
    module = importlib.import_module("prefscope.pipeline.prepare_dataset")
    resolved = "c" * 40
    monkeypatch.setattr(
        module,
        "load_hf_table",
        lambda *args, **kwargs: pd.DataFrame({"prompt": ["q"], "response": ["r"]}),
    )

    result = prepare_dataset(
        tmp_path / "canonical.parquet",
        hf_dataset="owner/data",
        revision=resolved,
        mapping=mapping_from_spec({"mode": "single"}),
    )

    assert result["source_spec"]["requested_revision"] == resolved
    assert result["source_spec"]["resolved_revision"] == resolved


def test_prepare_hf_can_load_a_pre_resolved_analysis_commit(monkeypatch, tmp_path):
    resolved = "c" * 40

    def fake_load(dataset_id, **kwargs):
        assert kwargs["revision"] == resolved
        return pd.DataFrame({"prompt": ["q"], "response": ["a"]})

    monkeypatch.setattr(
        "prefscope.pipeline.prepare_dataset.load_hf_table", fake_load)
    result = prepare_dataset(
        tmp_path / "data.parquet", hf_dataset="org/data", revision="main",
        resolved_revision=resolved,
    )

    assert result["source_spec"]["requested_revision"] == "main"
    assert result["source_spec"]["resolved_revision"] == resolved
