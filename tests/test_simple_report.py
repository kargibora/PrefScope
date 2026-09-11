import json

import pandas as pd
import pytest

from prefscope.reporting import Report


def test_reporting_exports_only_report():
    import prefscope.reporting as reporting

    assert reporting.__all__ == ["Report"]



def test_report_collects_caller_metrics_and_tables(tmp_path):
    report = Report(
        title="Lens run",
        metrics={"mean_l0": 3.5, "n_rows": 2},
        metadata={"model": "example"},
    )
    report.add_table(
        "activity",
        pd.DataFrame({"feature_id": [2, 5], "mean": [0.2, 0.4]}),
    )

    directory = report.save(tmp_path / "report")
    payload = json.loads((directory / "report.json").read_text())

    assert payload == {
        "title": "Lens run",
        "metrics": {"mean_l0": 3.5, "n_rows": 2},
        "metadata": {"model": "example"},
        "tables": {"activity": "tables/table_0.csv"},
    }
    saved = pd.read_csv(directory / payload["tables"]["activity"], index_col=0)
    pd.testing.assert_frame_equal(saved, report.tables["activity"])


@pytest.mark.parametrize("with_report", [False, True])
def test_report_save_refuses_existing_directory_without_changing_files(
    tmp_path, with_report
):
    directory = tmp_path / "report"
    if with_report:
        Report(tables={
            "public": pd.DataFrame({"value": [1]}),
            "private": pd.DataFrame({"text": ["keep private"]}),
        }).save(directory)
    else:
        directory.mkdir()
    before = {
        path.relative_to(directory): path.read_bytes()
        for path in directory.rglob("*") if path.is_file()
    }

    replacement = Report(tables={"public": pd.DataFrame({"value": [2]})})
    with pytest.raises(FileExistsError):
        replacement.save(directory)

    after = {
        path.relative_to(directory): path.read_bytes()
        for path in directory.rglob("*") if path.is_file()
    }
    assert after == before


def test_report_replaces_values_by_name_and_rejects_non_numeric_metrics():
    report = Report(metrics={"score": 1.0})
    report.add_metric("score", 2)
    assert report.metrics == {"score": 2}
    with pytest.raises(TypeError, match="finite real number"):
        report.add_metric("bad", True)


def test_report_keeps_caller_tables_without_interpreting_columns():
    table = pd.DataFrame({"anything": ["x"], "custom_score": [0.25]})
    report = Report(tables={"custom": table})
    assert list(report.tables["custom"].columns) == ["anything", "custom_score"]


def test_report_table_labels_cannot_alias_output_filenames(tmp_path):
    report = Report(
        tables={
            "A": pd.DataFrame({"value": [1]}),
            "a": pd.DataFrame({"value": [2]}),
        }
    )
    directory = report.save(tmp_path / "report")
    payload = json.loads((directory / "report.json").read_text())
    assert payload["tables"] == {
        "A": "tables/table_0.csv",
        "a": "tables/table_1.csv",
    }
