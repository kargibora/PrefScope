from __future__ import annotations

import pandas as pd
import pytest

from prefscope import AnalysisArtifact, OutcomeSpec, TableContract, analyze_dataset
from prefscope.api.analysis_schemas import BUILTIN_ANALYSIS_TABLE_CONTRACTS


def _contract(*, allow_extra=False):
    return TableContract(
        schema_name="sample_table",
        schema_version=1,
        required_columns=("name", "count", "optional_count", "score", "passed"),
        dtypes={
            "name": "string",
            "count": "integer",
            "optional_count": "nullable_integer",
            "score": "float",
            "passed": "boolean",
        },
        unique_key=("name",),
        orientation="as_declared",
        units={"score": "unitless"},
        allow_extra_columns=allow_extra,
    )


def _table():
    return pd.DataFrame({
        "name": ["a", "b"],
        "count": [1, 2],
        "optional_count": [1.0, float("nan")],
        "score": [0.1, 0.2],
        "passed": [True, False],
    })


def test_table_contract_validates_without_mutating_or_casting():
    contract = _contract()
    table = _table()
    before = table.copy(deep=True)
    dtypes = table.dtypes.copy()
    contract.validate(table)
    pd.testing.assert_frame_equal(table, before)
    assert table.dtypes.equals(dtypes)
    assert contract.identifier == "sample_table/v1"
    assert contract.to_manifest()["unique_key"] == ["name"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda table: table.drop(columns="score"), "missing required"),
        (lambda table: table.assign(extra=1), "unexpected columns"),
        (lambda table: table[["count", "name", "optional_count", "score", "passed"]],
         "canonical order"),
        (lambda table: table.assign(count=[1.5, 2.5]), "logical dtype"),
        (lambda table: table.assign(score=[1, 2]), "logical dtype"),
        (lambda table: table.assign(passed=[True, None]), "logical dtype"),
        (lambda table: table.assign(name=["a", "a"]), "duplicates"),
        (lambda table: table.assign(name=["a", None]), "missing values"),
    ],
)
def test_table_contract_rejects_schema_violations(change, message):
    with pytest.raises(ValueError, match=message):
        _contract().validate(change(_table()))


def test_float_contract_accepts_extension_and_object_float_values():
    extension = _table()
    extension["score"] = pd.Series([0.1, pd.NA], dtype="Float64")
    _contract().validate(extension)

    object_values = _table()
    object_values["score"] = pd.Series([0.1, 0.2], dtype=object)
    _contract().validate(object_values)


def test_empty_tables_still_enforce_declared_logical_dtypes():
    contract = _contract()
    valid = contract.empty_frame()
    contract.validate(valid)

    wrong_string = valid.copy()
    wrong_string["name"] = pd.Series(dtype="float64")
    with pytest.raises(ValueError, match="logical dtype"):
        contract.validate(wrong_string)

    wrong_integer = valid.copy()
    wrong_integer["count"] = pd.Series(dtype="float64")
    with pytest.raises(ValueError, match="logical dtype"):
        contract.validate(wrong_integer)

    wrong_boolean = valid.copy()
    wrong_boolean["passed"] = pd.Series(dtype=object)
    with pytest.raises(ValueError, match="logical dtype"):
        contract.validate(wrong_boolean)

    nullable_integer = valid.copy()
    nullable_integer["optional_count"] = pd.Series([float("nan")], dtype="float64")[:0]
    contract.validate(nullable_integer)


def test_open_contract_still_requires_valid_column_labels():
    table = _table().assign(extra=1)
    table.columns = ["name", "count", "optional_count", "score", "passed", 9]
    with pytest.raises(ValueError, match="unique non-empty strings"):
        _contract(allow_extra=True).validate(table)


def test_open_table_contract_allows_trailing_extra_columns():
    _contract(allow_extra=True).validate(_table().assign(extra=1))


def test_table_contract_definition_fails_closed():
    with pytest.raises(ValueError, match="lower_snake_case"):
        TableContract("Bad Name", 1, ("id",), {"id": "integer"}, ("id",), "none")
    with pytest.raises(ValueError, match="positive integer"):
        TableContract("bad_version", 0, ("id",), {"id": "integer"}, ("id",), "none")
    with pytest.raises(ValueError, match="every required column"):
        TableContract("missing_type", 1, ("id",), {}, ("id",), "none")


def test_analysis_artifact_contract_is_additive_and_old_call_shape_still_works():
    old_style = AnalysisArtifact("custom", pd.DataFrame({"x": [1]}), "one row", {})
    assert "table_schema" not in old_style.to_manifest()

    contract = TableContract(
        "custom", 1, ("x",), {"x": "integer"}, ("x",), "none")
    artifact = AnalysisArtifact(
        "custom", pd.DataFrame({"x": [1]}), "one row", {}, contract)
    assert artifact.to_manifest()["table_schema"]["version"] == 1


def test_builtin_contracts_are_versioned_and_outcome_empty_table_is_canonical():
    assert set(BUILTIN_ANALYSIS_TABLE_CONTRACTS) == {
        "outcome_associations",
        "feature_artifact_diagnostics",
        "preference_length_confounds",
        "paired_outcome_shifts",
        "prompt_conditioned_outcome_shifts",
        "paired_concept_shift",
    }
    outcome = OutcomeSpec([0.0, 1.0], row_ids=("a", "b"), kind="continuous")
    artifact = analyze_dataset(outcomes={"reward": outcome}).artifact(
        "outcome_associations")
    assert artifact.table.empty
    assert tuple(artifact.table.columns) == artifact.table_contract.required_columns
    assert artifact.to_manifest()["table_schema"]["name"] == "outcome_associations"
