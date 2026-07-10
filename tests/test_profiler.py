"""
Tests for src/audit/profiler.py.

These tests verify deterministic dataset loading and profiling behavior:
CSV loading, path validation, column type inference, numeric/categorical/boolean/
datetime summaries, target summary, warnings, and JSON-safe API/UI output.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.audit.profiler import (
    count_infinite_values,
    get_column_types,
    get_file_size_mb,
    infer_datetime_columns,
    load_dataset,
    normalize_allowed_extensions,
    profile_dataset,
    safe_percent,
    safe_round,
    summarize_categorical_columns,
    summarize_datetime_columns,
    summarize_numeric_columns,
    summarize_target,
    validate_dataset_path,
    validate_loaded_dataframe,
)
from src.utils.exceptions import AuditCopilotException


def test_load_dataset_reads_temp_csv(temp_classification_csv: Path) -> None:
    df = load_dataset(temp_classification_csv)

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert set(df.columns) == {"age", "income", "city", "approved"}
    assert len(df) == 40


def test_validate_dataset_path_accepts_csv(temp_classification_csv: Path) -> None:
    path = validate_dataset_path(temp_classification_csv)

    assert path.exists()
    assert path.suffix == ".csv"


def test_get_file_size_mb_returns_float(temp_classification_csv: Path) -> None:
    size_mb = get_file_size_mb(temp_classification_csv)

    assert isinstance(size_mb, float)
    assert size_mb >= 0.0


def test_load_dataset_missing_file_raises(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(AuditCopilotException):
        load_dataset(missing_path)


def test_load_dataset_unsupported_extension_raises(tmp_path: Path) -> None:
    txt_path = tmp_path / "sample.txt"
    txt_path.write_text("a,b\n1,2\n", encoding="utf-8")

    with pytest.raises(AuditCopilotException):
        load_dataset(txt_path)


def test_validate_loaded_dataframe_rejects_empty_dataframe() -> None:
    with pytest.raises(AuditCopilotException):
        validate_loaded_dataframe(pd.DataFrame())


def test_validate_loaded_dataframe_rejects_duplicate_columns() -> None:
    df = pd.DataFrame(
        [[1, 2, 0], [3, 4, 1]],
        columns=["feature", "feature", "target"],
    )

    with pytest.raises(AuditCopilotException):
        validate_loaded_dataframe(df)


def test_normalize_allowed_extensions() -> None:
    allowed = normalize_allowed_extensions(["csv", ".tsv", "  json  ", ""])

    assert allowed == {".csv", ".tsv", ".json"}


def test_safe_percent_handles_zero_and_rounding() -> None:
    assert safe_percent(1, 3) == 33.33
    assert safe_percent(10, 0) == 0.0


def test_safe_round_handles_nan_and_inf() -> None:
    assert safe_round(1.23456, digits=2) == 1.23
    assert safe_round(np.nan) == 0.0
    assert safe_round(np.inf) == 0.0
    assert safe_round(None) == 0.0


def test_profile_dataset_output_contract(classification_df: pd.DataFrame) -> None:
    profile = profile_dataset(classification_df, target_column="approved")

    required_keys = {
        "shape",
        "row_count",
        "column_count",
        "memory_usage_mb",
        "duplicate_rows",
        "duplicate_row_percent",
        "column_types",
        "numeric_columns",
        "categorical_columns",
        "boolean_columns",
        "datetime_columns",
        "unsupported_columns",
        "missing_cells",
        "missing_cells_percent",
        "numeric_summary",
        "categorical_summary",
        "boolean_summary",
        "datetime_summary",
        "target_summary",
        "warnings",
        "message",
    }

    assert required_keys.issubset(profile.keys())
    assert profile["shape"] == {"rows": 40, "columns": 4}
    assert profile["row_count"] == 40
    assert profile["column_count"] == 4
    assert profile["target_summary"]["target_column"] == "approved"
    assert profile["target_summary"]["exists"] is True
    assert profile["message"] == "Dataset profiling completed successfully."


def test_profile_dataset_is_json_serializable(classification_df: pd.DataFrame) -> None:
    profile = profile_dataset(classification_df, target_column="approved")

    json.dumps(profile, default=str)


def test_profile_dataset_without_target_is_supported(
    classification_df: pd.DataFrame,
) -> None:
    profile = profile_dataset(classification_df)

    assert profile["target_summary"]["target_column"] is None
    assert profile["target_summary"]["exists"] is False


def test_profile_dataset_missing_target_adds_warning(
    classification_df: pd.DataFrame,
) -> None:
    profile = profile_dataset(classification_df, target_column="missing_target")

    assert profile["target_summary"]["exists"] is False
    assert any("Target column" in warning for warning in profile["warnings"])


def test_profile_dataset_empty_dataframe_raises() -> None:
    with pytest.raises(AuditCopilotException):
        profile_dataset(pd.DataFrame(), target_column="target")


def test_get_column_types_detects_basic_types() -> None:
    df = pd.DataFrame(
        {
            "numeric_feature": [1, 2, 3],
            "category_feature": ["a", "b", "a"],
            "bool_feature": [True, False, True],
            "date_feature": pd.date_range("2024-01-01", periods=3),
        },
    )

    column_types = get_column_types(df)

    assert "numeric_feature" in column_types["numeric_columns"]
    assert "category_feature" in column_types["categorical_columns"]
    assert "bool_feature" in column_types["boolean_columns"]
    assert "date_feature" in column_types["datetime_columns"]


def test_infer_datetime_columns_detects_date_like_strings() -> None:
    df = pd.DataFrame(
        {
            "created_date": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
            ],
            "city": ["Delhi", "Mumbai", "Pune", "Bengaluru"],
        },
    )

    inferred = infer_datetime_columns(df)

    assert "created_date" in inferred
    assert "city" not in inferred


def test_summarize_numeric_columns_handles_infinite_values(
    infinite_values_df: pd.DataFrame,
) -> None:
    summary = summarize_numeric_columns(
        infinite_values_df,
        numeric_columns=["feature_a", "feature_b"],
    )

    assert summary["feature_a"]["infinite_count"] == 1
    assert summary["feature_b"]["infinite_count"] == 1
    assert summary["feature_a"]["missing_count"] >= 1


def test_count_infinite_values() -> None:
    series = pd.Series([1.0, np.inf, -np.inf, None, 5.0])

    assert count_infinite_values(series) == 2


def test_summarize_categorical_columns(classification_df: pd.DataFrame) -> None:
    summary = summarize_categorical_columns(classification_df, ["city"])

    assert "city" in summary
    assert summary["city"]["unique_count"] >= 1
    assert isinstance(summary["city"]["top_values"], list)
    assert summary["city"]["top_values"]


def test_summarize_datetime_columns(datetime_df: pd.DataFrame) -> None:
    summary = summarize_datetime_columns(datetime_df, ["signup_date"])

    assert "signup_date" in summary
    assert summary["signup_date"]["parse_success_count"] == len(datetime_df)
    assert summary["signup_date"]["min"] is not None
    assert summary["signup_date"]["max"] is not None


def test_summarize_target_existing_column(classification_df: pd.DataFrame) -> None:
    summary = summarize_target(classification_df, "approved")

    assert summary["target_column"] == "approved"
    assert summary["exists"] is True
    assert summary["unique_count"] == 2
    assert summary["missing_count"] == 0
    assert summary["is_numeric"] is True
    assert isinstance(summary["sample_values"], list)
    assert isinstance(summary["top_values"], list)


def test_summarize_target_missing_column(classification_df: pd.DataFrame) -> None:
    summary = summarize_target(classification_df, "missing")

    assert summary["target_column"] == "missing"
    assert summary["exists"] is False
    assert "not found" in summary["message"].lower()


def test_profile_dataset_warns_for_small_dataset() -> None:
    df = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "target": [0, 1, 0, 1],
        },
    )

    profile = profile_dataset(df, target_column="target")

    assert any(
        "fewer than 50 rows" in warning.lower() for warning in profile["warnings"]
    )


def test_profile_dataset_reports_duplicate_rows(messy_df: pd.DataFrame) -> None:
    profile = profile_dataset(messy_df, target_column="label")

    assert profile["duplicate_rows"] >= 1
    assert profile["duplicate_row_percent"] > 0
    assert any("duplicate rows" in warning.lower() for warning in profile["warnings"])
