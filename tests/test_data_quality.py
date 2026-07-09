"""
Tests for src/audit/data_quality.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.audit.data_quality import (
    calculate_quality_score,
    run_data_quality_audit,
    safe_percent,
)
from src.utils.exceptions import AuditCopilotException


def test_detects_missing_values(messy_df):
    result = run_data_quality_audit(messy_df, target_column="label")

    missing = result["missing_values"]

    assert "feature_a" in missing
    assert missing["feature_a"]["missing_count"] == 2
    assert missing["feature_a"]["missing_percent"] > 0


def test_detects_constant_column(messy_df):
    result = run_data_quality_audit(messy_df, target_column="label")

    assert "feature_b" in result["constant_columns"]


def test_detects_duplicate_rows(messy_df):
    result = run_data_quality_audit(messy_df, target_column="label")

    assert result["duplicate_rows"] >= 1
    assert result["duplicate_row_percent"] > 0


def test_detects_id_like_column_by_name(leaky_df):
    result = run_data_quality_audit(leaky_df, target_column="result")

    id_columns = {item["column"] for item in result["possible_id_columns"]}

    assert "student_id" in id_columns


def test_clean_dataset_has_no_warnings_about_missing(classification_df):
    result = run_data_quality_audit(classification_df, target_column="approved")

    assert result["missing_values"] == {}
    assert result["constant_columns"] == []
    assert "quality_score" in result


def test_quality_score_schema_is_present(messy_df):
    result = run_data_quality_audit(messy_df, target_column="label")
    quality_score = result["quality_score"]

    assert 0 <= quality_score["score"] <= 100
    assert quality_score["health_label"] in {"good", "needs_review", "poor", "critical"}
    assert isinstance(quality_score["penalties"], list)


def test_target_quality_schema_is_present(classification_df):
    result = run_data_quality_audit(classification_df, target_column="approved")
    target_quality = result["target_quality"]

    assert target_quality["target_column"] == "approved"
    assert target_quality["unique_count"] == 2
    assert target_quality["missing_count"] == 0


def test_missing_target_values_are_flagged(missing_target_df):
    result = run_data_quality_audit(missing_target_df, target_column="target")

    findings = result["findings"]
    categories = {finding["category"] for finding in findings}

    assert "target_missing_values" in categories
    assert result["target_quality"]["missing_count"] == 2


def test_infinite_values_are_detected():
    df = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, np.inf, 4.0],
            "feature_b": [1, 2, 3, 4],
            "target": [0, 1, 0, 1],
        }
    )

    result = run_data_quality_audit(df, target_column="target")

    assert "feature_a" in result["infinite_values"]
    assert result["infinite_values"]["feature_a"]["total_infinity_count"] == 1


def test_outliers_are_detected():
    df = pd.DataFrame(
        {
            "feature_a": [10, 11, 12, 13, 14, 15, 1_000],
            "target": [0, 1, 0, 1, 0, 1, 0],
        }
    )

    result = run_data_quality_audit(df, target_column="target")

    outlier_columns = {item["column"] for item in result["outlier_columns"]}
    assert "feature_a" in outlier_columns


def test_high_cardinality_column_is_detected():
    df = pd.DataFrame(
        {
            "text_id": [f"id_{i}" for i in range(60)],
            "feature": list(range(60)),
            "target": [0, 1] * 30,
        }
    )

    result = run_data_quality_audit(df, target_column="target")

    high_cardinality = {item["column"] for item in result["high_cardinality_columns"]}
    possible_ids = {item["column"] for item in result["possible_id_columns"]}

    assert "text_id" in high_cardinality
    assert "text_id" in possible_ids


def test_column_quality_summary_contains_all_columns(messy_df):
    result = run_data_quality_audit(messy_df, target_column="label")

    summary_columns = {item["column"] for item in result["column_quality_summary"]}

    assert summary_columns == set(messy_df.columns)


def test_recommended_actions_are_deduplicated(messy_df):
    result = run_data_quality_audit(messy_df, target_column="label")

    actions = result["recommended_actions"]

    assert len(actions) == len(set(actions))


def test_safe_percent_handles_zero_denominator():
    assert safe_percent(10, 0) == 0.0


def test_calculate_quality_score_penalizes_findings():
    findings = [
        {"severity": "critical", "category": "invalid_target", "column": "target"},
        {"severity": "high", "category": "possible_id_column", "column": "id"},
        {"severity": "medium", "category": "missing", "column": "x"},
    ]

    score = calculate_quality_score(findings)

    assert score["score"] < 100
    assert score["health_label"] in {"needs_review", "poor", "critical"}


def test_missing_target_column_raises(classification_df):
    with pytest.raises(AuditCopilotException):
        run_data_quality_audit(classification_df, target_column="does_not_exist")


def test_empty_dataframe_raises():
    with pytest.raises(AuditCopilotException):
        run_data_quality_audit(pd.DataFrame(), target_column="label")


def test_single_column_dataframe_raises():
    df = pd.DataFrame({"target": [0, 1, 0]})

    with pytest.raises(AuditCopilotException):
        run_data_quality_audit(df, target_column="target")
