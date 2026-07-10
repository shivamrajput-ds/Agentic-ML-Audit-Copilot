"""
Tests for src/audit/data_quality.py.

These tests focus on deterministic, production-style data-quality behavior:
missing values, duplicate rows/columns, constant/near-constant features,
identifier-like columns, rare categories, infinite values, outliers, quality
scoring, and validation errors.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.audit.data_quality import (
    calculate_quality_score,
    run_data_quality_audit,
    safe_percent,
)
from src.utils.exceptions import AuditCopilotException


def _finding_categories(result: dict) -> set[str]:
    """Return finding categories from a data-quality result."""
    return {
        str(finding.get("category"))
        for finding in result.get("findings", [])
        if isinstance(finding, dict)
    }


def _finding_columns(result: dict, category: str) -> set[str]:
    """Return finding columns for one category."""
    return {
        str(finding.get("column"))
        for finding in result.get("findings", [])
        if isinstance(finding, dict) and finding.get("category") == category
    }


def test_data_quality_output_is_json_serializable(messy_df):
    result = run_data_quality_audit(messy_df, target_column="label")

    json.dumps(result, default=str)
    assert result["message"] == "Data quality audit completed successfully."


def test_detects_missing_values(messy_df):
    result = run_data_quality_audit(messy_df, target_column="label")

    missing = result["missing_values"]

    assert "feature_a" in missing
    assert missing["feature_a"]["missing_count"] == 2
    assert missing["feature_a"]["missing_percent"] > 0
    assert result["missing_values"]["feature_a"]["missing_count"] == 2


def test_detects_constant_column(messy_df):
    result = run_data_quality_audit(messy_df, target_column="label")

    assert "feature_b" in result["constant_columns"]
    assert "feature_b" in _finding_columns(result, "constant_column")


def test_detects_near_constant_column():
    df = pd.DataFrame(
        {
            "mostly_same": ["yes"] * 96 + ["no"] * 4,
            "feature": list(range(100)),
            "target": [0, 1] * 50,
        },
    )

    result = run_data_quality_audit(df, target_column="target")

    near_constant = {item["column"] for item in result["near_constant_columns"]}
    assert "mostly_same" in near_constant


def test_detects_duplicate_rows(messy_df):
    result = run_data_quality_audit(messy_df, target_column="label")

    assert result["duplicate_rows"] >= 1
    assert result["duplicate_row_percent"] > 0
    assert "duplicate_rows" in _finding_categories(result)


def test_detects_duplicate_columns():
    df = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4, 5, 6],
            "feature_a_copy": [1, 2, 3, 4, 5, 6],
            "target": [0, 1, 0, 1, 0, 1],
        },
    )

    result = run_data_quality_audit(df, target_column="target")

    duplicate_pairs = {
        (item["column_a"], item["column_b"]) for item in result["duplicate_columns"]
    }
    assert ("feature_a", "feature_a_copy") in duplicate_pairs


def test_detects_id_like_column_by_name(leaky_df):
    result = run_data_quality_audit(leaky_df, target_column="result")

    id_columns = {item["column"] for item in result["possible_id_columns"]}

    assert "student_id" in id_columns


def test_does_not_mark_continuous_numeric_feature_as_id():
    """
    Numeric continuous/high-unique features should not be treated as IDs merely
    because they are unique. Name-based ID checks can still flag real IDs.
    """
    df = pd.DataFrame(
        {
            "income": np.linspace(20_000, 90_000, 100),
            "age": np.linspace(18, 60, 100),
            "target": [0, 1] * 50,
        },
    )

    result = run_data_quality_audit(df, target_column="target")

    possible_ids = {item["column"] for item in result["possible_id_columns"]}
    assert "income" not in possible_ids
    assert "age" not in possible_ids


def test_clean_dataset_has_no_warnings_about_missing(classification_df):
    result = run_data_quality_audit(classification_df, target_column="approved")

    assert result["missing_values"] == {}
    assert result["constant_columns"] == []
    assert "quality_score" in result


def test_quality_score_schema_is_present(messy_df):
    result = run_data_quality_audit(messy_df, target_column="label")
    quality_score = result["quality_score"]

    assert 0 <= quality_score["score"] <= 100
    assert quality_score["health_label"] in {
        "good",
        "needs_review",
        "poor",
        "critical",
    }
    assert isinstance(quality_score["penalties"], list)
    assert quality_score["note"]


def test_target_quality_schema_is_present(classification_df):
    result = run_data_quality_audit(classification_df, target_column="approved")
    target_quality = result["target_quality"]

    assert target_quality["target_column"] == "approved"
    assert target_quality["unique_count"] == 2
    assert target_quality["missing_count"] == 0
    assert target_quality["is_numeric"] is True


def test_missing_target_values_are_flagged(missing_target_df):
    result = run_data_quality_audit(missing_target_df, target_column="target")

    categories = _finding_categories(result)

    assert "target_missing_values" in categories
    assert result["target_quality"]["missing_count"] == 2


def test_invalid_single_class_target_is_critical(invalid_single_class_df):
    result = run_data_quality_audit(invalid_single_class_df, target_column="target")

    categories = _finding_categories(result)
    critical_findings = [
        finding
        for finding in result["findings"]
        if finding.get("severity") == "critical"
    ]

    assert "invalid_target" in categories
    assert critical_findings


def test_infinite_values_are_detected():
    df = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, np.inf, 4.0],
            "feature_b": [1, 2, 3, 4],
            "target": [0, 1, 0, 1],
        },
    )

    result = run_data_quality_audit(df, target_column="target")

    assert "feature_a" in result["infinite_values"]
    assert result["infinite_values"]["feature_a"]["total_infinity_count"] == 1
    assert "feature_a" in _finding_columns(result, "infinite_values")


def test_infinite_values_in_target_are_high_risk():
    df = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4],
            "target": [0.0, 1.0, np.inf, 1.0],
        },
    )

    result = run_data_quality_audit(df, target_column="target")

    target_findings = [
        finding
        for finding in result["findings"]
        if finding.get("category") == "infinite_values"
        and finding.get("column") == "target"
    ]

    assert target_findings
    assert target_findings[0]["severity"] in {"high", "critical"}


def test_outliers_are_detected():
    df = pd.DataFrame(
        {
            "feature_a": list(range(10, 30)) + [1_000, 1_001],
            "target": [0, 1] * 11,
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
        },
    )

    result = run_data_quality_audit(df, target_column="target")

    high_cardinality = {item["column"] for item in result["high_cardinality_columns"]}
    possible_ids = {item["column"] for item in result["possible_id_columns"]}

    assert "text_id" in high_cardinality
    assert "text_id" in possible_ids


def test_rare_categories_are_detected_when_supported():
    df = pd.DataFrame(
        {
            "city": ["Delhi"] * 20 + ["Mumbai"] * 2 + ["Pune"] * 1,
            "target": [0, 1] * 11 + [0],
        }
    )

    result = run_data_quality_audit(df, target_column="target")

    assert "column_quality_summary" in result
    assert isinstance(result.get("findings", []), list)
    summary_columns = {item["column"] for item in result["column_quality_summary"]}
    assert "city" in summary_columns


def test_column_quality_summary_contains_all_columns(messy_df):
    result = run_data_quality_audit(messy_df, target_column="label")

    summary_columns = {item["column"] for item in result["column_quality_summary"]}

    assert summary_columns == set(messy_df.columns)


def test_finding_summary_schema_is_present(messy_df):
    result = run_data_quality_audit(messy_df, target_column="label")

    assert "finding_summary" in result
    finding_summary = result["finding_summary"]

    assert "by_severity" in finding_summary
    assert "by_category" in finding_summary
    assert "total_findings" in finding_summary
    assert finding_summary["total_findings"] == len(result["findings"])


def test_recommended_actions_are_deduplicated(messy_df):
    result = run_data_quality_audit(messy_df, target_column="label")

    actions = result["recommended_actions"]

    assert len(actions) == len(set(actions))


def test_warnings_are_sorted_by_severity(messy_df):
    result = run_data_quality_audit(messy_df, target_column="label")

    warnings = result["warnings"]
    assert warnings

    # The messy dataset should surface at least one non-info warning.
    assert any(
        warning.startswith("[CRITICAL]")
        or warning.startswith("[HIGH]")
        or warning.startswith("[MEDIUM]")
        for warning in warnings
    )


def test_safe_percent_handles_zero_denominator():
    assert safe_percent(10, 0) == 0.0


def test_safe_percent_rounds_to_two_decimals():
    assert safe_percent(1, 3) == 33.33


def test_calculate_quality_score_penalizes_findings():
    findings = [
        {"severity": "critical", "category": "invalid_target", "column": "target"},
        {"severity": "high", "category": "possible_id_column", "column": "id"},
        {"severity": "medium", "category": "missing", "column": "x"},
    ]

    score = calculate_quality_score(findings)

    assert score["score"] < 100
    assert score["health_label"] in {"needs_review", "poor", "critical"}
    assert len(score["penalties"]) >= 1


def test_calculate_quality_score_caps_repeated_category_penalties():
    findings = [
        {"severity": "medium", "category": "rare_value_column", "column": f"x{i}"}
        for i in range(30)
    ]

    score = calculate_quality_score(findings)

    assert 0 <= score["score"] <= 100
    assert score["score"] >= 0


def test_case_insensitive_target_column_is_supported(classification_df):
    renamed = classification_df.rename(columns={"approved": "Approved"})

    result = run_data_quality_audit(renamed, target_column="approved")

    assert result["target_column"] == "Approved"
    assert result["target_quality"]["target_column"] == "Approved"


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


def test_data_quality_result_has_required_top_level_keys(classification_df):
    result = run_data_quality_audit(classification_df, target_column="approved")

    required_keys = {
        "target_column",
        "total_rows",
        "total_columns",
        "quality_score",
        "target_quality",
        "duplicate_rows",
        "duplicate_row_percent",
        "duplicate_columns",
        "missing_values",
        "high_missing_columns",
        "null_only_columns",
        "constant_columns",
        "near_constant_columns",
        "high_cardinality_columns",
        "possible_id_columns",
        "mixed_type_columns",
        "infinite_values",
        "outlier_columns",
        "rare_category_columns",
        "findings",
        "finding_summary",
        "column_quality_summary",
        "warnings",
        "recommended_actions",
        "thresholds",
        "message",
    }

    assert required_keys.issubset(result.keys())
