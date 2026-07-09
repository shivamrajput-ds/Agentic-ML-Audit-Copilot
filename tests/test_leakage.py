"""
Tests for src/audit/leakage.py.

These tests cover the most important leakage-audit behaviors:
- obvious leakage should be flagged
- clean data should not be flooded with duplicate-target risks
- the tool must report possible leakage, not confirmed leakage
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.audit.leakage import (
    find_direct_duplicate_target_columns,
    find_high_cardinality_review_columns,
    find_name_based_leakage_risks,
    find_numeric_correlation_risks,
    get_overall_severity,
    run_leakage_check,
    summarize_risks,
)
from src.utils.exceptions import AuditCopilotException


def test_flags_direct_duplicate_target_column(leaky_df):
    result = run_leakage_check(leaky_df, target_column="result")

    duplicate_risks = result["duplicate_target_risks"]
    flagged_columns = {risk["column"] for risk in duplicate_risks}

    assert "target_copy" in flagged_columns


def test_flags_target_like_column_name(leaky_df):
    result = run_leakage_check(leaky_df, target_column="result")

    name_based_columns = {risk["column"] for risk in result["name_based_risks"]}

    assert "total" in name_based_columns


def test_total_risk_count_is_positive_for_leaky_dataset(leaky_df):
    result = run_leakage_check(leaky_df, target_column="result")

    assert result["total_possible_leakage_risks"] > 0
    assert result["requires_human_review"] is True
    assert result["overall_severity"] in {"medium", "high", "critical"}


def test_clean_dataset_has_no_duplicate_target_risks(regression_df):
    result = run_leakage_check(regression_df, target_column="score")

    flagged_columns = {risk["column"] for risk in result["duplicate_target_risks"]}

    assert flagged_columns == set()


def test_never_claims_confirmed_leakage(leaky_df):
    result = run_leakage_check(leaky_df, target_column="result")

    warning_text = result.get("warning", "").lower()

    assert "possible leakage" in warning_text
    assert "human" in warning_text

    for risk in result["all_risks"]:
        assert risk["is_confirmed_leakage"] is False
        assert risk["requires_human_review"] is True


def test_numeric_correlation_risk_is_flagged():
    df = pd.DataFrame(
        {
            "target": [10, 20, 30, 40, 50, 60],
            "target_formula": [20, 40, 60, 80, 100, 120],
            "noise": [3, 1, 9, 2, 8, 4],
        }
    )

    result = run_leakage_check(df, target_column="target")

    flagged_columns = {risk["column"] for risk in result["numeric_correlation_risks"]}

    assert "target_formula" in flagged_columns


def test_mutual_information_risk_schema_exists(leaky_df):
    result = run_leakage_check(leaky_df, target_column="result")

    assert "mutual_information_risks" in result
    assert isinstance(result["mutual_information_risks"], list)


def test_high_cardinality_identifier_risk_is_flagged():
    df = pd.DataFrame(
        {
            "user_token": [f"user_{idx}" for idx in range(100)],
            "feature": list(range(100)),
            "target": [0, 1] * 50,
        }
    )

    result = run_leakage_check(df, target_column="target")

    risk_columns = {risk["column"] for risk in result["high_cardinality_review_risks"]}

    assert "user_token" in risk_columns


def test_name_based_checker_does_not_flag_target_column_itself(leaky_df):
    risks = find_name_based_leakage_risks(leaky_df, target_column="result")

    flagged_columns = {risk["column"] for risk in risks}

    assert "result" not in flagged_columns


def test_direct_duplicate_helper_respects_threshold(leaky_df):
    risks = find_direct_duplicate_target_columns(
        leaky_df,
        target_column="result",
        threshold=0.95,
    )

    flagged_columns = {risk["column"] for risk in risks}

    assert "target_copy" in flagged_columns


def test_numeric_correlation_helper_skips_non_numeric_target(leaky_df):
    risks = find_numeric_correlation_risks(
        leaky_df,
        target_column="result",
        threshold=0.90,
    )

    assert risks == []


def test_high_cardinality_helper_uses_thresholds():
    df = pd.DataFrame(
        {
            "id_col": [f"id_{idx}" for idx in range(20)],
            "target": [0, 1] * 10,
        }
    )

    risks = find_high_cardinality_review_columns(
        df,
        target_column="target",
        thresholds={
            "high_cardinality_threshold": 10,
            "id_unique_percent_threshold": 95,
        },
    )

    assert len(risks) == 1
    assert risks[0]["column"] == "id_col"


def test_risk_summary_and_overall_severity():
    risks = [
        {"risk_level": "medium"},
        {"risk_level": "critical"},
        {"risk_level": "low"},
    ]

    summary = summarize_risks(risks)

    assert summary["critical"] == 1
    assert summary["medium"] == 1
    assert get_overall_severity(summary) == "critical"


def test_missing_target_column_raises(leaky_df):
    with pytest.raises(AuditCopilotException):
        run_leakage_check(leaky_df, target_column="does_not_exist")


def test_empty_dataframe_raises():
    with pytest.raises(AuditCopilotException):
        run_leakage_check(pd.DataFrame(), target_column="target")


def test_target_with_only_missing_values_raises():
    df = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "target": [None, None, None],
        }
    )

    with pytest.raises(AuditCopilotException):
        run_leakage_check(df, target_column="target")
