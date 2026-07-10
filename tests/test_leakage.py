"""
Tests for src/audit/leakage.py.

These tests cover the most important leakage-audit behaviors:
- obvious leakage should be flagged
- clean data should not be flooded with duplicate-target risks
- the tool must report possible leakage, not confirmed leakage
- high-cardinality/identifier heuristics should be conservative
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import pytest

from src.audit.leakage import (
    find_classification_proxy_risks,
    find_direct_duplicate_target_columns,
    find_high_cardinality_review_columns,
    find_name_based_leakage_risks,
    find_numeric_correlation_risks,
    get_overall_severity,
    run_leakage_check,
    summarize_risk_types,
    summarize_risks,
)
from src.utils.exceptions import AuditCopilotException


def _risk_columns(risks: list[dict[str, Any]]) -> set[str]:
    """Return columns from risk records."""
    return {str(risk.get("column")) for risk in risks}


def _assert_possible_leakage_only(result: dict[str, Any]) -> None:
    """Assert the leakage module never claims confirmed leakage."""
    warning_text = str(result.get("warning", "")).lower()

    assert "possible leakage" in warning_text or "no obvious leakage" in warning_text
    assert result.get("leakage_policy", {}).get("confirmed_leakage_claimed") is False

    for risk in result.get("all_risks", []):
        assert risk["is_confirmed_leakage"] is False
        assert risk["requires_human_review"] is True


def test_leakage_output_is_json_serializable(leaky_df: pd.DataFrame) -> None:
    result = run_leakage_check(leaky_df, target_column="result")

    json.dumps(result, default=str)
    assert result["message"] == "Leakage check completed successfully."


def test_leakage_result_has_required_top_level_keys(leaky_df: pd.DataFrame) -> None:
    result = run_leakage_check(leaky_df, target_column="result")

    required_keys = {
        "target_column",
        "total_possible_leakage_risks",
        "overall_severity",
        "risk_summary",
        "risk_type_summary",
        "all_risks",
        "requires_human_review",
        "review_columns",
        "recommended_actions",
        "warning",
        "leakage_policy",
        "message",
    }

    assert required_keys.issubset(result.keys())

    optional_risk_buckets = {
        "name_based_risks",
        "numeric_correlation_risks",
        "duplicate_target_risks",
        "classification_proxy_risks",
        "encoded_target_correlation_risks",
        "mutual_information_risks",
        "high_cardinality_review_risks",
    }
    for key in optional_risk_buckets.intersection(result.keys()):
        assert isinstance(result[key], list)


def test_leakage_report_schema_for_leaky_dataset(leaky_df: pd.DataFrame) -> None:
    result = run_leakage_check(leaky_df, target_column="result")

    assert result["target_column"] == "result"
    assert result["message"] == "Leakage check completed successfully."
    assert result["total_possible_leakage_risks"] == len(result["all_risks"])
    assert result["overall_severity"] in {"none", "low", "medium", "high", "critical"}
    assert isinstance(result["risk_summary"], dict)
    assert isinstance(result["risk_type_summary"], dict)
    assert isinstance(result["review_columns"], list)
    assert isinstance(result["recommended_actions"], list)
    assert result["leakage_policy"]["deterministic_only"] is True
    _assert_possible_leakage_only(result)


def test_flags_direct_duplicate_target_column(leaky_df: pd.DataFrame) -> None:
    result = run_leakage_check(leaky_df, target_column="result")

    assert "target_copy" in _risk_columns(result["duplicate_target_risks"])
    assert result["risk_type_summary"].get("duplicate_target_risk", 0) >= 1
    assert result["overall_severity"] == "critical"


def test_flags_target_like_column_name(leaky_df: pd.DataFrame) -> None:
    result = run_leakage_check(leaky_df, target_column="result")

    name_based_columns = _risk_columns(result["name_based_risks"])

    assert "total" in name_based_columns


def test_total_risk_count_is_positive_for_leaky_dataset(leaky_df: pd.DataFrame) -> None:
    result = run_leakage_check(leaky_df, target_column="result")

    assert result["total_possible_leakage_risks"] > 0
    assert result["requires_human_review"] is True
    assert result["overall_severity"] in {"medium", "high", "critical"}


def test_clean_dataset_has_no_duplicate_target_risks(
    regression_df: pd.DataFrame,
) -> None:
    result = run_leakage_check(regression_df, target_column="score")

    assert _risk_columns(result["duplicate_target_risks"]) == set()
    _assert_possible_leakage_only(result)


def test_never_claims_confirmed_leakage(leaky_df: pd.DataFrame) -> None:
    result = run_leakage_check(leaky_df, target_column="result")
    _assert_possible_leakage_only(result)


def test_numeric_correlation_risk_is_flagged() -> None:
    df = pd.DataFrame(
        {
            "target": list(range(1, 31)),
            "target_formula": [value * 2 for value in range(1, 31)],
            "noise": [3, 1, 9, 2, 8, 4, 7, 5, 6, 10] * 3,
        },
    )

    result = run_leakage_check(df, target_column="target")

    assert "target_formula" in _risk_columns(result["numeric_correlation_risks"])


def test_numeric_correlation_helper_skips_non_numeric_target(
    leaky_df: pd.DataFrame,
) -> None:
    risks = find_numeric_correlation_risks(
        leaky_df,
        target_column="result",
        threshold=0.90,
        min_compare_rows=10,
    )

    assert risks == []


def test_numeric_correlation_helper_respects_min_compare_rows() -> None:
    df = pd.DataFrame(
        {
            "target": [1, 2, 3, 4, 5, None, None, None, None, None],
            "perfect_proxy": [2, 4, 6, 8, 10, None, None, None, None, None],
        },
    )

    risks = find_numeric_correlation_risks(
        df,
        target_column="target",
        threshold=0.90,
        min_compare_rows=10,
    )

    assert risks == []


def test_mutual_information_risk_schema_exists(leaky_df: pd.DataFrame) -> None:
    result = run_leakage_check(leaky_df, target_column="result")

    assert "mutual_information_risks" in result
    assert isinstance(result["mutual_information_risks"], list)


def test_classification_proxy_risk_can_be_flagged() -> None:
    df = pd.DataFrame(
        {
            "feature_proxy": [
                1,
                2,
                1,
                2,
                1,
                2,
                1,
                2,
                1,
                2,
                90,
                95,
                92,
                91,
                96,
                94,
                93,
                97,
                98,
                99,
            ],
            "noise": list(range(20)),
            "target": ["no"] * 10 + ["yes"] * 10,
        },
    )

    risks = find_classification_proxy_risks(
        df,
        target_column="target",
        threshold=0.75,
        min_compare_rows=10,
    )

    assert "feature_proxy" in _risk_columns(risks)


def test_high_cardinality_identifier_risk_is_flagged_for_string_identifier() -> None:
    df = pd.DataFrame(
        {
            "user_token": [f"user_{idx}" for idx in range(100)],
            "feature": list(range(100)),
            "target": [0, 1] * 50,
        },
    )

    result = run_leakage_check(df, target_column="target")

    assert "user_token" in _risk_columns(result["high_cardinality_review_risks"])


def test_high_cardinality_identifier_does_not_flag_continuous_numeric_feature() -> None:
    df = pd.DataFrame(
        {
            "income": list(range(10_000, 10_100)),
            "age": list(range(20, 120)),
            "target": [0, 1] * 50,
        },
    )

    risks = find_high_cardinality_review_columns(
        df,
        target_column="target",
        thresholds={
            "high_cardinality_threshold": 10,
            "id_unique_percent_threshold": 95,
        },
    )

    assert _risk_columns(risks) == set()


def test_high_cardinality_helper_uses_thresholds() -> None:
    df = pd.DataFrame(
        {
            "id_col": [f"id_{idx}" for idx in range(20)],
            "target": [0, 1] * 10,
        },
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
    assert risks[0]["risk_type"] == "high_cardinality_identifier_risk"


def test_name_based_checker_does_not_flag_target_column_itself(
    leaky_df: pd.DataFrame,
) -> None:
    risks = find_name_based_leakage_risks(leaky_df, target_column="result")

    assert "result" not in _risk_columns(risks)


def test_name_based_checker_uses_token_safe_matching() -> None:
    df = pd.DataFrame(
        {
            "classroom_size": [30, 31, 32, 33],
            "income": [100, 110, 120, 130],
            "target": [0, 1, 0, 1],
        },
    )

    risks = find_name_based_leakage_risks(df, target_column="target")

    assert "classroom_size" not in _risk_columns(risks)
    assert "income" not in _risk_columns(risks)


def test_direct_duplicate_helper_respects_threshold(leaky_df: pd.DataFrame) -> None:
    risks = find_direct_duplicate_target_columns(
        leaky_df,
        target_column="result",
        threshold=0.95,
        min_compare_rows=10,
    )

    assert "target_copy" in _risk_columns(risks)


def test_direct_duplicate_helper_ignores_missing_value_false_matches() -> None:
    df = pd.DataFrame(
        {
            "target": ["A", "B", None, None, None, None],
            "feature": ["A", "B", None, None, None, None],
        },
    )

    risks = find_direct_duplicate_target_columns(
        df,
        target_column="target",
        threshold=0.95,
        min_compare_rows=3,
    )

    assert risks == []


def test_review_columns_are_compact_and_sorted(leaky_df: pd.DataFrame) -> None:
    result = run_leakage_check(leaky_df, target_column="result")
    review_columns = result["review_columns"]

    assert review_columns
    assert all("column" in item for item in review_columns)
    assert all("risk_types" in item for item in review_columns)
    assert all(item["requires_human_review"] is True for item in review_columns)

    first = review_columns[0]
    assert first["highest_risk_level"] in {"critical", "high", "medium", "low"}


def test_risk_summary_type_summary_and_overall_severity() -> None:
    risks = [
        {"risk_level": "medium", "risk_type": "name_based_risk"},
        {"risk_level": "critical", "risk_type": "duplicate_target_risk"},
        {"risk_level": "low", "risk_type": "high_cardinality_identifier_risk"},
    ]

    summary = summarize_risks(risks)
    type_summary = summarize_risk_types(risks)

    assert summary["critical"] == 1
    assert summary["medium"] == 1
    assert get_overall_severity(summary) == "critical"
    assert type_summary["duplicate_target_risk"] == 1
    assert type_summary["name_based_risk"] == 1


def test_no_risks_has_none_overall_severity() -> None:
    summary = summarize_risks([])

    assert summary == {"critical": 0, "high": 0, "medium": 0, "low": 0}
    assert get_overall_severity(summary) == "none"


def test_case_insensitive_target_column_is_resolved() -> None:
    df = pd.DataFrame(
        {
            "Feature": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "Target": [0, 1] * 5,
        },
    )

    result = run_leakage_check(df, target_column="target")

    assert result["target_column"] == "Target"


def test_missing_target_column_raises(leaky_df: pd.DataFrame) -> None:
    with pytest.raises(AuditCopilotException):
        run_leakage_check(leaky_df, target_column="does_not_exist")


def test_empty_dataframe_raises() -> None:
    with pytest.raises(AuditCopilotException):
        run_leakage_check(pd.DataFrame(), target_column="target")


def test_target_with_only_missing_values_raises() -> None:
    df = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "target": [None, None, None],
        },
    )

    with pytest.raises(AuditCopilotException):
        run_leakage_check(df, target_column="target")


def test_single_unique_target_raises() -> None:
    df = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "target": [1, 1, 1, 1],
        },
    )

    with pytest.raises(AuditCopilotException):
        run_leakage_check(df, target_column="target")


def test_duplicate_column_names_raise() -> None:
    df = pd.DataFrame(
        [[1, 10, 0], [2, 20, 1], [3, 30, 0]],
        columns=["feature", "feature", "target"],
    )

    with pytest.raises(AuditCopilotException):
        run_leakage_check(df, target_column="target")


def test_all_risk_records_have_required_policy_fields(leaky_df: pd.DataFrame) -> None:
    result = run_leakage_check(leaky_df, target_column="result")

    for risk in result["all_risks"]:
        assert {"column", "risk_type", "risk_level", "reason"}.issubset(risk.keys())
        assert risk["risk_level"] in {"low", "medium", "high", "critical"}
        assert risk["requires_human_review"] is True
        assert risk["is_confirmed_leakage"] is False
