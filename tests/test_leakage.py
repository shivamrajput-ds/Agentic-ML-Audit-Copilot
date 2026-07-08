"""
Tests for src/audit/leakage.py

These tests focus on verifying that leakage detection actually fires on
an obviously leaky dataset, and stays quiet on a clean one — the two
failure modes that matter most for an audit tool (missed leakage, and
false positives on clean data).
"""
from __future__ import annotations

import pytest

from src.audit.leakage import run_leakage_check
from src.utils.exceptions import AuditCopilotException


def test_flags_direct_duplicate_target_column(leaky_df):
    result = run_leakage_check(leaky_df, target_column="result")
    duplicate_risks = result["duplicate_target_risks"]
    flagged_columns = {risk["column"] for risk in duplicate_risks}
    assert "target_copy" in flagged_columns


def test_flags_target_like_column_name(leaky_df):
    """
    leakage.py's name-based check flags columns whose name overlaps with
    the target name or known target-like keywords (e.g. "total" as a
    likely target-adjacent aggregate column) — it does NOT do ID-column
    detection, that's data_quality.py's job (find_possible_id_columns).
    """
    result = run_leakage_check(leaky_df, target_column="result")
    name_based_columns = {risk["column"] for risk in result["name_based_risks"]}
    assert "total" in name_based_columns


def test_total_risk_count_is_positive_for_leaky_dataset(leaky_df):
    result = run_leakage_check(leaky_df, target_column="result")
    assert result["total_possible_leakage_risks"] > 0


def test_clean_dataset_reports_no_or_few_risks(regression_df):
    """
    A dataset with genuinely independent features should not be flooded
    with false-positive leakage risks.
    """
    result = run_leakage_check(regression_df, target_column="score")
    # hours_studied is deliberately correlated with score by construction
    # in the fixture, so some risk is expected — but student_id-style
    # name/duplicate risks should not appear.
    flagged_columns = {risk["column"] for risk in result["duplicate_target_risks"]}
    assert flagged_columns == set()


def test_never_claims_confirmed_leakage(leaky_df):
    """
    Philosophy check: the tool must only ever report *possible* leakage
    risk, never assert confirmed leakage.
    """
    result = run_leakage_check(leaky_df, target_column="result")
    warning_text = result.get("warning", "").lower()
    assert "possible" in warning_text
    assert "this is confirmed" not in warning_text


def test_missing_target_column_raises(leaky_df):
    with pytest.raises(AuditCopilotException):
        run_leakage_check(leaky_df, target_column="does_not_exist")