"""
Tests for src/audit/data_quality.py
"""
from __future__ import annotations

import pytest

from src.audit.data_quality import run_data_quality_audit
from src.utils.exceptions import AuditCopilotException


def test_detects_missing_values(messy_df):
    result = run_data_quality_audit(messy_df, target_column="label")
    missing = result["missing_values"]
    assert "feature_a" in missing
    assert missing["feature_a"]["missing_count"] == 2


def test_detects_constant_column(messy_df):
    result = run_data_quality_audit(messy_df, target_column="label")
    assert "feature_b" in result["constant_columns"]


def test_detects_duplicate_rows(messy_df):
    result = run_data_quality_audit(messy_df, target_column="label")
    assert result["duplicate_rows"] >= 1


def test_clean_dataset_has_no_warnings_about_missing(classification_df):
    result = run_data_quality_audit(classification_df, target_column="approved")
    assert result["missing_values"] == {}
    assert result["constant_columns"] == []


def test_missing_target_column_raises(classification_df):
    with pytest.raises(AuditCopilotException):
        run_data_quality_audit(classification_df, target_column="does_not_exist")


def test_empty_dataframe_raises():
    import pandas as pd

    with pytest.raises(AuditCopilotException):
        run_data_quality_audit(pd.DataFrame(), target_column="label")


def test_detects_id_like_column_by_name(leaky_df):
    result = run_data_quality_audit(leaky_df, target_column="result")
    id_columns = {item["column"] for item in result["possible_id_columns"]}
    assert "student_id" in id_columns