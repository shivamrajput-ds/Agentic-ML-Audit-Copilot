"""
Tests for src/audit/problem_detector.py
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.audit.problem_detector import detect_problem_type
from src.utils.exceptions import AuditCopilotException


def test_detects_binary_classification(classification_df):
    result = detect_problem_type(classification_df, target_column="approved")
    assert result["problem_type"] == "binary_classification"
    assert result["unique_values"] == 2


def test_detects_regression(regression_df):
    result = detect_problem_type(regression_df, target_column="score")
    assert result["problem_type"] == "regression"


def test_detects_multiclass_for_string_target(leaky_df):
    result = detect_problem_type(leaky_df, target_column="student_id")
    # student_id has as many unique values as rows -> multiclass by the
    # current heuristic (non-numeric dtype always -> classification)
    assert result["problem_type"] == "multiclass_classification"


def test_missing_target_column_raises(classification_df):
    with pytest.raises(AuditCopilotException):
        detect_problem_type(classification_df, target_column="does_not_exist")


def test_constant_target_raises():
    """
    A target with only one unique value cannot be used for classification
    or regression and must raise a clear error instead of being silently
    mislabeled.
    """
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": ["same", "same", "same", "same"]})
    with pytest.raises(AuditCopilotException):
        detect_problem_type(df, target_column="y")


def test_all_missing_target_raises():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [None, None, None]})
    with pytest.raises(AuditCopilotException):
        detect_problem_type(df, target_column="y")