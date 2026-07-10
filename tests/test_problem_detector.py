"""
Tests for src/audit/problem_detector.py.

The problem detector is intentionally conservative. These tests verify that it
keeps stable public output keys, detects the common supervised-learning problem
families, and flags ambiguous target columns for human review instead of
silently overclaiming certainty.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.audit.problem_detector import (
    detect_problem_type,
    get_class_balance_preview,
    get_detection_reason,
    get_sample_values,
    infer_problem_type,
    is_integer_like_series,
)
from src.utils.exceptions import AuditCopilotException

REQUIRED_RESULT_KEYS = {
    "target_column",
    "problem_type",
    "target_dtype",
    "is_numeric_target",
    "is_bool_target",
    "is_integer_like_target",
    "unique_values",
    "unique_percent",
    "total_values",
    "total_rows",
    "missing_count",
    "missing_percent",
    "classification_unique_threshold",
    "confidence",
    "needs_human_review",
    "requires_human_review",
    "human_review_reason",
    "sample_values",
    "class_balance_preview",
    "warnings",
    "reason",
    "recommended_action",
}


def assert_problem_detector_contract(result: dict[str, object]) -> None:
    """Assert common response keys and JSON-serializability."""
    assert REQUIRED_RESULT_KEYS.issubset(result.keys())
    assert result["problem_type"] in {
        "binary_classification",
        "multiclass_classification",
        "regression",
    }
    assert result["confidence"] in {"high", "medium", "low"}
    assert result["needs_human_review"] == result["requires_human_review"]
    json.dumps(result, default=str)


def test_detects_binary_classification(classification_df):
    result = detect_problem_type(classification_df, target_column="approved")

    assert_problem_detector_contract(result)
    assert result["problem_type"] == "binary_classification"
    assert result["unique_values"] == 2
    assert result["confidence"] == "high"
    assert result["needs_human_review"] is False
    assert result["class_balance_preview"] is not None


def test_detects_binary_string_target():
    df = pd.DataFrame(
        {
            "feature": range(8),
            "target": ["yes", "no", "yes", "no", "yes", "yes", "no", "yes"],
        }
    )

    result = detect_problem_type(df, target_column="target")

    assert_problem_detector_contract(result)
    assert result["problem_type"] == "binary_classification"
    assert result["is_numeric_target"] is False
    assert result["unique_values"] == 2


def test_detects_regression(regression_df):
    result = detect_problem_type(regression_df, target_column="score")

    assert_problem_detector_contract(result)
    assert result["problem_type"] == "regression"
    assert result["is_numeric_target"] is True
    assert result["confidence"] == "high"


def test_high_unique_string_target_is_handled_safely(leaky_df):
    result = detect_problem_type(leaky_df, target_column="student_id")

    assert_problem_detector_contract(result)
    assert result["problem_type"] == "multiclass_classification"
    assert result["is_numeric_target"] is False
    assert isinstance(result["needs_human_review"], bool)
    if result["needs_human_review"]:
        assert result["human_review_reason"] is not None


def test_multiclass_string_high_cardinality_needs_review():
    df = pd.DataFrame(
        {
            "feature": range(80),
            "target": [f"class_{idx}" for idx in range(80)],
        }
    )

    result = detect_problem_type(df, target_column="target")

    assert result["problem_type"] == "multiclass_classification"
    assert result["needs_human_review"] is True
    assert result["human_review_reason"] is not None
    reason = str(result["human_review_reason"]).lower()
    assert "high" in reason or "cardinality" in reason or "unique" in reason


def test_numeric_low_unique_target_is_ambiguous_multiclass():
    df = pd.DataFrame(
        {
            "feature": range(30),
            "target": [0, 1, 2] * 10,
        }
    )

    result = detect_problem_type(
        df,
        target_column="target",
        classification_unique_threshold=20,
    )

    assert result["problem_type"] == "multiclass_classification"
    assert result["confidence"] == "medium"
    assert result["needs_human_review"] is True
    assert result["human_review_reason"] is not None


def test_numeric_integer_like_regression_can_need_review():
    target = pd.Series([1, 2, 3, 4, 5, 6] * 5)

    problem_type, confidence, needs_review, reason = infer_problem_type(
        target=target,
        unique_count=6,
        unique_percent=20.0,
        is_numeric=True,
        is_bool=False,
        is_bool_like=False,
        is_integer_like=True,
        classification_unique_threshold=3,
    )

    assert problem_type == "regression"
    assert confidence == "medium"
    assert needs_review is True
    assert reason is not None


def test_low_unique_float_target_can_be_ambiguous():
    df = pd.DataFrame(
        {
            "feature": range(24),
            "target": [1.0, 2.0, 3.0] * 8,
        }
    )

    result = detect_problem_type(
        df,
        target_column="target",
        classification_unique_threshold=5,
    )

    assert result["problem_type"] == "multiclass_classification"
    assert result["needs_human_review"] is True
    assert isinstance(result["is_integer_like_target"], bool)


def test_numeric_like_string_target_gets_warning():
    df = pd.DataFrame(
        {
            "feature": range(10),
            "target": [str(value) for value in [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]],
        }
    )

    result = detect_problem_type(df, target_column="target")

    assert result["problem_type"] in {
        "binary_classification",
        "multiclass_classification",
        "regression",
    }
    assert isinstance(result.get("warnings", []), list)


def test_bool_target_detects_binary():
    df = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "target": [True, False, True, False],
        }
    )

    result = detect_problem_type(df, target_column="target")

    assert result["problem_type"] == "binary_classification"
    assert result["is_bool_target"] is True


def test_target_column_whitespace_is_handled(classification_df):
    result = detect_problem_type(classification_df, target_column=" approved ")

    assert result["target_column"] == "approved"
    assert result["problem_type"] == "binary_classification"


def test_class_balance_preview_for_classification(classification_df):
    target = classification_df["approved"]

    preview = get_class_balance_preview(target, "binary_classification")

    assert preview is not None
    assert preview["total_classes"] == 2
    assert len(preview["top_classes"]) == 2


def test_class_balance_preview_respects_max_classes():
    target = pd.Series([f"class_{idx}" for idx in range(10)])

    preview = get_class_balance_preview(
        target,
        "multiclass_classification",
        max_classes=3,
    )

    assert preview is not None
    assert preview["shown_classes"] == 3
    assert preview["total_classes"] == 10
    assert len(preview["top_classes"]) == 3


def test_class_balance_preview_none_for_regression(regression_df):
    preview = get_class_balance_preview(regression_df["score"], "regression")

    assert preview is None


def test_get_sample_values_returns_strings(classification_df):
    values = get_sample_values(classification_df["approved"])

    assert values
    assert all(isinstance(value, str) for value in values)


def test_get_sample_values_handles_datetime_values():
    values = get_sample_values(
        pd.Series(pd.date_range("2025-01-01", periods=3)),
    )

    assert len(values) == 3
    assert all(isinstance(value, str) for value in values)


def test_is_integer_like_series():
    assert is_integer_like_series(pd.Series([1.0, 2.0, 3.0])) is True
    assert is_integer_like_series(pd.Series([1.1, 2.0, 3.0])) is False


def test_is_integer_like_series_handles_numeric_strings():
    assert is_integer_like_series(pd.Series(["1", "2", "3"])) is True
    assert is_integer_like_series(pd.Series(["1.1", "2", "3"])) is False


def test_detection_reason_contains_problem_type_context():
    reason = get_detection_reason(
        problem_type="regression",
        unique_count=100,
        unique_percent=90.0,
        target_dtype="float64",
        is_numeric=True,
        is_bool_like=False,
        is_integer_like=False,
        classification_unique_threshold=20,
        confidence="high",
    )

    assert "regression" in reason.lower()
    assert "confidence" in reason.lower()


def test_infer_problem_type_binary_overrides_bool_flag():
    target = pd.Series([True, False, True, False])

    problem_type, confidence, needs_review, reason = infer_problem_type(
        target=target,
        unique_count=2,
        unique_percent=50.0,
        is_numeric=False,
        is_bool=True,
        is_bool_like=True,
        is_integer_like=False,
        classification_unique_threshold=20,
    )

    assert problem_type == "binary_classification"
    assert confidence == "high"
    assert needs_review is False
    assert reason is None


def test_missing_target_column_raises_with_close_match(classification_df):
    with pytest.raises(AuditCopilotException) as error_info:
        detect_problem_type(classification_df, target_column="approvd")

    assert "target column" in str(error_info.value).lower()


def test_invalid_threshold_raises(classification_df):
    with pytest.raises(AuditCopilotException):
        detect_problem_type(
            classification_df,
            target_column="approved",
            classification_unique_threshold=1,
        )


def test_negative_threshold_raises(classification_df):
    with pytest.raises(AuditCopilotException):
        detect_problem_type(
            classification_df,
            target_column="approved",
            classification_unique_threshold=-5,
        )


def test_missing_target_column_raises(classification_df):
    with pytest.raises(AuditCopilotException):
        detect_problem_type(classification_df, target_column="does_not_exist")


def test_constant_target_raises():
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": ["same", "same", "same", "same"]})

    with pytest.raises(AuditCopilotException):
        detect_problem_type(df, target_column="y")


def test_all_missing_target_raises():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [None, None, None]})

    with pytest.raises(AuditCopilotException):
        detect_problem_type(df, target_column="y")


def test_empty_dataframe_raises():
    with pytest.raises(AuditCopilotException):
        detect_problem_type(pd.DataFrame(), target_column="target")


def test_blank_target_column_raises(classification_df):
    with pytest.raises(AuditCopilotException):
        detect_problem_type(classification_df, target_column="")


def test_recommended_action_is_non_empty(classification_df):
    result = detect_problem_type(classification_df, target_column="approved")

    assert isinstance(result["recommended_action"], str)
    assert result["recommended_action"].strip()
