"""
Tests for src/audit/class_imbalance.py.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.audit.class_imbalance import (
    calculate_effective_class_count,
    calculate_entropy,
    calculate_gini_impurity,
    detect_class_imbalance,
    get_imbalance_severity,
    get_rare_classes,
)
from src.utils.exceptions import AuditCopilotException


def test_detects_imbalance_ratio(classification_df):
    result = detect_class_imbalance(
        classification_df,
        target_column="approved",
        problem_type="binary_classification",
    )

    assert result["is_applicable"] is True
    assert result["imbalance_ratio"] == 1.67
    assert result["majority_class"] == "1"
    assert result["minority_class"] == "0"
    assert result["num_classes"] == 2
    assert "recommended_metrics" in result
    assert "recommended_actions" in result


def test_not_applicable_for_regression(regression_df):
    result = detect_class_imbalance(
        regression_df,
        target_column="score",
        problem_type="regression",
    )

    assert result["is_applicable"] is False
    assert result["problem_type"] == "regression"


def test_perfectly_balanced_data_has_low_severity():
    df = pd.DataFrame({"x": range(20), "label": [0, 1] * 10})

    result = detect_class_imbalance(
        df,
        target_column="label",
        problem_type="binary_classification",
    )

    assert result["imbalance_ratio"] == 1.0
    assert result["imbalance_severity"] == "low"
    assert result["requires_human_review"] is False


def test_strongly_imbalanced_data_detects_high_or_severe(imbalanced_classification_df):
    result = detect_class_imbalance(
        imbalanced_classification_df,
        target_column="approved",
        problem_type="binary_classification",
    )

    assert result["is_applicable"] is True
    assert result["imbalance_ratio"] > 10
    assert result["imbalance_severity"] == "severe"
    assert result["requires_human_review"] is True
    assert result["rare_classes"] == {}


def test_multiclass_imbalance_is_supported(multiclass_df):
    result = detect_class_imbalance(
        multiclass_df,
        target_column="risk_band",
        problem_type="multiclass_classification",
    )

    assert result["is_applicable"] is True
    assert result["num_classes"] == 3
    assert "class_counts" in result
    assert "distribution_metrics" in result


def test_severity_thresholds_are_monotonic():
    severity_order = ["low", "moderate", "high", "severe"]
    ratios = [1.0, 2.0, 5.0, 20.0]

    severities = [
        get_imbalance_severity(ratio, rare_classes={})
        for ratio in ratios
    ]

    indices = [severity_order.index(severity) for severity in severities]
    assert indices == sorted(indices)


def test_rare_classes_can_raise_low_ratio_to_moderate():
    severity = get_imbalance_severity(
        imbalance_ratio=2.0,
        rare_classes={"rare_label": 3.0},
    )

    assert severity == "moderate"


def test_get_rare_classes_returns_classes_below_threshold():
    percentages = pd.Series({"A": 90.0, "B": 7.0, "C": 3.0})

    rare = get_rare_classes(percentages, rare_class_threshold_percent=5)

    assert rare == {"C": 3.0}


def test_distribution_metrics_are_reasonable():
    percentages = pd.Series({"A": 50.0, "B": 50.0})

    entropy = calculate_entropy(percentages)
    gini = calculate_gini_impurity(percentages)
    effective_classes = calculate_effective_class_count(percentages)

    assert entropy == 1.0
    assert gini == 0.5
    assert effective_classes == 2.0


def test_missing_target_column_raises(classification_df):
    with pytest.raises(AuditCopilotException):
        detect_class_imbalance(
            classification_df,
            target_column="missing_column",
            problem_type="binary_classification",
        )


def test_single_class_target_raises(invalid_single_class_df):
    with pytest.raises(AuditCopilotException):
        detect_class_imbalance(
            invalid_single_class_df,
            target_column="target",
            problem_type="binary_classification",
        )


def test_unsupported_problem_type_raises(classification_df):
    with pytest.raises(AuditCopilotException):
        detect_class_imbalance(
            classification_df,
            target_column="approved",
            problem_type="clustering",
        )
