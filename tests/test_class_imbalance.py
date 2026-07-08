"""
Tests for src/audit/class_imbalance.py
"""
from __future__ import annotations

import pandas as pd

from src.audit.class_imbalance import detect_class_imbalance, get_imbalance_severity


def test_detects_imbalance_ratio(classification_df):
    result = detect_class_imbalance(
        classification_df,
        target_column="approved",
        problem_type="binary_classification",
    )
    assert result["is_applicable"] is True
    # 25 vs 15 -> ratio = 1.67
    assert result["imbalance_ratio"] == 1.67
    assert result["majority_class"] == "1"
    assert result["minority_class"] == "0"


def test_not_applicable_for_regression(regression_df):
    result = detect_class_imbalance(
        regression_df,
        target_column="score",
        problem_type="regression",
    )
    assert result["is_applicable"] is False


def test_perfectly_balanced_data_has_low_severity():
    df = pd.DataFrame({"x": range(20), "label": [0, 1] * 10})
    result = detect_class_imbalance(df, target_column="label", problem_type="binary_classification")
    assert result["imbalance_ratio"] == 1.0
    assert result["imbalance_severity"] == "low"


def test_severity_thresholds_are_monotonic():
    """
    Sanity check on the severity function directly: as the ratio grows,
    severity should never become "less severe".
    """
    severity_order = ["low", "moderate", "high", "severe"]
    ratios = [1.0, 2.0, 5.0, 20.0]
    severities = [get_imbalance_severity(r) for r in ratios]
    indices = [severity_order.index(s) for s in severities]
    assert indices == sorted(indices)