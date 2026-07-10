"""
Tests for src/audit/class_imbalance.py.

These tests focus on the public contract of detect_class_imbalance() and the
small deterministic helper functions used by the audit workflow. The assertions
avoid overfitting to implementation details while still checking production
behavior: JSON-safe outputs, safe edge-case handling, and useful review signals.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import pytest

from src.audit.class_imbalance import (
    calculate_effective_class_count,
    calculate_entropy,
    calculate_gini_impurity,
    detect_class_imbalance,
    get_imbalance_severity,
    get_rare_classes,
    safe_percent,
)
from src.utils.exceptions import AuditCopilotException

CLASSIFICATION_TYPES = {"binary_classification", "multiclass_classification"}


def assert_common_imbalance_contract(result: dict[str, Any]) -> None:
    """Validate common response contract for applicable imbalance reports."""
    assert result["is_applicable"] is True
    assert result["problem_type"] in CLASSIFICATION_TYPES
    assert isinstance(result["target_column"], str)
    assert isinstance(result["total_rows"], int)
    assert isinstance(result["valid_target_rows"], int)
    assert isinstance(result["missing_target_rows"], int)
    assert isinstance(result["missing_target_percent"], float)
    assert isinstance(result["num_classes"], int)
    assert isinstance(result["class_counts"], dict)
    assert isinstance(result["class_percentages"], dict)
    assert isinstance(result["imbalance_ratio"], float)
    assert result["imbalance_severity"] in {"low", "moderate", "high", "severe"}
    assert isinstance(result["rare_classes"], dict)
    assert isinstance(result["distribution_metrics"], dict)
    assert isinstance(result["findings"], list)
    assert isinstance(result["recommended_metrics"], list)
    assert isinstance(result["recommended_actions"], list)
    assert isinstance(result["warning"], str)
    assert result["message"] == "Class imbalance detection completed successfully."


def test_class_imbalance_output_is_json_serializable(
    classification_df: pd.DataFrame,
) -> None:
    result = detect_class_imbalance(
        classification_df,
        target_column="approved",
        problem_type="binary_classification",
    )

    json.dumps(result, default=str)
    assert result["message"] == "Class imbalance detection completed successfully."


def test_class_imbalance_result_has_required_top_level_keys(
    classification_df: pd.DataFrame,
) -> None:
    result = detect_class_imbalance(
        classification_df,
        target_column="approved",
        problem_type="binary_classification",
    )

    required_keys = {
        "is_applicable",
        "problem_type",
        "target_column",
        "total_rows",
        "valid_target_rows",
        "missing_target_rows",
        "missing_target_percent",
        "num_classes",
        "class_counts",
        "class_percentages",
        "majority_class",
        "minority_class",
        "majority_count",
        "minority_count",
        "min_class_count",
        "max_class_count",
        "imbalance_ratio",
        "imbalance_severity",
        "rare_classes",
        "distribution_metrics",
        "requires_human_review",
        "findings",
        "recommended_metrics",
        "recommended_actions",
        "warning",
        "message",
    }

    assert required_keys.issubset(result.keys())


def test_detects_binary_imbalance_ratio(classification_df: pd.DataFrame) -> None:
    result = detect_class_imbalance(
        classification_df,
        target_column="approved",
        problem_type="binary_classification",
    )

    assert_common_imbalance_contract(result)
    assert result["imbalance_ratio"] == pytest.approx(1.67, abs=0.01)
    assert result["majority_class"] == "1"
    assert result["minority_class"] == "0"
    assert result["majority_count"] == 25
    assert result["minority_count"] == 15
    assert result["num_classes"] == 2
    assert result["class_counts"] == {"1": 25, "0": 15}
    assert result["missing_target_rows"] == 0


def test_not_applicable_for_regression(regression_df: pd.DataFrame) -> None:
    result = detect_class_imbalance(
        regression_df,
        target_column="score",
        problem_type="regression",
    )

    assert result["is_applicable"] is False
    assert result["problem_type"] == "regression"
    assert result["target_column"] == "score"
    assert "not applicable" in result["message"].lower()


def test_perfectly_balanced_data_has_low_severity() -> None:
    df = pd.DataFrame({"x": range(20), "label": [0, 1] * 10})

    result = detect_class_imbalance(
        df,
        target_column="label",
        problem_type="binary_classification",
    )

    assert_common_imbalance_contract(result)
    assert result["imbalance_ratio"] == 1.0
    assert result["imbalance_severity"] == "low"
    assert result["requires_human_review"] is False
    assert result["findings"] == []
    assert result["rare_classes"] == {}


def test_strongly_imbalanced_data_detects_severe(
    imbalanced_classification_df: pd.DataFrame,
) -> None:
    result = detect_class_imbalance(
        imbalanced_classification_df,
        target_column="approved",
        problem_type="binary_classification",
    )

    assert_common_imbalance_contract(result)
    assert result["imbalance_ratio"] > 10
    assert result["imbalance_severity"] == "severe"
    assert result["requires_human_review"] is True
    assert any(
        finding.get("category") == "class_imbalance" for finding in result["findings"]
    )


def test_multiclass_imbalance_is_supported(multiclass_df: pd.DataFrame) -> None:
    result = detect_class_imbalance(
        multiclass_df,
        target_column="risk_band",
        problem_type="multiclass_classification",
    )

    assert_common_imbalance_contract(result)
    assert result["num_classes"] == 3
    assert set(result["class_counts"]) == {"low", "medium", "high"}
    assert set(result["distribution_metrics"]) == {
        "normalized_entropy",
        "gini_impurity",
        "effective_class_count",
    }
    assert "Macro F1 Score" in result["recommended_metrics"]


def test_missing_target_values_are_reported(missing_target_df: pd.DataFrame) -> None:
    result = detect_class_imbalance(
        missing_target_df,
        target_column="target",
        problem_type="binary_classification",
    )

    assert_common_imbalance_contract(result)
    assert result["total_rows"] == 6
    assert result["valid_target_rows"] == 4
    assert result["missing_target_rows"] == 2
    assert result["missing_target_percent"] == 33.33


def test_tiny_class_adds_review_findings(tiny_class_df: pd.DataFrame) -> None:
    result = detect_class_imbalance(
        tiny_class_df,
        target_column="target",
        problem_type="binary_classification",
    )

    assert_common_imbalance_contract(result)
    assert result["min_class_count"] == 1
    assert result["requires_human_review"] is True
    assert any(
        finding.get("category") == "insufficient_class_samples"
        for finding in result["findings"]
    )
    assert "fewer than 2 samples" in result["warning"]


def test_optional_class_distribution_records_when_available(
    classification_df: pd.DataFrame,
) -> None:
    """Newer reports expose list records for Streamlit/API tables."""
    result = detect_class_imbalance(
        classification_df,
        target_column="approved",
        problem_type="binary_classification",
    )

    class_distribution = result.get("class_distribution")
    if class_distribution is None:
        pytest.skip("class_distribution is optional for older compatible outputs.")

    assert isinstance(class_distribution, list)
    assert class_distribution
    assert {"class", "count", "percent"}.issubset(class_distribution[0])


def test_optional_split_viability_summary_when_available(
    tiny_class_df: pd.DataFrame,
) -> None:
    """Newer reports expose split/CV viability warnings for modeling."""
    result = detect_class_imbalance(
        tiny_class_df,
        target_column="target",
        problem_type="binary_classification",
    )

    split_viability = result.get("split_viability")
    if split_viability is None:
        pytest.skip("split_viability is optional for older compatible outputs.")

    assert isinstance(split_viability, dict)
    assert split_viability.get("can_stratify_train_test_split") is False
    assert split_viability.get("can_use_3_fold_stratified_cv") is False
    assert split_viability.get("can_use_5_fold_stratified_cv") is False


def test_severity_thresholds_are_monotonic() -> None:
    severity_order = ["low", "moderate", "high", "severe"]
    ratios = [1.0, 2.0, 5.0, 20.0]

    severities = [get_imbalance_severity(ratio, rare_classes={}) for ratio in ratios]

    indices = [severity_order.index(severity) for severity in severities]
    assert indices == sorted(indices)


def test_rare_classes_can_raise_low_ratio_to_moderate() -> None:
    severity = get_imbalance_severity(
        imbalance_ratio=2.0,
        rare_classes={"rare_label": 3.0},
    )

    assert severity == "moderate"


def test_get_rare_classes_returns_classes_below_threshold() -> None:
    percentages = pd.Series({"A": 90.0, "B": 7.0, "C": 3.0})

    rare = get_rare_classes(percentages, rare_class_threshold_percent=5)

    assert rare == {"C": 3.0}


def test_get_rare_classes_excludes_exact_threshold() -> None:
    percentages = pd.Series({"A": 95.0, "B": 5.0})

    rare = get_rare_classes(percentages, rare_class_threshold_percent=5)

    assert rare == {}


def test_distribution_metrics_are_reasonable_for_balanced_binary() -> None:
    percentages = pd.Series({"A": 50.0, "B": 50.0})

    entropy = calculate_entropy(percentages)
    gini = calculate_gini_impurity(percentages)
    effective_classes = calculate_effective_class_count(percentages)

    assert entropy == 1.0
    assert gini == 0.5
    assert effective_classes == 2.0


def test_distribution_metrics_handle_empty_or_single_class_series() -> None:
    empty = pd.Series(dtype=float)
    single = pd.Series({"A": 100.0})

    assert calculate_entropy(empty) == 0.0
    assert calculate_entropy(single) == 0.0
    assert calculate_gini_impurity(empty) == 0.0
    assert calculate_effective_class_count(empty) == 0.0
    assert calculate_effective_class_count(single) == 1.0


def test_safe_percent_handles_zero_denominator() -> None:
    assert safe_percent(10, 0) == 0.0
    assert safe_percent(1, 4) == 25.0


def test_target_column_is_trimmed_when_supported(
    classification_df: pd.DataFrame,
) -> None:
    result = detect_class_imbalance(
        classification_df,
        target_column=" approved ",
        problem_type=" binary_classification ",
    )

    assert_common_imbalance_contract(result)
    assert result["target_column"] == "approved"
    assert result["problem_type"] == "binary_classification"


def test_missing_target_column_raises(classification_df: pd.DataFrame) -> None:
    with pytest.raises(AuditCopilotException):
        detect_class_imbalance(
            classification_df,
            target_column="missing_column",
            problem_type="binary_classification",
        )


def test_single_class_target_raises(invalid_single_class_df: pd.DataFrame) -> None:
    with pytest.raises(AuditCopilotException):
        detect_class_imbalance(
            invalid_single_class_df,
            target_column="target",
            problem_type="binary_classification",
        )


def test_unsupported_problem_type_raises(classification_df: pd.DataFrame) -> None:
    with pytest.raises(AuditCopilotException):
        detect_class_imbalance(
            classification_df,
            target_column="approved",
            problem_type="clustering",
        )


def test_empty_dataframe_raises() -> None:
    with pytest.raises(AuditCopilotException):
        detect_class_imbalance(
            pd.DataFrame(),
            target_column="target",
            problem_type="binary_classification",
        )


def test_regression_not_applicable_output_is_json_serializable(
    regression_df: pd.DataFrame,
) -> None:
    result = detect_class_imbalance(
        regression_df,
        target_column="score",
        problem_type="regression",
    )

    json.dumps(result, default=str)
    assert result["is_applicable"] is False
    assert result.get("imbalance_severity", "not_applicable") == "not_applicable"
