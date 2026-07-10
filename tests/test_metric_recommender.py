"""
Tests for src/audit/metric_recommender.py.

These tests verify stable public output contract, problem-type normalization,
classification/regression scoring compatibility, imbalance-aware choices, and
safe config fallback behavior.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest

import src.audit.metric_recommender as metric_module
from src.audit.metric_recommender import (
    get_safe_scoring_metric,
    metric_display_name,
    normalize_imbalance_severity,
    normalize_key,
    normalize_problem_type,
    recommend_metrics,
)
from src.utils.exceptions import AuditCopilotException

ConfigGetter = Callable[[str, Any], Any]

REQUIRED_RESULT_KEYS = {
    "problem_type",
    "imbalance_severity",
    "recommended_metrics",
    "primary_metric",
    "primary_metric_key",
    "scoring_metric",
    "sklearn_scoring_metric",
    "selection_metric_key",
    "configured_default",
    "secondary_metrics",
    "probability_metrics",
    "higher_is_better",
    "optimization_direction",
    "reason",
    "notes",
    "warnings",
}


def _patch_config(monkeypatch: pytest.MonkeyPatch, overrides: dict[str, Any]) -> None:
    """Patch metric module config lookups for deterministic tests."""
    original_get_config_value: ConfigGetter = metric_module.get_config_value

    def fake_get_config_value(path: str, default: Any = None) -> Any:
        return overrides.get(path, original_get_config_value(path, default))

    monkeypatch.setattr(metric_module, "get_config_value", fake_get_config_value)


def assert_metric_contract(result: dict[str, Any]) -> None:
    """Validate common API/UI metric recommendation output contract."""
    assert REQUIRED_RESULT_KEYS.issubset(result.keys())
    assert result["problem_type"] in {
        "binary_classification",
        "multiclass_classification",
        "regression",
    }
    assert isinstance(result["recommended_metrics"], list)
    assert result["recommended_metrics"]
    assert isinstance(result["secondary_metrics"], list)
    assert isinstance(result["probability_metrics"], list)
    assert isinstance(result["higher_is_better"], bool)
    assert result["optimization_direction"] in {"maximize", "minimize"}
    assert result["sklearn_scoring_metric"] == result["scoring_metric"]
    assert isinstance(result["reason"], str)
    assert result["reason"].strip()
    assert isinstance(result["notes"], list)
    assert isinstance(result["warnings"], list)
    json.dumps(result, default=str)


def test_binary_classification_metrics_default_contract() -> None:
    result = recommend_metrics(
        problem_type="binary_classification",
        imbalance_severity="low",
    )

    assert_metric_contract(result)
    assert result["problem_type"] == "binary_classification"
    assert result["primary_metric"] == "F1 Score"
    assert result["selection_metric_key"] == "f1_score"
    assert result["higher_is_better"] is True
    assert result["optimization_direction"] == "maximize"
    assert "PR-AUC" in result["probability_metrics"]
    assert "ROC-AUC" in result["probability_metrics"]


def test_binary_classification_imbalanced_reason_mentions_accuracy() -> None:
    result = recommend_metrics(
        problem_type="binary_classification",
        imbalance_severity="severe",
    )

    assert_metric_contract(result)
    assert result["imbalance_severity"] == "severe"
    assert result["primary_metric"] == "F1 Score"
    assert "accuracy" in result["reason"].lower()


def test_binary_metric_config_f1_weighted_warns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_config(
        monkeypatch,
        {
            "metrics.classification_default": "f1_weighted",
        },
    )

    result = recommend_metrics(
        problem_type="binary_classification",
        imbalance_severity="low",
    )

    assert_metric_contract(result)
    assert result["scoring_metric"] == "f1_weighted"
    assert result["warnings"]


def test_multiclass_balanced_uses_weighted_f1() -> None:
    result = recommend_metrics(
        problem_type="multiclass_classification",
        imbalance_severity="low",
    )

    assert_metric_contract(result)
    assert result["problem_type"] == "multiclass_classification"
    assert result["primary_metric"] == "Weighted F1 Score"
    assert result["scoring_metric"] in {"f1_weighted", "f1_macro", "balanced_accuracy"}
    assert result["selection_metric_key"] == "f1_score"
    assert "Macro F1 Score" in result["recommended_metrics"]
    assert "Weighted F1 Score" in result["recommended_metrics"]


def test_multiclass_imbalanced_uses_macro_f1() -> None:
    result = recommend_metrics(
        problem_type="multiclass_classification",
        imbalance_severity="high",
    )

    assert_metric_contract(result)
    assert result["primary_metric"] == "Macro F1 Score"
    assert result["scoring_metric"] == "f1_macro"
    assert result["selection_metric_key"] == "f1_macro"
    assert "minority" in result["reason"].lower()


def test_multiclass_binary_config_falls_back_safely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_config(
        monkeypatch,
        {
            "metrics.classification_default": "f1",
        },
    )

    result = recommend_metrics(
        problem_type="multiclass_classification",
        imbalance_severity="low",
    )

    assert_metric_contract(result)
    assert result["scoring_metric"] == "f1_weighted"
    assert result["warnings"]


def test_regression_default_rmse_contract() -> None:
    result = recommend_metrics(problem_type="regression")

    assert_metric_contract(result)
    assert result["problem_type"] == "regression"
    assert result["imbalance_severity"] == "not_applicable"
    assert result["primary_metric"] == "RMSE"
    assert result["scoring_metric"] == "neg_root_mean_squared_error"
    assert result["selection_metric_key"] == "rmse"
    assert result["higher_is_better"] is False
    assert result["optimization_direction"] == "minimize"


def test_regression_r2_config_is_maximize(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_config(
        monkeypatch,
        {
            "metrics.regression_default": "r2",
        },
    )

    result = recommend_metrics(problem_type="regression")

    assert_metric_contract(result)
    assert result["primary_metric"] == "R2 Score"
    assert result["scoring_metric"] == "r2"
    assert result["selection_metric_key"] == "r2_score"
    assert result["higher_is_better"] is True
    assert result["optimization_direction"] == "maximize"


def test_regression_unsupported_config_falls_back_to_rmse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_config(
        monkeypatch,
        {
            "metrics.regression_default": "unsupported_metric",
        },
    )

    result = recommend_metrics(problem_type="regression")

    assert_metric_contract(result)
    assert result["primary_metric"] == "RMSE"
    assert result["scoring_metric"] == "neg_root_mean_squared_error"
    assert result["warnings"]


def test_regression_mape_alias_maps_to_sklearn_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_config(
        monkeypatch,
        {
            "metrics.regression_default": "mape",
        },
    )

    result = recommend_metrics(problem_type="regression")

    assert_metric_contract(result)
    assert result["primary_metric"] == "MAPE"
    assert result["scoring_metric"] == "neg_mean_absolute_percentage_error"
    assert result["selection_metric_key"] == "mape"
    assert result["higher_is_better"] is False


@pytest.mark.parametrize(
    ("raw_problem_type", "expected"),
    [
        ("binary", "binary_classification"),
        ("binary classification", "binary_classification"),
        ("multi-class", "multiclass_classification"),
        ("multi_class", "multiclass_classification"),
        ("classification", "multiclass_classification"),
        ("regression", "regression"),
    ],
)
def test_normalize_problem_type_aliases(raw_problem_type: str, expected: str) -> None:
    assert normalize_problem_type(raw_problem_type) == expected


@pytest.mark.parametrize(
    ("raw_severity", "expected"),
    [
        ("medium", "moderate"),
        ("moderate", "moderate"),
        ("high", "high"),
        ("severe", "severe"),
        ("not_applicable", "not_applicable"),
        (None, "unknown"),
        ("weird", "unknown"),
    ],
)
def test_normalize_imbalance_severity(raw_severity: str | None, expected: str) -> None:
    assert normalize_imbalance_severity(raw_severity) == expected


def test_normalize_key_collapses_spaces_hyphens_and_slashes() -> None:
    assert normalize_key(" Macro-F1 / Score ") == "macro_f1_score"
    assert normalize_key("///") == "unknown"


def test_metric_display_names() -> None:
    assert metric_display_name("f1_weighted") == "Weighted F1 Score"
    assert metric_display_name("pr_auc") == "PR-AUC"
    assert metric_display_name("prauc") == "PR-AUC"
    assert metric_display_name("rmse") == "RMSE"
    assert metric_display_name("unknown_metric") == "UNKNOWN_METRIC"


def test_get_safe_scoring_metric_binary_aliases() -> None:
    assert (
        get_safe_scoring_metric(
            "binary_classification",
            configured_metric="pr_auc",
            fallback="f1_score",
        )
        == "average_precision"
    )
    assert (
        get_safe_scoring_metric(
            "binary_classification",
            configured_metric="f1_score",
            fallback="accuracy",
        )
        == "f1"
    )


def test_get_safe_scoring_metric_multiclass_aliases() -> None:
    assert (
        get_safe_scoring_metric(
            "multiclass_classification",
            configured_metric="macro_f1",
            fallback="f1_weighted",
        )
        == "f1_macro"
    )
    assert (
        get_safe_scoring_metric(
            "multiclass_classification",
            configured_metric="macro_precision",
            fallback="f1_weighted",
        )
        == "precision_macro"
    )


def test_get_safe_scoring_metric_regression_aliases() -> None:
    assert (
        get_safe_scoring_metric(
            "regression",
            configured_metric="mae",
            fallback="rmse",
        )
        == "neg_mean_absolute_error"
    )
    assert (
        get_safe_scoring_metric(
            "regression",
            configured_metric="r2_score",
            fallback="rmse",
        )
        == "r2"
    )


def test_get_safe_scoring_metric_unknown_falls_back() -> None:
    assert (
        get_safe_scoring_metric(
            "binary_classification",
            configured_metric="does_not_exist",
            fallback="f1_score",
        )
        == "f1"
    )


def test_unsupported_problem_type_raises() -> None:
    with pytest.raises(AuditCopilotException):
        recommend_metrics(problem_type="clustering")


def test_blank_problem_type_raises() -> None:
    with pytest.raises(AuditCopilotException):
        recommend_metrics(problem_type="")


def test_config_error_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_get_config_value(_path: str, _default: Any = None) -> Any:
        raise RuntimeError("config failed")

    monkeypatch.setattr(metric_module, "get_config_value", broken_get_config_value)

    result = recommend_metrics(
        problem_type="binary_classification",
        imbalance_severity="low",
    )

    assert_metric_contract(result)
    assert result["configured_default"] == "f1_score"
    assert result["scoring_metric"] == "f1"
