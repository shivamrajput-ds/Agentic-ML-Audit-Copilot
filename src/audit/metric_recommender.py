from __future__ import annotations

from typing import Any

from src.utils.config import get_config_value
from src.utils.exceptions import MetricRecommendationError
from src.utils.logger import get_logger

logger = get_logger(__name__)

CLASSIFICATION_TYPES = {"binary_classification", "multiclass_classification"}
SUPPORTED_PROBLEM_TYPES = CLASSIFICATION_TYPES | {"regression"}

PROBLEM_TYPE_ALIASES = {
    "binary": "binary_classification",
    "binary_classification": "binary_classification",
    "binary classification": "binary_classification",
    "multiclass": "multiclass_classification",
    "multi_class": "multiclass_classification",
    "multi-class": "multiclass_classification",
    "multiclass_classification": "multiclass_classification",
    "multiclass classification": "multiclass_classification",
    "classification": "multiclass_classification",
    "regression": "regression",
}

IMBALANCE_LEVELS = {
    "none",
    "not_applicable",
    "low",
    "moderate",
    "medium",
    "high",
    "severe",
    "unknown",
}

IMBALANCED_LEVELS = {"moderate", "medium", "high", "severe"}

# sklearn scoring names are not always the same as the human-facing metric keys
# stored in baseline model outputs. Keep this mapping explicit.
SKLEARN_SCORING_MAP = {
    "binary_classification": {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1": "f1",
        "f1_score": "f1",
        "f1_binary": "f1",
        "f1_weighted": "f1_weighted",
        "precision": "precision",
        "precision_binary": "precision",
        "recall": "recall",
        "recall_binary": "recall",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "prauc": "average_precision",
        "average_precision": "average_precision",
    },
    "multiclass_classification": {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1_macro": "f1_macro",
        "macro_f1": "f1_macro",
        "f1_weighted": "f1_weighted",
        "weighted_f1": "f1_weighted",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "macro_recall": "recall_macro",
        "macro_precision": "precision_macro",
    },
    "regression": {
        "rmse": "neg_root_mean_squared_error",
        "root_mean_squared_error": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "mean_absolute_error": "neg_mean_absolute_error",
        "mse": "neg_mean_squared_error",
        "mean_squared_error": "neg_mean_squared_error",
        "r2": "r2",
        "r2_score": "r2",
        "mape": "neg_mean_absolute_percentage_error",
        "mean_absolute_percentage_error": "neg_mean_absolute_percentage_error",
    },
}

DISPLAY_METRIC_NAMES = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced Accuracy",
    "f1": "F1 Score",
    "f1_score": "F1 Score",
    "f1_binary": "Binary F1 Score",
    "f1_macro": "Macro F1 Score",
    "f1_weighted": "Weighted F1 Score",
    "precision": "Precision",
    "precision_binary": "Binary Precision",
    "precision_macro": "Macro Precision",
    "recall": "Recall",
    "recall_binary": "Binary Recall",
    "recall_macro": "Macro Recall",
    "roc_auc": "ROC-AUC",
    "average_precision": "PR-AUC",
    "pr_auc": "PR-AUC",
    "prauc": "PR-AUC",
    "rmse": "RMSE",
    "mae": "MAE",
    "mse": "MSE",
    "r2": "R2 Score",
    "r2_score": "R2 Score",
    "mape": "MAPE",
}

LOWER_IS_BETTER_REGRESSION_METRICS = {"rmse", "mae", "mse", "mape"}
HIGHER_IS_BETTER_REGRESSION_METRICS = {"r2", "r2_score"}


def normalize_text(value: Any, default: str = "unknown") -> str:
    """Normalize small config/user text values."""
    if value is None or not str(value).strip():
        return default

    return str(value).strip().lower()


def normalize_key(value: Any, default: str = "unknown") -> str:
    """Normalize text into a stable key for metric/problem aliases."""
    text = normalize_text(value, default=default)
    key = text.replace("-", "_").replace(" ", "_").replace("/", "_")
    while "__" in key:
        key = key.replace("__", "_")
    return key.strip("_") or default


def get_config_text(path: str, default: str) -> str:
    """Read config value as clean lower-case text."""
    try:
        value = get_config_value(path, default)
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        logger.warning(
            "Invalid config value for %s. Using default=%s. Error=%s",
            path,
            default,
            error,
        )
        return normalize_key(default, default=default)

    return normalize_key(value, default=default)


def normalize_problem_type(problem_type: Any) -> str:
    """Normalize problem type while preserving official output names."""
    raw = normalize_text(problem_type)
    key = normalize_key(problem_type)

    if raw in PROBLEM_TYPE_ALIASES:
        return PROBLEM_TYPE_ALIASES[raw]

    return PROBLEM_TYPE_ALIASES.get(key, key)


def normalize_imbalance_severity(imbalance_severity: str | None) -> str:
    """Normalize imbalance level from class imbalance module/config."""
    normalized = normalize_key(imbalance_severity, default="unknown")

    if normalized == "medium":
        return "moderate"

    if normalized not in IMBALANCE_LEVELS:
        return "unknown"

    return normalized


def get_safe_scoring_metric(
    problem_type: str,
    configured_metric: str,
    fallback: str,
) -> str:
    """Return a sklearn-compatible scoring metric."""
    metric_map = SKLEARN_SCORING_MAP.get(problem_type, {})
    normalized_metric = normalize_key(configured_metric, fallback)
    fallback_metric = metric_map.get(normalize_key(fallback), fallback)
    return metric_map.get(normalized_metric, fallback_metric)


def metric_display_name(metric_key: str) -> str:
    """Return a human-friendly metric name."""
    return DISPLAY_METRIC_NAMES.get(normalize_key(metric_key), metric_key.upper())


def build_common_output(
    *,
    problem_type: str,
    imbalance_severity: str,
    recommended_metrics: list[str],
    primary_metric: str,
    scoring_metric: str,
    configured_default: str,
    secondary_metrics: list[str],
    higher_is_better: bool,
    reason: str,
    notes: list[str],
    selection_metric_key: str,
    probability_metrics: list[str] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Build a backward-compatible metric recommendation response."""
    optimization_direction = "maximize" if higher_is_better else "minimize"

    return {
        "problem_type": problem_type,
        "imbalance_severity": imbalance_severity,
        "recommended_metrics": recommended_metrics,
        "primary_metric": primary_metric,
        "primary_metric_key": normalize_key(primary_metric),
        "scoring_metric": scoring_metric,
        "sklearn_scoring_metric": scoring_metric,
        "selection_metric_key": selection_metric_key,
        "configured_default": configured_default,
        "secondary_metrics": secondary_metrics,
        "probability_metrics": probability_metrics or [],
        "higher_is_better": higher_is_better,
        "optimization_direction": optimization_direction,
        "reason": reason,
        "notes": notes,
        "warnings": warnings or [],
    }


def recommend_metrics(
    problem_type: str,
    imbalance_severity: str | None = None,
) -> dict[str, Any]:
    """Recommend evaluation metrics based on problem type and imbalance severity."""
    try:
        logger.info("Starting metric recommendation")

        normalized_problem_type = normalize_problem_type(problem_type)

        if normalized_problem_type not in SUPPORTED_PROBLEM_TYPES:
            raise MetricRecommendationError(
                f"Unsupported problem type: {normalized_problem_type}",
            )

        normalized_imbalance = normalize_imbalance_severity(imbalance_severity)

        if normalized_problem_type == "binary_classification":
            result = _binary_classification_metrics(normalized_imbalance)
        elif normalized_problem_type == "multiclass_classification":
            result = _multiclass_classification_metrics(normalized_imbalance)
        else:
            result = _regression_metrics()

        logger.info(
            "Metric recommendation completed. Problem type=%s Primary metric=%s Scoring=%s",
            result["problem_type"],
            result["primary_metric"],
            result["scoring_metric"],
        )
        return result

    except MetricRecommendationError:
        raise
    except (TypeError, ValueError, KeyError, AttributeError) as error:
        logger.exception("Metric recommendation failed.")
        raise MetricRecommendationError(
            "Metric recommendation failed.",
            error_detail=str(error),
        ) from error


def _binary_classification_metrics(imbalance_severity: str) -> dict[str, Any]:
    configured_default = get_config_text(
        "metrics.classification_default",
        "f1_score",
    )

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC",
        "PR-AUC",
        "Balanced Accuracy",
        "Confusion Matrix",
    ]

    warnings: list[str] = []

    if configured_default == "f1_weighted":
        warnings.append(
            "Configured classification_default='f1_weighted' is valid, but binary "
            "classification usually reports plain F1 as the primary metric.",
        )

    if imbalance_severity in IMBALANCED_LEVELS:
        reason = (
            "Binary classification with class imbalance should not rely on accuracy. "
            "F1 balances precision and recall, while PR-AUC is useful when the "
            "positive class is rare."
        )
        selection_metric_key = "f1_score"
    else:
        reason = (
            "Binary classification needs both overall correctness and class-wise "
            "performance. Precision, Recall, F1, ROC-AUC, PR-AUC, Balanced Accuracy, "
            "and Confusion Matrix provide a balanced view."
        )
        selection_metric_key = "f1_score"

    scoring_metric = get_safe_scoring_metric(
        "binary_classification",
        configured_default,
        "f1_score",
    )

    return build_common_output(
        problem_type="binary_classification",
        imbalance_severity=imbalance_severity,
        recommended_metrics=metrics,
        primary_metric="F1 Score",
        scoring_metric=scoring_metric,
        configured_default=configured_default,
        secondary_metrics=[
            "Precision",
            "Recall",
            "ROC-AUC",
            "PR-AUC",
            "Balanced Accuracy",
        ],
        probability_metrics=["ROC-AUC", "PR-AUC"],
        higher_is_better=True,
        selection_metric_key=selection_metric_key,
        reason=reason,
        notes=[
            "Use Accuracy only as a supporting metric.",
            "Use PR-AUC when the positive class is rare.",
            "Use Confusion Matrix to inspect false positives and false negatives.",
            "Check the business cost of false positives versus false negatives before final model selection.",
        ],
        warnings=warnings,
    )


def _multiclass_classification_metrics(imbalance_severity: str) -> dict[str, Any]:
    configured_default = get_config_text(
        "metrics.classification_default",
        "f1_weighted",
    )

    metrics = [
        "Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1 Score",
        "Weighted F1 Score",
        "Balanced Accuracy",
        "Confusion Matrix",
    ]

    warnings: list[str] = []

    if configured_default in {"f1", "f1_score", "f1_binary"}:
        warnings.append(
            "Configured classification_default maps to binary F1. For multiclass, "
            "using f1_weighted/f1_macro is safer.",
        )

    if imbalance_severity in IMBALANCED_LEVELS:
        primary_metric = "Macro F1 Score"
        scoring_metric = "f1_macro"
        selection_metric_key = "f1_macro"
        reason = (
            "Multiclass classification with imbalance should focus on minority-class "
            "performance. Macro F1 treats every class equally, while Weighted F1 "
            "reflects class frequency."
        )
    else:
        primary_metric = "Weighted F1 Score"
        scoring_metric = get_safe_scoring_metric(
            "multiclass_classification",
            configured_default,
            "f1_weighted",
        )
        if scoring_metric in {"f1", "precision", "recall", "roc_auc"}:
            scoring_metric = "f1_weighted"
        selection_metric_key = "f1_score"
        reason = (
            "Multiclass classification needs metrics that evaluate all classes. "
            "Macro metrics treat all classes equally, while weighted metrics account "
            "for class frequency."
        )

    return build_common_output(
        problem_type="multiclass_classification",
        imbalance_severity=imbalance_severity,
        recommended_metrics=metrics,
        primary_metric=primary_metric,
        scoring_metric=scoring_metric,
        configured_default=configured_default,
        secondary_metrics=[
            "Accuracy",
            "Macro F1 Score",
            "Weighted F1 Score",
            "Balanced Accuracy",
        ],
        probability_metrics=[],
        higher_is_better=True,
        selection_metric_key=selection_metric_key,
        reason=reason,
        notes=[
            "Use Macro F1 to check minority-class performance.",
            "Use Weighted F1 to reflect real class distribution.",
            "Use Confusion Matrix to inspect which classes are confused.",
            "For many classes, also inspect per-class recall for business-critical labels.",
        ],
        warnings=warnings,
    )


def _regression_metrics() -> dict[str, Any]:
    configured_default = get_config_text(
        "metrics.regression_default",
        "rmse",
    )

    metric_key = configured_default

    if metric_key not in SKLEARN_SCORING_MAP["regression"]:
        metric_key = "rmse"

    scoring_metric = get_safe_scoring_metric(
        "regression",
        metric_key,
        "rmse",
    )

    higher_is_better = metric_key in HIGHER_IS_BETTER_REGRESSION_METRICS
    primary_metric = metric_display_name(metric_key)

    reason = (
        "Regression problems need error-based metrics. RMSE penalizes large errors, "
        "MAE is easy to explain, and R2 shows how much variance the model explains."
    )

    if metric_key in HIGHER_IS_BETTER_REGRESSION_METRICS:
        reason = (
            "R2 is useful for explaining variance captured by the model, but it should "
            "be interpreted alongside absolute error metrics such as MAE and RMSE."
        )

    return build_common_output(
        problem_type="regression",
        imbalance_severity="not_applicable",
        recommended_metrics=[
            "MAE",
            "RMSE",
            "R2 Score",
            "MAPE",
            "Median Absolute Error",
        ],
        primary_metric=primary_metric,
        scoring_metric=scoring_metric,
        configured_default=configured_default,
        secondary_metrics=["MAE", "RMSE", "R2 Score", "Median Absolute Error"],
        probability_metrics=[],
        higher_is_better=higher_is_better,
        selection_metric_key=(
            "r2_score" if metric_key in {"r2", "r2_score"} else metric_key
        ),
        reason=reason,
        notes=[
            "Use MAPE only when target values are positive and not close to zero.",
            "Use MAE when explainability matters more than penalizing large errors.",
            "RMSE is lower-is-better, but sklearn scoring uses negative RMSE for optimization.",
            "Use residual plots before trusting a regression baseline.",
        ],
        warnings=[]
        if configured_default == metric_key
        else [
            f"Unsupported regression_default='{configured_default}' was replaced with 'rmse'.",
        ],
    )


if __name__ == "__main__":
    output = recommend_metrics(
        problem_type="multiclass_classification",
        imbalance_severity="moderate",
    )
    print(output)
