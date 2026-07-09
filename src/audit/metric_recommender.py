from __future__ import annotations

from typing import Any

from src.utils.config import get_config_value
from src.utils.exceptions import MetricRecommendationError
from src.utils.logger import get_logger


logger = get_logger(__name__)

SUPPORTED_PROBLEM_TYPES = {
    "binary_classification",
    "multiclass_classification",
    "regression",
}

IMBALANCE_LEVELS = {
    "none",
    "low",
    "moderate",
    "high",
    "severe",
    "unknown",
}


SKLEARN_SCORING_MAP = {
    "binary_classification": {
        "f1": "f1",
        "f1_score": "f1",
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "balanced_accuracy": "balanced_accuracy",
    },
    "multiclass_classification": {
        "f1_macro": "f1_macro",
        "f1_weighted": "f1_weighted",
        "balanced_accuracy": "balanced_accuracy",
        "accuracy": "accuracy",
    },
    "regression": {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
        "r2_score": "r2",
    },
}


def normalize_text(value: str | None, default: str = "unknown") -> str:
    """
    Normalize small config/user text values.
    """
    if value is None or str(value).strip() == "":
        return default

    return str(value).lower().strip()


def get_config_text(path: str, default: str) -> str:
    """
    Read config value as clean lower-case text.
    """
    value = get_config_value(path, default)
    return normalize_text(str(value), default=default)


def get_safe_scoring_metric(problem_type: str, configured_metric: str, fallback: str) -> str:
    """
    Return a sklearn-compatible scoring metric.
    """
    metric_map = SKLEARN_SCORING_MAP.get(problem_type, {})
    normalized_metric = normalize_text(configured_metric, fallback)
    return metric_map.get(normalized_metric, fallback)


def recommend_metrics(
    problem_type: str,
    imbalance_severity: str | None = None,
) -> dict[str, Any]:
    """
    Recommend evaluation metrics based on the detected ML problem type.
    """
    try:
        logger.info("Starting metric recommendation")

        normalized_problem_type = normalize_text(problem_type)

        if normalized_problem_type not in SUPPORTED_PROBLEM_TYPES:
            raise MetricRecommendationError(
                f"Unsupported problem type: {normalized_problem_type}"
            )

        normalized_imbalance = normalize_text(imbalance_severity)

        if normalized_imbalance not in IMBALANCE_LEVELS:
            normalized_imbalance = "unknown"

        if normalized_problem_type == "binary_classification":
            result = _binary_classification_metrics(normalized_imbalance)

        elif normalized_problem_type == "multiclass_classification":
            result = _multiclass_classification_metrics(normalized_imbalance)

        else:
            result = _regression_metrics()

        logger.info(
            "Metric recommendation completed. Problem type=%s Primary metric=%s",
            result["problem_type"],
            result["primary_metric"],
        )

        return result

    except MetricRecommendationError:
        raise

    except Exception as error:
        logger.exception("Metric recommendation failed.")
        raise MetricRecommendationError(
            "Metric recommendation failed.",
            error_detail=str(error),
        ) from error


def _binary_classification_metrics(imbalance_severity: str) -> dict[str, Any]:
    configured_default = get_config_text(
        "metrics.classification_default",
        "f1_weighted",
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

    if imbalance_severity in {"moderate", "high", "severe"}:
        primary_metric = "F1 Score"
        scoring_metric = get_safe_scoring_metric(
            "binary_classification",
            configured_default,
            "f1",
        )
        reason = (
            "Binary classification with imbalance should not rely on accuracy. "
            "F1 balances precision and recall. PR-AUC is also important when "
            "the positive class is rare."
        )
    else:
        primary_metric = "F1 Score"
        scoring_metric = "f1"
        reason = (
            "Binary classification needs both overall correctness and class-wise "
            "performance. Precision, Recall, F1, ROC-AUC, PR-AUC, Balanced Accuracy, "
            "and Confusion Matrix provide a balanced view."
        )

    return {
        "problem_type": "binary_classification",
        "imbalance_severity": imbalance_severity,
        "recommended_metrics": metrics,
        "primary_metric": primary_metric,
        "scoring_metric": scoring_metric,
        "configured_default": configured_default,
        "secondary_metrics": [
            "Precision",
            "Recall",
            "ROC-AUC",
            "PR-AUC",
            "Balanced Accuracy",
        ],
        "higher_is_better": True,
        "reason": reason,
        "notes": [
            "Use Accuracy only as a supporting metric.",
            "Use PR-AUC when the positive class is rare.",
            "Use Confusion Matrix to inspect false positives and false negatives.",
        ],
    }


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

    if imbalance_severity in {"moderate", "high", "severe"}:
        primary_metric = "Macro F1 Score"
        scoring_metric = "f1_macro"
        reason = (
            "Multiclass classification with imbalance should focus on minority classes. "
            "Macro F1 treats every class equally, while Weighted F1 reflects class frequency."
        )
    else:
        primary_metric = "Weighted F1 Score"
        scoring_metric = get_safe_scoring_metric(
            "multiclass_classification",
            configured_default,
            "f1_weighted",
        )
        reason = (
            "Multiclass classification needs metrics that evaluate all classes. "
            "Macro metrics treat all classes equally, while weighted metrics account "
            "for class frequency."
        )

    return {
        "problem_type": "multiclass_classification",
        "imbalance_severity": imbalance_severity,
        "recommended_metrics": metrics,
        "primary_metric": primary_metric,
        "scoring_metric": scoring_metric,
        "configured_default": configured_default,
        "secondary_metrics": [
            "Accuracy",
            "Macro F1 Score",
            "Weighted F1 Score",
            "Balanced Accuracy",
        ],
        "higher_is_better": True,
        "reason": reason,
        "notes": [
            "Use Macro F1 to check minority-class performance.",
            "Use Weighted F1 to reflect real class distribution.",
            "Use Confusion Matrix to inspect which classes are confused.",
        ],
    }


def _regression_metrics() -> dict[str, Any]:
    configured_default = get_config_text(
        "metrics.regression_default",
        "rmse",
    )

    scoring_metric = get_safe_scoring_metric(
        "regression",
        configured_default,
        "neg_root_mean_squared_error",
    )

    metrics = [
        "MAE",
        "RMSE",
        "R2 Score",
        "MAPE",
        "Median Absolute Error",
    ]

    return {
        "problem_type": "regression",
        "imbalance_severity": "not_applicable",
        "recommended_metrics": metrics,
        "primary_metric": "RMSE",
        "scoring_metric": scoring_metric,
        "configured_default": configured_default,
        "secondary_metrics": ["MAE", "R2 Score", "Median Absolute Error"],
        "higher_is_better": False,
        "reason": (
            "Regression problems need error-based metrics. RMSE penalizes large errors, "
            "MAE is easy to explain, and R2 shows how much variance the model explains."
        ),
        "notes": [
            "Use MAPE only when target values are positive and not close to zero.",
            "Use MAE when explainability matters more than penalizing large errors.",
            "RMSE is lower-is-better, but sklearn scoring uses negative RMSE for optimization.",
        ],
    }


if __name__ == "__main__":
    output = recommend_metrics(
        problem_type="multiclass_classification",
        imbalance_severity="moderate",
    )
    print(output)
