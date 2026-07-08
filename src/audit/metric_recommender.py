from __future__ import annotations

from typing import Any

from src.utils.exceptions import MetricRecommendationError
from src.utils.logger import get_logger


logger = get_logger(__name__)

SUPPORTED_PROBLEM_TYPES = {
    "binary_classification",
    "multiclass_classification",
    "regression",
}


def recommend_metrics(
    problem_type: str,
    imbalance_severity: str | None = None,
) -> dict[str, Any]:
    """
    Recommend evaluation metrics based on detected ML problem type.
    """
    try:
        logger.info("Starting metric recommendation")

        if problem_type is None or str(problem_type).strip() == "":
            raise MetricRecommendationError(
                "Problem type is required for metric recommendation."
            )

        normalized_problem_type = problem_type.lower().strip()

        if normalized_problem_type not in SUPPORTED_PROBLEM_TYPES:
            raise MetricRecommendationError(
                f"Unsupported problem type: {normalized_problem_type}"
            )

        normalized_imbalance = (
            imbalance_severity.lower().strip()
            if imbalance_severity
            else "unknown"
        )

        if normalized_problem_type == "binary_classification":
            return _binary_classification_metrics(normalized_imbalance)

        if normalized_problem_type == "multiclass_classification":
            return _multiclass_classification_metrics(normalized_imbalance)

        return _regression_metrics()

    except MetricRecommendationError:
        raise

    except Exception as error:
        logger.error(f"Metric recommendation failed: {error}")
        raise MetricRecommendationError(
            "Metric recommendation failed.",
            error_detail=str(error),
        ) from error


def _binary_classification_metrics(imbalance_severity: str) -> dict[str, Any]:
    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC",
        "PR-AUC",
        "Confusion Matrix",
    ]

    if imbalance_severity in {"moderate", "high", "severe"}:
        primary_metric = "F1 Score"
        reason = (
            "Binary classification with possible imbalance should not rely only on accuracy. "
            "F1 balances precision and recall, while PR-AUC is useful when the positive class is rare."
        )
    else:
        primary_metric = "F1 Score"
        reason = (
            "Binary classification needs both overall correctness and class-wise performance. "
            "Precision, Recall, F1, ROC-AUC, PR-AUC, and Confusion Matrix provide a balanced view."
        )

    return {
        "problem_type": "binary_classification",
        "recommended_metrics": metrics,
        "primary_metric": primary_metric,
        "secondary_metrics": ["Precision", "Recall", "ROC-AUC", "PR-AUC"],
        "reason": reason,
        "notes": [
            "Use Accuracy only as a supporting metric.",
            "Use PR-AUC especially when the positive class is rare.",
        ],
    }


def _multiclass_classification_metrics(imbalance_severity: str) -> dict[str, Any]:
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
        reason = (
            "Multiclass classification with imbalance should focus on minority classes. "
            "Macro F1 treats every class equally, while Weighted F1 reflects class frequency."
        )
    else:
        primary_metric = "Weighted F1 Score"
        reason = (
            "Multiclass classification needs metrics that evaluate all classes. "
            "Macro metrics treat all classes equally, while weighted metrics account for class frequency."
        )

    return {
        "problem_type": "multiclass_classification",
        "recommended_metrics": metrics,
        "primary_metric": primary_metric,
        "secondary_metrics": ["Accuracy", "Macro F1 Score", "Weighted F1 Score"],
        "reason": reason,
        "notes": [
            "Use Macro F1 to check minority-class performance.",
            "Use Weighted F1 to reflect real class distribution.",
        ],
    }


def _regression_metrics() -> dict[str, Any]:
    metrics = [
        "MAE",
        "RMSE",
        "R2 Score",
        "MAPE",
        "Median Absolute Error",
    ]

    return {
        "problem_type": "regression",
        "recommended_metrics": metrics,
        "primary_metric": "RMSE",
        "secondary_metrics": ["MAE", "R2 Score"],
        "reason": (
            "Regression problems need error-based metrics. RMSE penalizes large errors, "
            "MAE is easy to explain, and R2 shows how much variance the model explains."
        ),
        "notes": [
            "Use MAPE only when target values are positive and not close to zero.",
            "Use MAE when explainability matters more than penalizing large errors.",
        ],
    }


if __name__ == "__main__":
    output = recommend_metrics(
        problem_type="multiclass_classification",
        imbalance_severity="moderate",
    )
    print(output)