from typing import Any

from src.utils.exceptions import MetricRecommendationError
from src.utils.logger import get_logger


logger = get_logger(__name__)


SUPPORTED_PROBLEM_TYPES = {
    "binary_classification",
    "multiclass_classification",
    "regression",
}


def recommend_metrics(problem_type: str) -> dict[str, Any]:
    """
    Recommend evaluation metrics based on detected ML problem type.
    """
    try:
        logger.info("Starting metric recommendation")

        if not problem_type:
            raise MetricRecommendationError(
                "Problem type is required for metric recommendation."
            )

        normalized_problem_type = problem_type.lower().strip()

        if normalized_problem_type not in SUPPORTED_PROBLEM_TYPES:
            raise MetricRecommendationError(
                f"Unsupported problem type: {normalized_problem_type}"
            )

        if normalized_problem_type == "binary_classification":
            metrics = [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "ROC-AUC",
                "PR-AUC",
            ]
            primary_metric = "F1 Score"
            reason = (
                "Binary classification needs both overall correctness and "
                "class-wise performance metrics. Precision, Recall, F1, "
                "ROC-AUC, and PR-AUC are useful when classes may be imbalanced."
            )

        elif normalized_problem_type == "multiclass_classification":
            metrics = [
                "Accuracy",
                "Macro Precision",
                "Macro Recall",
                "Macro F1 Score",
                "Weighted F1 Score",
            ]
            primary_metric = "Weighted F1 Score"
            reason = (
                "Multiclass classification needs metrics that evaluate all "
                "classes. Macro metrics treat all classes equally, while "
                "weighted metrics account for class frequency."
            )

        else:
            metrics = [
                "MAE",
                "RMSE",
                "R2 Score",
                "MAPE",
            ]
            primary_metric = "RMSE"
            reason = (
                "Regression problems need error-based metrics. MAE is easy "
                "to explain, RMSE penalizes large errors, R2 shows explained "
                "variance, and MAPE shows percentage error when target values "
                "are suitable."
            )

        result = {
            "problem_type": normalized_problem_type,
            "recommended_metrics": metrics,
            "primary_metric": primary_metric,
            "reason": reason,
        }

        logger.info("Metric recommendation completed successfully")
        return result

    except MetricRecommendationError:
        raise

    except Exception as error:
        logger.error(f"Metric recommendation failed: {error}")
        raise MetricRecommendationError(
            "Metric recommendation failed",
            error_detail=str(error),
        ) from error


if __name__ == "__main__":
    sample_problem_type = "multiclass_classification"
    output = recommend_metrics(sample_problem_type)
    print(output)