from typing import Any

import pandas as pd

from src.utils.config import get_config_value
from src.utils.exceptions import AuditCopilotException
from src.utils.logger import get_logger


logger = get_logger(__name__)


def detect_class_imbalance(
    df: pd.DataFrame,
    target_column: str,
    problem_type: str,
) -> dict[str, Any]:
    """
    Detect class imbalance for classification problems.
    """
    try:
        logger.info("Starting class imbalance detection")

        if df is None or df.empty:
            raise AuditCopilotException("Input dataframe is empty.")

        if not target_column:
            raise AuditCopilotException("Target column is required.")

        if target_column not in df.columns:
            raise AuditCopilotException(f"Target column not found: {target_column}")

        if not problem_type:
            raise AuditCopilotException("Problem type is required.")

        normalized_problem_type = problem_type.lower().strip()

        if normalized_problem_type == "regression":
            return {
                "problem_type": normalized_problem_type,
                "target_column": target_column,
                "is_applicable": False,
                "message": (
                    "Class imbalance detection is not applicable for regression problems."
                ),
            }

        if normalized_problem_type not in {
            "binary_classification",
            "multiclass_classification",
        }:
            raise AuditCopilotException(
                f"Unsupported problem type: {normalized_problem_type}"
            )

        target_series = df[target_column].dropna()

        if target_series.empty:
            raise AuditCopilotException("Target column has no valid non-null values.")

        value_counts = target_series.value_counts()
        value_percentages = target_series.value_counts(normalize=True).mul(100).round(2)

        majority_class = value_counts.idxmax()
        minority_class = value_counts.idxmin()

        majority_count = int(value_counts.max())
        minority_count = int(value_counts.min())

        if minority_count == 0:
            imbalance_ratio = float("inf")
        else:
            imbalance_ratio = round(majority_count / minority_count, 2)

        severity = _get_imbalance_severity(imbalance_ratio)

        recommended_metrics = _recommend_metrics_for_imbalance(
            problem_type=normalized_problem_type,
            severity=severity,
        )

        result = {
            "problem_type": normalized_problem_type,
            "target_column": target_column,
            "is_applicable": True,
            "class_counts": {
                str(label): int(count)
                for label, count in value_counts.items()
            },
            "class_percentages": {
                str(label): float(percent)
                for label, percent in value_percentages.items()
            },
            "majority_class": str(majority_class),
            "minority_class": str(minority_class),
            "majority_count": majority_count,
            "minority_count": minority_count,
            "imbalance_ratio": imbalance_ratio,
            "imbalance_severity": severity,
            "recommended_metrics": recommended_metrics,
            "warning": _get_warning(severity),
        }

        logger.info("Class imbalance detection completed successfully")
        return result

    except AuditCopilotException:
        raise

    except Exception as error:
        logger.error(f"Class imbalance detection failed: {error}")
        raise AuditCopilotException(
            "Class imbalance detection failed",
            error_detail=str(error),
        ) from error


def _get_imbalance_severity(imbalance_ratio: float) -> str:
    """
    Decide imbalance severity using config-driven ratio thresholds.
    """
    low_threshold = float(get_config_value("imbalance.low_ratio_threshold", 1.5))
    moderate_threshold = float(get_config_value("imbalance.moderate_ratio_threshold", 3))
    high_threshold = float(get_config_value("imbalance.high_ratio_threshold", 10))

    if imbalance_ratio < low_threshold:
        return "low"

    if imbalance_ratio < moderate_threshold:
        return "moderate"

    if imbalance_ratio < high_threshold:
        return "high"

    return "severe"

def _recommend_metrics_for_imbalance(
    problem_type: str,
    severity: str,
) -> list[str]:
    """
    Recommend evaluation metrics based on imbalance severity.
    """
    if severity == "low":
        if problem_type == "binary_classification":
            return ["Accuracy", "F1 Score", "ROC-AUC"]

        return ["Accuracy", "Macro F1 Score", "Weighted F1 Score"]

    if problem_type == "binary_classification":
        return ["Precision", "Recall", "F1 Score", "PR-AUC", "ROC-AUC"]

    return ["Macro Precision", "Macro Recall", "Macro F1 Score", "Weighted F1 Score"]


def _get_warning(severity: str) -> str:
    """
    Generate human-readable imbalance warning.
    """
    if severity == "low":
        return "No major class imbalance detected."

    if severity == "moderate":
        return "Moderate class imbalance detected. Accuracy alone may be misleading."

    if severity == "high":
        return (
            "High class imbalance detected. Prefer F1, Recall, Precision, "
            "PR-AUC, or Macro F1."
        )

    return (
        "Severe class imbalance detected. Accuracy is likely misleading and "
        "the model may ignore minority classes."
    )


if __name__ == "__main__":
    sample_df = pd.DataFrame(
        {
            "text": ["a", "b", "c", "d", "e", "f", "g", "h"],
            "target": ["yes", "yes", "yes", "yes", "yes", "yes", "no", "no"],
        }
    )

    output = detect_class_imbalance(
        df=sample_df,
        target_column="target",
        problem_type="binary_classification",
    )

    print(output)