from __future__ import annotations

from typing import Any

import pandas as pd

from src.utils.config import get_config_value
from src.utils.exceptions import AuditCopilotException
from src.utils.logger import get_logger


logger = get_logger(__name__)

CLASSIFICATION_TYPES = {"binary_classification", "multiclass_classification"}


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

        _validate_inputs(df, target_column, problem_type)

        normalized_problem_type = problem_type.lower().strip()

        if normalized_problem_type == "regression":
            return {
                "problem_type": normalized_problem_type,
                "target_column": target_column,
                "is_applicable": False,
                "message": "Class imbalance detection is not applicable for regression problems.",
            }

        if normalized_problem_type not in CLASSIFICATION_TYPES:
            raise AuditCopilotException(
                f"Unsupported problem type: {normalized_problem_type}"
            )

        target_series = df[target_column].dropna()

        if target_series.empty:
            raise AuditCopilotException("Target column has no valid non-null values.")

        if target_series.nunique(dropna=True) < 2:
            raise AuditCopilotException(
                "Target column must contain at least 2 unique classes."
            )

        value_counts = target_series.value_counts()
        value_percentages = value_counts.div(len(target_series)).mul(100).round(2)

        majority_class = value_counts.idxmax()
        minority_class = value_counts.idxmin()

        majority_count = int(value_counts.max())
        minority_count = int(value_counts.min())
        total_valid_rows = int(len(target_series))

        imbalance_ratio = round(majority_count / minority_count, 2)

        severity = get_imbalance_severity(imbalance_ratio)
        rare_classes = get_rare_classes(value_percentages)

        result: dict[str, Any] = {
            "problem_type": normalized_problem_type,
            "target_column": target_column,
            "is_applicable": True,
            "total_rows": int(len(df)),
            "valid_target_rows": total_valid_rows,
            "missing_target_rows": int(df[target_column].isna().sum()),
            "missing_target_percent": float(round(df[target_column].isna().mean() * 100, 2)),
            "num_classes": int(value_counts.shape[0]),
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
            "imbalance_ratio": float(imbalance_ratio),
            "imbalance_severity": severity,
            "rare_classes": rare_classes,
            "recommended_metrics": recommend_metrics_for_imbalance(
                problem_type=normalized_problem_type,
                severity=severity,
            ),
            "recommended_actions": recommend_actions(severity, rare_classes),
            "warning": get_warning(severity, rare_classes),
        }

        logger.info("Class imbalance detection completed successfully")
        return result

    except AuditCopilotException:
        raise

    except Exception as error:
        logger.error(f"Class imbalance detection failed: {error}")
        raise AuditCopilotException(
            "Class imbalance detection failed.",
            error_detail=str(error),
        ) from error


def _validate_inputs(
    df: pd.DataFrame,
    target_column: str,
    problem_type: str,
) -> None:
    if df is None or df.empty:
        raise AuditCopilotException("Input dataframe is empty.")

    if target_column is None or str(target_column).strip() == "":
        raise AuditCopilotException("Target column is required.")

    if target_column not in df.columns:
        raise AuditCopilotException(f"Target column not found: {target_column}")

    if problem_type is None or str(problem_type).strip() == "":
        raise AuditCopilotException("Problem type is required.")


def get_imbalance_severity(imbalance_ratio: float) -> str:
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


def get_rare_classes(
    class_percentages: pd.Series,
    rare_class_threshold_percent: float | None = None,
) -> dict[str, float]:
    """
    Detect classes whose percentage is very small.
    """
    if rare_class_threshold_percent is None:
        rare_class_threshold_percent = float(
            get_config_value("imbalance.rare_class_threshold_percent", 5)
        )

    rare = class_percentages[class_percentages < rare_class_threshold_percent]

    return {
        str(label): float(percent)
        for label, percent in rare.items()
    }


def recommend_metrics_for_imbalance(
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

    return [
        "Macro Precision",
        "Macro Recall",
        "Macro F1 Score",
        "Weighted F1 Score",
        "Balanced Accuracy",
    ]


def recommend_actions(
    severity: str,
    rare_classes: dict[str, float],
) -> list[str]:
    """
    Recommend practical actions for handling imbalance.
    """
    actions: list[str] = []

    if severity == "low" and not rare_classes:
        return ["No major imbalance handling required."]

    actions.append("Avoid relying only on accuracy.")
    actions.append("Use stratified train-test split for classification.")

    if severity in {"moderate", "high", "severe"}:
        actions.append("Compare macro and weighted metrics.")
        actions.append("Consider class_weight='balanced' for supported models.")

    if severity in {"high", "severe"}:
        actions.append("Inspect minority-class recall carefully.")
        actions.append("Consider resampling only after creating a proper train split.")

    if rare_classes:
        actions.append("Rare classes detected; verify whether they are valid labels or data errors.")

    return actions


def get_warning(
    severity: str,
    rare_classes: dict[str, float] | None = None,
) -> str:
    """
    Generate human-readable imbalance warning.
    """
    rare_classes = rare_classes or {}

    if severity == "low" and not rare_classes:
        return "No major class imbalance detected."

    if severity == "low" and rare_classes:
        return "Overall imbalance is low, but some rare classes are present."

    if severity == "moderate":
        return "Moderate class imbalance detected. Accuracy alone may be misleading."

    if severity == "high":
        return (
            "High class imbalance detected. Prefer F1, Recall, Precision, "
            "PR-AUC, Balanced Accuracy, or Macro F1."
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