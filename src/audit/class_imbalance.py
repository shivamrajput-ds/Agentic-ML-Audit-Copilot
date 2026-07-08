from __future__ import annotations

import math
from typing import Any

import pandas as pd

from src.utils.config import get_config_value
from src.utils.exceptions import ClassImbalanceError
from src.utils.logger import get_logger


logger = get_logger(__name__)

CLASSIFICATION_TYPES = {"binary_classification", "multiclass_classification"}


def validate_inputs(
    df: pd.DataFrame,
    target_column: str,
    problem_type: str,
) -> None:
    if df is None or df.empty:
        raise ClassImbalanceError("Input dataframe is empty.")

    if target_column is None or str(target_column).strip() == "":
        raise ClassImbalanceError("Target column is required.")

    if target_column not in df.columns:
        raise ClassImbalanceError(f"Target column not found: {target_column}")

    if problem_type is None or str(problem_type).strip() == "":
        raise ClassImbalanceError("Problem type is required.")


def safe_percent(count: int | float, total: int | float) -> float:
    if total == 0:
        return 0.0
    return round((float(count) / float(total)) * 100, 2)


def get_imbalance_thresholds() -> dict[str, float]:
    return {
        "low_ratio_threshold": float(get_config_value("imbalance.low_ratio_threshold", 1.5)),
        "moderate_ratio_threshold": float(get_config_value("imbalance.moderate_ratio_threshold", 3)),
        "high_ratio_threshold": float(get_config_value("imbalance.high_ratio_threshold", 10)),
        "rare_class_threshold_percent": float(
            get_config_value("imbalance.rare_class_threshold_percent", 5)
        ),
    }


def calculate_entropy(class_percentages: pd.Series) -> float:
    """
    Shannon entropy normalized to [0, 1].
    Higher means more even class distribution.
    """
    probabilities = class_percentages.div(100)

    if probabilities.empty or len(probabilities) <= 1:
        return 0.0

    entropy = -sum(
        float(p) * math.log(float(p), 2)
        for p in probabilities
        if float(p) > 0
    )
    max_entropy = math.log(len(probabilities), 2)

    if max_entropy == 0:
        return 0.0

    return round(entropy / max_entropy, 4)


def calculate_gini_impurity(class_percentages: pd.Series) -> float:
    """
    Gini impurity: higher means more diverse target distribution.
    """
    probabilities = class_percentages.div(100)
    gini = 1 - sum(float(p) ** 2 for p in probabilities)
    return round(float(gini), 4)


def calculate_effective_class_count(class_percentages: pd.Series) -> float:
    """
    Effective number of classes using inverse Simpson index.
    If class distribution is perfectly balanced, this approaches num_classes.
    """
    probabilities = class_percentages.div(100)
    denominator = sum(float(p) ** 2 for p in probabilities)

    if denominator == 0:
        return 0.0

    return round(1 / denominator, 4)


def get_imbalance_severity(
    imbalance_ratio: float,
    rare_classes: dict[str, float],
) -> str:
    thresholds = get_imbalance_thresholds()

    if imbalance_ratio < thresholds["low_ratio_threshold"] and not rare_classes:
        return "low"

    if imbalance_ratio < thresholds["moderate_ratio_threshold"]:
        return "moderate" if rare_classes else "low"

    if imbalance_ratio < thresholds["high_ratio_threshold"]:
        return "high"

    return "severe"


def get_rare_classes(
    class_percentages: pd.Series,
    rare_class_threshold_percent: float | None = None,
) -> dict[str, float]:
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
    if problem_type == "binary_classification":
        if severity == "low":
            return ["Accuracy", "F1 Score", "ROC-AUC", "Confusion Matrix"]

        return [
            "Precision",
            "Recall",
            "F1 Score",
            "PR-AUC",
            "ROC-AUC",
            "Balanced Accuracy",
            "Confusion Matrix",
        ]

    if severity == "low":
        return [
            "Accuracy",
            "Macro F1 Score",
            "Weighted F1 Score",
            "Confusion Matrix",
        ]

    return [
        "Macro Precision",
        "Macro Recall",
        "Macro F1 Score",
        "Weighted F1 Score",
        "Balanced Accuracy",
        "Per-class Recall",
        "Confusion Matrix",
    ]


def recommend_actions(
    severity: str,
    rare_classes: dict[str, float],
    min_class_count: int,
    num_classes: int,
) -> list[str]:
    actions: list[str] = []

    if severity == "low" and not rare_classes:
        return ["No major imbalance handling required."]

    actions.append("Avoid relying only on accuracy.")
    actions.append("Use stratified train-test split for classification.")
    actions.append("Inspect confusion matrix and per-class recall.")

    if severity in {"moderate", "high", "severe"}:
        actions.append("Compare macro and weighted metrics.")
        actions.append("Consider class_weight='balanced' for supported models.")

    if severity in {"high", "severe"}:
        actions.append("Inspect minority-class recall carefully.")
        actions.append("Consider resampling only after creating a proper train split.")

    if rare_classes:
        actions.append("Rare classes detected; verify whether they are valid labels or data errors.")

    if min_class_count < 5:
        actions.append(
            "At least one class has fewer than 5 samples. Cross-validation and model evaluation may be unstable."
        )

    if num_classes > 20:
        actions.append(
            "Target has many classes. Review whether labels are meaningful and not identifiers."
        )

    return actions


def get_warning(
    severity: str,
    rare_classes: dict[str, float] | None = None,
    min_class_count: int | None = None,
) -> str:
    rare_classes = rare_classes or {}

    if min_class_count is not None and min_class_count < 2:
        return (
            "At least one class has fewer than 2 samples. Stratified splitting and reliable evaluation may fail."
        )

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


def build_findings(
    severity: str,
    imbalance_ratio: float,
    rare_classes: dict[str, float],
    min_class_count: int,
    majority_class: str,
    minority_class: str,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []

    if severity in {"moderate", "high", "severe"}:
        findings.append(
            {
                "severity": severity,
                "category": "class_imbalance",
                "message": f"Class imbalance ratio is {imbalance_ratio}.",
                "evidence": {
                    "majority_class": majority_class,
                    "minority_class": minority_class,
                    "imbalance_ratio": imbalance_ratio,
                },
                "requires_human_review": True,
            }
        )

    if rare_classes:
        findings.append(
            {
                "severity": "medium",
                "category": "rare_classes",
                "message": "Rare classes detected below configured threshold.",
                "evidence": rare_classes,
                "requires_human_review": True,
            }
        )

    if min_class_count < 2:
        findings.append(
            {
                "severity": "critical",
                "category": "insufficient_class_samples",
                "message": "At least one class has fewer than 2 samples.",
                "evidence": {"min_class_count": min_class_count},
                "requires_human_review": True,
            }
        )
    elif min_class_count < 5:
        findings.append(
            {
                "severity": "high",
                "category": "very_small_minority_class",
                "message": "At least one class has fewer than 5 samples.",
                "evidence": {"min_class_count": min_class_count},
                "requires_human_review": True,
            }
        )

    return findings


def detect_class_imbalance(
    df: pd.DataFrame,
    target_column: str,
    problem_type: str,
) -> dict[str, Any]:
    """
    Detect class imbalance for classification problems.

    Regression returns is_applicable=False.
    """
    try:
        logger.info("Starting class imbalance detection")

        validate_inputs(df, target_column, problem_type)
        normalized_problem_type = problem_type.lower().strip()

        if normalized_problem_type == "regression":
            return {
                "problem_type": normalized_problem_type,
                "target_column": target_column,
                "is_applicable": False,
                "message": "Class imbalance detection is not applicable for regression problems.",
            }

        if normalized_problem_type not in CLASSIFICATION_TYPES:
            raise ClassImbalanceError(
                f"Unsupported problem type: {normalized_problem_type}"
            )

        target_series = df[target_column].dropna()

        if target_series.empty:
            raise ClassImbalanceError("Target column has no valid non-null values.")

        if target_series.nunique(dropna=True) < 2:
            raise ClassImbalanceError(
                "Target column must contain at least 2 unique classes."
            )

        value_counts = target_series.value_counts()
        value_percentages = value_counts.div(len(target_series)).mul(100).round(2)

        majority_class = value_counts.idxmax()
        minority_class = value_counts.idxmin()

        majority_count = int(value_counts.max())
        minority_count = int(value_counts.min())
        total_valid_rows = int(len(target_series))
        num_classes = int(value_counts.shape[0])

        imbalance_ratio = round(majority_count / minority_count, 2)

        rare_classes = get_rare_classes(value_percentages)
        severity = get_imbalance_severity(imbalance_ratio, rare_classes)

        entropy_score = calculate_entropy(value_percentages)
        gini_impurity = calculate_gini_impurity(value_percentages)
        effective_class_count = calculate_effective_class_count(value_percentages)

        findings = build_findings(
            severity=severity,
            imbalance_ratio=float(imbalance_ratio),
            rare_classes=rare_classes,
            min_class_count=minority_count,
            majority_class=str(majority_class),
            minority_class=str(minority_class),
        )

        result: dict[str, Any] = {
            "problem_type": normalized_problem_type,
            "target_column": target_column,
            "is_applicable": True,
            "total_rows": int(len(df)),
            "valid_target_rows": total_valid_rows,
            "missing_target_rows": int(df[target_column].isna().sum()),
            "missing_target_percent": float(
                round(df[target_column].isna().mean() * 100, 2)
            ),
            "num_classes": num_classes,
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
            "min_class_count": minority_count,
            "max_class_count": majority_count,
            "imbalance_ratio": float(imbalance_ratio),
            "imbalance_severity": severity,
            "rare_classes": rare_classes,
            "rare_class_threshold_percent": float(
                get_config_value("imbalance.rare_class_threshold_percent", 5)
            ),
            "distribution_metrics": {
                "normalized_entropy": entropy_score,
                "gini_impurity": gini_impurity,
                "effective_class_count": effective_class_count,
            },
            "findings": findings,
            "requires_human_review": bool(findings),
            "recommended_metrics": recommend_metrics_for_imbalance(
                problem_type=normalized_problem_type,
                severity=severity,
            ),
            "recommended_actions": recommend_actions(
                severity=severity,
                rare_classes=rare_classes,
                min_class_count=minority_count,
                num_classes=num_classes,
            ),
            "warning": get_warning(
                severity=severity,
                rare_classes=rare_classes,
                min_class_count=minority_count,
            ),
            "message": "Class imbalance detection completed successfully.",
        }

        logger.info("Class imbalance detection completed successfully")
        return result

    except ClassImbalanceError:
        raise

    except Exception as error:
        logger.exception("Class imbalance detection failed.")
        raise ClassImbalanceError(
            "Class imbalance detection failed.",
            error_detail=str(error),
        ) from error


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
