from __future__ import annotations

import math
from difflib import get_close_matches
from typing import Any

import pandas as pd

from src.utils.config import get_config_value
from src.utils.exceptions import ClassImbalanceError
from src.utils.logger import get_logger

logger = get_logger(__name__)

CLASSIFICATION_TYPES = {"binary_classification", "multiclass_classification"}
SUPPORTED_PROBLEM_TYPES = CLASSIFICATION_TYPES | {"regression"}


def get_float_config(path: str, default: float) -> float:
    """Read float config values with a safe fallback."""
    try:
        value = float(get_config_value(path, default))
    except (KeyError, TypeError, ValueError, RuntimeError):
        logger.warning(
            "Invalid float config for %s. Falling back to %s.", path, default
        )
        return float(default)

    if not math.isfinite(value):
        logger.warning(
            "Non-finite float config for %s. Falling back to %s.", path, default
        )
        return float(default)

    return value


def clamp_float(value: float, minimum: float, maximum: float) -> float:
    """Clamp a float value into a valid range."""
    return max(minimum, min(maximum, value))


def normalize_problem_type(problem_type: str) -> str:
    """Normalize and validate supported problem type."""
    normalized = str(problem_type).strip().lower()

    if not normalized:
        raise ClassImbalanceError("Problem type is required.")

    if normalized not in SUPPORTED_PROBLEM_TYPES:
        raise ClassImbalanceError(f"Unsupported problem type: {normalized}")

    return normalized


def resolve_target_column(df: pd.DataFrame, target_column: str) -> str:
    """Resolve target column while preserving backward-compatible exact matching."""
    cleaned_target = str(target_column).strip()

    if not cleaned_target:
        raise ClassImbalanceError("Target column is required.")

    if cleaned_target in df.columns:
        return cleaned_target

    case_insensitive_matches = [
        str(column)
        for column in df.columns
        if str(column).lower() == cleaned_target.lower()
    ]
    if case_insensitive_matches:
        return case_insensitive_matches[0]

    close_matches = get_close_matches(
        cleaned_target,
        [str(column) for column in df.columns],
        n=3,
        cutoff=0.7,
    )
    suggestion = (
        f" Did you mean one of these: {close_matches}?" if close_matches else ""
    )

    raise ClassImbalanceError(f"Target column not found: {cleaned_target}.{suggestion}")


def validate_inputs(
    df: pd.DataFrame,
    target_column: str,
    problem_type: str,
) -> str:
    """Validate inputs for class imbalance detection and return resolved target name."""
    if df is None or df.empty:
        raise ClassImbalanceError("Input dataframe is empty.")

    if not isinstance(df, pd.DataFrame):
        raise ClassImbalanceError("Input must be a pandas DataFrame.")

    if df.columns.empty:
        raise ClassImbalanceError("Input dataframe has no columns.")

    duplicate_columns = df.columns[df.columns.duplicated()].tolist()
    if duplicate_columns:
        raise ClassImbalanceError(f"Duplicate column names found: {duplicate_columns}")

    resolved_target_column = resolve_target_column(df, target_column)
    normalize_problem_type(problem_type)

    return resolved_target_column


def safe_percent(count: int | float, total: int | float) -> float:
    """Calculate percentage safely."""
    try:
        count_value = float(count)
        total_value = float(total)
    except (TypeError, ValueError):
        return 0.0

    if (
        total_value == 0
        or not math.isfinite(total_value)
        or not math.isfinite(count_value)
    ):
        return 0.0

    return round((count_value / total_value) * 100, 2)


def get_target_series(df: pd.DataFrame, target_column: str) -> pd.Series:
    """Return target column as a Series with a stable pandas access pattern."""
    target = df.loc[:, target_column]

    if isinstance(target, pd.DataFrame):
        raise ClassImbalanceError(
            f"Target column '{target_column}' resolved to multiple columns.",
        )

    return target


def make_label_text(value: Any) -> str:
    """Convert target labels into stable JSON-safe string keys."""
    if value is None:
        return "None"

    try:
        if pd.isna(value):
            return "NaN"
    except (TypeError, ValueError):
        pass

    return str(value)


def value_counts_safely(series: pd.Series) -> pd.Series:
    """Return target value counts, falling back to string labels if needed."""
    try:
        return series.value_counts(dropna=False)
    except (TypeError, ValueError):
        return series.map(make_label_text).value_counts(dropna=False)


def nunique_safely(series: pd.Series, dropna: bool = True) -> int:
    """Return unique target count safely."""
    try:
        return int(series.nunique(dropna=dropna))
    except (TypeError, ValueError):
        return int(series.map(make_label_text).nunique(dropna=dropna))


def get_imbalance_thresholds() -> dict[str, float]:
    """Read and sanitize class imbalance thresholds from config."""
    low_ratio_threshold = max(
        1.0,
        get_float_config("imbalance.low_ratio_threshold", 1.5),
    )
    moderate_ratio_threshold = max(
        low_ratio_threshold,
        get_float_config("imbalance.moderate_ratio_threshold", 3.0),
    )
    high_ratio_threshold = max(
        moderate_ratio_threshold,
        get_float_config("imbalance.high_ratio_threshold", 10.0),
    )
    rare_class_threshold_percent = clamp_float(
        get_float_config("imbalance.rare_class_threshold_percent", 5.0),
        minimum=0.0,
        maximum=100.0,
    )

    return {
        "low_ratio_threshold": low_ratio_threshold,
        "moderate_ratio_threshold": moderate_ratio_threshold,
        "high_ratio_threshold": high_ratio_threshold,
        "rare_class_threshold_percent": rare_class_threshold_percent,
    }


def _clean_percentages(class_percentages: pd.Series) -> pd.Series:
    """Convert class percentages into valid non-negative percentages."""
    numeric_percentages = pd.to_numeric(class_percentages, errors="coerce")
    numeric_percentages = numeric_percentages.dropna()
    numeric_percentages = numeric_percentages[numeric_percentages >= 0]
    return numeric_percentages


def calculate_entropy(class_percentages: pd.Series) -> float:
    """
    Calculate normalized Shannon entropy in [0, 1].

    Higher values indicate a more even class distribution.
    """
    percentages = _clean_percentages(class_percentages)
    probabilities = percentages.div(100)
    probabilities = probabilities[probabilities > 0]

    if probabilities.empty or len(probabilities) <= 1:
        return 0.0

    entropy = -sum(
        float(probability) * math.log(float(probability), 2)
        for probability in probabilities
    )
    max_entropy = math.log(len(probabilities), 2)

    if max_entropy == 0:
        return 0.0

    return round(entropy / max_entropy, 4)


def calculate_gini_impurity(class_percentages: pd.Series) -> float:
    """Calculate Gini impurity; higher values mean more diverse distribution."""
    percentages = _clean_percentages(class_percentages)
    probabilities = percentages.div(100)

    if probabilities.empty:
        return 0.0

    gini = 1 - sum(float(probability) ** 2 for probability in probabilities)
    return round(max(0.0, float(gini)), 4)


def calculate_effective_class_count(class_percentages: pd.Series) -> float:
    """
    Calculate effective number of classes using inverse Simpson index.

    If class distribution is perfectly balanced, this approaches num_classes.
    """
    percentages = _clean_percentages(class_percentages)
    probabilities = percentages.div(100)
    denominator = sum(float(probability) ** 2 for probability in probabilities)

    if denominator == 0:
        return 0.0

    return round(1 / denominator, 4)


def get_imbalance_severity(
    imbalance_ratio: float,
    rare_classes: dict[str, float],
) -> str:
    """Classify class imbalance severity."""
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
    """Return classes below the configured rare-class percentage threshold."""
    threshold = (
        clamp_float(float(rare_class_threshold_percent), 0.0, 100.0)
        if rare_class_threshold_percent is not None
        else get_imbalance_thresholds()["rare_class_threshold_percent"]
    )

    rare = _clean_percentages(class_percentages)
    rare = rare[rare < threshold]

    return {str(label): float(round(percent, 2)) for label, percent in rare.items()}


def recommend_metrics_for_imbalance(
    problem_type: str,
    severity: str,
) -> list[str]:
    """Recommend metrics based on problem type and imbalance severity."""
    normalized_problem_type = normalize_problem_type(problem_type)

    if normalized_problem_type == "binary_classification":
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

    if normalized_problem_type == "multiclass_classification":
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

    return []


def recommend_actions(
    severity: str,
    rare_classes: dict[str, float],
    min_class_count: int,
    num_classes: int,
) -> list[str]:
    """Recommend practical next actions for class imbalance."""
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
        actions.append(
            "Rare classes detected; verify whether they are valid labels or data errors.",
        )

    if min_class_count < 2:
        actions.append(
            "At least one class has fewer than 2 samples. Stratified splitting may fail.",
        )
    elif min_class_count < 5:
        actions.append(
            "At least one class has fewer than 5 samples. Cross-validation and "
            "model evaluation may be unstable.",
        )

    if num_classes > 20:
        actions.append(
            "Target has many classes. Review whether labels are meaningful and not "
            "identifiers.",
        )

    return actions


def get_warning(
    severity: str,
    rare_classes: dict[str, float] | None = None,
    min_class_count: int | None = None,
) -> str:
    """Generate a human-readable class imbalance warning."""
    rare_classes = rare_classes or {}

    if min_class_count is not None and min_class_count < 2:
        return (
            "At least one class has fewer than 2 samples. Stratified splitting and "
            "reliable evaluation may fail."
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


def _severity_to_human_review(severity: str) -> bool:
    """Return whether a severity should trigger human review."""
    return severity in {"critical", "severe", "high", "medium", "moderate"}


def build_findings(
    severity: str,
    imbalance_ratio: float,
    rare_classes: dict[str, float],
    min_class_count: int,
    majority_class: str,
    minority_class: str,
) -> list[dict[str, Any]]:
    """Build standardized findings for imbalance results."""
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
                "requires_human_review": _severity_to_human_review(severity),
            },
        )

    if rare_classes:
        findings.append(
            {
                "severity": "medium",
                "category": "rare_classes",
                "message": "Rare classes detected below configured threshold.",
                "evidence": rare_classes,
                "requires_human_review": True,
            },
        )

    if min_class_count < 2:
        findings.append(
            {
                "severity": "critical",
                "category": "insufficient_class_samples",
                "message": "At least one class has fewer than 2 samples.",
                "evidence": {"min_class_count": min_class_count},
                "requires_human_review": True,
            },
        )
    elif min_class_count < 5:
        findings.append(
            {
                "severity": "high",
                "category": "very_small_minority_class",
                "message": "At least one class has fewer than 5 samples.",
                "evidence": {"min_class_count": min_class_count},
                "requires_human_review": True,
            },
        )

    return findings


def _validate_target_series(target_series: pd.Series) -> None:
    """Validate non-null target values for imbalance detection."""
    if target_series.empty:
        raise ClassImbalanceError("Target column has no valid non-null values.")

    if nunique_safely(target_series, dropna=True) < 2:
        raise ClassImbalanceError(
            "Target column must contain at least 2 unique classes.",
        )


def _build_regression_response(
    normalized_problem_type: str,
    target_column: str,
) -> dict[str, Any]:
    """Return a standard not-applicable response for regression problems."""
    return {
        "problem_type": normalized_problem_type,
        "target_column": target_column,
        "is_applicable": False,
        "requires_human_review": False,
        "findings": [],
        "recommended_metrics": [],
        "recommended_actions": [],
        "message": "Class imbalance detection is not applicable for regression problems.",
    }


def _get_tied_classes(value_counts: pd.Series) -> dict[str, list[str]]:
    """Return tied majority/minority classes for transparent reporting."""
    max_count = value_counts.max()
    min_count = value_counts.min()

    majority_classes = [
        make_label_text(label)
        for label, count in value_counts.items()
        if int(count) == int(max_count)
    ]
    minority_classes = [
        make_label_text(label)
        for label, count in value_counts.items()
        if int(count) == int(min_count)
    ]

    return {
        "majority_classes": majority_classes,
        "minority_classes": minority_classes,
    }


def _build_split_viability(
    min_class_count: int,
    num_classes: int,
    total_valid_rows: int,
) -> dict[str, Any]:
    """Summarize whether stratified splitting and CV are likely viable."""
    return {
        "can_stratify_train_test_split": min_class_count >= 2,
        "can_use_3_fold_stratified_cv": min_class_count >= 3,
        "can_use_5_fold_stratified_cv": min_class_count >= 5,
        "min_class_count": min_class_count,
        "num_classes": num_classes,
        "valid_target_rows": total_valid_rows,
    }


def _build_distribution_table(
    value_counts: pd.Series,
    value_percentages: pd.Series,
) -> list[dict[str, Any]]:
    """Build list-based class distribution for UI tables."""
    rows: list[dict[str, Any]] = []

    for label, count in value_counts.items():
        percent = value_percentages.get(label, 0.0)
        try:
            percent_value = float(percent)
        except (TypeError, ValueError):
            percent_value = 0.0

        rows.append(
            {
                "class": make_label_text(label),
                "count": int(count),
                "percent": round(percent_value, 2),
            },
        )

    return rows


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

        resolved_target_column = validate_inputs(df, target_column, problem_type)
        normalized_problem_type = normalize_problem_type(problem_type)

        if normalized_problem_type == "regression":
            return _build_regression_response(
                normalized_problem_type, resolved_target_column
            )

        raw_target_series = get_target_series(df, resolved_target_column)
        target_series = raw_target_series.dropna()
        _validate_target_series(target_series)

        value_counts = value_counts_safely(target_series)
        value_percentages = value_counts.div(len(target_series)).mul(100).round(2)

        majority_class = value_counts.idxmax()
        minority_class = value_counts.idxmin()

        majority_count = int(value_counts.max())
        minority_count = int(value_counts.min())
        total_valid_rows = int(len(target_series))
        missing_target_rows = int(raw_target_series.isna().sum())
        num_classes = int(value_counts.shape[0])

        if minority_count <= 0:
            raise ClassImbalanceError("Minority class count cannot be zero.")

        imbalance_ratio = round(majority_count / minority_count, 2)
        thresholds = get_imbalance_thresholds()
        rare_class_threshold_percent = thresholds["rare_class_threshold_percent"]

        rare_classes = get_rare_classes(
            value_percentages,
            rare_class_threshold_percent=rare_class_threshold_percent,
        )
        severity = get_imbalance_severity(imbalance_ratio, rare_classes)

        entropy_score = calculate_entropy(value_percentages)
        gini_impurity = calculate_gini_impurity(value_percentages)
        effective_class_count = calculate_effective_class_count(value_percentages)
        tied_classes = _get_tied_classes(value_counts)
        split_viability = _build_split_viability(
            min_class_count=minority_count,
            num_classes=num_classes,
            total_valid_rows=total_valid_rows,
        )

        findings = build_findings(
            severity=severity,
            imbalance_ratio=float(imbalance_ratio),
            rare_classes=rare_classes,
            min_class_count=minority_count,
            majority_class=make_label_text(majority_class),
            minority_class=make_label_text(minority_class),
        )

        result: dict[str, Any] = {
            "problem_type": normalized_problem_type,
            "target_column": resolved_target_column,
            "is_applicable": True,
            "total_rows": int(len(df)),
            "valid_target_rows": total_valid_rows,
            "missing_target_rows": missing_target_rows,
            "missing_target_percent": safe_percent(missing_target_rows, len(df)),
            "num_classes": num_classes,
            "class_counts": {
                make_label_text(label): int(count)
                for label, count in value_counts.items()
            },
            "class_percentages": {
                make_label_text(label): float(percent)
                for label, percent in value_percentages.items()
            },
            "class_distribution": _build_distribution_table(
                value_counts=value_counts,
                value_percentages=value_percentages,
            ),
            "majority_class": make_label_text(majority_class),
            "minority_class": make_label_text(minority_class),
            "majority_classes": tied_classes["majority_classes"],
            "minority_classes": tied_classes["minority_classes"],
            "majority_count": majority_count,
            "minority_count": minority_count,
            "min_class_count": minority_count,
            "max_class_count": majority_count,
            "imbalance_ratio": float(imbalance_ratio),
            "imbalance_severity": severity,
            "rare_classes": rare_classes,
            "rare_class_threshold_percent": rare_class_threshold_percent,
            "thresholds": thresholds,
            "distribution_metrics": {
                "normalized_entropy": entropy_score,
                "gini_impurity": gini_impurity,
                "effective_class_count": effective_class_count,
            },
            "split_viability": split_viability,
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
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        ZeroDivisionError,
    ) as error:
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
        },
    )

    output = detect_class_imbalance(
        df=sample_df,
        target_column="target",
        problem_type="binary_classification",
    )

    print(output)
