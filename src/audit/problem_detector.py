from __future__ import annotations

from typing import Any

import pandas as pd

from src.utils.config import get_config_value
from src.utils.exceptions import InvalidTargetColumnError, ProblemTypeDetectionError
from src.utils.logger import get_logger


logger = get_logger(__name__)


CLASSIFICATION_TYPES = {"binary_classification", "multiclass_classification"}
SUPPORTED_PROBLEM_TYPES = {"binary_classification", "multiclass_classification", "regression"}


def validate_target(df: pd.DataFrame, target_column: str) -> pd.Series:
    """
    Validate target column and return non-null target values.
    """
    if df is None or df.empty:
        raise InvalidTargetColumnError("Dataset is empty.")

    if target_column is None or str(target_column).strip() == "":
        raise InvalidTargetColumnError("Target column is required.")

    if target_column not in df.columns:
        raise InvalidTargetColumnError(
            f"Target column '{target_column}' not found in dataset."
        )

    target = df[target_column].dropna()

    if target.empty:
        raise InvalidTargetColumnError(
            f"Target column '{target_column}' has only missing values."
        )

    if target.nunique(dropna=True) < 2:
        raise InvalidTargetColumnError(
            f"Target column '{target_column}' has only one unique value "
            "and cannot be used for classification or regression."
        )

    return target


def get_classification_unique_threshold() -> int:
    """
    Read classification threshold from config.
    """
    threshold = int(get_config_value("problem_detection.classification_unique_threshold", 20))

    if threshold < 2:
        raise ProblemTypeDetectionError(
            "problem_detection.classification_unique_threshold must be at least 2."
        )

    return threshold


def detect_problem_type(
    df: pd.DataFrame,
    target_column: str,
    classification_unique_threshold: int | None = None,
) -> dict[str, Any]:
    """
    Detect ML problem type from target column.

    Supported outputs:
    - binary_classification
    - multiclass_classification
    - regression

    Important:
    Numeric targets with low/medium unique values are ambiguous. This module flags
    them for human review instead of overclaiming certainty.
    """
    try:
        if classification_unique_threshold is None:
            classification_unique_threshold = get_classification_unique_threshold()

        if classification_unique_threshold < 2:
            raise ProblemTypeDetectionError(
                "classification_unique_threshold must be at least 2."
            )

        target = validate_target(df, target_column)

        unique_count = int(target.nunique(dropna=True))
        total_count = int(target.shape[0])
        total_rows = int(len(df))
        missing_count = int(df[target_column].isna().sum())
        missing_percent = float(round(df[target_column].isna().mean() * 100, 2))
        target_dtype = str(target.dtype)

        is_numeric = bool(pd.api.types.is_numeric_dtype(target))
        is_bool = bool(pd.api.types.is_bool_dtype(target))
        is_integer_like = is_integer_like_series(target) if is_numeric else False
        unique_percent = float(round((unique_count / total_count) * 100, 2))

        problem_type, confidence, needs_human_review, review_reason = infer_problem_type(
            target=target,
            unique_count=unique_count,
            unique_percent=unique_percent,
            is_numeric=is_numeric,
            is_bool=is_bool,
            is_integer_like=is_integer_like,
            classification_unique_threshold=classification_unique_threshold,
        )

        warnings = generate_warnings(
            total_rows=total_rows,
            total_count=total_count,
            missing_percent=missing_percent,
            unique_count=unique_count,
            unique_percent=unique_percent,
            is_numeric=is_numeric,
            is_integer_like=is_integer_like,
            problem_type=problem_type,
            needs_human_review=needs_human_review,
        )

        result: dict[str, Any] = {
            "target_column": target_column,
            "problem_type": problem_type,
            "target_dtype": target_dtype,
            "is_numeric_target": is_numeric,
            "is_bool_target": is_bool,
            "is_integer_like_target": is_integer_like,
            "unique_values": unique_count,
            "unique_percent": unique_percent,
            "total_values": total_count,
            "total_rows": total_rows,
            "missing_count": missing_count,
            "missing_percent": missing_percent,
            "classification_unique_threshold": int(classification_unique_threshold),
            "confidence": confidence,
            "needs_human_review": needs_human_review,
            "requires_human_review": needs_human_review,
            "human_review_reason": review_reason,
            "sample_values": get_sample_values(target),
            "class_balance_preview": get_class_balance_preview(target, problem_type),
            "warnings": warnings,
            "reason": get_detection_reason(
                problem_type=problem_type,
                unique_count=unique_count,
                unique_percent=unique_percent,
                target_dtype=target_dtype,
                is_numeric=is_numeric,
                is_integer_like=is_integer_like,
                classification_unique_threshold=classification_unique_threshold,
                confidence=confidence,
            ),
            "recommended_action": get_recommended_action(
                problem_type=problem_type,
                needs_human_review=needs_human_review,
            ),
        }

        logger.info(
            "Problem type detected: %s | confidence=%s | human_review=%s",
            problem_type,
            confidence,
            needs_human_review,
        )
        return result

    except (InvalidTargetColumnError, ProblemTypeDetectionError):
        raise

    except Exception as error:
        logger.exception("Problem type detection failed.")
        raise ProblemTypeDetectionError(
            "Failed to detect problem type.",
            error_detail=str(error),
        ) from error


def is_integer_like_series(series: pd.Series) -> bool:
    """
    Check whether numeric target values are integer-like.
    """
    try:
        numeric = pd.to_numeric(series.dropna(), errors="coerce").dropna()

        if numeric.empty:
            return False

        return bool((numeric % 1 == 0).all())

    except Exception:
        return False


def infer_problem_type(
    target: pd.Series,
    unique_count: int,
    unique_percent: float,
    is_numeric: bool,
    is_bool: bool,
    is_integer_like: bool,
    classification_unique_threshold: int,
) -> tuple[str, str, bool, str | None]:
    """
    Infer problem type and confidence.

    Returns:
    (problem_type, confidence, needs_human_review, review_reason)
    """
    if unique_count == 2 or is_bool:
        return (
            "binary_classification",
            "high",
            False,
            None,
        )

    if not is_numeric:
        return (
            "multiclass_classification",
            "high" if unique_count <= 50 else "medium",
            unique_count > 50,
            "Non-numeric target has high cardinality; confirm whether this is a valid classification target."
            if unique_count > 50
            else None,
        )

    if is_numeric:
        if unique_count <= classification_unique_threshold:
            # Numeric class labels like 0/1/2 are common, but numeric discrete
            # targets can also be ordinal/regression. Flag medium ambiguity.
            return (
                "multiclass_classification",
                "medium",
                True,
                (
                    "Numeric target has limited unique values. It may represent "
                    "class labels or an ordinal/regression target. Confirm with domain context."
                ),
            )

        # Numeric targets with 5-20 unique values are especially ambiguous,
        # but this branch only hits if threshold was set lower than unique_count.
        if 5 <= unique_count <= 20 and is_integer_like:
            return (
                "regression",
                "medium",
                True,
                (
                    "Numeric integer-like target has 5-20 unique values. It may be "
                    "ordinal classification or regression. Confirm manually."
                ),
            )

        if unique_percent < 2 and is_integer_like:
            return (
                "regression",
                "medium",
                True,
                (
                    "Numeric target has low unique percentage and integer-like values. "
                    "Confirm whether it is continuous regression or ordinal classes."
                ),
            )

        return (
            "regression",
            "high",
            False,
            None,
        )

    return (
        "multiclass_classification",
        "medium",
        True,
        "Problem type could not be determined with high confidence.",
    )


def get_sample_values(target: pd.Series, max_values: int = 10) -> list[str]:
    """
    Return JSON-safe sample target values.
    """
    values = target.dropna().unique()[:max_values]
    return [str(value) for value in values]


def get_class_balance_preview(
    target: pd.Series,
    problem_type: str,
    max_classes: int = 15,
) -> dict[str, Any] | None:
    """
    Return class distribution preview for classification targets only.
    """
    if problem_type not in CLASSIFICATION_TYPES:
        return None

    counts = target.value_counts(dropna=False).head(max_classes)
    total = len(target)

    return {
        "top_classes": [
            {
                "class": str(label),
                "count": int(count),
                "percent": float(round((int(count) / total) * 100, 2)) if total else 0.0,
            }
            for label, count in counts.items()
        ],
        "shown_classes": int(len(counts)),
        "total_classes": int(target.nunique(dropna=True)),
    }


def generate_warnings(
    total_rows: int,
    total_count: int,
    missing_percent: float,
    unique_count: int,
    unique_percent: float,
    is_numeric: bool,
    is_integer_like: bool,
    problem_type: str,
    needs_human_review: bool,
) -> list[str]:
    """
    Generate warnings for detected target properties.
    """
    warnings: list[str] = []

    if total_rows < 50:
        warnings.append(
            "Dataset has fewer than 50 rows. Problem type detection and model evaluation may be unreliable."
        )

    if missing_percent > 0:
        warnings.append(
            f"Target column has {missing_percent}% missing values. Rows with missing target should be removed before training."
        )

    if needs_human_review:
        warnings.append(
            "Problem type needs human review because target properties are ambiguous."
        )

    if problem_type in CLASSIFICATION_TYPES and unique_count > 50:
        warnings.append(
            "Classification target has high cardinality. Confirm whether labels are meaningful and not identifiers."
        )

    if is_numeric and is_integer_like and 5 <= unique_count <= 20:
        warnings.append(
            "Numeric integer-like target has 5-20 unique values. It may be ordinal classification or regression."
        )

    if unique_percent > 80 and problem_type in CLASSIFICATION_TYPES:
        warnings.append(
            "Classification target has very high unique percentage. This may not be a valid classification problem."
        )

    if total_count == 0:
        warnings.append("Target has zero valid values.")

    return warnings


def get_detection_reason(
    problem_type: str,
    unique_count: int,
    unique_percent: float,
    target_dtype: str,
    is_numeric: bool,
    is_integer_like: bool,
    classification_unique_threshold: int,
    confidence: str,
) -> str:
    """
    Explain why the problem type was selected.
    """
    if problem_type == "binary_classification":
        return (
            "Target has exactly 2 unique values, so it is treated as binary classification. "
            f"Confidence: {confidence}."
        )

    if problem_type == "multiclass_classification" and is_numeric:
        return (
            f"Target is numeric with {unique_count} unique values "
            f"({unique_percent}% unique), which is less than or equal to the "
            f"configured classification threshold ({classification_unique_threshold}). "
            "It is treated as multiclass classification, but numeric discrete targets "
            "should be reviewed by a human."
        )

    if problem_type == "multiclass_classification":
        return (
            f"Target has {unique_count} unique non-numeric values with dtype "
            f"{target_dtype}. It is treated as multiclass classification. "
            f"Confidence: {confidence}."
        )

    if problem_type == "regression":
        extra = (
            " Target is integer-like, so confirm it is not ordinal classification."
            if is_integer_like
            else ""
        )
        return (
            f"Target is numeric with {unique_count} unique values "
            f"({unique_percent}% unique), which is greater than the configured "
            f"classification threshold ({classification_unique_threshold}). "
            f"It is treated as regression.{extra} Confidence: {confidence}."
        )

    return "Problem type detected from target column properties."


def get_recommended_action(
    problem_type: str,
    needs_human_review: bool,
) -> str:
    """
    Return next recommended action.
    """
    if needs_human_review:
        return (
            "Confirm the problem type with domain context before training baseline models. "
            "Numeric discrete targets are often ambiguous."
        )

    if problem_type in CLASSIFICATION_TYPES:
        return (
            "Proceed with class imbalance checks, stratified split validation, "
            "and classification metrics."
        )

    if problem_type == "regression":
        return (
            "Proceed with regression metrics and inspect target distribution/outliers."
        )

    return "Review target column manually."


if __name__ == "__main__":
    from src.audit.profiler import load_dataset

    dataset_path = "data/sample/student_mark.csv"
    target_column = "Grade"

    df = load_dataset(dataset_path)
    result = detect_problem_type(df, target_column)

    print(result)
