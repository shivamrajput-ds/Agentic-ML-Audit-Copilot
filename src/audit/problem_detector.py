from __future__ import annotations

from typing import Any

import pandas as pd

from src.utils.exceptions import InvalidTargetColumnError, ProblemTypeDetectionError
from src.utils.logger import get_logger


logger = get_logger(__name__)


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


def detect_problem_type(
    df: pd.DataFrame,
    target_column: str,
    classification_unique_threshold: int = 20,
) -> dict[str, Any]:
    """
    Detect ML problem type from target column.

    Supported outputs:
    - binary_classification
    - multiclass_classification
    - regression
    """
    try:
        if classification_unique_threshold < 2:
            raise ProblemTypeDetectionError(
                "classification_unique_threshold must be at least 2."
            )

        target = validate_target(df, target_column)

        unique_count = int(target.nunique(dropna=True))
        total_count = int(target.shape[0])
        missing_count = int(df[target_column].isna().sum())
        missing_percent = float(round(df[target_column].isna().mean() * 100, 2))
        target_dtype = str(target.dtype)

        is_numeric = bool(pd.api.types.is_numeric_dtype(target))
        is_bool = bool(pd.api.types.is_bool_dtype(target))

        if unique_count == 2:
            problem_type = "binary_classification"

        elif is_bool:
            problem_type = "binary_classification"

        elif is_numeric:
            if unique_count <= classification_unique_threshold:
                problem_type = "multiclass_classification"
            else:
                problem_type = "regression"

        else:
            problem_type = "multiclass_classification"

        result: dict[str, Any] = {
            "target_column": target_column,
            "problem_type": problem_type,
            "target_dtype": target_dtype,
            "is_numeric_target": is_numeric,
            "unique_values": unique_count,
            "total_values": total_count,
            "missing_count": missing_count,
            "missing_percent": missing_percent,
            "classification_unique_threshold": int(classification_unique_threshold),
            "sample_values": get_sample_values(target),
            "reason": get_detection_reason(
                problem_type=problem_type,
                unique_count=unique_count,
                target_dtype=target_dtype,
                is_numeric=is_numeric,
                classification_unique_threshold=classification_unique_threshold,
            ),
        }

        logger.info(f"Problem type detected: {problem_type}")
        return result

    except (InvalidTargetColumnError, ProblemTypeDetectionError):
        raise

    except Exception as error:
        raise ProblemTypeDetectionError(
            "Failed to detect problem type.",
            error_detail=str(error),
        ) from error


def get_sample_values(target: pd.Series, max_values: int = 10) -> list[str]:
    """
    Return JSON-safe sample target values.
    """
    values = target.dropna().unique()[:max_values]
    return [str(value) for value in values]


def get_detection_reason(
    problem_type: str,
    unique_count: int,
    target_dtype: str,
    is_numeric: bool,
    classification_unique_threshold: int,
) -> str:
    """
    Explain why the problem type was selected.
    """
    if problem_type == "binary_classification":
        return "Target has exactly 2 unique values, so it is treated as binary classification."

    if problem_type == "multiclass_classification" and is_numeric:
        return (
            f"Target is numeric with {unique_count} unique values, which is less than "
            f"or equal to the configured classification threshold "
            f"({classification_unique_threshold}). It is treated as multiclass classification."
        )

    if problem_type == "multiclass_classification":
        return (
            f"Target has {unique_count} unique non-numeric values with dtype "
            f"{target_dtype}. It is treated as multiclass classification."
        )

    if problem_type == "regression":
        return (
            f"Target is numeric with {unique_count} unique values, which is greater than "
            f"the configured classification threshold ({classification_unique_threshold}). "
            "It is treated as regression."
        )

    return "Problem type detected from target column properties."


if __name__ == "__main__":
    from src.audit.profiler import load_dataset

    dataset_path = "data/sample/student_mark.csv"
    target_column = "Grade"

    df = load_dataset(dataset_path)
    result = detect_problem_type(df, target_column)

    print(result)