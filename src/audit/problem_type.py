from typing import Any

import pandas as pd

from src.utils.exceptions import ProblemTypeDetectionError, InvalidTargetColumnError
from src.utils.logger import get_logger


logger = get_logger(__name__)


def validate_target(df: pd.DataFrame, target_column: str) -> pd.Series:
    """
    Validate the target column and return its non-null values.
    """
    if not target_column:
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

    # EDGE CASE FIX: a target with only one distinct value (e.g. every
    # row is "Yes", or every row is 0) cannot be used for classification
    # or regression — there is nothing for a model to learn. Without this
    # check, the old logic silently labeled a constant target as
    # "multiclass_classification", which would later fail confusingly
    # deep inside train_test_split (stratify) or model training instead
    # of failing clearly here with a helpful message.
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
    Detect whether ML problem is binary classification,
    multiclass classification, or regression.

    Detection rules (in order):
    1. Exactly 2 unique values -> binary classification.
    2. Numeric target with <= classification_unique_threshold unique
       values -> multiclass classification (e.g. a 1-5 star rating
       stored as int).
    3. Numeric target with more unique values than the threshold ->
       regression.
    4. Non-numeric target (strings/categories) with more than 2 unique
       values -> multiclass classification.
    """
    try:
        target = validate_target(df, target_column)

        unique_count = target.nunique()
        total_count = int(target.shape[0])
        target_dtype = str(target.dtype)

        if unique_count == 2:
            problem_type = "binary_classification"

        elif pd.api.types.is_numeric_dtype(target):
            if unique_count <= classification_unique_threshold:
                problem_type = "multiclass_classification"
            else:
                problem_type = "regression"

        else:
            problem_type = "multiclass_classification"

        result = {
            "target_column": target_column,
            "problem_type": problem_type,
            "target_dtype": target_dtype,
            "unique_values": int(unique_count),
            "total_values": total_count,
            "sample_values": [str(value) for value in target.unique()[:10]],
            "reason": _get_reason(problem_type, unique_count, target_dtype),
        }

        logger.info(f"Problem type detected: {problem_type}")
        return result

    except (InvalidTargetColumnError, ProblemTypeDetectionError):
        raise

    except Exception as error:
        raise ProblemTypeDetectionError(
            "Failed to detect problem type",
            error_detail=str(error),
        ) from error


def _get_reason(problem_type: str, unique_count: int, target_dtype: str) -> str:
    """
    Explain in plain language why this problem type was chosen.
    """
    if problem_type == "binary_classification":
        return "Target has exactly 2 unique values."

    if problem_type == "multiclass_classification":
        return (
            f"Target has {unique_count} unique values with dtype {target_dtype}. "
            "This is suitable for classification."
        )

    if problem_type == "regression":
        return (
            f"Target is numeric with {unique_count} unique values. "
            "This is suitable for regression."
        )

    return "Problem type detected from target column properties."


if __name__ == "__main__":
    from src.audit.profiler import load_dataset

    dataset_path = "data/sample/student_mark.csv"
    target_column = "Grade"

    df = load_dataset(dataset_path)
    result = detect_problem_type(df, target_column)

    print(result)