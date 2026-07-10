from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from src.utils.config import get_config_value
from src.utils.exceptions import InvalidTargetColumnError, ProblemTypeDetectionError
from src.utils.logger import get_logger

logger = get_logger(__name__)

CLASSIFICATION_TYPES = {"binary_classification", "multiclass_classification"}
SUPPORTED_PROBLEM_TYPES = {
    "binary_classification",
    "multiclass_classification",
    "regression",
}

_TRUE_LABELS = {"true", "yes", "y", "1", "positive", "pos"}
_FALSE_LABELS = {"false", "no", "n", "0", "negative", "neg"}
_BOOL_LIKE_LABELS = _TRUE_LABELS | _FALSE_LABELS
_DEFAULT_CLASSIFICATION_UNIQUE_THRESHOLD = 20
_DEFAULT_HIGH_CARDINALITY_TARGET_THRESHOLD = 50
_DEFAULT_HIGH_UNIQUE_PERCENT_THRESHOLD = 80.0
_DEFAULT_LOW_UNIQUE_PERCENT_THRESHOLD = 2.0


class TargetStats(dict[str, Any]):
    """Dictionary-like container for target diagnostics."""


def safe_percent(numerator: int | float, denominator: int | float) -> float:
    """Calculate a safe rounded percentage."""
    try:
        numerator_value = float(numerator)
        denominator_value = float(denominator)
    except (TypeError, ValueError):
        return 0.0

    if denominator_value <= 0:
        return 0.0

    return float(round((numerator_value / denominator_value) * 100, 2))


def make_label_text(value: Any) -> str:
    """Convert labels into stable JSON-safe strings."""
    if value is None:
        return "None"

    try:
        if pd.isna(value):
            return "NaN"
    except (TypeError, ValueError):
        pass

    return str(value)


def get_target_series(df: pd.DataFrame, target_column: str) -> pd.Series:
    """Return target column as a Series with duplicate-column protection."""
    target = df.loc[:, target_column]

    if isinstance(target, pd.DataFrame):
        raise InvalidTargetColumnError(
            f"Target column '{target_column}' resolved to multiple columns.",
        )

    return target


def nunique_safely(series: pd.Series, dropna: bool = True) -> int:
    """Return unique count safely for target labels."""
    try:
        return int(series.nunique(dropna=dropna))
    except (TypeError, ValueError):
        return int(series.map(make_label_text).nunique(dropna=dropna))


def value_counts_safely(series: pd.Series, dropna: bool = False) -> pd.Series:
    """Return value counts safely for target labels."""
    try:
        return series.value_counts(dropna=dropna)
    except (TypeError, ValueError):
        return series.map(make_label_text).value_counts(dropna=dropna)


def safe_int_config(path: str, default: int, minimum: int | None = None) -> int:
    """Read an integer config value with validation."""
    try:
        value = int(get_config_value(path, default))
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise ProblemTypeDetectionError(
            f"{path} must be an integer.",
            error_detail=str(error),
        ) from error

    if minimum is not None and value < minimum:
        raise ProblemTypeDetectionError(f"{path} must be at least {minimum}.")

    return value


def safe_float_config(path: str, default: float, minimum: float | None = None) -> float:
    """Read a float config value with validation."""
    try:
        value = float(get_config_value(path, default))
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise ProblemTypeDetectionError(
            f"{path} must be a number.",
            error_detail=str(error),
        ) from error

    if minimum is not None and value < minimum:
        raise ProblemTypeDetectionError(f"{path} must be at least {minimum}.")

    return value


def normalize_target_column(target_column: str) -> str:
    """Normalize target column input while preserving exact column semantics."""
    if target_column is None or not str(target_column).strip():
        raise InvalidTargetColumnError("Target column is required.")

    return str(target_column).strip()


def validate_target(df: pd.DataFrame, target_column: str) -> pd.Series:
    """Validate target column and return non-null target values."""
    if df is None or df.empty:
        raise InvalidTargetColumnError("Dataset is empty.")

    normalized_target = normalize_target_column(target_column)

    if normalized_target not in df.columns:
        raise InvalidTargetColumnError(
            f"Target column '{normalized_target}' not found in dataset.",
        )

    target = get_target_series(df, normalized_target).dropna()

    if target.empty:
        raise InvalidTargetColumnError(
            f"Target column '{normalized_target}' has only missing values.",
        )

    if nunique_safely(target, dropna=True) < 2:
        raise InvalidTargetColumnError(
            f"Target column '{normalized_target}' has only one unique value "
            "and cannot be used for classification or regression.",
        )

    return target


def get_classification_unique_threshold() -> int:
    """Read classification threshold from config."""
    return safe_int_config(
        path="problem_detection.classification_unique_threshold",
        default=_DEFAULT_CLASSIFICATION_UNIQUE_THRESHOLD,
        minimum=2,
    )


def get_high_cardinality_threshold() -> int:
    """Read high-cardinality target threshold from config."""
    return safe_int_config(
        path="problem_detection.high_cardinality_threshold",
        default=_DEFAULT_HIGH_CARDINALITY_TARGET_THRESHOLD,
        minimum=2,
    )


def get_high_unique_percent_threshold() -> float:
    """Read high unique percentage warning threshold from config."""
    value = safe_float_config(
        path="problem_detection.high_unique_percent_threshold",
        default=_DEFAULT_HIGH_UNIQUE_PERCENT_THRESHOLD,
        minimum=0.0,
    )
    return min(100.0, value)


def get_low_unique_percent_threshold() -> float:
    """Read low unique percentage warning threshold from config."""
    value = safe_float_config(
        path="problem_detection.low_unique_percent_threshold",
        default=_DEFAULT_LOW_UNIQUE_PERCENT_THRESHOLD,
        minimum=0.0,
    )
    return min(100.0, value)


def detect_problem_type(
    df: pd.DataFrame,
    target_column: str,
    classification_unique_threshold: int | None = None,
) -> dict[str, Any]:
    """
    Detect ML problem type from the target column.

    Supported outputs:
    - binary_classification
    - multiclass_classification
    - regression

    Numeric targets with limited unique values are intentionally flagged for
    human review instead of overclaiming certainty.
    """
    try:
        normalized_target = normalize_target_column(target_column)
        threshold = (
            get_classification_unique_threshold()
            if classification_unique_threshold is None
            else int(classification_unique_threshold)
        )

        if threshold < 2:
            raise ProblemTypeDetectionError(
                "classification_unique_threshold must be at least 2.",
            )

        target = validate_target(df, normalized_target)
        high_cardinality_threshold = get_high_cardinality_threshold()
        high_unique_percent_threshold = get_high_unique_percent_threshold()
        low_unique_percent_threshold = get_low_unique_percent_threshold()

        stats = build_target_stats(
            df=df,
            target=target,
            target_column=normalized_target,
        )

        problem_type, confidence, needs_human_review, review_reason = (
            infer_problem_type(
                target=target,
                unique_count=int(stats["unique_values"]),
                unique_percent=float(stats["unique_percent"]),
                is_numeric=bool(stats["is_numeric_target"]),
                is_bool=bool(stats["is_bool_target"]),
                is_bool_like=bool(stats["is_bool_like_target"]),
                is_integer_like=bool(stats["is_integer_like_target"]),
                classification_unique_threshold=threshold,
                high_cardinality_threshold=high_cardinality_threshold,
                low_unique_percent_threshold=low_unique_percent_threshold,
            )
        )

        warnings = generate_warnings(
            total_rows=int(stats["total_rows"]),
            total_count=int(stats["total_values"]),
            missing_percent=float(stats["missing_percent"]),
            unique_count=int(stats["unique_values"]),
            unique_percent=float(stats["unique_percent"]),
            is_numeric=bool(stats["is_numeric_target"]),
            is_bool_like=bool(stats["is_bool_like_target"]),
            is_integer_like=bool(stats["is_integer_like_target"]),
            problem_type=problem_type,
            needs_human_review=needs_human_review,
            high_cardinality_threshold=high_cardinality_threshold,
            high_unique_percent_threshold=high_unique_percent_threshold,
        )

        result: dict[str, Any] = {
            "target_column": normalized_target,
            "problem_type": problem_type,
            "target_dtype": stats["target_dtype"],
            "is_numeric_target": stats["is_numeric_target"],
            "is_bool_target": stats["is_bool_target"],
            "is_bool_like_target": stats["is_bool_like_target"],
            "is_integer_like_target": stats["is_integer_like_target"],
            "unique_values": stats["unique_values"],
            "unique_percent": stats["unique_percent"],
            "total_values": stats["total_values"],
            "total_rows": stats["total_rows"],
            "missing_count": stats["missing_count"],
            "missing_percent": stats["missing_percent"],
            "classification_unique_threshold": int(threshold),
            "high_cardinality_threshold": high_cardinality_threshold,
            "confidence": confidence,
            "needs_human_review": needs_human_review,
            "requires_human_review": needs_human_review,
            "human_review_reason": review_reason,
            "sample_values": get_sample_values(target),
            "class_balance_preview": get_class_balance_preview(target, problem_type),
            "warnings": warnings,
            "reason": get_detection_reason(
                problem_type=problem_type,
                unique_count=int(stats["unique_values"]),
                unique_percent=float(stats["unique_percent"]),
                target_dtype=str(stats["target_dtype"]),
                is_numeric=bool(stats["is_numeric_target"]),
                is_bool_like=bool(stats["is_bool_like_target"]),
                is_integer_like=bool(stats["is_integer_like_target"]),
                classification_unique_threshold=threshold,
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
    except (TypeError, ValueError, KeyError, AttributeError) as error:
        logger.exception("Problem type detection failed.")
        raise ProblemTypeDetectionError(
            "Failed to detect problem type.",
            error_detail=str(error),
        ) from error


def build_target_stats(
    df: pd.DataFrame,
    target: pd.Series,
    target_column: str,
) -> TargetStats:
    """Build deterministic target diagnostics used by problem detection."""
    unique_count = nunique_safely(target, dropna=True)
    total_count = int(target.shape[0])
    total_rows = int(len(df))
    full_target = get_target_series(df, target_column)
    missing_count = int(full_target.isna().sum())
    target_dtype = str(target.dtype)

    is_numeric = bool(pd.api.types.is_numeric_dtype(target))
    is_bool = bool(pd.api.types.is_bool_dtype(target))
    is_bool_like = bool(is_bool or is_bool_like_series(target))
    is_integer_like = is_integer_like_series(target) if is_numeric else False

    return TargetStats(
        target_dtype=target_dtype,
        is_numeric_target=is_numeric,
        is_bool_target=is_bool,
        is_bool_like_target=is_bool_like,
        is_integer_like_target=is_integer_like,
        unique_values=unique_count,
        unique_percent=safe_percent(unique_count, total_count),
        total_values=total_count,
        total_rows=total_rows,
        missing_count=missing_count,
        missing_percent=safe_percent(missing_count, total_rows),
    )


def is_integer_like_series(series: pd.Series) -> bool:
    """Check whether numeric target values are integer-like."""
    try:
        numeric = pd.to_numeric(series.dropna(), errors="coerce").dropna()

        if numeric.empty:
            return False

        return bool((numeric % 1 == 0).all())

    except (TypeError, ValueError):
        return False


def is_bool_like_series(series: pd.Series) -> bool:
    """Detect common yes/no or true/false string-like binary targets."""
    try:
        normalized_values = normalize_values_for_label_checks(series.dropna().unique())
    except (TypeError, ValueError):
        return False

    if len(normalized_values) != 2:
        return False

    return normalized_values.issubset(_BOOL_LIKE_LABELS)


def normalize_values_for_label_checks(values: Iterable[Any]) -> set[str]:
    """Normalize labels to lowercase strings for heuristic checks."""
    normalized: set[str] = set()

    for value in values:
        if pd.isna(value):
            continue
        normalized.add(str(value).strip().lower())

    return normalized


def infer_problem_type(
    target: pd.Series,
    unique_count: int,
    unique_percent: float,
    is_numeric: bool,
    is_bool: bool | None = None,
    is_bool_like: bool | None = None,
    is_integer_like: bool | int | None = None,
    classification_unique_threshold: int | None = None,
    high_cardinality_threshold: int = _DEFAULT_HIGH_CARDINALITY_TARGET_THRESHOLD,
    low_unique_percent_threshold: float = _DEFAULT_LOW_UNIQUE_PERCENT_THRESHOLD,
) -> tuple[str, str, bool, str | None]:
    """
    Infer problem type and confidence.

    Returns:
        (problem_type, confidence, needs_human_review, review_reason)

    Backward compatibility:
    older tests/callers passed no is_bool argument:
    infer_problem_type(target, unique_count, unique_percent, is_numeric,
                       is_bool_like, is_integer_like, threshold)
    This function still supports that positional style.
    """
    _ = target

    if (
        classification_unique_threshold is None
        and isinstance(is_integer_like, int)
        and not isinstance(is_integer_like, bool)
    ):
        old_threshold = int(is_integer_like)
        old_is_integer_like = bool(is_bool_like)
        old_is_bool_like = bool(is_bool)
        is_bool = False
        is_bool_like = old_is_bool_like
        is_integer_like = old_is_integer_like
        classification_unique_threshold = old_threshold

    is_bool = bool(is_bool) if is_bool is not None else False
    is_bool_like = bool(is_bool_like) if is_bool_like is not None else False
    is_integer_like = bool(is_integer_like) if is_integer_like is not None else False
    classification_unique_threshold = (
        get_classification_unique_threshold()
        if classification_unique_threshold is None
        else int(classification_unique_threshold)
    )

    if unique_count == 2 or is_bool or is_bool_like:
        confidence = "high" if is_bool or is_bool_like else "medium"
        review_reason = (
            None
            if confidence == "high"
            else (
                "Target has exactly 2 unique values. It is treated as binary "
                "classification; confirm label semantics if values are numeric."
            )
        )
        return (
            "binary_classification",
            confidence,
            confidence != "high" and is_numeric,
            review_reason,
        )

    if not is_numeric:
        needs_review = unique_count > high_cardinality_threshold
        return (
            "multiclass_classification",
            "medium" if needs_review else "high",
            needs_review,
            (
                "Non-numeric target has high cardinality; confirm whether this is "
                "a valid classification target and not an identifier."
            )
            if needs_review
            else None,
        )

    if unique_count <= classification_unique_threshold:
        return (
            "multiclass_classification",
            "medium",
            True,
            (
                "Numeric target has limited unique values. It may represent class "
                "labels, ordinal labels, ratings, buckets, or a regression target. "
                "Confirm with domain context."
            ),
        )

    if is_integer_like and unique_percent <= low_unique_percent_threshold:
        return (
            "regression",
            "medium",
            True,
            (
                "Numeric target has low unique percentage and integer-like values. "
                "Confirm whether it is continuous regression or ordinal classes."
            ),
        )

    if is_integer_like and classification_unique_threshold < unique_count <= 50:
        return (
            "regression",
            "medium",
            True,
            (
                "Numeric integer-like target has a moderate number of unique values. "
                "It may be regression, ordinal classification, or bucketed output. "
                "Confirm manually."
            ),
        )

    return (
        "regression",
        "high",
        False,
        None,
    )


def get_sample_values(target: pd.Series, max_values: int = 10) -> list[str]:
    """Return JSON-safe sample target values."""
    values = target.dropna().unique()[:max_values]
    return [make_label_text(value) for value in values]


def get_class_balance_preview(
    target: pd.Series,
    problem_type: str,
    max_classes: int = 15,
) -> dict[str, Any] | None:
    """Return class distribution preview for classification targets only."""
    if problem_type not in CLASSIFICATION_TYPES:
        return None

    counts = value_counts_safely(target, dropna=False).head(max_classes)
    total = len(target)
    total_classes = nunique_safely(target, dropna=True)

    return {
        "top_classes": [
            {
                "class": make_label_text(label),
                "count": int(count),
                "percent": safe_percent(int(count), total),
            }
            for label, count in counts.items()
        ],
        "shown_classes": int(len(counts)),
        "total_classes": int(total_classes),
        "is_truncated": bool(total_classes > len(counts)),
    }


def generate_warnings(
    total_rows: int,
    total_count: int,
    missing_percent: float,
    unique_count: int,
    unique_percent: float,
    is_numeric: bool,
    is_bool_like: bool,
    is_integer_like: bool,
    problem_type: str,
    needs_human_review: bool,
    high_cardinality_threshold: int = _DEFAULT_HIGH_CARDINALITY_TARGET_THRESHOLD,
    high_unique_percent_threshold: float = _DEFAULT_HIGH_UNIQUE_PERCENT_THRESHOLD,
) -> list[str]:
    """Generate warnings for detected target properties."""
    warnings: list[str] = []

    if total_rows < 50:
        warnings.append(
            "Dataset has fewer than 50 rows. Problem type detection and model "
            "evaluation may be unreliable.",
        )

    if missing_percent > 0:
        warnings.append(
            f"Target column has {missing_percent}% missing values. Rows with "
            "missing target should be removed before training.",
        )

    if needs_human_review:
        warnings.append(
            "Problem type needs human review because target properties are ambiguous.",
        )

    if (
        problem_type in CLASSIFICATION_TYPES
        and unique_count > high_cardinality_threshold
    ):
        warnings.append(
            "Classification target has high cardinality. Confirm whether labels "
            "are meaningful and not identifiers.",
        )

    if is_numeric and is_integer_like and 5 <= unique_count <= 50:
        warnings.append(
            "Numeric integer-like target has limited/moderate unique values. It may "
            "be ordinal classification, bucketed regression, or regular regression.",
        )

    if is_bool_like and problem_type == "binary_classification":
        warnings.append(
            "Target looks boolean-like. Confirm which label represents the positive class.",
        )

    if (
        unique_percent > high_unique_percent_threshold
        and problem_type in CLASSIFICATION_TYPES
    ):
        warnings.append(
            "Classification target has very high unique percentage. This may not "
            "be a valid classification problem.",
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
    is_bool_like: bool,
    is_integer_like: bool,
    classification_unique_threshold: int,
    confidence: str,
) -> str:
    """Explain why the problem type was selected."""
    if problem_type == "binary_classification":
        bool_like_text = " Target values look boolean-like." if is_bool_like else ""
        return (
            "Target has exactly 2 unique values, so it is treated as binary "
            f"classification.{bool_like_text} Confidence: {confidence}."
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
    """Return next recommended action."""
    if needs_human_review:
        return (
            "Confirm the problem type with domain context before training baseline "
            "models. Numeric discrete targets are often ambiguous."
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

    dataframe = load_dataset(dataset_path)
    output = detect_problem_type(dataframe, target_column)

    print(output)
