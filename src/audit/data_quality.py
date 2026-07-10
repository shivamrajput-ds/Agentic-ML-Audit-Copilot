from __future__ import annotations

from collections import defaultdict
from difflib import get_close_matches
from typing import Any

import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object

from src.utils.config import get_config_value
from src.utils.exceptions import DataQualityError, InvalidTargetColumnError
from src.utils.logger import get_logger

logger = get_logger(__name__)

SEVERITY_RANK = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
    "info": 4,
}

TRUE_VALUES = {"true", "1", "yes", "y", "on"}
FALSE_VALUES = {"false", "0", "no", "n", "off"}


def as_bool(value: Any, default: bool = False) -> bool:
    """Convert config/env-like values safely into booleans."""
    if isinstance(value, bool):
        return value

    if value is None:
        return float(default)

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in TRUE_VALUES:
            return True
        if normalized in FALSE_VALUES:
            return False
        return default

    return bool(value)


def get_float_config(path: str, default: float) -> float:
    """Read float config values with a safe fallback."""
    try:
        value = float(get_config_value(path, default))
    except (KeyError, TypeError, ValueError, RuntimeError):
        logger.warning("Invalid float config for %s. Using default=%s", path, default)
        return float(default)

    if not np.isfinite(value):
        logger.warning(
            "Non-finite float config for %s. Using default=%s", path, default
        )
        return default

    return value


def get_int_config(path: str, default: int) -> int:
    """Read integer config values with a safe fallback."""
    try:
        value = int(get_config_value(path, default))
    except (KeyError, TypeError, ValueError, RuntimeError):
        logger.warning("Invalid integer config for %s. Using default=%s", path, default)
        return int(default)

    return value


def clamp(value: float, lower: float, upper: float) -> float:
    """Clamp a numeric value into a configured range."""
    return max(lower, min(upper, value))


def resolve_target_column(df: pd.DataFrame, target_column: str) -> str:
    """Resolve and validate target column name with helpful error messages."""
    if target_column is None or not str(target_column).strip():
        raise InvalidTargetColumnError("Target column is required.")

    requested = str(target_column).strip()
    columns = [str(column) for column in df.columns]

    if requested in df.columns:
        return requested

    case_insensitive_matches = [
        column for column in columns if column.lower() == requested.lower()
    ]
    if len(case_insensitive_matches) == 1:
        resolved = case_insensitive_matches[0]
        logger.warning(
            "Resolved target column '%s' to '%s' using case-insensitive match.",
            target_column,
            resolved,
        )
        return resolved

    close_matches = get_close_matches(requested, columns, n=3, cutoff=0.65)
    suggestion = f" Did you mean one of: {close_matches}?" if close_matches else ""
    raise InvalidTargetColumnError(
        f"Target column '{target_column}' not found in dataset.{suggestion}",
    )


def validate_inputs(df: pd.DataFrame, target_column: str) -> None:
    """Validate inputs for data quality audit."""
    if df is None or df.empty:
        raise DataQualityError("Input dataframe is empty.")

    duplicate_names = [str(column) for column in df.columns[df.columns.duplicated()]]
    if duplicate_names:
        raise DataQualityError(f"Duplicate column names found: {duplicate_names}")

    resolved_target = resolve_target_column(df, target_column)

    if len(df.columns) <= 1:
        raise DataQualityError("Dataset must contain at least one feature column.")

    if resolved_target not in df.columns:
        raise InvalidTargetColumnError(
            f"Target column '{target_column}' not found in dataset.",
        )


def get_thresholds() -> dict[str, float]:
    """Read and sanitize thresholds from config.yaml."""
    high_missing_threshold = clamp(
        get_float_config("audit.high_missing_threshold", 50.0),
        0.0,
        100.0,
    )
    warning_missing_threshold = clamp(
        get_float_config("missing_values.warning_threshold", 20.0),
        0.0,
        high_missing_threshold,
    )

    return {
        "high_missing_threshold": high_missing_threshold,
        "warning_missing_threshold": warning_missing_threshold,
        "high_cardinality_threshold": max(
            1.0,
            get_float_config("audit.high_cardinality_threshold", 50.0),
        ),
        "id_unique_percent_threshold": clamp(
            get_float_config("audit.id_unique_percent_threshold", 95.0),
            0.0,
            100.0,
        ),
        "near_constant_threshold": clamp(
            get_float_config("audit.near_constant_threshold", 95.0),
            0.0,
            100.0,
        ),
        "rare_value_threshold_percent": clamp(
            get_float_config("data_quality.rare_value_threshold_percent", 1.0),
            0.0,
            100.0,
        ),
        "iqr_multiplier": max(0.1, get_float_config("outliers.iqr_multiplier", 1.5)),
    }


def safe_percent(numerator: float, denominator: float) -> float:
    """Calculate percentage safely."""
    if denominator == 0:
        return 0.0

    try:
        return round((float(numerator) / float(denominator)) * 100, 2)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def make_json_safe_value(value: Any) -> str | int | float | bool | None:
    """Convert common pandas/numpy scalar values into JSON-safe values."""
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return round(value, 6)

    if isinstance(value, int | bool | str):
        return value

    return str(value)


def add_finding(
    findings: list[dict[str, Any]],
    severity: str,
    category: str,
    message: str,
    column: str | None = None,
    evidence: dict[str, Any] | None = None,
    recommendation: str | None = None,
) -> None:
    """Add a standardized finding."""
    normalized_severity = str(severity).lower().strip()
    if normalized_severity not in SEVERITY_RANK:
        normalized_severity = "info"

    findings.append(
        {
            "severity": normalized_severity,
            "category": category,
            "column": column,
            "message": message,
            "evidence": evidence or {},
            "recommendation": recommendation,
            "requires_human_review": normalized_severity
            in {"critical", "high", "medium"},
        },
    )


def sort_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort findings by severity and category for stable API/UI output."""
    return sorted(
        findings,
        key=lambda item: (
            SEVERITY_RANK.get(str(item.get("severity", "info")), 99),
            str(item.get("category", "")),
            str(item.get("column", "")),
        ),
    )


def value_counts_safely(series: pd.Series, dropna: bool = False) -> pd.Series:
    """Return value counts, falling back to string conversion for unhashable values."""
    try:
        return series.value_counts(dropna=dropna)
    except (TypeError, ValueError):
        return (
            series.map(make_json_safe_value)
            .astype("string")
            .value_counts(dropna=dropna)
        )


def nunique_safely(series: pd.Series, dropna: bool = True) -> int:
    """Return unique count safely for columns with unhashable object values."""
    try:
        return int(series.nunique(dropna=dropna))
    except (TypeError, ValueError):
        return int(
            series.map(make_json_safe_value).astype("string").nunique(dropna=dropna)
        )


def is_string_like_series(series: pd.Series) -> bool:
    """Return True for object/string/category columns."""
    dtype = series.dtype
    return bool(
        pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
        or isinstance(dtype, pd.CategoricalDtype)
    )


def analyze_missing_values(
    df: pd.DataFrame,
    target_column: str,
    thresholds: dict[str, float],
    findings: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Analyze missing values for all columns."""
    missing_values: dict[str, dict[str, Any]] = {}

    for column in df.columns:
        column_name = str(column)
        missing_count = int(df[column].isna().sum())
        missing_percent = safe_percent(missing_count, len(df))

        if missing_count > 0:
            missing_values[column_name] = {
                "missing_count": missing_count,
                "missing_percent": missing_percent,
            }

        if column_name == target_column:
            if missing_percent > 0:
                add_finding(
                    findings=findings,
                    severity="high",
                    category="target_missing_values",
                    column=column_name,
                    message=(
                        f"Target column has {missing_percent}% missing values. "
                        "Rows with missing target cannot be used for supervised training."
                    ),
                    evidence={
                        "missing_count": missing_count,
                        "missing_percent": missing_percent,
                    },
                    recommendation="Drop rows with missing target before model training.",
                )
            continue

        if missing_percent >= thresholds["high_missing_threshold"]:
            add_finding(
                findings=findings,
                severity="high",
                category="high_missing_values",
                column=column_name,
                message=f"Column has high missing values: {missing_percent}%.",
                evidence={
                    "missing_count": missing_count,
                    "missing_percent": missing_percent,
                },
                recommendation=(
                    "Review whether this feature is useful. Consider dropping it "
                    "or using domain-specific imputation."
                ),
            )

        elif missing_percent >= thresholds["warning_missing_threshold"]:
            add_finding(
                findings=findings,
                severity="medium",
                category="moderate_missing_values",
                column=column_name,
                message=f"Column has moderate missing values: {missing_percent}%.",
                evidence={
                    "missing_count": missing_count,
                    "missing_percent": missing_percent,
                },
                recommendation="Use appropriate imputation and monitor missingness pattern.",
            )

    return missing_values


def detect_duplicate_rows(
    df: pd.DataFrame,
    findings: list[dict[str, Any]],
) -> int:
    """Detect duplicate rows."""
    try:
        duplicate_rows = int(df.duplicated().sum())
    except TypeError as error:
        logger.warning("Duplicate row check skipped: %s", error)
        add_finding(
            findings=findings,
            severity="info",
            category="duplicate_rows_check_skipped",
            message=(
                "Duplicate row check was skipped because some columns contain "
                "unhashable Python objects."
            ),
            evidence={"error": str(error)},
            recommendation="Convert unhashable values to strings before duplicate checking.",
        )
        return 0

    duplicate_percent = safe_percent(duplicate_rows, len(df))

    if duplicate_rows > 0:
        severity = "medium" if duplicate_percent < 5 else "high"
        add_finding(
            findings=findings,
            severity=severity,
            category="duplicate_rows",
            message=(
                f"Dataset contains {duplicate_rows} duplicate rows "
                f"({duplicate_percent}%)."
            ),
            evidence={
                "duplicate_rows": duplicate_rows,
                "duplicate_percent": duplicate_percent,
            },
            recommendation=(
                "Review duplicates and remove them if they are not valid repeated "
                "observations."
            ),
        )

    return duplicate_rows


def get_column_fingerprint(series: pd.Series) -> tuple[Any, ...]:
    """Build a stable fingerprint for duplicate-column detection."""
    try:
        hashed = hash_pandas_object(series, index=False, categorize=True)
        return tuple(hashed.astype("uint64").tolist())
    except (TypeError, ValueError):
        safe_series = series.map(make_json_safe_value).astype("string")
        hashed = hash_pandas_object(safe_series, index=False, categorize=True)
        return tuple(hashed.astype("uint64").tolist())


def detect_duplicate_columns(
    df: pd.DataFrame,
    findings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Detect duplicate columns by exact value equality."""
    duplicate_columns: list[dict[str, Any]] = []
    seen_fingerprints: dict[tuple[Any, ...], str] = {}

    max_columns = max(
        1, get_int_config("data_quality.max_duplicate_column_checks", 300)
    )
    if len(df.columns) > max_columns:
        add_finding(
            findings=findings,
            severity="info",
            category="duplicate_columns_check_limited",
            message=(
                "Duplicate column check was limited because the dataset has many "
                f"columns ({len(df.columns)})."
            ),
            evidence={
                "total_columns": int(len(df.columns)),
                "checked_columns": max_columns,
            },
            recommendation="Run a focused duplicate-column check on selected feature groups.",
        )

    columns_to_check = list(df.columns[:max_columns])

    for column in columns_to_check:
        column_name = str(column)
        try:
            fingerprint = get_column_fingerprint(df[column])
        except (TypeError, ValueError) as error:
            logger.warning(
                "Skipping duplicate column check for %s: %s", column_name, error
            )
            continue

        if fingerprint in seen_fingerprints:
            original_column = seen_fingerprints[fingerprint]
            try:
                is_duplicate = bool(df[original_column].equals(df[column]))
            except (TypeError, ValueError):
                is_duplicate = True

            if not is_duplicate:
                continue

            record = {
                "column_a": original_column,
                "column_b": column_name,
            }
            duplicate_columns.append(record)
            add_finding(
                findings=findings,
                severity="medium",
                category="duplicate_columns",
                column=column_name,
                message=f"Column '{column_name}' is an exact duplicate of '{original_column}'.",
                evidence=record,
                recommendation="Drop one of the duplicate columns before modeling.",
            )
        else:
            seen_fingerprints[fingerprint] = column_name

    return duplicate_columns


def detect_constant_columns(
    df: pd.DataFrame,
    target_column: str,
    findings: list[dict[str, Any]],
) -> list[str]:
    """Detect constant feature columns."""
    constant_columns: list[str] = []

    for column in df.columns:
        column_name = str(column)
        if column_name == target_column:
            continue

        unique_count = nunique_safely(df[column], dropna=False)

        if unique_count <= 1:
            constant_columns.append(column_name)
            add_finding(
                findings=findings,
                severity="medium",
                category="constant_column",
                column=column_name,
                message="Column has only one unique value and provides no predictive signal.",
                evidence={"unique_count": unique_count},
                recommendation="Drop this column before training.",
            )

    return constant_columns


def detect_near_constant_columns(
    df: pd.DataFrame,
    target_column: str,
    thresholds: dict[str, float],
    findings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Detect near-constant feature columns."""
    near_constant_columns: list[dict[str, Any]] = []

    for column in df.columns:
        column_name = str(column)
        if column_name == target_column:
            continue

        value_counts = value_counts_safely(df[column], dropna=False)

        if value_counts.empty:
            continue

        dominant_count = int(value_counts.iloc[0])
        dominant_percent = safe_percent(dominant_count, len(df))

        if dominant_percent >= thresholds["near_constant_threshold"]:
            record = {
                "column": column_name,
                "dominant_value": make_json_safe_value(value_counts.index[0]),
                "dominant_count": dominant_count,
                "dominant_percent": dominant_percent,
            }
            near_constant_columns.append(record)
            add_finding(
                findings=findings,
                severity="low",
                category="near_constant_column",
                column=column_name,
                message=(
                    "Column is near-constant. Dominant value appears in "
                    f"{dominant_percent}% rows."
                ),
                evidence=record,
                recommendation=(
                    "Review usefulness; near-constant columns often add little signal."
                ),
            )

    return near_constant_columns


def detect_high_cardinality_columns(
    df: pd.DataFrame,
    target_column: str,
    thresholds: dict[str, float],
    findings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Detect high-cardinality categorical columns."""
    high_cardinality_columns: list[dict[str, Any]] = []
    feature_df = df.drop(columns=[target_column])

    categorical_columns = feature_df.select_dtypes(
        include=["object", "category", "string"],
    ).columns

    for column in categorical_columns:
        column_name = str(column)
        unique_count = nunique_safely(df[column], dropna=True)
        unique_percent = safe_percent(unique_count, len(df))

        if unique_count >= thresholds["high_cardinality_threshold"]:
            record = {
                "column": column_name,
                "unique_count": unique_count,
                "unique_percent": unique_percent,
            }
            high_cardinality_columns.append(record)

            severity = (
                "high"
                if unique_percent >= thresholds["id_unique_percent_threshold"]
                else "medium"
            )

            add_finding(
                findings=findings,
                severity=severity,
                category="high_cardinality_column",
                column=column_name,
                message=(
                    f"Column has high cardinality: {unique_count} unique values "
                    f"({unique_percent}%)."
                ),
                evidence=record,
                recommendation=(
                    "Avoid naive one-hot encoding if cardinality is high. "
                    "Consider dropping ID-like columns or using target-independent "
                    "encodings."
                ),
            )

    return high_cardinality_columns


def column_name_suggests_id(column: str) -> bool:
    """Detect identifier-like column names with safe token/suffix matching."""
    normalized = (
        str(column)
        .lower()
        .strip()
        .replace("-", "_")
        .replace(" ", "_")
        .replace(".", "_")
    )

    exact_names = {
        "id",
        "uuid",
        "guid",
        "identifier",
        "serial",
        "roll",
        "roll_no",
        "zipcode",
        "zip",
        "phone",
        "mobile",
        "email",
        "aadhaar",
        "pan",
        "ssn",
    }

    suffixes = (
        "_id",
        "_uuid",
        "_guid",
        "_identifier",
        "_serial",
        "_roll",
        "_roll_no",
        "_zipcode",
        "_zip",
        "_phone",
        "_mobile",
        "_email",
        "_aadhaar",
        "_pan",
        "_ssn",
    )

    return normalized in exact_names or normalized.endswith(suffixes)


def detect_possible_id_columns(
    df: pd.DataFrame,
    target_column: str,
    thresholds: dict[str, float],
    findings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Detect possible ID/identifier columns."""
    possible_id_columns: list[dict[str, Any]] = []
    flag_numeric_unique_ids = as_bool(
        get_config_value("data_quality.flag_numeric_unique_as_id", False),
        default=False,
    )

    for column in df.columns:
        column_name = str(column)
        if column_name == target_column:
            continue

        series = df[column]
        unique_count = nunique_safely(series, dropna=True)
        unique_percent = safe_percent(unique_count, len(df))

        name_suggests_id = column_name_suggests_id(column_name)
        uniqueness_suggests_id = (
            unique_percent >= thresholds["id_unique_percent_threshold"]
        )
        type_safely_supports_id_uniqueness = (
            is_string_like_series(series) or flag_numeric_unique_ids
        )

        if name_suggests_id or (
            uniqueness_suggests_id and type_safely_supports_id_uniqueness
        ):
            record = {
                "column": column_name,
                "unique_count": unique_count,
                "unique_percent": unique_percent,
                "name_suggests_id": name_suggests_id,
                "uniqueness_suggests_id": bool(
                    uniqueness_suggests_id and type_safely_supports_id_uniqueness,
                ),
                "dtype": str(series.dtype),
            }
            possible_id_columns.append(record)
            add_finding(
                findings=findings,
                severity="high" if record["uniqueness_suggests_id"] else "medium",
                category="possible_id_column",
                column=column_name,
                message=(
                    "Column may be an identifier and can cause memorization or "
                    "high-dimensional noise."
                ),
                evidence=record,
                recommendation=(
                    "Review this column. Drop it unless it has a legitimate, "
                    "prediction-time meaning."
                ),
            )

    return possible_id_columns


def detect_mixed_type_columns(
    df: pd.DataFrame,
    target_column: str,
    findings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Detect object columns with mixed Python value types."""
    mixed_type_columns: list[dict[str, Any]] = []
    object_columns = df.select_dtypes(include=["object"]).columns

    for column in object_columns:
        column_name = str(column)
        if column_name == target_column:
            continue

        type_counts = (
            df[column]
            .dropna()
            .map(lambda value: type(value).__name__)
            .value_counts()
            .to_dict()
        )

        if len(type_counts) > 1:
            record = {
                "column": column_name,
                "type_counts": {
                    str(key): int(value) for key, value in type_counts.items()
                },
            }
            mixed_type_columns.append(record)
            add_finding(
                findings=findings,
                severity="medium",
                category="mixed_type_column",
                column=column_name,
                message="Column contains mixed Python value types.",
                evidence=record,
                recommendation="Standardize this column before modeling.",
            )

    return mixed_type_columns


def detect_infinite_values(
    df: pd.DataFrame,
    target_column: str,
    findings: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Detect np.inf and -np.inf in numeric columns."""
    infinite_values: dict[str, dict[str, Any]] = {}
    numeric_columns = df.select_dtypes(include=["number"]).columns

    for column in numeric_columns:
        column_name = str(column)
        numeric = pd.to_numeric(df[column], errors="coerce")
        pos_inf_count = int(np.isposinf(numeric).sum())
        neg_inf_count = int(np.isneginf(numeric).sum())
        total_inf_count = pos_inf_count + neg_inf_count

        if total_inf_count > 0:
            record = {
                "positive_infinity_count": pos_inf_count,
                "negative_infinity_count": neg_inf_count,
                "total_infinity_count": total_inf_count,
            }
            infinite_values[column_name] = record
            severity = "high" if column_name == target_column else "medium"
            recommendation = (
                "Replace infinite target values or remove affected rows before training."
                if column_name == target_column
                else "Replace infinite values with NaN before imputation/modeling."
            )
            add_finding(
                findings=findings,
                severity=severity,
                category="infinite_values",
                column=column_name,
                message=(
                    "Column contains infinite values that can break "
                    "preprocessing/model training."
                ),
                evidence=record,
                recommendation=recommendation,
            )

    return infinite_values


def detect_outlier_columns(
    df: pd.DataFrame,
    target_column: str,
    thresholds: dict[str, float],
    findings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Detect numeric outliers using IQR."""
    outliers_enabled = as_bool(
        get_config_value("outliers.enabled", True),
        default=True,
    )
    if not outliers_enabled:
        return []

    outlier_columns: list[dict[str, Any]] = []

    numeric_columns = (
        df.drop(columns=[target_column]).select_dtypes(include=["number"]).columns
    )

    min_non_null = max(3, get_int_config("outliers.min_non_null_values", 10))

    for column in numeric_columns:
        column_name = str(column)
        series = (
            pd.to_numeric(df[column], errors="coerce")
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
        )

        if len(series) < min_non_null or series.nunique() <= 1:
            continue

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1

        if iqr == 0:
            continue

        lower_bound = q1 - thresholds["iqr_multiplier"] * iqr
        upper_bound = q3 + thresholds["iqr_multiplier"] * iqr

        outlier_mask = (series < lower_bound) | (series > upper_bound)
        outlier_count = int(outlier_mask.sum())
        outlier_percent = safe_percent(outlier_count, len(series))

        if outlier_count > 0:
            record = {
                "column": column_name,
                "outlier_count": outlier_count,
                "outlier_percent": outlier_percent,
                "lower_bound": round(float(lower_bound), 4),
                "upper_bound": round(float(upper_bound), 4),
                "method": "iqr",
            }
            outlier_columns.append(record)

            severity = "medium" if outlier_percent >= 5 else "low"
            add_finding(
                findings=findings,
                severity=severity,
                category="outliers",
                column=column_name,
                message=(
                    f"Column has {outlier_count} possible IQR outliers "
                    f"({outlier_percent}%)."
                ),
                evidence=record,
                recommendation=(
                    "Review outliers. Do not remove automatically unless domain "
                    "context supports it."
                ),
            )

    return outlier_columns


def detect_rare_categories(
    df: pd.DataFrame,
    target_column: str,
    thresholds: dict[str, float],
    findings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Detect categorical columns with many rare categories."""
    rare_category_columns: list[dict[str, Any]] = []
    threshold_percent = thresholds["rare_value_threshold_percent"]

    if threshold_percent <= 0:
        return rare_category_columns

    categorical_columns = (
        df.drop(columns=[target_column])
        .select_dtypes(
            include=["object", "category", "string"],
        )
        .columns
    )

    for column in categorical_columns:
        column_name = str(column)
        counts = value_counts_safely(df[column].dropna(), dropna=True)
        if counts.empty:
            continue

        percents = counts.apply(lambda count: safe_percent(int(count), len(df)))
        rare_mask = percents < threshold_percent
        rare_count = int(rare_mask.sum())

        if rare_count == 0:
            continue

        total_categories = int(len(counts))
        rare_category_percent = safe_percent(rare_count, total_categories)

        if rare_category_percent < 20:
            continue

        record = {
            "column": column_name,
            "rare_categories_count": rare_count,
            "total_categories": total_categories,
            "rare_category_percent": rare_category_percent,
            "threshold_percent": threshold_percent,
        }
        rare_category_columns.append(record)
        add_finding(
            findings=findings,
            severity="low",
            category="rare_categories",
            column=column_name,
            message=(
                f"Column has {rare_count} rare categories below "
                f"{threshold_percent}% frequency."
            ),
            evidence=record,
            recommendation=(
                "Consider grouping rare categories into an 'Other' bucket before "
                "modeling if they cause sparse features."
            ),
        )

    return rare_category_columns


def analyze_target_quality(
    df: pd.DataFrame,
    target_column: str,
    findings: list[dict[str, Any]],
) -> dict[str, Any]:
    """Analyze target column quality."""
    target = df[target_column]
    missing_count = int(target.isna().sum())
    missing_percent = safe_percent(missing_count, len(df))
    unique_count = nunique_safely(target, dropna=True)
    is_numeric = bool(pd.api.types.is_numeric_dtype(target))

    target_quality = {
        "target_column": target_column,
        "dtype": str(target.dtype),
        "missing_count": missing_count,
        "missing_percent": missing_percent,
        "unique_count": unique_count,
        "unique_percent": safe_percent(unique_count, len(df)),
        "is_numeric": is_numeric,
    }

    if is_numeric:
        numeric_target = pd.to_numeric(target, errors="coerce")
        numeric_array = np.asarray(numeric_target, dtype=float)
        target_quality["infinite_count"] = int(np.isinf(numeric_array).sum())
        valid_target = numeric_target.replace([np.inf, -np.inf], np.nan).dropna()
        target_quality["valid_numeric_count"] = int(valid_target.count())
    else:
        target_quality["infinite_count"] = 0

    if unique_count < 2:
        add_finding(
            findings=findings,
            severity="critical",
            category="invalid_target",
            column=target_column,
            message="Target column has fewer than 2 unique non-null values.",
            evidence=target_quality,
            recommendation="Choose a valid target column with at least 2 classes/values.",
        )

    return target_quality


def calculate_quality_score(findings: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Calculate quality score from findings.

    This is a triage score, not a statistical guarantee.
    """
    score = 100.0

    penalty_map = {
        "critical": 35,
        "high": 15,
        "medium": 7,
        "low": 2,
        "info": 0,
    }
    category_caps = {
        "high_missing_values": 30,
        "moderate_missing_values": 21,
        "near_constant_column": 10,
        "outliers": 12,
        "rare_categories": 8,
        "possible_id_column": 30,
        "high_cardinality_column": 30,
    }

    penalties: list[dict[str, Any]] = []
    category_penalty_totals: dict[str, float] = defaultdict(float)

    for finding in findings:
        severity = str(finding.get("severity", "info")).lower()
        category = str(finding.get("category", "unknown"))
        base_penalty = float(penalty_map.get(severity, 0))

        if base_penalty <= 0:
            continue

        category_cap = category_caps.get(category)
        if category_cap is not None:
            remaining_cap = max(
                0.0, float(category_cap) - category_penalty_totals[category]
            )
            penalty = min(base_penalty, remaining_cap)
        else:
            penalty = base_penalty

        if penalty <= 0:
            continue

        category_penalty_totals[category] += penalty
        score -= penalty
        penalties.append(
            {
                "category": finding.get("category"),
                "column": finding.get("column"),
                "severity": severity,
                "penalty": round(penalty, 2),
            },
        )

    final_score = max(0.0, min(100.0, score))

    if final_score >= 90:
        health_label = "good"
    elif final_score >= 75:
        health_label = "needs_review"
    elif final_score >= 50:
        health_label = "poor"
    else:
        health_label = "critical"

    return {
        "score": round(final_score, 2),
        "health_label": health_label,
        "penalties": penalties,
        "note": (
            "Quality score is a practical triage score, not a guarantee of model "
            "readiness."
        ),
    }


def generate_warnings(findings: list[dict[str, Any]]) -> list[str]:
    """Generate human-readable warnings."""
    if not findings:
        return ["No major basic data quality issues detected."]

    warnings: list[str] = []

    for finding in sort_findings(findings):
        severity = str(finding.get("severity", "info")).upper()
        column = finding.get("column")
        message = finding.get("message")

        if column:
            warnings.append(f"[{severity}] {column}: {message}")
        else:
            warnings.append(f"[{severity}] {message}")

    return warnings


def generate_recommended_actions(findings: list[dict[str, Any]]) -> list[str]:
    """Generate deduplicated recommended actions."""
    actions: list[str] = []

    for finding in sort_findings(findings):
        recommendation = finding.get("recommendation")
        if recommendation and recommendation not in actions:
            actions.append(str(recommendation))

    if not actions:
        return ["No major data quality preprocessing action required."]

    return actions


def build_column_quality_summary(
    df: pd.DataFrame,
    target_column: str,
    findings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build column-wise quality summary."""
    finding_count_by_column: dict[str, int] = {}

    for finding in findings:
        column = finding.get("column")
        if column:
            column_name = str(column)
            finding_count_by_column[column_name] = (
                finding_count_by_column.get(column_name, 0) + 1
            )

    summary: list[dict[str, Any]] = []

    for column in df.columns:
        column_name = str(column)
        missing_count = int(df[column].isna().sum())
        unique_count = nunique_safely(df[column], dropna=True)

        summary.append(
            {
                "column": column_name,
                "dtype": str(df[column].dtype),
                "is_target": column_name == target_column,
                "missing_count": missing_count,
                "missing_percent": safe_percent(missing_count, len(df)),
                "unique_count": unique_count,
                "unique_percent": safe_percent(unique_count, len(df)),
                "finding_count": finding_count_by_column.get(column_name, 0),
            },
        )

    return summary


def build_finding_summary(findings: list[dict[str, Any]]) -> dict[str, Any]:
    """Build counts by severity and category for dashboards/API consumers."""
    by_severity: dict[str, int] = {severity: 0 for severity in SEVERITY_RANK}
    by_category: dict[str, int] = {}

    for finding in findings:
        severity = str(finding.get("severity", "info")).lower()
        category = str(finding.get("category", "unknown"))
        by_severity[severity] = by_severity.get(severity, 0) + 1
        by_category[category] = by_category.get(category, 0) + 1

    return {
        "total_findings": len(findings),
        "by_severity": by_severity,
        "by_category": dict(sorted(by_category.items())),
        "requires_human_review": any(
            bool(finding.get("requires_human_review", False)) for finding in findings
        ),
    }


def run_data_quality_audit(
    df: pd.DataFrame,
    target_column: str,
) -> dict[str, Any]:
    """
    Run deterministic data quality audit.

    This module reports data quality risks only. It does not automatically mutate data.
    """
    try:
        logger.info("Starting data quality audit")

        validate_inputs(df, target_column)
        target_column = resolve_target_column(df, target_column)

        thresholds = get_thresholds()
        findings: list[dict[str, Any]] = []

        target_quality = analyze_target_quality(df, target_column, findings)
        missing_values = analyze_missing_values(df, target_column, thresholds, findings)
        duplicate_rows = detect_duplicate_rows(df, findings)
        duplicate_columns = detect_duplicate_columns(df, findings)
        constant_columns = detect_constant_columns(df, target_column, findings)
        near_constant_columns = detect_near_constant_columns(
            df,
            target_column,
            thresholds,
            findings,
        )
        high_cardinality_columns = detect_high_cardinality_columns(
            df,
            target_column,
            thresholds,
            findings,
        )
        possible_id_columns = detect_possible_id_columns(
            df,
            target_column,
            thresholds,
            findings,
        )
        mixed_type_columns = detect_mixed_type_columns(df, target_column, findings)
        infinite_values = detect_infinite_values(df, target_column, findings)
        outlier_columns = detect_outlier_columns(
            df,
            target_column,
            thresholds,
            findings,
        )
        rare_category_columns = detect_rare_categories(
            df,
            target_column,
            thresholds,
            findings,
        )

        null_only_columns = [
            column
            for column, values in missing_values.items()
            if values["missing_percent"] == 100
        ]

        high_missing_columns = [
            {
                "column": column,
                **values,
            }
            for column, values in missing_values.items()
            if values["missing_percent"] >= thresholds["high_missing_threshold"]
        ]

        sorted_findings = sort_findings(findings)
        quality_score = calculate_quality_score(sorted_findings)
        warnings = generate_warnings(sorted_findings)
        recommended_actions = generate_recommended_actions(sorted_findings)
        column_quality_summary = build_column_quality_summary(
            df,
            target_column,
            sorted_findings,
        )

        report: dict[str, Any] = {
            "target_column": target_column,
            "total_rows": int(len(df)),
            "total_columns": int(len(df.columns)),
            "quality_score": quality_score,
            "target_quality": target_quality,
            "duplicate_rows": duplicate_rows,
            "duplicate_row_percent": safe_percent(duplicate_rows, len(df)),
            "duplicate_columns": duplicate_columns,
            "missing_values": missing_values,
            "high_missing_columns": high_missing_columns,
            "null_only_columns": null_only_columns,
            "constant_columns": constant_columns,
            "near_constant_columns": near_constant_columns,
            "high_cardinality_columns": high_cardinality_columns,
            "possible_id_columns": possible_id_columns,
            "mixed_type_columns": mixed_type_columns,
            "infinite_values": infinite_values,
            "outlier_columns": outlier_columns,
            "rare_category_columns": rare_category_columns,
            "findings": sorted_findings,
            "finding_summary": build_finding_summary(sorted_findings),
            "column_quality_summary": column_quality_summary,
            "warnings": warnings,
            "recommended_actions": recommended_actions,
            "thresholds": thresholds,
            "message": "Data quality audit completed successfully.",
        }

        logger.info("Data quality audit completed successfully")
        return report

    except (DataQualityError, InvalidTargetColumnError):
        raise
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        pd.errors.ParserError,
    ) as error:
        logger.exception("Data quality audit failed.")
        raise DataQualityError(
            "Data quality audit failed.",
            error_detail=str(error),
        ) from error


if __name__ == "__main__":
    sample_df = pd.DataFrame(
        {
            "student_id": [1, 2, 3, 4, 5, 5],
            "age": [18, 19, 20, np.inf, 22, 22],
            "gender": ["M", "F", "M", None, "F", "F"],
            "constant_col": ["x", "x", "x", "x", "x", "x"],
            "grade": ["A", "B", "A", "B", "A", "A"],
        },
    )

    output = run_data_quality_audit(sample_df, "grade")
    print(output)
