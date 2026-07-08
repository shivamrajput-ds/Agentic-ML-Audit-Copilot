from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import get_config_value
from src.utils.exceptions import DataQualityError, InvalidTargetColumnError
from src.utils.logger import get_logger


logger = get_logger(__name__)


ID_NAME_PATTERNS = {
    "id",
    "uid",
    "uuid",
    "key",
    "identifier",
    "user_id",
    "customer_id",
    "record_id",
    "transaction_id",
    "order_id",
    "account_id",
    "student_id",
    "employee_id",
}


def validate_inputs(df: pd.DataFrame, target_column: str) -> None:
    """
    Validate inputs for data quality audit.
    """
    if df is None or df.empty:
        raise DataQualityError("Dataset is empty.")

    if target_column is None or str(target_column).strip() == "":
        raise InvalidTargetColumnError("Target column is required.")

    if target_column not in df.columns:
        raise InvalidTargetColumnError(
            f"Target column '{target_column}' not found in dataset."
        )

    if len(df.columns) != len(set(df.columns)):
        raise DataQualityError("Dataset contains duplicate column names.")


def get_memory_usage_mb(df: pd.DataFrame) -> float:
    """
    Return dataframe memory usage in MB.
    """
    memory_bytes = df.memory_usage(deep=True).sum()
    return float(round(memory_bytes / (1024 * 1024), 4))


def find_missing_values(df: pd.DataFrame) -> dict[str, Any]:
    """
    Find missing value count and percentage for columns with missing values.
    """
    missing_count = df.isna().sum()
    missing_percent = (missing_count / len(df) * 100).round(2)

    return {
        column: {
            "missing_count": int(missing_count[column]),
            "missing_percent": float(missing_percent[column]),
        }
        for column in df.columns
        if missing_count[column] > 0
    }


def find_high_missing_columns(
    df: pd.DataFrame,
    threshold: float,
) -> list[dict[str, Any]]:
    """
    Find columns whose missing percentage is above configured threshold.
    """
    missing_percent = (df.isna().mean() * 100).round(2)

    return [
        {
            "column": column,
            "missing_percent": float(percent),
            "threshold": float(threshold),
        }
        for column, percent in missing_percent.items()
        if percent >= threshold
    ]


def find_null_only_columns(df: pd.DataFrame) -> list[str]:
    """
    Find columns containing only missing values.
    """
    return [column for column in df.columns if df[column].isna().all()]


def find_constant_columns(df: pd.DataFrame) -> list[str]:
    """
    Find columns with only one unique value, including missing values.
    """
    return [
        column
        for column in df.columns
        if df[column].nunique(dropna=False) <= 1
    ]


def find_near_constant_columns(
    df: pd.DataFrame,
    dominance_threshold: float,
) -> list[dict[str, Any]]:
    """
    Find columns where one value dominates the column.
    """
    results: list[dict[str, Any]] = []

    for column in df.columns:
        value_percent = df[column].value_counts(normalize=True, dropna=False).mul(100)

        if value_percent.empty:
            continue

        top_value = value_percent.index[0]
        top_percent = float(round(value_percent.iloc[0], 2))
        unique_values = int(df[column].nunique(dropna=False))

        if top_percent >= dominance_threshold and unique_values > 1:
            results.append(
                {
                    "column": column,
                    "dominant_value": str(top_value),
                    "dominant_value_percent": top_percent,
                    "unique_values": unique_values,
                    "threshold": float(dominance_threshold),
                }
            )

    return results


def find_high_cardinality_columns(
    df: pd.DataFrame,
    threshold: int,
) -> list[dict[str, Any]]:
    """
    Find categorical columns with many unique values.
    """
    categorical_columns = df.select_dtypes(
        include=["object", "string", "category"]
    ).columns

    results: list[dict[str, Any]] = []

    for column in categorical_columns:
        unique_count = int(df[column].nunique(dropna=True))
        unique_percent = float(round(unique_count / len(df) * 100, 2))

        if unique_count >= threshold:
            results.append(
                {
                    "column": column,
                    "unique_values": unique_count,
                    "unique_percent": unique_percent,
                    "threshold": int(threshold),
                }
            )

    return results


def get_id_reason(name_looks_like_id: bool, almost_all_unique: bool) -> str:
    if name_looks_like_id and almost_all_unique:
        return "Column name looks like an ID and most values are unique."

    if name_looks_like_id:
        return "Column name looks like an ID."

    return "Most values are unique, so this may behave like an ID column."


def find_possible_id_columns(
    df: pd.DataFrame,
    unique_percent_threshold: float,
) -> list[dict[str, Any]]:
    """
    Find columns that look like identifiers by name or uniqueness.
    """
    possible_id_columns: list[dict[str, Any]] = []

    for column in df.columns:
        column_lower = str(column).lower().strip()
        unique_count = int(df[column].nunique(dropna=True))
        non_null_count = int(df[column].notna().sum())

        if non_null_count == 0:
            continue

        unique_percent = float(round(unique_count / non_null_count * 100, 2))

        name_looks_like_id = (
            column_lower in ID_NAME_PATTERNS
            or column_lower.endswith("_id")
            or column_lower.endswith("id")
            or column_lower.startswith("id_")
            or "uuid" in column_lower
        )

        almost_all_unique = unique_percent >= unique_percent_threshold

        if name_looks_like_id or almost_all_unique:
            possible_id_columns.append(
                {
                    "column": column,
                    "unique_values": unique_count,
                    "non_null_count": non_null_count,
                    "unique_percent": unique_percent,
                    "name_looks_like_id": bool(name_looks_like_id),
                    "almost_all_unique": bool(almost_all_unique),
                    "threshold": float(unique_percent_threshold),
                    "reason": get_id_reason(name_looks_like_id, almost_all_unique),
                }
            )

    return possible_id_columns


def find_infinite_values(df: pd.DataFrame) -> dict[str, Any]:
    """
    Find positive/negative infinity values in numeric columns.
    """
    numeric_df = df.select_dtypes(include=["number"])

    if numeric_df.empty:
        return {}

    inf_mask = np.isinf(numeric_df.to_numpy(dtype=float, copy=True))
    inf_df = pd.DataFrame(inf_mask, columns=numeric_df.columns, index=numeric_df.index)
    inf_count = inf_df.sum()

    return {
        column: {
            "infinite_count": int(inf_count[column]),
            "infinite_percent": float(round(inf_count[column] / len(df) * 100, 2)),
        }
        for column in numeric_df.columns
        if inf_count[column] > 0
    }


def find_outlier_columns_iqr(
    df: pd.DataFrame,
    multiplier: float,
) -> list[dict[str, Any]]:
    """
    Detect numeric columns with IQR-based outliers.
    """
    numeric_df = df.select_dtypes(include=["number"])
    results: list[dict[str, Any]] = []

    for column in numeric_df.columns:
        series = numeric_df[column].replace([np.inf, -np.inf], np.nan).dropna()

        if series.empty or series.nunique(dropna=True) < 2:
            continue

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1

        if iqr == 0:
            continue

        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr

        outlier_mask = (series < lower_bound) | (series > upper_bound)
        outlier_count = int(outlier_mask.sum())

        if outlier_count > 0:
            results.append(
                {
                    "column": column,
                    "outlier_count": outlier_count,
                    "outlier_percent": float(round(outlier_count / len(series) * 100, 2)),
                    "q1": float(round(q1, 4)),
                    "q3": float(round(q3, 4)),
                    "iqr": float(round(iqr, 4)),
                    "lower_bound": float(round(lower_bound, 4)),
                    "upper_bound": float(round(upper_bound, 4)),
                    "method": "iqr",
                    "iqr_multiplier": float(multiplier),
                }
            )

    return results


def detect_datetime_like_columns(df: pd.DataFrame) -> list[str]:
    """
    Detect datetime-like object columns.
    """
    datetime_columns: list[str] = []

    for column in df.columns:
        series = df[column]

        if pd.api.types.is_datetime64_any_dtype(series):
            datetime_columns.append(column)
            continue

        if not pd.api.types.is_object_dtype(series):
            continue

        sample = series.dropna().astype(str).head(100)

        if sample.empty:
            continue

        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        valid_ratio = float(parsed.notna().mean())

        if valid_ratio >= 0.8:
            datetime_columns.append(column)

    return datetime_columns


def get_datetime_quality(
    df: pd.DataFrame,
    datetime_columns: list[str],
) -> dict[str, Any]:
    """
    Return datetime parsing quality and min/max dates.
    """
    results: dict[str, Any] = {}

    for column in datetime_columns:
        parsed = pd.to_datetime(df[column], errors="coerce", format="mixed")
        valid_count = int(parsed.notna().sum())

        results[column] = {
            "valid_datetime_count": valid_count,
            "valid_datetime_percent": float(round(parsed.notna().mean() * 100, 2)),
            "invalid_or_missing_count": int(parsed.isna().sum()),
            "min_date": str(parsed.min()) if valid_count else None,
            "max_date": str(parsed.max()) if valid_count else None,
        }

    return results


def find_mixed_type_columns(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Detect object columns that contain mixed Python value types.
    """
    results: list[dict[str, Any]] = []

    object_columns = df.select_dtypes(include=["object", "string"]).columns

    for column in object_columns:
        non_null = df[column].dropna()

        if non_null.empty:
            continue

        type_counts = non_null.map(lambda value: type(value).__name__).value_counts()

        if len(type_counts) > 1:
            results.append(
                {
                    "column": column,
                    "types": {
                        str(type_name): int(count)
                        for type_name, count in type_counts.items()
                    },
                    "reason": "Column contains multiple Python value types.",
                }
            )

    return results


def get_boolean_quality(df: pd.DataFrame) -> dict[str, Any]:
    """
    Summarize boolean columns.
    """
    bool_columns = df.select_dtypes(include=["bool"]).columns
    results: dict[str, Any] = {}

    for column in bool_columns:
        counts = df[column].value_counts(dropna=False)
        percentages = df[column].value_counts(dropna=False, normalize=True).mul(100).round(2)

        results[column] = {
            str(value): {
                "count": int(counts[value]),
                "percent": float(percentages[value]),
            }
            for value in counts.index
        }

    return results


def get_target_quality_summary(df: pd.DataFrame, target_column: str) -> dict[str, Any]:
    """
    Summarize target quality.
    """
    target = df[target_column]

    return {
        "target_column": target_column,
        "missing_count": int(target.isna().sum()),
        "missing_percent": float(round(target.isna().mean() * 100, 2)),
        "unique_values": int(target.nunique(dropna=True)),
        "dtype": str(target.dtype),
    }


def calculate_quality_score(
    high_missing_columns: list[dict[str, Any]],
    duplicate_rows_percent: float,
    constant_columns: list[str],
    near_constant_columns: list[dict[str, Any]],
    high_cardinality_columns: list[dict[str, Any]],
    possible_id_columns: list[dict[str, Any]],
    infinite_values: dict[str, Any],
    null_only_columns: list[str],
    outlier_columns: list[dict[str, Any]],
    target_quality: dict[str, Any],
) -> dict[str, Any]:
    """
    Calculate a simple audit health score from 0 to 100.
    """
    score = 100.0
    penalties: list[dict[str, Any]] = []

    def apply_penalty(name: str, value: float, reason: str) -> None:
        nonlocal score
        score -= value
        penalties.append(
            {
                "check": name,
                "penalty": float(round(value, 2)),
                "reason": reason,
            }
        )

    if target_quality.get("missing_count", 0) > 0:
        apply_penalty("target_missing", 15, "Target column contains missing values.")

    if target_quality.get("unique_values", 0) < 2:
        apply_penalty("target_unique_values", 30, "Target has fewer than 2 unique values.")

    if null_only_columns:
        apply_penalty("null_only_columns", min(20, len(null_only_columns) * 5), "Null-only columns found.")

    if high_missing_columns:
        apply_penalty(
            "high_missing_columns",
            min(20, len(high_missing_columns) * 4),
            "High-missing columns found.",
        )

    if duplicate_rows_percent > 0:
        apply_penalty(
            "duplicate_rows",
            min(15, duplicate_rows_percent / 2),
            "Duplicate rows found.",
        )

    if constant_columns:
        apply_penalty(
            "constant_columns",
            min(15, len(constant_columns) * 3),
            "Constant columns found.",
        )

    if near_constant_columns:
        apply_penalty(
            "near_constant_columns",
            min(10, len(near_constant_columns) * 2),
            "Near-constant columns found.",
        )

    if high_cardinality_columns:
        apply_penalty(
            "high_cardinality_columns",
            min(10, len(high_cardinality_columns) * 2),
            "High-cardinality categorical columns found.",
        )

    if possible_id_columns:
        apply_penalty(
            "possible_id_columns",
            min(10, len(possible_id_columns) * 2),
            "Possible ID-like columns found.",
        )

    if infinite_values:
        apply_penalty(
            "infinite_values",
            min(10, len(infinite_values) * 3),
            "Infinite numeric values found.",
        )

    if outlier_columns:
        apply_penalty(
            "outlier_columns",
            min(8, len(outlier_columns)),
            "IQR outlier columns found.",
        )

    final_score = max(0.0, min(100.0, score))

    if final_score >= 85:
        health_label = "good"
    elif final_score >= 70:
        health_label = "needs_review"
    elif final_score >= 50:
        health_label = "poor"
    else:
        health_label = "critical"

    return {
        "score": float(round(final_score, 2)),
        "health_label": health_label,
        "penalties": penalties,
    }


def generate_quality_warnings(
    missing_values: dict[str, Any],
    high_missing_columns: list[dict[str, Any]],
    duplicate_rows: int,
    constant_columns: list[str],
    near_constant_columns: list[dict[str, Any]],
    high_cardinality_columns: list[dict[str, Any]],
    possible_id_columns: list[dict[str, Any]],
    infinite_values: dict[str, Any],
    null_only_columns: list[str],
    outlier_columns: list[dict[str, Any]],
    mixed_type_columns: list[dict[str, Any]],
    target_quality: dict[str, Any],
) -> list[str]:
    """
    Generate human-readable data quality warnings.
    """
    warnings: list[str] = []

    if target_quality.get("missing_count", 0) > 0:
        warnings.append("Target column contains missing values.")

    if target_quality.get("unique_values", 0) < 2:
        warnings.append("Target column has fewer than 2 unique values.")

    if missing_values:
        warnings.append("Feature columns contain missing values.")

    if high_missing_columns:
        warnings.append("Some feature columns have very high missing percentages.")

    if null_only_columns:
        warnings.append("Some feature columns contain only missing values.")

    if duplicate_rows > 0:
        warnings.append("Dataset contains duplicate rows.")

    if constant_columns:
        warnings.append("Some feature columns have only one unique value.")

    if near_constant_columns:
        warnings.append("Some feature columns are near-constant.")

    if high_cardinality_columns:
        warnings.append(
            "Some categorical feature columns have high cardinality and may need careful encoding."
        )

    if possible_id_columns:
        warnings.append(
            "Some feature columns look like ID columns and should usually not be used for modeling."
        )

    if infinite_values:
        warnings.append("Some numeric feature columns contain infinite values.")

    if outlier_columns:
        warnings.append("Some numeric feature columns contain IQR-based outliers.")

    if mixed_type_columns:
        warnings.append("Some object columns contain mixed Python value types.")

    if not warnings:
        warnings.append("No major basic data quality issues detected.")

    return warnings


def generate_recommended_actions(
    high_missing_columns: list[dict[str, Any]],
    duplicate_rows: int,
    constant_columns: list[str],
    near_constant_columns: list[dict[str, Any]],
    high_cardinality_columns: list[dict[str, Any]],
    possible_id_columns: list[dict[str, Any]],
    infinite_values: dict[str, Any],
    null_only_columns: list[str],
    outlier_columns: list[dict[str, Any]],
    mixed_type_columns: list[dict[str, Any]],
    target_quality: dict[str, Any],
) -> list[str]:
    """
    Generate practical recommended actions.
    """
    actions: list[str] = []

    if target_quality.get("missing_count", 0) > 0:
        actions.append("Remove rows with missing target values before training.")

    if target_quality.get("unique_values", 0) < 2:
        actions.append("Choose a target column with at least 2 unique values.")

    if null_only_columns:
        actions.append("Drop columns that contain only missing values.")

    if high_missing_columns:
        actions.append("Review high-missing columns before imputation or dropping.")

    if duplicate_rows > 0:
        actions.append("Check whether duplicate rows are valid or should be removed.")

    if constant_columns:
        actions.append("Drop constant feature columns before model training.")

    if near_constant_columns:
        actions.append("Review near-constant columns because they may add little signal.")

    if high_cardinality_columns:
        actions.append("Use careful encoding for high-cardinality categorical columns.")

    if possible_id_columns:
        actions.append(
            "Exclude ID-like columns from modeling unless they have real predictive meaning."
        )

    if infinite_values:
        actions.append("Replace infinite values with NaN and handle them during preprocessing.")

    if outlier_columns:
        actions.append("Review outliers and decide whether to cap, transform, or keep them.")

    if mixed_type_columns:
        actions.append("Standardize mixed-type columns before preprocessing.")

    if not actions:
        actions.append("Proceed to leakage detection and baseline modeling.")

    return actions


def run_data_quality_audit(df: pd.DataFrame, target_column: str) -> dict[str, Any]:
    """
    Run deterministic data quality checks before model training.
    """
    try:
        logger.info("Starting data quality audit")

        validate_inputs(df, target_column)

        high_missing_threshold = float(
            get_config_value("audit.high_missing_threshold", 50)
        )
        high_cardinality_threshold = int(
            get_config_value("audit.high_cardinality_threshold", 50)
        )
        id_unique_percent_threshold = float(
            get_config_value("audit.id_unique_percent_threshold", 95)
        )
        near_constant_threshold = float(
            get_config_value("audit.near_constant_threshold", 95)
        )

        outliers_enabled = bool(get_config_value("outliers.enabled", True))
        iqr_multiplier = float(get_config_value("outliers.iqr_multiplier", 1.5))

        feature_df = df.drop(columns=[target_column])

        target_quality = get_target_quality_summary(df, target_column)
        missing_values = find_missing_values(feature_df)

        high_missing_columns = find_high_missing_columns(
            feature_df,
            threshold=high_missing_threshold,
        )

        null_only_columns = find_null_only_columns(feature_df)

        duplicate_rows = int(df.duplicated().sum())
        duplicate_rows_percent = float(round(df.duplicated().mean() * 100, 2))

        constant_columns = find_constant_columns(feature_df)

        near_constant_columns = find_near_constant_columns(
            feature_df,
            dominance_threshold=near_constant_threshold,
        )

        high_cardinality_columns = find_high_cardinality_columns(
            feature_df,
            threshold=high_cardinality_threshold,
        )

        possible_id_columns = find_possible_id_columns(
            feature_df,
            unique_percent_threshold=id_unique_percent_threshold,
        )

        infinite_values = find_infinite_values(feature_df)

        outlier_columns = (
            find_outlier_columns_iqr(feature_df, multiplier=iqr_multiplier)
            if outliers_enabled
            else []
        )

        datetime_columns = detect_datetime_like_columns(feature_df)
        datetime_quality = get_datetime_quality(feature_df, datetime_columns)

        mixed_type_columns = find_mixed_type_columns(feature_df)
        boolean_quality = get_boolean_quality(feature_df)

        warnings = generate_quality_warnings(
            missing_values=missing_values,
            high_missing_columns=high_missing_columns,
            duplicate_rows=duplicate_rows,
            constant_columns=constant_columns,
            near_constant_columns=near_constant_columns,
            high_cardinality_columns=high_cardinality_columns,
            possible_id_columns=possible_id_columns,
            infinite_values=infinite_values,
            null_only_columns=null_only_columns,
            outlier_columns=outlier_columns,
            mixed_type_columns=mixed_type_columns,
            target_quality=target_quality,
        )

        recommended_actions = generate_recommended_actions(
            high_missing_columns=high_missing_columns,
            duplicate_rows=duplicate_rows,
            constant_columns=constant_columns,
            near_constant_columns=near_constant_columns,
            high_cardinality_columns=high_cardinality_columns,
            possible_id_columns=possible_id_columns,
            infinite_values=infinite_values,
            null_only_columns=null_only_columns,
            outlier_columns=outlier_columns,
            mixed_type_columns=mixed_type_columns,
            target_quality=target_quality,
        )

        quality_score = calculate_quality_score(
            high_missing_columns=high_missing_columns,
            duplicate_rows_percent=duplicate_rows_percent,
            constant_columns=constant_columns,
            near_constant_columns=near_constant_columns,
            high_cardinality_columns=high_cardinality_columns,
            possible_id_columns=possible_id_columns,
            infinite_values=infinite_values,
            null_only_columns=null_only_columns,
            outlier_columns=outlier_columns,
            target_quality=target_quality,
        )

        report: dict[str, Any] = {
            "metadata": {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "audit_module": "data_quality",
            },
            "total_rows": int(df.shape[0]),
            "total_columns": int(df.shape[1]),
            "feature_columns": int(feature_df.shape[1]),
            "target_column": target_column,
            "memory_usage_mb": get_memory_usage_mb(df),
            "thresholds": {
                "high_missing_threshold": high_missing_threshold,
                "high_cardinality_threshold": high_cardinality_threshold,
                "id_unique_percent_threshold": id_unique_percent_threshold,
                "near_constant_threshold": near_constant_threshold,
                "outliers_enabled": outliers_enabled,
                "iqr_multiplier": iqr_multiplier,
            },
            "quality_score": quality_score,
            "target_quality": target_quality,
            "missing_values": missing_values,
            "high_missing_columns": high_missing_columns,
            "null_only_columns": null_only_columns,
            "duplicate_rows": duplicate_rows,
            "duplicate_rows_percent": duplicate_rows_percent,
            "constant_columns": constant_columns,
            "near_constant_columns": near_constant_columns,
            "high_cardinality_columns": high_cardinality_columns,
            "possible_id_columns": possible_id_columns,
            "infinite_values": infinite_values,
            "outlier_columns": outlier_columns,
            "datetime_columns": datetime_columns,
            "datetime_quality": datetime_quality,
            "mixed_type_columns": mixed_type_columns,
            "boolean_quality": boolean_quality,
            "warnings": warnings,
            "recommended_actions": recommended_actions,
        }

        logger.info(
            "Data quality audit completed. Score=%s Health=%s",
            quality_score["score"],
            quality_score["health_label"],
        )

        return report

    except (DataQualityError, InvalidTargetColumnError):
        raise

    except Exception as error:
        logger.exception("Data quality audit failed.")
        raise DataQualityError(
            "Data quality audit failed.",
            error_detail=str(error),
        ) from error


if __name__ == "__main__":
    from src.audit.profiler import load_dataset

    dataset_path = "data/sample/student_mark.csv"
    target_column = "Grade"

    df = load_dataset(dataset_path)
    quality_report = run_data_quality_audit(df, target_column)

    print(quality_report)
