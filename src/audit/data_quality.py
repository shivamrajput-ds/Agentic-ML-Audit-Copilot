from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import get_config_value
from src.utils.exceptions import DataQualityError, InvalidTargetColumnError
from src.utils.logger import get_logger


logger = get_logger(__name__)


def validate_inputs(df: pd.DataFrame, target_column: str) -> None:
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


def find_missing_values(df: pd.DataFrame) -> dict[str, Any]:
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


def find_high_missing_columns(df: pd.DataFrame, threshold: float) -> list[dict[str, Any]]:
    missing_percent = (df.isna().mean() * 100).round(2)

    return [
        {"column": column, "missing_percent": float(percent)}
        for column, percent in missing_percent.items()
        if percent >= threshold
    ]


def find_constant_columns(df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in df.columns
        if df[column].nunique(dropna=False) <= 1
    ]


def find_high_cardinality_columns(
    df: pd.DataFrame,
    threshold: int,
) -> list[dict[str, Any]]:
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
                }
            )

    return results


def find_possible_id_columns(
    df: pd.DataFrame,
    unique_percent_threshold: float = 95.0,
) -> list[dict[str, Any]]:
    possible_id_columns: list[dict[str, Any]] = []

    known_id_names = {
        "id",
        "uid",
        "uuid",
        "user_id",
        "customer_id",
        "record_id",
        "transaction_id",
        "order_id",
        "account_id",
        "student_id",
        "employee_id",
    }

    for column in df.columns:
        column_lower = str(column).lower().strip()
        unique_count = int(df[column].nunique(dropna=True))
        unique_percent = float(round(unique_count / len(df) * 100, 2))

        name_looks_like_id = (
            column_lower in known_id_names
            or column_lower.endswith("_id")
            or column_lower.endswith("id")
            or column_lower.startswith("id_")
        )

        almost_all_unique = unique_percent >= unique_percent_threshold

        if name_looks_like_id or almost_all_unique:
            possible_id_columns.append(
                {
                    "column": column,
                    "unique_values": unique_count,
                    "unique_percent": unique_percent,
                    "reason": get_id_reason(name_looks_like_id, almost_all_unique),
                }
            )

    return possible_id_columns


def find_infinite_values(df: pd.DataFrame) -> dict[str, Any]:
    numeric_df = df.select_dtypes(include=["number"])

    if numeric_df.empty:
        return {}

    inf_mask = numeric_df.replace([np.inf, -np.inf], np.nan).isna() & numeric_df.notna()
    inf_count = inf_mask.sum()

    return {
        column: {
            "infinite_count": int(inf_count[column]),
            "infinite_percent": float(round(inf_count[column] / len(df) * 100, 2)),
        }
        for column in numeric_df.columns
        if inf_count[column] > 0
    }


def find_near_constant_columns(
    df: pd.DataFrame,
    dominance_threshold: float = 95.0,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for column in df.columns:
        value_percent = df[column].value_counts(normalize=True, dropna=False).mul(100)

        if value_percent.empty:
            continue

        top_percent = float(round(value_percent.iloc[0], 2))

        if top_percent >= dominance_threshold and df[column].nunique(dropna=False) > 1:
            results.append(
                {
                    "column": column,
                    "dominant_value_percent": top_percent,
                    "unique_values": int(df[column].nunique(dropna=False)),
                }
            )

    return results


def get_target_quality_summary(df: pd.DataFrame, target_column: str) -> dict[str, Any]:
    target = df[target_column]

    return {
        "target_column": target_column,
        "missing_count": int(target.isna().sum()),
        "missing_percent": float(round(target.isna().mean() * 100, 2)),
        "unique_values": int(target.nunique(dropna=True)),
        "dtype": str(target.dtype),
    }


def get_id_reason(name_looks_like_id: bool, almost_all_unique: bool) -> str:
    if name_looks_like_id and almost_all_unique:
        return "Column name looks like an ID and most values are unique."

    if name_looks_like_id:
        return "Column name looks like an ID."

    return "Most values are unique, so this may behave like an ID column."


def generate_quality_warnings(
    missing_values: dict[str, Any],
    high_missing_columns: list[dict[str, Any]],
    duplicate_rows: int,
    constant_columns: list[str],
    near_constant_columns: list[dict[str, Any]],
    high_cardinality_columns: list[dict[str, Any]],
    possible_id_columns: list[dict[str, Any]],
    infinite_values: dict[str, Any],
    target_quality: dict[str, Any],
) -> list[str]:
    warnings: list[str] = []

    if target_quality.get("missing_count", 0) > 0:
        warnings.append("Target column contains missing values.")

    if target_quality.get("unique_values", 0) < 2:
        warnings.append("Target column has fewer than 2 unique values.")

    if missing_values:
        warnings.append("Feature columns contain missing values.")

    if high_missing_columns:
        warnings.append("Some feature columns have very high missing percentages.")

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
    target_quality: dict[str, Any],
) -> list[str]:
    actions: list[str] = []

    if target_quality.get("missing_count", 0) > 0:
        actions.append("Remove rows with missing target values before training.")

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
        actions.append("Exclude ID-like columns from modeling unless they have real predictive meaning.")

    if infinite_values:
        actions.append("Replace infinite values with NaN and handle them during preprocessing.")

    if not actions:
        actions.append("Proceed to leakage detection and baseline modeling.")

    return actions


def run_data_quality_audit(df: pd.DataFrame, target_column: str) -> dict[str, Any]:
    """
    Run deterministic data quality checks before model training.
    """
    try:
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

        feature_df = df.drop(columns=[target_column])

        target_quality = get_target_quality_summary(df, target_column)
        missing_values = find_missing_values(feature_df)

        high_missing_columns = find_high_missing_columns(
            feature_df,
            threshold=high_missing_threshold,
        )

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

        warnings = generate_quality_warnings(
            missing_values=missing_values,
            high_missing_columns=high_missing_columns,
            duplicate_rows=duplicate_rows,
            constant_columns=constant_columns,
            near_constant_columns=near_constant_columns,
            high_cardinality_columns=high_cardinality_columns,
            possible_id_columns=possible_id_columns,
            infinite_values=infinite_values,
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
            target_quality=target_quality,
        )

        report: dict[str, Any] = {
            "total_rows": int(df.shape[0]),
            "total_columns": int(df.shape[1]),
            "feature_columns": int(feature_df.shape[1]),
            "target_column": target_column,
            "thresholds": {
                "high_missing_threshold": high_missing_threshold,
                "high_cardinality_threshold": high_cardinality_threshold,
                "id_unique_percent_threshold": id_unique_percent_threshold,
                "near_constant_threshold": near_constant_threshold,
            },
            "target_quality": target_quality,
            "missing_values": missing_values,
            "high_missing_columns": high_missing_columns,
            "duplicate_rows": duplicate_rows,
            "duplicate_rows_percent": duplicate_rows_percent,
            "constant_columns": constant_columns,
            "near_constant_columns": near_constant_columns,
            "high_cardinality_columns": high_cardinality_columns,
            "possible_id_columns": possible_id_columns,
            "infinite_values": infinite_values,
            "warnings": warnings,
            "recommended_actions": recommended_actions,
        }

        logger.info("Data quality audit completed")
        return report

    except (DataQualityError, InvalidTargetColumnError):
        raise

    except Exception as error:
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