from typing import Any

import pandas as pd

from src.utils.config import get_config_value
from src.utils.exceptions import DataQualityError, InvalidTargetColumnError
from src.utils.logger import get_logger


logger = get_logger(__name__)


def validate_inputs(df: pd.DataFrame, target_column: str) -> None:
    """
    Validate dataset and target column before running data quality checks.
    """
    if df.empty:
        raise DataQualityError("Dataset is empty.")

    if not target_column:
        raise InvalidTargetColumnError("Target column is required.")

    if target_column not in df.columns:
        raise InvalidTargetColumnError(
            f"Target column '{target_column}' not found in dataset."
        )


def find_missing_values(df: pd.DataFrame) -> dict[str, Any]:
    """
    Find missing value count and percentage for each column.
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
    Find columns where missing percentage is greater than or equal to threshold.
    """
    missing_percent = (df.isna().mean() * 100).round(2)

    return [
        {
            "column": column,
            "missing_percent": float(percent),
        }
        for column, percent in missing_percent.items()
        if percent >= threshold
    ]


def find_constant_columns(df: pd.DataFrame) -> list[str]:
    """
    Find columns with only one unique value.
    """
    return [
        column
        for column in df.columns
        if df[column].nunique(dropna=False) <= 1
    ]


def find_high_cardinality_columns(
    df: pd.DataFrame,
    threshold: int,
) -> list[dict[str, Any]]:
    """
    Find categorical columns with high unique value count.
    """
    categorical_columns = df.select_dtypes(
        include=["object", "string", "category"]
    ).columns

    results: list[dict[str, Any]] = []

    for column in categorical_columns:
        unique_count = df[column].nunique(dropna=True)

        if unique_count >= threshold:
            results.append(
                {
                    "column": column,
                    "unique_values": int(unique_count),
                    "unique_percent": float(
                        round(unique_count / len(df) * 100, 2)
                    ),
                }
            )

    return results


def find_possible_id_columns(
    df: pd.DataFrame,
    unique_percent_threshold: float = 95.0,
) -> list[dict[str, Any]]:
    """
    Find columns that look like identifiers based on name or uniqueness.
    """
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
    }

    for column in df.columns:
        column_lower = column.lower().strip()
        unique_count = df[column].nunique(dropna=True)
        unique_percent = unique_count / len(df) * 100

        name_looks_like_id = (
            column_lower in known_id_names
            or column_lower.endswith("_id")
            or column_lower.startswith("id_")
        )

        almost_all_unique = unique_percent >= unique_percent_threshold

        if name_looks_like_id or almost_all_unique:
            possible_id_columns.append(
                {
                    "column": column,
                    "unique_values": int(unique_count),
                    "unique_percent": float(round(unique_percent, 2)),
                    "reason": _get_id_reason(
                        name_looks_like_id=name_looks_like_id,
                        almost_all_unique=almost_all_unique,
                    ),
                }
            )

    return possible_id_columns


def get_target_quality_summary(
    df: pd.DataFrame,
    target_column: str,
) -> dict[str, Any]:
    """
    Create a separate quality summary for the target column.
    """
    target = df[target_column]

    return {
        "target_column": target_column,
        "missing_count": int(target.isna().sum()),
        "missing_percent": float(round(target.isna().mean() * 100, 2)),
        "unique_values": int(target.nunique(dropna=True)),
        "dtype": str(target.dtype),
    }


def _get_id_reason(
    name_looks_like_id: bool,
    almost_all_unique: bool,
) -> str:
    """
    Explain why a column is flagged as a possible ID column.
    """
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
    high_cardinality_columns: list[dict[str, Any]],
    possible_id_columns: list[dict[str, Any]],
    target_quality: dict[str, Any],
) -> list[str]:
    """
    Convert detected data quality issues into human-readable warnings.
    """
    warnings: list[str] = []

    if target_quality.get("missing_count", 0) > 0:
        warnings.append("Target column contains missing values.")

    if missing_values:
        warnings.append("Feature columns contain missing values.")

    if high_missing_columns:
        warnings.append("Some feature columns have very high missing percentages.")

    if duplicate_rows > 0:
        warnings.append("Dataset contains duplicate rows.")

    if constant_columns:
        warnings.append("Some feature columns have only one unique value.")

    if high_cardinality_columns:
        warnings.append(
            "Some categorical feature columns have high cardinality and may need careful encoding."
        )

    if possible_id_columns:
        warnings.append(
            "Some feature columns look like ID columns and should usually not be used for modeling."
        )

    if not warnings:
        warnings.append("No major basic data quality issues detected.")

    return warnings


def run_data_quality_audit(
    df: pd.DataFrame,
    target_column: str,
) -> dict[str, Any]:
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

        feature_df = df.drop(columns=[target_column])

        target_quality = get_target_quality_summary(
            df=df,
            target_column=target_column,
        )

        missing_values = find_missing_values(feature_df)

        high_missing_columns = find_high_missing_columns(
            df=feature_df,
            threshold=high_missing_threshold,
        )

        duplicate_rows = int(df.duplicated().sum())
        duplicate_rows_percent = float(round(df.duplicated().mean() * 100, 2))

        constant_columns = find_constant_columns(feature_df)

        high_cardinality_columns = find_high_cardinality_columns(
            df=feature_df,
            threshold=high_cardinality_threshold,
        )

        possible_id_columns = find_possible_id_columns(
            df=feature_df,
            unique_percent_threshold=id_unique_percent_threshold,
        )

        warnings = generate_quality_warnings(
            missing_values=missing_values,
            high_missing_columns=high_missing_columns,
            duplicate_rows=duplicate_rows,
            constant_columns=constant_columns,
            high_cardinality_columns=high_cardinality_columns,
            possible_id_columns=possible_id_columns,
            target_quality=target_quality,
        )

        report = {
            "total_rows": int(df.shape[0]),
            "total_columns": int(df.shape[1]),
            "feature_columns": int(feature_df.shape[1]),
            "target_column": target_column,
            "target_quality": target_quality,
            "missing_values": missing_values,
            "high_missing_columns": high_missing_columns,
            "duplicate_rows": duplicate_rows,
            "duplicate_rows_percent": duplicate_rows_percent,
            "constant_columns": constant_columns,
            "high_cardinality_columns": high_cardinality_columns,
            "possible_id_columns": possible_id_columns,
            "warnings": warnings,
        }

        logger.info("Data quality audit completed")
        return report

    except (DataQualityError, InvalidTargetColumnError):
        raise

    except Exception as error:
        raise DataQualityError(
            "Data quality audit failed",
            error_detail=str(error),
        ) from error


if __name__ == "__main__":
    from src.audit.profiler import load_dataset

    dataset_path = "data/sample/student_mark.csv"
    target_column = "Grade"

    df = load_dataset(dataset_path)
    quality_report = run_data_quality_audit(df, target_column)

    print(quality_report)