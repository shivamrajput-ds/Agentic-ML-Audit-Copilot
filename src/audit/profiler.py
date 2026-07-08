from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.exceptions import InvalidDatasetError, InvalidTargetColumnError
from src.utils.logger import get_logger


logger = get_logger(__name__)
ROOT_DIR = Path(__file__).resolve().parents[2]


def load_dataset(file_path: str | Path) -> pd.DataFrame:
    file_path = Path(file_path)

    if not file_path.is_absolute():
        file_path = ROOT_DIR / file_path

    if not file_path.exists():
        raise InvalidDatasetError(f"Dataset file not found: {file_path}")

    if file_path.suffix.lower() != ".csv":
        raise InvalidDatasetError("Only CSV files are supported in MVP.")

    try:
        df = pd.read_csv(file_path)
    except Exception as error:
        raise InvalidDatasetError(
            "Failed to read CSV file",
            error_detail=str(error),
        ) from error

    if df.empty:
        raise InvalidDatasetError("Dataset is empty.")

    logger.info(f"Dataset loaded successfully: {file_path}")
    logger.info(f"Dataset shape: {df.shape}")

    return df


def validate_target_column(df: pd.DataFrame, target_column: str) -> None:
    if not target_column:
        raise InvalidTargetColumnError("Target column is required.")

    if target_column not in df.columns:
        raise InvalidTargetColumnError(
            f"Target column '{target_column}' not found in dataset."
        )

    if df[target_column].isna().all():
        raise InvalidTargetColumnError(
            f"Target column '{target_column}' contains only missing values."
        )


def get_column_types(df: pd.DataFrame, target_column: str) -> dict[str, list[str]]:
    feature_df = df.drop(columns=[target_column])

    numeric_columns = feature_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = feature_df.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()

    datetime_columns = feature_df.select_dtypes(
        include=["datetime64", "datetimetz"]
    ).columns.tolist()

    other_columns = [
        col
        for col in feature_df.columns
        if col not in numeric_columns + categorical_columns + datetime_columns
    ]

    return {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": datetime_columns,
        "other_columns": other_columns,
    }


def get_missing_values_summary(df: pd.DataFrame) -> dict[str, dict[str, float]]:
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


def get_target_summary(df: pd.DataFrame, target_column: str) -> dict[str, Any]:
    target_series = df[target_column]

    summary: dict[str, Any] = {
        "target_column": target_column,
        "missing_count": int(target_series.isna().sum()),
        "missing_percent": float((target_series.isna().mean() * 100).round(2)),
        "unique_values": int(target_series.nunique(dropna=True)),
        "dtype": str(target_series.dtype),
    }

    if target_series.nunique(dropna=True) <= 20:
        counts = target_series.value_counts(dropna=False)
        percentages = (target_series.value_counts(dropna=False, normalize=True) * 100).round(2)

        summary["distribution"] = {
            str(index): {
                "count": int(counts[index]),
                "percent": float(percentages[index]),
            }
            for index in counts.index
        }
    else:
        summary["distribution"] = "Too many unique values to display"

    return summary


def profile_dataset(df: pd.DataFrame, target_column: str) -> dict[str, Any]:
    validate_target_column(df, target_column)

    column_types = get_column_types(df, target_column)

    profile = {
    "shape": {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
    },
    "columns": df.columns.tolist(),
    "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
    "duplicate_rows": int(df.duplicated().sum()),
    "duplicate_rows_percent": float((df.duplicated().mean() * 100).round(2)),
    "missing_values": get_missing_values_summary(df),

    "numeric_columns": column_types["numeric_columns"],
    "categorical_columns": column_types["categorical_columns"],
    "datetime_columns": column_types["datetime_columns"],
    "other_columns": column_types["other_columns"],

    "column_types": column_types,

    "target_summary": get_target_summary(df, target_column),
}
    logger.info("Dataset profiling completed")

    return profile


if __name__ == "__main__":
    dataset_path = "data/sample/student_mark.csv"
    target_column = "Grade"

    df = load_dataset(dataset_path)
    profile = profile_dataset(df, target_column)

    print(profile)
    