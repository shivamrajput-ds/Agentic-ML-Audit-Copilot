from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.exceptions import InvalidDatasetError, InvalidTargetColumnError
from src.utils.logger import get_logger


logger = get_logger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]


def resolve_file_path(file_path: str | Path) -> Path:
    path = Path(file_path)

    if not path.is_absolute():
        path = ROOT_DIR / path

    return path


def load_dataset(file_path: str | Path) -> pd.DataFrame:
    """
    Load and validate a CSV dataset.
    """
    path = resolve_file_path(file_path)

    if not path.exists():
        raise InvalidDatasetError(f"Dataset file not found: {path}")

    if path.suffix.lower() != ".csv":
        raise InvalidDatasetError("Only CSV files are supported currently.")

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as error:
        raise InvalidDatasetError("CSV file is empty.") from error
    except pd.errors.ParserError as error:
        raise InvalidDatasetError(
            "CSV file could not be parsed properly.",
            error_detail=str(error),
        ) from error
    except Exception as error:
        raise InvalidDatasetError(
            "Failed to read CSV file.",
            error_detail=str(error),
        ) from error

    if df.empty:
        raise InvalidDatasetError("Dataset is empty.")

    df.columns = [str(col).strip() for col in df.columns]

    if len(df.columns) != len(set(df.columns)):
        raise InvalidDatasetError("Dataset contains duplicate column names.")

    logger.info(f"Dataset loaded successfully: {path}")
    logger.info(f"Dataset shape: {df.shape}")

    return df


def validate_target_column(df: pd.DataFrame, target_column: str) -> None:
    """
    Validate that the selected target column exists and is usable.
    """
    if df is None or df.empty:
        raise InvalidDatasetError("Dataset is empty.")

    if target_column is None or str(target_column).strip() == "":
        raise InvalidTargetColumnError("Target column is required.")

    if target_column not in df.columns:
        raise InvalidTargetColumnError(
            f"Target column '{target_column}' not found in dataset."
        )

    target_series = df[target_column]

    if target_series.isna().all():
        raise InvalidTargetColumnError(
            f"Target column '{target_column}' contains only missing values."
        )

    if target_series.nunique(dropna=True) < 2:
        raise InvalidTargetColumnError(
            f"Target column '{target_column}' must contain at least 2 unique values."
        )


def detect_datetime_columns(df: pd.DataFrame) -> list[str]:
    """
    Detect datetime-like columns safely.
    """
    datetime_columns: list[str] = []

    for column in df.columns:
        series = df[column]

        if pd.api.types.is_datetime64_any_dtype(series):
            datetime_columns.append(column)
            continue

        if not pd.api.types.is_object_dtype(series):
            continue

        non_null_series = series.dropna()

        if non_null_series.empty:
            continue

        sample = non_null_series.astype(str).head(100)

        try:
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
            valid_ratio = parsed.notna().mean()

            if valid_ratio >= 0.8:
                datetime_columns.append(column)
        except Exception:
            continue

    return datetime_columns


def get_column_types(df: pd.DataFrame, target_column: str) -> dict[str, list[str]]:
    """
    Detect feature column types excluding the target column.
    """
    feature_df = df.drop(columns=[target_column])

    datetime_columns = detect_datetime_columns(feature_df)

    numeric_columns = feature_df.select_dtypes(include=["number"]).columns.tolist()

    categorical_columns = feature_df.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()

    categorical_columns = [
        col for col in categorical_columns if col not in datetime_columns
    ]

    known_columns = set(numeric_columns + categorical_columns + datetime_columns)

    other_columns = [
        column for column in feature_df.columns if column not in known_columns
    ]

    return {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": datetime_columns,
        "other_columns": other_columns,
    }


def get_missing_values_summary(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """
    Return missing value count and percentage for columns with missing values.
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


def get_numeric_summary(df: pd.DataFrame, numeric_columns: list[str]) -> dict[str, Any]:
    """
    Return basic numeric statistics for numeric feature columns.
    """
    if not numeric_columns:
        return {}

    summary: dict[str, Any] = {}

    for column in numeric_columns:
        series = df[column].dropna()

        if series.empty:
            continue

        summary[column] = {
            "mean": float(round(series.mean(), 4)),
            "median": float(round(series.median(), 4)),
            "std": float(round(series.std(), 4)) if len(series) > 1 else 0.0,
            "min": float(series.min()),
            "max": float(series.max()),
            "q1": float(series.quantile(0.25)),
            "q3": float(series.quantile(0.75)),
        }

    return summary


def get_categorical_summary(
    df: pd.DataFrame,
    categorical_columns: list[str],
    max_categories: int = 10,
) -> dict[str, Any]:
    """
    Return top category distribution for categorical columns.
    """
    summary: dict[str, Any] = {}

    for column in categorical_columns:
        series = df[column]

        value_counts = series.value_counts(dropna=False).head(max_categories)
        value_percent = (
            series.value_counts(dropna=False, normalize=True).head(max_categories) * 100
        ).round(2)

        summary[column] = {
            "unique_values": int(series.nunique(dropna=True)),
            "top_values": {
                str(index): {
                    "count": int(value_counts[index]),
                    "percent": float(value_percent[index]),
                }
                for index in value_counts.index
            },
        }

    return summary


def get_target_summary(df: pd.DataFrame, target_column: str) -> dict[str, Any]:
    """
    Build a summary of the target column.
    """
    target_series = df[target_column]

    summary: dict[str, Any] = {
        "target_column": target_column,
        "missing_count": int(target_series.isna().sum()),
        "missing_percent": float(round(target_series.isna().mean() * 100, 2)),
        "unique_values": int(target_series.nunique(dropna=True)),
        "dtype": str(target_series.dtype),
    }

    unique_count = target_series.nunique(dropna=True)

    if unique_count <= 20:
        counts = target_series.value_counts(dropna=False)
        percentages = (
            target_series.value_counts(dropna=False, normalize=True) * 100
        ).round(2)

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


def infer_problem_hint(df: pd.DataFrame, target_column: str) -> str:
    """
    Give a lightweight problem type hint.
    Final decision should be handled by a dedicated problem detector module.
    """
    target = df[target_column]
    unique_count = target.nunique(dropna=True)

    if pd.api.types.is_numeric_dtype(target):
        if unique_count <= 20:
            return "classification_possible"
        return "regression_possible"

    return "classification_possible"


def profile_dataset(df: pd.DataFrame, target_column: str) -> dict[str, Any]:
    """
    Generate a deterministic profile of the dataset before modeling.
    """
    validate_target_column(df, target_column)

    column_types = get_column_types(df, target_column)

    numeric_columns = column_types["numeric_columns"]
    categorical_columns = column_types["categorical_columns"]

    profile: dict[str, Any] = {
        "shape": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
        },
        "columns": df.columns.tolist(),
        "dtypes": {
            column: str(dtype)
            for column, dtype in df.dtypes.items()
        },
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_rows_percent": float(round(df.duplicated().mean() * 100, 2)),
        "missing_values": get_missing_values_summary(df),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": column_types["datetime_columns"],
        "other_columns": column_types["other_columns"],
        "column_types": column_types,
        "numeric_summary": get_numeric_summary(df, numeric_columns),
        "categorical_summary": get_categorical_summary(df, categorical_columns),
        "target_summary": get_target_summary(df, target_column),
        "problem_type_hint": infer_problem_hint(df, target_column),
    }

    logger.info("Dataset profiling completed")
    return profile


if __name__ == "__main__":
    dataset_path = "data/sample/student_mark.csv"
    target_column = "Grade"

    df = load_dataset(dataset_path)
    profile = profile_dataset(df, target_column)

    print(profile)