from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import get_config_value
from src.utils.exceptions import InvalidDatasetError
from src.utils.logger import get_logger

logger = get_logger(__name__)

_TRUE_VALUES = {"true", "1", "yes", "y", "on"}


def as_bool(value: Any) -> bool:
    """Convert config values safely into boolean."""
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() in _TRUE_VALUES

    return bool(value)


def safe_round(value: Any, digits: int = 4, default: float = 0.0) -> float:
    """Safely round pandas/numpy scalar values."""
    try:
        if value is None or pd.isna(value):
            return default
        return round(float(value), digits)
    except (TypeError, ValueError):
        return default


def safe_percent(numerator: float, denominator: float) -> float:
    """Return percentage safely rounded to 2 decimals."""
    if denominator == 0:
        return 0.0

    return round((numerator / denominator) * 100, 2)


def validate_dataset_path(dataset_path: str | Path) -> Path:
    """Validate dataset path and extension."""
    if dataset_path is None or not str(dataset_path).strip():
        raise InvalidDatasetError("Dataset path is required.")

    path = Path(dataset_path)

    if not path.exists():
        raise InvalidDatasetError(f"Dataset file not found: {path}")

    if not path.is_file():
        raise InvalidDatasetError(f"Dataset path is not a file: {path}")

    allowed_extensions = get_config_value("dataset.allowed_extensions", [".csv"])
    allowed = {str(ext).strip().lower() for ext in allowed_extensions}

    if path.suffix.lower() not in allowed:
        raise InvalidDatasetError(
            f"Unsupported dataset extension '{path.suffix}'. Allowed: {sorted(allowed)}"
        )

    if path.stat().st_size == 0:
        raise InvalidDatasetError("Dataset file is empty.")

    return path


def get_file_size_mb(path: Path) -> float:
    """Return file size in MB."""
    return round(path.stat().st_size / (1024 * 1024), 2)


def read_csv_with_fallback(path: Path, **kwargs: Any) -> pd.DataFrame:
    """Read CSV with common encoding fallback."""
    try:
        return pd.read_csv(path, **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1", **kwargs)
    except pd.errors.EmptyDataError as error:
        raise InvalidDatasetError("CSV file is empty or has no columns.") from error
    except pd.errors.ParserError as error:
        raise InvalidDatasetError(
            "CSV parsing failed. File may be corrupt or malformed.",
            error_detail=str(error),
        ) from error


def validate_loaded_dataframe(df: pd.DataFrame) -> None:
    """Validate dataframe after loading."""
    if df.empty:
        raise InvalidDatasetError("Loaded dataset is empty.")

    if df.columns.empty:
        raise InvalidDatasetError("Loaded dataset has no columns.")

    duplicate_columns = df.columns[df.columns.duplicated()].tolist()
    if duplicate_columns:
        raise InvalidDatasetError(f"Duplicate column names found: {duplicate_columns}")


def load_dataset(dataset_path: str | Path) -> pd.DataFrame:
    """
    Load tabular dataset.

    For normal files, the full CSV is loaded.
    For very large files, only max_rows are loaded when sampling is enabled.
    """
    try:
        path = validate_dataset_path(dataset_path)

        max_rows = int(get_config_value("dataset.max_rows", 1_000_000))
        max_file_size_mb = float(get_config_value("dataset.max_file_size_mb", 500))
        sample_large_dataset = as_bool(
            get_config_value("dataset.sample_large_dataset", True)
        )

        if path.suffix.lower() != ".csv":
            raise InvalidDatasetError("Only CSV files are currently supported.")

        file_size_mb = get_file_size_mb(path)
        read_kwargs: dict[str, Any] = {"low_memory": False}

        if file_size_mb > max_file_size_mb:
            if not sample_large_dataset:
                raise InvalidDatasetError(
                    f"Dataset file is {file_size_mb} MB, exceeding configured "
                    f"max_file_size_mb={max_file_size_mb}."
                )

            read_kwargs["nrows"] = max_rows
            logger.warning(
                "Large dataset detected: %s MB. Loaded first %s rows for audit.",
                file_size_mb,
                max_rows,
            )

        df = read_csv_with_fallback(path, **read_kwargs)
        validate_loaded_dataframe(df)

        if len(df) > max_rows:
            if not sample_large_dataset:
                raise InvalidDatasetError(
                    f"Dataset has {len(df)} rows, exceeding configured "
                    f"max_rows={max_rows}."
                )

            random_state = int(get_config_value("random_seed", 42))
            df = df.sample(n=max_rows, random_state=random_state).reset_index(drop=True)
            logger.warning(
                "Dataset exceeded max_rows. Sampled %s rows for audit.",
                max_rows,
            )

        logger.info("Dataset loaded successfully: %s", path)
        logger.info("Dataset shape: %s", df.shape)
        return df

    except InvalidDatasetError:
        raise
    except (
        OSError,
        UnicodeDecodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        TypeError,
        ValueError,
    ) as error:
        logger.exception("Dataset loading failed.")
        raise InvalidDatasetError(
            "Failed to load dataset.",
            error_detail=str(error),
        ) from error


def infer_datetime_columns(df: pd.DataFrame, max_check_columns: int = 50) -> list[str]:
    """Heuristically infer datetime-like object columns."""
    datetime_columns: list[str] = []
    object_columns = df.select_dtypes(include=["object", "string"]).columns.tolist()

    for column in object_columns[:max_check_columns]:
        series = df[column].dropna().astype(str)

        if series.empty:
            continue

        sample = series.head(min(200, len(series)))

        try:
            parsed = pd.to_datetime(sample, errors="coerce", utc=False)
            parse_rate = float(parsed.notna().mean())
        except (TypeError, ValueError, OverflowError):
            continue

        if parse_rate >= 0.8:
            datetime_columns.append(column)

    return datetime_columns


def get_column_types(df: pd.DataFrame) -> dict[str, list[str]]:
    """Get column type groups."""
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = df.select_dtypes(
        include=["object", "category", "string"],
    ).columns.tolist()
    boolean_columns = df.select_dtypes(include=["bool"]).columns.tolist()
    native_datetime_columns = df.select_dtypes(
        include=["datetime64", "datetimetz"],
    ).columns.tolist()
    inferred_datetime_columns = infer_datetime_columns(df)

    datetime_columns = sorted(set(native_datetime_columns + inferred_datetime_columns))

    categorical_columns = [
        column for column in categorical_columns if column not in datetime_columns
    ]

    supported_columns = set(
        numeric_columns + categorical_columns + boolean_columns + datetime_columns,
    )
    unsupported_columns = [
        column for column in df.columns if column not in supported_columns
    ]

    return {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "boolean_columns": boolean_columns,
        "datetime_columns": datetime_columns,
        "unsupported_columns": unsupported_columns,
    }


def summarize_numeric_columns(
    df: pd.DataFrame,
    numeric_columns: list[str],
) -> dict[str, Any]:
    """Summarize numeric columns."""
    summary: dict[str, Any] = {}

    for column in numeric_columns:
        series = df[column].replace([np.inf, -np.inf], np.nan)
        valid = series.dropna()
        missing_count = int(series.isna().sum())

        if valid.empty:
            summary[column] = {
                "count": 0,
                "missing_count": missing_count,
                "missing_percent": safe_percent(missing_count, len(series)),
                "mean": None,
                "std": None,
                "min": None,
                "q25": None,
                "median": None,
                "q75": None,
                "max": None,
                "skewness": None,
                "kurtosis": None,
                "infinite_count": int(np.isinf(df[column]).sum()),
            }
            continue

        summary[column] = {
            "count": int(valid.count()),
            "missing_count": missing_count,
            "missing_percent": safe_percent(missing_count, len(series)),
            "mean": safe_round(valid.mean()),
            "std": safe_round(valid.std()),
            "min": safe_round(valid.min()),
            "q25": safe_round(valid.quantile(0.25)),
            "median": safe_round(valid.median()),
            "q75": safe_round(valid.quantile(0.75)),
            "max": safe_round(valid.max()),
            "skewness": safe_round(valid.skew()) if len(valid) > 2 else 0.0,
            "kurtosis": safe_round(valid.kurtosis()) if len(valid) > 3 else 0.0,
            "infinite_count": int(np.isinf(df[column]).sum()),
        }

    return summary


def summarize_categorical_columns(
    df: pd.DataFrame,
    categorical_columns: list[str],
) -> dict[str, Any]:
    """Summarize categorical columns."""
    max_categories = int(get_config_value("dataset.max_categories", 10))
    summary: dict[str, Any] = {}

    for column in categorical_columns:
        series = df[column]
        missing_count = int(series.isna().sum())
        unique_count = int(series.nunique(dropna=True))
        value_counts = series.value_counts(dropna=False).head(max_categories)

        top_values = [
            {
                "value": str(index),
                "count": int(count),
                "percent": safe_percent(int(count), len(series)),
            }
            for index, count in value_counts.items()
        ]

        summary[column] = {
            "count": int(series.count()),
            "missing_count": missing_count,
            "missing_percent": safe_percent(missing_count, len(series)),
            "unique_count": unique_count,
            "unique_percent": safe_percent(unique_count, len(series)),
            "top_values": top_values,
        }

    return summary


def summarize_datetime_columns(
    df: pd.DataFrame,
    datetime_columns: list[str],
) -> dict[str, Any]:
    """Summarize datetime columns."""
    summary: dict[str, Any] = {}

    for column in datetime_columns:
        parsed = pd.to_datetime(df[column], errors="coerce")
        valid = parsed.dropna()
        parse_success_count = int(parsed.notna().sum())

        summary[column] = {
            "parse_success_count": parse_success_count,
            "parse_success_percent": safe_percent(parse_success_count, len(parsed)),
            "missing_or_invalid_count": int(parsed.isna().sum()),
            "min": str(valid.min()) if not valid.empty else None,
            "max": str(valid.max()) if not valid.empty else None,
        }

    return summary


def summarize_target(df: pd.DataFrame, target_column: str) -> dict[str, Any]:
    """
    Summarize target column safely.
    """
    if target_column not in df.columns:
        return {
            "target_column": target_column,
            "exists": False,
            "message": "Target column not found.",
        }

    series = df[target_column]
    missing_count = int(series.isna().sum())
    unique_count = int(series.nunique(dropna=True))
    top_values = series.value_counts(dropna=False).head(20)

    return {
        "target_column": target_column,
        "exists": True,
        "dtype": str(series.dtype),
        "missing_count": missing_count,
        "missing_percent": safe_percent(missing_count, len(series)),
        "unique_count": unique_count,
        "unique_percent": safe_percent(unique_count, len(series)),
        "sample_values": [str(value) for value in series.dropna().unique()[:10]],
        "top_values": [
            {
                "value": str(index),
                "count": int(count),
                "percent": safe_percent(int(count), len(series)),
            }
            for index, count in top_values.items()
        ],
    }


def profile_dataset(
    df: pd.DataFrame,
    target_column: str | None = None,
) -> dict[str, Any]:
    """
    Generate deterministic dataset profile.
    """
    try:
        logger.info("Starting dataset profiling")

        if df is None or df.empty:
            raise InvalidDatasetError("Input dataframe is empty.")

        rows, columns = df.shape
        column_types = get_column_types(df)

        numeric_columns = column_types["numeric_columns"]
        categorical_columns = column_types["categorical_columns"]
        boolean_columns = column_types["boolean_columns"]
        datetime_columns = column_types["datetime_columns"]
        unsupported_columns = column_types["unsupported_columns"]

        duplicate_rows = int(df.duplicated().sum())
        missing_cells = int(df.isna().sum().sum())
        memory_usage_mb = safe_round(df.memory_usage(deep=True).sum() / (1024 * 1024))

        warnings: list[str] = []

        if rows < 50:
            warnings.append(
                "Dataset has fewer than 50 rows. Model evaluation may be unreliable."
            )

        if columns < 2:
            warnings.append("Dataset has fewer than 2 columns.")

        if duplicate_rows > 0:
            warnings.append(f"Dataset contains {duplicate_rows} duplicate rows.")

        if unsupported_columns:
            warnings.append(
                "Some unsupported columns were detected and may be dropped during "
                "preprocessing."
            )

        target_summary = (
            summarize_target(df, target_column)
            if target_column is not None
            else {"target_column": None, "exists": False}
        )

        profile: dict[str, Any] = {
            "shape": {
                "rows": int(rows),
                "columns": int(columns),
            },
            "memory_usage_mb": memory_usage_mb,
            "duplicate_rows": duplicate_rows,
            "duplicate_row_percent": safe_percent(duplicate_rows, rows),
            "column_types": column_types,
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "boolean_columns": boolean_columns,
            "datetime_columns": datetime_columns,
            "unsupported_columns": unsupported_columns,
            "missing_cells": missing_cells,
            "missing_cells_percent": safe_percent(missing_cells, rows * columns),
            "numeric_summary": summarize_numeric_columns(df, numeric_columns),
            "categorical_summary": summarize_categorical_columns(
                df,
                categorical_columns,
            ),
            "datetime_summary": summarize_datetime_columns(df, datetime_columns),
            "target_summary": target_summary,
            "warnings": warnings,
            "message": "Dataset profiling completed successfully.",
        }

        logger.info("Dataset profiling completed successfully")
        return profile

    except InvalidDatasetError:
        raise
    except (TypeError, ValueError, KeyError, AttributeError) as error:
        logger.exception("Dataset profiling failed.")
        raise InvalidDatasetError(
            "Dataset profiling failed.",
            error_detail=str(error),
        ) from error


if __name__ == "__main__":
    dataset_path = "data/sample/student_mark.csv"
    dataframe = load_dataset(dataset_path)
    print(profile_dataset(dataframe, target_column="Grade"))
