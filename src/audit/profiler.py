from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import get_config_value
from src.utils.exceptions import InvalidDatasetError
from src.utils.logger import get_logger

logger = get_logger(__name__)

_TRUE_VALUES = {"true", "1", "yes", "y", "on"}
_FALSE_VALUES = {"false", "0", "no", "n", "off"}
_DEFAULT_ALLOWED_EXTENSIONS = [".csv"]
_DEFAULT_MAX_ROWS = 1_000_000
_DEFAULT_MAX_FILE_SIZE_MB = 500.0
_DEFAULT_RANDOM_SEED = 42
_DEFAULT_MAX_CATEGORIES = 10
_DEFAULT_DATETIME_PARSE_THRESHOLD = 0.8
_DEFAULT_DATETIME_SAMPLE_SIZE = 200
_DEFAULT_MAX_DATETIME_CHECK_COLUMNS = 50


def as_bool(value: Any) -> bool:
    """Convert config values safely into boolean."""
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_VALUES:
            return True
        if normalized in _FALSE_VALUES:
            return False

    return bool(value)


def get_int_config(path: str, default: int, minimum: int | None = None) -> int:
    """Read integer config with safe fallback and optional lower bound."""
    try:
        value = int(get_config_value(path, default))
    except (KeyError, TypeError, ValueError, RuntimeError):
        logger.warning(
            "Invalid integer config for %s. Using default=%s.", path, default
        )
        value = int(default)

    if minimum is not None:
        return max(minimum, value)

    return value


def get_float_config(path: str, default: float, minimum: float | None = None) -> float:
    """Read float config with safe fallback and optional lower bound."""
    try:
        value = float(get_config_value(path, default))
    except (KeyError, TypeError, ValueError, RuntimeError):
        logger.warning("Invalid float config for %s. Using default=%s.", path, default)
        value = float(default)

    if not np.isfinite(value):
        logger.warning(
            "Non-finite float config for %s. Using default=%s.", path, default
        )
        value = float(default)

    if minimum is not None:
        return max(minimum, value)

    return value


def get_list_config(path: str, default: list[str]) -> list[str]:
    """Read list-like config safely.

    Supports real lists/tuples/sets and comma-separated strings. This prevents
    accidental character-by-character parsing when YAML/env config returns a
    single string such as ".csv,.tsv".
    """
    try:
        value = get_config_value(path, default)
    except (KeyError, TypeError, ValueError, RuntimeError):
        logger.warning("Invalid list config for %s. Using default=%s.", path, default)
        return default

    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
        return [item for item in items if item] or default

    if isinstance(value, dict):
        logger.warning("Invalid list config for %s. Using default=%s.", path, default)
        return default

    if isinstance(value, Iterable):
        items = [str(item).strip() for item in value]
        return [item for item in items if item] or default

    logger.warning("Invalid list config for %s. Using default=%s.", path, default)
    return default


def safe_round(value: Any, digits: int = 4, default: float = 0.0) -> float:
    """Safely round pandas/numpy scalar values."""
    try:
        if value is None:
            return default

        if pd.isna(value):
            return default

        numeric_value = float(value)
        if not np.isfinite(numeric_value):
            return default

        return round(numeric_value, digits)
    except (TypeError, ValueError):
        return default


def safe_percent(numerator: float, denominator: float) -> float:
    """Return percentage safely rounded to 2 decimals."""
    try:
        denominator_value = float(denominator)
        if denominator_value <= 0:
            return 0.0

        numerator_value = float(numerator)
        if not np.isfinite(numerator_value) or not np.isfinite(denominator_value):
            return 0.0

        return round((numerator_value / denominator_value) * 100, 2)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def value_to_json_safe_string(value: Any) -> str | None:
    """Convert pandas/numpy values into JSON-safe display strings."""
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    return str(value)


def get_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a dataframe column as Series with duplicate-column protection."""
    series = df.loc[:, column]

    if isinstance(series, pd.DataFrame):
        raise InvalidDatasetError(f"Column '{column}' resolved to multiple columns.")

    return series


def nunique_safely(series: pd.Series, dropna: bool = True) -> int:
    """Return unique count safely for object columns."""
    try:
        return int(series.nunique(dropna=dropna))
    except (TypeError, ValueError):
        return int(series.astype("string").nunique(dropna=dropna))


def value_counts_safely(series: pd.Series, dropna: bool = False) -> pd.Series:
    """Return value counts safely for object columns."""
    try:
        return series.value_counts(dropna=dropna)
    except (TypeError, ValueError):
        return series.astype("string").value_counts(dropna=dropna)


def duplicate_rows_safely(df: pd.DataFrame) -> int:
    """Count duplicate rows safely."""
    try:
        return int(df.duplicated().sum())
    except (TypeError, ValueError):
        safe_df = df.astype("string")
        return int(safe_df.duplicated().sum())


def normalize_allowed_extensions(extensions: list[str]) -> set[str]:
    """Normalize configured file extensions."""
    normalized: set[str] = set()

    for extension in extensions:
        ext = str(extension).strip().lower()
        if not ext:
            continue
        normalized.add(ext if ext.startswith(".") else f".{ext}")

    return normalized or set(_DEFAULT_ALLOWED_EXTENSIONS)


def validate_dataset_path(dataset_path: str | Path) -> Path:
    """Validate dataset path and extension."""
    if dataset_path is None or not str(dataset_path).strip():
        raise InvalidDatasetError("Dataset path is required.")

    path = Path(dataset_path).expanduser()

    if not path.exists():
        raise InvalidDatasetError(f"Dataset file not found: {path}")

    if not path.is_file():
        raise InvalidDatasetError(f"Dataset path is not a file: {path}")

    allowed_extensions = get_list_config(
        "dataset.allowed_extensions",
        _DEFAULT_ALLOWED_EXTENSIONS,
    )
    allowed = normalize_allowed_extensions(allowed_extensions)

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
    read_kwargs = dict(kwargs)
    configured_encoding = read_kwargs.pop("encoding", None)

    encodings: list[str | None]
    if configured_encoding:
        encodings = [str(configured_encoding), "utf-8-sig", "latin1"]
    else:
        encodings = [None, "utf-8-sig", "latin1"]

    seen_encodings: set[str | None] = set()
    last_unicode_error: UnicodeDecodeError | None = None

    for encoding in encodings:
        if encoding in seen_encodings:
            continue
        seen_encodings.add(encoding)

        try:
            if encoding is None:
                return pd.read_csv(path, **read_kwargs)
            return pd.read_csv(path, encoding=encoding, **read_kwargs)
        except UnicodeDecodeError as error:
            last_unicode_error = error
            logger.warning(
                "CSV read failed due to encoding issue. path=%s encoding=%s",
                path,
                encoding or "default",
            )
            continue
        except pd.errors.EmptyDataError as error:
            raise InvalidDatasetError("CSV file is empty or has no columns.") from error
        except pd.errors.ParserError as error:
            raise InvalidDatasetError(
                "CSV parsing failed. File may be corrupt or malformed.",
                error_detail=str(error),
            ) from error

    if last_unicode_error is not None:
        raise InvalidDatasetError(
            "CSV decoding failed for supported encodings.",
            error_detail=str(last_unicode_error),
        ) from last_unicode_error

    raise InvalidDatasetError("CSV loading failed for an unknown reason.")


def validate_loaded_dataframe(df: pd.DataFrame) -> None:
    """Validate dataframe after loading."""
    if df.empty:
        raise InvalidDatasetError("Loaded dataset is empty.")

    if df.columns.empty:
        raise InvalidDatasetError("Loaded dataset has no columns.")

    duplicate_columns = df.columns[df.columns.duplicated()].tolist()
    if duplicate_columns:
        raise InvalidDatasetError(f"Duplicate column names found: {duplicate_columns}")

    blank_columns = [column for column in df.columns if not str(column).strip()]
    if blank_columns:
        raise InvalidDatasetError("Dataset contains blank column names.")


def load_dataset(dataset_path: str | Path) -> pd.DataFrame:
    """
    Load tabular dataset.

    For normal files, the full CSV is loaded.
    For very large files, only max_rows are loaded when sampling is enabled.
    """
    try:
        path = validate_dataset_path(dataset_path)

        max_rows = get_int_config("dataset.max_rows", _DEFAULT_MAX_ROWS, minimum=1)
        max_file_size_mb = get_float_config(
            "dataset.max_file_size_mb",
            _DEFAULT_MAX_FILE_SIZE_MB,
            minimum=0.0,
        )
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

            random_state = get_int_config("random_seed", _DEFAULT_RANDOM_SEED)
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


def parse_datetime_series(series: pd.Series) -> pd.Series:
    """Parse a series as datetime with a pandas-version-safe fallback."""
    try:
        return pd.to_datetime(series, errors="coerce", utc=False, format="mixed")
    except (TypeError, ValueError):
        return pd.to_datetime(series, errors="coerce", utc=False)


def infer_datetime_columns(df: pd.DataFrame, max_check_columns: int = 50) -> list[str]:
    """Heuristically infer datetime-like object columns."""
    datetime_columns: list[str] = []
    object_columns = df.select_dtypes(include=["object", "string"]).columns.tolist()

    max_columns = max(1, int(max_check_columns))
    sample_size = get_int_config(
        "dataset.datetime_sample_size",
        _DEFAULT_DATETIME_SAMPLE_SIZE,
        minimum=10,
    )
    parse_threshold = get_float_config(
        "dataset.datetime_parse_threshold",
        _DEFAULT_DATETIME_PARSE_THRESHOLD,
        minimum=0.0,
    )
    parse_threshold = min(1.0, parse_threshold)

    for column in object_columns[:max_columns]:
        series = df[column].dropna().astype(str).str.strip()
        series = series[series != ""]

        if series.empty:
            continue

        sample = series.head(min(sample_size, len(series)))

        try:
            parsed = parse_datetime_series(sample)
            parse_rate = float(parsed.notna().mean())
        except (TypeError, ValueError, OverflowError):
            continue

        if parse_rate >= parse_threshold:
            datetime_columns.append(str(column))

    return datetime_columns


def get_column_types(df: pd.DataFrame) -> dict[str, list[str]]:
    """Get column type groups."""
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = df.select_dtypes(
        include=["object", "category", "string"],
    ).columns.tolist()
    boolean_columns = df.select_dtypes(include=["bool", "boolean"]).columns.tolist()
    native_datetime_columns = df.select_dtypes(
        include=["datetime64", "datetimetz"],
    ).columns.tolist()
    inferred_datetime_columns = infer_datetime_columns(
        df,
        max_check_columns=get_int_config(
            "dataset.max_datetime_check_columns",
            _DEFAULT_MAX_DATETIME_CHECK_COLUMNS,
            minimum=1,
        ),
    )

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
        "numeric_columns": [str(column) for column in numeric_columns],
        "categorical_columns": [str(column) for column in categorical_columns],
        "boolean_columns": [str(column) for column in boolean_columns],
        "datetime_columns": [str(column) for column in datetime_columns],
        "unsupported_columns": [str(column) for column in unsupported_columns],
    }


def count_infinite_values(series: pd.Series) -> int:
    """Count infinite values in a numeric-like series safely."""
    try:
        numeric = pd.to_numeric(series, errors="coerce")
        return int(np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).sum())
    except (TypeError, ValueError):
        return 0


def summarize_numeric_columns(
    df: pd.DataFrame,
    numeric_columns: list[str],
) -> dict[str, Any]:
    """Summarize numeric columns."""
    summary: dict[str, Any] = {}

    for column in numeric_columns:
        raw_series = get_series(df, column)
        infinite_count = count_infinite_values(raw_series)
        series = pd.to_numeric(raw_series, errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        )
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
                "infinite_count": infinite_count,
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
            "infinite_count": infinite_count,
        }

    return summary


def summarize_categorical_columns(
    df: pd.DataFrame,
    categorical_columns: list[str],
) -> dict[str, Any]:
    """Summarize categorical columns."""
    max_categories = get_int_config(
        "dataset.max_categories",
        _DEFAULT_MAX_CATEGORIES,
        minimum=1,
    )
    summary: dict[str, Any] = {}

    for column in categorical_columns:
        series = get_series(df, column)
        missing_count = int(series.isna().sum())
        unique_count = nunique_safely(series, dropna=True)
        value_counts = value_counts_safely(series, dropna=False).head(max_categories)

        top_values = [
            {
                "value": value_to_json_safe_string(index),
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


def summarize_boolean_columns(
    df: pd.DataFrame,
    boolean_columns: list[str],
) -> dict[str, Any]:
    """Summarize boolean columns without changing existing type-group outputs."""
    summary: dict[str, Any] = {}

    for column in boolean_columns:
        series = get_series(df, column)
        missing_count = int(series.isna().sum())
        value_counts = value_counts_safely(series, dropna=False)

        summary[column] = {
            "count": int(series.count()),
            "missing_count": missing_count,
            "missing_percent": safe_percent(missing_count, len(series)),
            "true_count": int((series == True).sum()),  # noqa: E712
            "false_count": int((series == False).sum()),  # noqa: E712
            "top_values": [
                {
                    "value": value_to_json_safe_string(index),
                    "count": int(count),
                    "percent": safe_percent(int(count), len(series)),
                }
                for index, count in value_counts.items()
            ],
        }

    return summary


def summarize_datetime_columns(
    df: pd.DataFrame,
    datetime_columns: list[str],
) -> dict[str, Any]:
    """Summarize datetime columns."""
    summary: dict[str, Any] = {}

    for column in datetime_columns:
        parsed = parse_datetime_series(get_series(df, column))
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
    """Summarize target column safely."""
    if target_column not in df.columns:
        return {
            "target_column": target_column,
            "exists": False,
            "message": "Target column not found.",
        }

    series = get_series(df, target_column)
    missing_count = int(series.isna().sum())
    unique_count = nunique_safely(series, dropna=True)
    top_values = value_counts_safely(series, dropna=False).head(20)

    return {
        "target_column": target_column,
        "exists": True,
        "dtype": str(series.dtype),
        "missing_count": missing_count,
        "missing_percent": safe_percent(missing_count, len(series)),
        "unique_count": unique_count,
        "unique_percent": safe_percent(unique_count, len(series)),
        "is_numeric": bool(pd.api.types.is_numeric_dtype(series)),
        "is_boolean": bool(pd.api.types.is_bool_dtype(series)),
        "sample_values": [
            value_to_json_safe_string(value) for value in series.dropna().unique()[:10]
        ],
        "top_values": [
            {
                "value": value_to_json_safe_string(index),
                "count": int(count),
                "percent": safe_percent(int(count), len(series)),
            }
            for index, count in top_values.items()
        ],
    }


def build_profile_warnings(
    rows: int,
    columns: int,
    duplicate_rows: int,
    unsupported_columns: list[str],
    target_summary: dict[str, Any],
) -> list[str]:
    """Build deterministic profile warnings for UI/report usage."""
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

    if not target_summary.get("exists", False):
        warnings.append("Target column was not found in the dataset.")
    elif target_summary.get("missing_count", 0) > 0:
        warnings.append("Target column contains missing values.")

    return warnings


def profile_dataset(
    df: pd.DataFrame,
    target_column: str | None = None,
) -> dict[str, Any]:
    """Generate deterministic dataset profile."""
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

        duplicate_rows = duplicate_rows_safely(df)
        missing_cells = int(df.isna().sum().sum())
        memory_usage_mb = safe_round(df.memory_usage(deep=True).sum() / (1024 * 1024))

        normalized_target_column = (
            target_column.strip() if isinstance(target_column, str) else target_column
        )
        target_summary = (
            summarize_target(df, normalized_target_column)
            if normalized_target_column is not None
            else {"target_column": None, "exists": False}
        )
        warnings = build_profile_warnings(
            rows=rows,
            columns=columns,
            duplicate_rows=duplicate_rows,
            unsupported_columns=unsupported_columns,
            target_summary=target_summary,
        )

        profile: dict[str, Any] = {
            "shape": {
                "rows": int(rows),
                "columns": int(columns),
            },
            "row_count": int(rows),
            "column_count": int(columns),
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
            "boolean_summary": summarize_boolean_columns(df, boolean_columns),
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
