from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import get_config_value
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

    allowed_extensions = get_config_value("dataset.allowed_extensions", [".csv"])
    max_upload_mb = float(get_config_value("api.max_upload_mb", 25))

    if not path.exists():
        raise InvalidDatasetError(f"Dataset file not found: {path}")

    if path.suffix.lower() not in allowed_extensions:
        raise InvalidDatasetError(
            f"Unsupported file type '{path.suffix}'. Allowed: {allowed_extensions}"
        )

    file_size_mb = path.stat().st_size / (1024 * 1024)

    if file_size_mb > max_upload_mb:
        logger.warning(
            "Dataset file size %.2f MB exceeds configured upload limit %.2f MB",
            file_size_mb,
            max_upload_mb,
        )

    try:
        df = pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError as error:
        logger.exception("CSV file is empty.")
        raise InvalidDatasetError("CSV file is empty.") from error
    except pd.errors.ParserError as error:
        logger.exception("CSV file could not be parsed properly.")
        raise InvalidDatasetError(
            "CSV file could not be parsed properly.",
            error_detail=str(error),
        ) from error
    except Exception as error:
        logger.exception("Failed to read CSV file.")
        raise InvalidDatasetError(
            "Failed to read CSV file.",
            error_detail=str(error),
        ) from error

    if df.empty:
        raise InvalidDatasetError("Dataset is empty.")

    df.columns = [str(col).strip() for col in df.columns]

    if len(df.columns) != len(set(df.columns)):
        raise InvalidDatasetError("Dataset contains duplicate column names.")

    logger.info("Dataset loaded successfully: %s", path)
    logger.info("Dataset shape: %s", df.shape)

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

    boolean_columns = feature_df.select_dtypes(include=["bool"]).columns.tolist()

    numeric_columns = feature_df.select_dtypes(include=["number"]).columns.tolist()

    categorical_columns = feature_df.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    categorical_columns = [
        col for col in categorical_columns if col not in datetime_columns
    ]

    known_columns = set(
        numeric_columns + categorical_columns + datetime_columns + boolean_columns
    )

    other_columns = [
        column for column in feature_df.columns if column not in known_columns
    ]

    return {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": datetime_columns,
        "boolean_columns": boolean_columns,
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
    Return numeric statistics for numeric feature columns.
    """
    if not numeric_columns:
        return {}

    summary: dict[str, Any] = {}

    for column in numeric_columns:
        series = df[column].dropna()

        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        summary[column] = {
            "mean": float(round(series.mean(), 4)),
            "median": float(round(series.median(), 4)),
            "std": float(round(series.std(), 4)) if len(series) > 1 else 0.0,
            "min": float(series.min()),
            "max": float(series.max()),
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(round(iqr, 4)),
            "skewness": float(round(series.skew(), 4)) if len(series) > 2 else 0.0,
            "kurtosis": float(round(series.kurtosis(), 4)) if len(series) > 3 else 0.0,
        }

    return summary


def get_categorical_summary(
    df: pd.DataFrame,
    categorical_columns: list[str],
    max_categories: int | None = None,
) -> dict[str, Any]:
    """
    Return top category distribution for categorical columns.
    """
    if max_categories is None:
        max_categories = int(get_config_value("dataset.max_categories", 10))

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


def get_datetime_summary(
    df: pd.DataFrame,
    datetime_columns: list[str],
) -> dict[str, Any]:
    """
    Return min date, max date and valid parse percentage for datetime columns.
    """
    summary: dict[str, Any] = {}

    for column in datetime_columns:
        parsed = pd.to_datetime(df[column], errors="coerce", format="mixed")

        valid_count = int(parsed.notna().sum())
        valid_percent = float(round(parsed.notna().mean() * 100, 2))

        if valid_count == 0:
            summary[column] = {
                "valid_datetime_count": 0,
                "valid_datetime_percent": 0.0,
                "min_date": None,
                "max_date": None,
            }
            continue

        summary[column] = {
            "valid_datetime_count": valid_count,
            "valid_datetime_percent": valid_percent,
            "min_date": str(parsed.min()),
            "max_date": str(parsed.max()),
        }

    return summary


def get_target_summary(df: pd.DataFrame, target_column: str) -> dict[str, Any]:
    """
    Build a summary of the target column.
    """
    target_series = df[target_column]
    unique_count = target_series.nunique(dropna=True)

    summary: dict[str, Any] = {
        "target_column": target_column,
        "missing_count": int(target_series.isna().sum()),
        "missing_percent": float(round(target_series.isna().mean() * 100, 2)),
        "unique_values": int(unique_count),
        "dtype": str(target_series.dtype),
    }

    if pd.api.types.is_numeric_dtype(target_series):
        clean_target = target_series.dropna()

        if not clean_target.empty:
            summary["numeric_summary"] = {
                "mean": float(round(clean_target.mean(), 4)),
                "median": float(round(clean_target.median(), 4)),
                "std": float(round(clean_target.std(), 4))
                if len(clean_target) > 1
                else 0.0,
                "min": float(clean_target.min()),
                "max": float(clean_target.max()),
            }

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


def detect_constant_columns(df: pd.DataFrame) -> list[str]:
    """
    Detect columns with only one unique non-null value.
    """
    return [
        column
        for column in df.columns
        if df[column].nunique(dropna=True) <= 1
    ]


def detect_high_cardinality_columns(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Detect categorical columns with high unique value count.
    """
    threshold = int(get_config_value("audit.high_cardinality_threshold", 50))
    results: list[dict[str, Any]] = []

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            continue

        unique_count = int(df[column].nunique(dropna=True))

        if unique_count >= threshold:
            results.append(
                {
                    "column": column,
                    "unique_values": unique_count,
                    "threshold": threshold,
                }
            )

    return results


def detect_near_constant_columns(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Detect columns where one value dominates the column.
    """
    threshold = float(get_config_value("audit.near_constant_threshold", 95))
    results: list[dict[str, Any]] = []

    for column in df.columns:
        series = df[column].dropna()

        if series.empty:
            continue

        top_percent = float(round(series.value_counts(normalize=True).iloc[0] * 100, 2))

        if top_percent >= threshold:
            results.append(
                {
                    "column": column,
                    "top_value_percent": top_percent,
                    "threshold": threshold,
                }
            )

    return results


def detect_identifier_columns(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Detect likely ID columns using uniqueness ratio and column name.
    """
    threshold = float(get_config_value("audit.id_unique_percent_threshold", 95))
    id_keywords = ["id", "uuid", "key", "identifier"]

    results: list[dict[str, Any]] = []

    for column in df.columns:
        series = df[column].dropna()

        if series.empty:
            continue

        unique_percent = float(round(series.nunique(dropna=True) / len(series) * 100, 2))
        name_matches = any(keyword in column.lower() for keyword in id_keywords)

        if unique_percent >= threshold or name_matches:
            results.append(
                {
                    "column": column,
                    "unique_percent": unique_percent,
                    "name_matches_id_pattern": name_matches,
                    "threshold": threshold,
                }
            )

    return results


def detect_null_only_columns(df: pd.DataFrame) -> list[str]:
    """
    Detect columns containing only missing values.
    """
    return [column for column in df.columns if df[column].isna().all()]


def get_memory_usage_mb(df: pd.DataFrame) -> float:
    """
    Return dataframe memory usage in MB.
    """
    memory_bytes = df.memory_usage(deep=True).sum()
    return float(round(memory_bytes / (1024 * 1024), 4))


def build_profile_warnings(df: pd.DataFrame) -> list[str]:
    """
    Build human-readable profiling warnings.
    """
    warnings: list[str] = []

    max_rows = int(get_config_value("dataset.max_rows", 1_000_000))
    high_missing_threshold = float(
        get_config_value("audit.high_missing_threshold", 50)
    )

    if len(df) > max_rows:
        warnings.append(
            f"Dataset has {len(df)} rows, which exceeds configured max_rows={max_rows}."
        )

    missing_percent = df.isna().mean() * 100

    for column, percent in missing_percent.items():
        if percent >= high_missing_threshold:
            warnings.append(
                f"Column '{column}' has high missing values: {round(percent, 2)}%."
            )

    for column in detect_null_only_columns(df):
        warnings.append(f"Column '{column}' contains only missing values.")

    for column in detect_constant_columns(df):
        warnings.append(f"Column '{column}' is constant or has only one unique value.")

    for item in detect_near_constant_columns(df):
        warnings.append(
            f"Column '{item['column']}' is near-constant "
            f"({item['top_value_percent']}% same value)."
        )

    for item in detect_identifier_columns(df):
        warnings.append(
            f"Column '{item['column']}' looks like an identifier "
            f"({item['unique_percent']}% unique)."
        )

    for item in detect_high_cardinality_columns(df):
        warnings.append(
            f"Column '{item['column']}' has high cardinality "
            f"({item['unique_values']} unique values)."
        )

    return warnings


def profile_dataset(df: pd.DataFrame, target_column: str) -> dict[str, Any]:
    """
    Generate a deterministic profile of the dataset before modeling.
    """
    validate_target_column(df, target_column)

    column_types = get_column_types(df, target_column)

    numeric_columns = column_types["numeric_columns"]
    categorical_columns = column_types["categorical_columns"]
    datetime_columns = column_types["datetime_columns"]

    profile: dict[str, Any] = {
        "metadata": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "audit_version": get_config_value("project.version", "unknown"),
            "pandas_version": pd.__version__,
        },
        "shape": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
        },
        "memory_usage_mb": get_memory_usage_mb(df),
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
        "datetime_columns": datetime_columns,
        "boolean_columns": column_types["boolean_columns"],
        "other_columns": column_types["other_columns"],
        "column_types": column_types,
        "numeric_summary": get_numeric_summary(df, numeric_columns),
        "categorical_summary": get_categorical_summary(df, categorical_columns),
        "datetime_summary": get_datetime_summary(df, datetime_columns),
        "target_summary": get_target_summary(df, target_column),
        "constant_columns": detect_constant_columns(df),
        "near_constant_columns": detect_near_constant_columns(df),
        "high_cardinality_columns": detect_high_cardinality_columns(df),
        "identifier_columns": detect_identifier_columns(df),
        "null_only_columns": detect_null_only_columns(df),
        "problem_type_hint": infer_problem_hint(df, target_column),
        "warnings": build_profile_warnings(df),
    }

    logger.info("Dataset profiling completed")
    return profile


if __name__ == "__main__":
    dataset_path = "data/sample/student_mark.csv"
    target_column = "Grade"

    df = load_dataset(dataset_path)
    profile = profile_dataset(df, target_column)

    print(profile)