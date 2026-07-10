from __future__ import annotations

import math
import re
from difflib import get_close_matches
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.utils.config import get_config_value
from src.utils.exceptions import PreprocessingError
from src.utils.logger import get_logger

logger = get_logger(__name__)

CLASSIFICATION_TYPES = {"binary_classification", "multiclass_classification"}
TRUE_VALUES = {"true", "1", "yes", "y", "on"}
FALSE_VALUES = {"false", "0", "no", "n", "off", "none", "null", ""}

NUMERIC_IMPUTER_STRATEGIES = {"mean", "median", "most_frequent", "constant"}
CATEGORICAL_IMPUTER_STRATEGIES = {"most_frequent", "constant"}
DATE_PATTERN = re.compile(
    r"(\d{1,4}[-/]\d{1,2}[-/]\d{1,4})|(\d{1,2}:\d{2})", re.IGNORECASE
)


def as_bool(value: Any) -> bool:
    """Convert config values safely into boolean."""
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in TRUE_VALUES:
            return True
        if normalized in FALSE_VALUES:
            return False

    return bool(value)


def get_int_config_value(
    key_path: str,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Read integer config values with safe fallback and optional bounds."""
    try:
        value = int(get_config_value(key_path, default))
    except (KeyError, TypeError, ValueError, RuntimeError):
        logger.warning(
            "Invalid integer config for %s. Using default=%s", key_path, default
        )
        value = int(default)

    if minimum is not None and value < minimum:
        logger.warning(
            "Integer config %s=%s is below minimum=%s. Using default=%s",
            key_path,
            value,
            minimum,
            default,
        )
        value = default

    if maximum is not None and value > maximum:
        logger.warning(
            "Integer config %s=%s is above maximum=%s. Using default=%s",
            key_path,
            value,
            maximum,
            default,
        )
        value = default

    return value


def get_float_config_value(
    key_path: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Read float config values with safe fallback and optional bounds."""
    try:
        value = float(get_config_value(key_path, default))
    except (KeyError, TypeError, ValueError, RuntimeError):
        logger.warning(
            "Invalid float config for %s. Using default=%s", key_path, default
        )
        value = float(default)

    if not math.isfinite(value):
        logger.warning(
            "Non-finite float config for %s. Using default=%s", key_path, default
        )
        value = float(default)

    if minimum is not None and value < minimum:
        logger.warning(
            "Float config %s=%s is below minimum=%s. Using default=%s",
            key_path,
            value,
            minimum,
            default,
        )
        value = default

    if maximum is not None and value > maximum:
        logger.warning(
            "Float config %s=%s is above maximum=%s. Using default=%s",
            key_path,
            value,
            maximum,
            default,
        )
        value = default

    return value


def get_string_config_value(
    key_path: str,
    default: str,
    allowed_values: set[str] | None = None,
) -> str:
    """Read string config values with safe fallback and optional allow-list."""
    try:
        raw_value = get_config_value(key_path, default)
    except (KeyError, TypeError, ValueError, RuntimeError):
        raw_value = default

    value = str(raw_value).strip().lower()

    if allowed_values is not None and value not in allowed_values:
        logger.warning(
            "Invalid string config for %s=%s. Allowed=%s. Using default=%s",
            key_path,
            value,
            sorted(allowed_values),
            default,
        )
        return default

    return value


def resolve_target_column(df: pd.DataFrame, target_column: str) -> str:
    """Resolve target column safely while tolerating accidental whitespace."""
    if target_column is None or not str(target_column).strip():
        raise PreprocessingError("Target column is required.")

    normalized_target = str(target_column).strip()

    if normalized_target in df.columns:
        return normalized_target

    stripped_lookup = {str(column).strip(): str(column) for column in df.columns}
    if normalized_target in stripped_lookup:
        resolved = stripped_lookup[normalized_target]
        logger.warning(
            "Target column resolved after trimming whitespace: requested=%s resolved=%s",
            target_column,
            resolved,
        )
        return resolved

    close_matches = get_close_matches(
        normalized_target, [str(column) for column in df.columns], n=3
    )
    suggestion = f" Did you mean one of {close_matches}?" if close_matches else ""
    raise PreprocessingError(
        f"Target column not found: {normalized_target}.{suggestion}"
    )


def validate_preprocessing_inputs(df: pd.DataFrame, target_column: str) -> str:
    """Validate preprocessing inputs before building pipelines or splitting data."""
    if df is None or df.empty:
        raise PreprocessingError("Input dataframe is empty.")

    if not isinstance(df, pd.DataFrame):
        raise PreprocessingError("Input must be a pandas DataFrame.")

    resolved_target = resolve_target_column(df, target_column)

    if len(df.columns) <= 1:
        raise PreprocessingError("Dataset must contain at least one feature column.")

    duplicate_columns = df.columns[df.columns.duplicated()].astype(str).tolist()
    if duplicate_columns:
        raise PreprocessingError(f"Duplicate column names found: {duplicate_columns}")

    return resolved_target


def clean_infinite_values(df: pd.DataFrame) -> pd.DataFrame:
    """Replace np.inf and -np.inf with NaN before imputation/modeling."""
    cleaned = df.copy()
    numeric_columns = cleaned.select_dtypes(include=["number"]).columns

    if len(numeric_columns) > 0:
        cleaned[numeric_columns] = cleaned[numeric_columns].replace(
            [np.inf, -np.inf],
            np.nan,
        )

    return cleaned


def get_target_series(df: pd.DataFrame, target_column: str) -> pd.Series:
    """Return target column as Series with duplicate-column protection."""
    target = df.loc[:, target_column]

    if isinstance(target, pd.DataFrame):
        raise PreprocessingError(
            f"Target column '{target_column}' resolved to multiple columns.",
        )

    return target


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


def is_string_like_series(series: pd.Series) -> bool:
    """
    Return True for string/categorical columns.

    Numeric columns can also be highly unique, but that does not make them IDs.
    This prevents valid numeric features like income, sales, or attendance from
    being dropped just because they have many unique values.
    """
    dtype = series.dtype
    return bool(
        pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
        or isinstance(dtype, pd.CategoricalDtype)
    )


def column_name_suggests_id(column: str) -> bool:
    """
    Detect identifier-like column names using safer token matching.

    Avoid broad substring mistakes such as dropping "income" only because it
    contains "id" somewhere in the text.
    """
    normalized = (
        str(column)
        .lower()
        .strip()
        .replace("-", "_")
        .replace(" ", "_")
        .replace(".", "_")
    )

    exact_id_names = {
        "id",
        "uuid",
        "guid",
        "identifier",
        "serial",
        "email",
        "phone",
        "mobile",
        "zipcode",
        "zip",
        "roll",
        "roll_no",
        "student_id",
        "customer_id",
        "user_id",
        "record_id",
    }

    if normalized in exact_id_names:
        return True

    id_suffixes = (
        "_id",
        "_uuid",
        "_guid",
        "_identifier",
        "_serial",
        "_email",
        "_phone",
        "_mobile",
        "_zipcode",
        "_zip",
        "_roll",
        "_roll_no",
    )

    return normalized.endswith(id_suffixes)


def column_name_suggests_datetime(column: str) -> bool:
    """Return True when a column name strongly suggests date/time semantics."""
    normalized = (
        str(column)
        .lower()
        .strip()
        .replace("-", "_")
        .replace(" ", "_")
        .replace(".", "_")
    )

    tokens = set(normalized.split("_"))
    datetime_tokens = {
        "date",
        "time",
        "datetime",
        "timestamp",
        "created",
        "updated",
        "modified",
        "month",
        "year",
    }

    return bool(tokens & datetime_tokens) or normalized.endswith(
        ("_date", "_time", "_at")
    )


def detect_id_like_columns(features: pd.DataFrame) -> list[str]:
    """
    Detect ID-like columns that should usually be dropped before modeling.

    Important:
    - Name-based ID detection applies to all dtypes.
    - High-uniqueness detection applies only to string/categorical columns.
    """
    id_unique_percent_threshold = get_float_config_value(
        "audit.id_unique_percent_threshold",
        95.0,
        minimum=1.0,
        maximum=100.0,
    )

    id_like_columns: list[str] = []

    for column in features.columns:
        series = features[column]
        unique_count = nunique_safely(series, dropna=True)
        unique_percent = (unique_count / len(features)) * 100 if len(features) else 0.0

        name_suggests_id = column_name_suggests_id(str(column))
        uniqueness_suggests_id = (
            is_string_like_series(series)
            and unique_percent >= id_unique_percent_threshold
            and unique_count > 1
        )

        if name_suggests_id or uniqueness_suggests_id:
            id_like_columns.append(str(column))

    return id_like_columns


def detect_high_cardinality_columns(features: pd.DataFrame) -> list[str]:
    """Detect high-cardinality categorical columns that can explode one-hot encoding."""
    high_cardinality_threshold = get_int_config_value(
        "audit.high_cardinality_threshold",
        50,
        minimum=2,
    )

    categorical_columns = features.select_dtypes(
        include=["object", "category", "string"],
    ).columns

    high_cardinality_columns: list[str] = []

    for column in categorical_columns:
        unique_count = nunique_safely(features[column], dropna=True)

        if unique_count >= high_cardinality_threshold:
            high_cardinality_columns.append(str(column))

    return high_cardinality_columns


def infer_datetime_feature_columns(features: pd.DataFrame) -> list[str]:
    """Infer datetime-like string columns conservatively for preprocessing."""
    infer_datetime = as_bool(
        get_config_value("preprocessing.infer_datetime_columns", True)
    )
    if not infer_datetime:
        return []

    max_check_columns = get_int_config_value(
        "preprocessing.datetime_max_check_columns",
        50,
        minimum=1,
    )
    sample_size = get_int_config_value(
        "preprocessing.datetime_sample_size",
        200,
        minimum=20,
    )
    parse_threshold = get_float_config_value(
        "preprocessing.datetime_parse_threshold",
        0.8,
        minimum=0.1,
        maximum=1.0,
    )

    candidate_columns = features.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()
    inferred_columns: list[str] = []

    for column in candidate_columns[:max_check_columns]:
        series = features[column].dropna().astype(str)
        if series.empty:
            continue

        sample = series.head(min(sample_size, len(series)))
        sample_joined = " ".join(sample.head(20).tolist())
        has_date_like_values = bool(DATE_PATTERN.search(sample_joined))
        name_suggests_datetime = column_name_suggests_datetime(str(column))

        if not has_date_like_values and not name_suggests_datetime:
            continue

        try:
            parsed = pd.to_datetime(sample, errors="coerce", utc=False, format="mixed")
        except (TypeError, ValueError, OverflowError):
            continue

        parse_rate = float(parsed.notna().mean())
        if parse_rate >= parse_threshold:
            inferred_columns.append(str(column))

    return inferred_columns


def get_feature_columns(df: pd.DataFrame, target_column: str) -> dict[str, list[str]]:
    """
    Identify feature column groups for preprocessing.

    High-cardinality and ID-like columns can optionally be dropped by config.
    """
    resolved_target = validate_preprocessing_inputs(df, target_column)
    features = df.drop(columns=[resolved_target]).copy()

    drop_id_like = as_bool(get_config_value("preprocessing.drop_id_like_columns", True))
    drop_high_cardinality = as_bool(
        get_config_value("preprocessing.drop_high_cardinality_columns", False),
    )

    id_like_columns = detect_id_like_columns(features) if drop_id_like else []
    high_cardinality_columns = (
        detect_high_cardinality_columns(features) if drop_high_cardinality else []
    )

    columns_to_drop = sorted(set(id_like_columns + high_cardinality_columns))
    features_for_model = features.drop(columns=columns_to_drop, errors="ignore")

    native_datetime_columns = features_for_model.select_dtypes(
        include=["datetime", "datetimetz"],
    ).columns.tolist()
    inferred_datetime_columns = infer_datetime_feature_columns(features_for_model)
    datetime_columns = sorted(
        set(map(str, native_datetime_columns + inferred_datetime_columns))
    )

    numeric_columns = features_for_model.select_dtypes(
        include=["number"],
    ).columns.tolist()

    categorical_columns = features_for_model.select_dtypes(
        include=["object", "category", "string", "bool"],
    ).columns.tolist()
    categorical_columns = [
        str(column)
        for column in categorical_columns
        if str(column) not in datetime_columns
    ]

    numeric_columns = [str(column) for column in numeric_columns]
    known_columns = set(numeric_columns + categorical_columns + datetime_columns)
    unsupported_columns = [
        str(column)
        for column in features_for_model.columns
        if str(column) not in known_columns
    ]

    return {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": datetime_columns,
        "unsupported_columns": unsupported_columns,
        "id_like_columns_dropped": id_like_columns,
        "high_cardinality_columns_dropped": high_cardinality_columns,
        "columns_dropped_before_modeling": columns_to_drop,
    }


def extract_datetime_features(datetime_df: pd.DataFrame) -> pd.DataFrame:
    """Convert datetime columns into numeric calendar features."""
    output = pd.DataFrame(index=datetime_df.index)

    for column in datetime_df.columns:
        series = pd.to_datetime(datetime_df[column], errors="coerce", format="mixed")

        output[f"{column}_year"] = series.dt.year
        output[f"{column}_month"] = series.dt.month
        output[f"{column}_day"] = series.dt.day
        output[f"{column}_dayofweek"] = series.dt.dayofweek
        output[f"{column}_quarter"] = series.dt.quarter
        output[f"{column}_is_month_start"] = series.dt.is_month_start.astype(float)
        output[f"{column}_is_month_end"] = series.dt.is_month_end.astype(float)

    return output


def build_numeric_pipeline() -> Pipeline:
    """Build numeric preprocessing pipeline."""
    numeric_strategy = get_string_config_value(
        "preprocessing.numeric_imputer_strategy",
        "median",
        NUMERIC_IMPUTER_STRATEGIES,
    )
    scale_numeric = as_bool(get_config_value("preprocessing.scale_numeric", True))

    imputer_kwargs: dict[str, Any] = {"strategy": numeric_strategy}
    if numeric_strategy == "constant":
        imputer_kwargs["fill_value"] = get_config_value(
            "preprocessing.numeric_fill_value",
            0,
        )

    steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(**imputer_kwargs)),
    ]

    if scale_numeric:
        steps.append(("scaler", StandardScaler()))

    return Pipeline(steps=steps)


def build_categorical_pipeline() -> Pipeline:
    """Build categorical preprocessing pipeline."""
    categorical_strategy = get_string_config_value(
        "preprocessing.categorical_imputer_strategy",
        "most_frequent",
        CATEGORICAL_IMPUTER_STRATEGIES,
    )

    imputer_kwargs: dict[str, Any] = {"strategy": categorical_strategy}
    if categorical_strategy == "constant":
        imputer_kwargs["fill_value"] = str(
            get_config_value("preprocessing.categorical_fill_value", "missing"),
        )

    max_categories = get_int_config_value(
        "preprocessing.one_hot_max_categories",
        0,
        minimum=0,
    )
    encoder_kwargs: dict[str, Any] = {
        "handle_unknown": "ignore",
        "sparse_output": True,
    }
    if max_categories > 1:
        encoder_kwargs["max_categories"] = max_categories

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(**imputer_kwargs)),
            ("encoder", OneHotEncoder(**encoder_kwargs)),
        ],
    )


def build_datetime_pipeline() -> Pipeline:
    """Build datetime preprocessing pipeline."""
    return Pipeline(
        steps=[
            (
                "datetime_features",
                FunctionTransformer(
                    extract_datetime_features,
                    validate=False,
                    feature_names_out=None,
                ),
            ),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ],
    )


def build_preprocessing_pipeline(
    df: pd.DataFrame,
    target_column: str,
) -> dict[str, Any]:
    """
    Build reusable sklearn preprocessing pipeline.

    The returned preprocessor is not fitted here. It is fitted inside each model
    Pipeline, which avoids train-test preprocessing leakage.
    """
    try:
        logger.info("Starting preprocessing pipeline creation")

        resolved_target = validate_preprocessing_inputs(df, target_column)

        working_df = clean_infinite_values(df)
        column_info = get_feature_columns(working_df, resolved_target)

        numeric_columns = column_info["numeric_columns"]
        categorical_columns = column_info["categorical_columns"]
        datetime_columns = column_info["datetime_columns"]
        unsupported_columns = column_info["unsupported_columns"]
        columns_dropped_before_modeling = column_info["columns_dropped_before_modeling"]

        transformers: list[tuple[str, Any, list[str]]] = []

        if numeric_columns:
            transformers.append(("numeric", build_numeric_pipeline(), numeric_columns))

        if categorical_columns:
            transformers.append(
                ("categorical", build_categorical_pipeline(), categorical_columns),
            )

        if datetime_columns:
            transformers.append(
                ("datetime", build_datetime_pipeline(), datetime_columns),
            )

        if not transformers:
            raise PreprocessingError(
                "No supported feature columns found for preprocessing.",
            )

        sparse_threshold = get_float_config_value(
            "preprocessing.sparse_threshold",
            0.3,
            minimum=0.0,
            maximum=1.0,
        )

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            sparse_threshold=sparse_threshold,
        )

        warnings: list[str] = []

        if unsupported_columns:
            warnings.append(
                "Unsupported feature columns will be dropped during preprocessing.",
            )

        if columns_dropped_before_modeling:
            warnings.append(
                "Some ID-like/high-cardinality columns were dropped before modeling.",
            )

        if datetime_columns:
            warnings.append(
                "Datetime columns were converted into numeric calendar features.",
            )

        result: dict[str, Any] = {
            "preprocessor": preprocessor,
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "datetime_columns": datetime_columns,
            "unsupported_columns_dropped": unsupported_columns,
            "id_like_columns_dropped": column_info["id_like_columns_dropped"],
            "high_cardinality_columns_dropped": column_info[
                "high_cardinality_columns_dropped"
            ],
            "columns_dropped_before_modeling": columns_dropped_before_modeling,
            "total_features_before_encoding": int(
                len(numeric_columns) + len(categorical_columns) + len(datetime_columns),
            ),
            "warnings": warnings,
            "message": "Preprocessing pipeline created successfully.",
        }

        logger.info("Preprocessing pipeline created successfully")
        return result

    except PreprocessingError:
        raise
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        pd.errors.ParserError,
    ) as error:
        logger.exception("Preprocessing pipeline creation failed.")
        raise PreprocessingError(
            "Preprocessing pipeline creation failed.",
            error_detail=str(error),
        ) from error


def clean_target_rows(
    df: pd.DataFrame,
    target_column: str,
) -> pd.DataFrame:
    """Remove rows where target is missing after invalid numeric targets are normalized."""
    resolved_target = validate_preprocessing_inputs(df, target_column)
    working_df = clean_infinite_values(df)

    before_rows = len(working_df)
    cleaned_df = working_df.dropna(subset=[resolved_target]).copy()
    dropped_rows = before_rows - len(cleaned_df)

    if dropped_rows > 0:
        logger.warning(
            "Dropped %s rows with missing/invalid target values.", dropped_rows
        )

    if cleaned_df.empty:
        raise PreprocessingError("No rows left after removing missing target values.")

    return cleaned_df


def split_features_target(
    df: pd.DataFrame,
    target_column: str,
    drop_missing_target: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split dataframe into feature matrix X and target vector y.

    Also cleans infinite values and drops configured ID/high-cardinality columns.
    """
    try:
        logger.info("Splitting features and target")

        resolved_target = validate_preprocessing_inputs(df, target_column)

        working_df = clean_infinite_values(df)
        if drop_missing_target:
            working_df = clean_target_rows(working_df, resolved_target)

        if working_df.empty:
            raise PreprocessingError("No rows available for feature-target split.")

        features = working_df.drop(columns=[resolved_target])
        target = get_target_series(working_df, resolved_target)

        if target.isna().any():
            raise PreprocessingError(
                "Target contains missing values. Set drop_missing_target=True before modeling.",
            )

        column_info = get_feature_columns(working_df, resolved_target)
        columns_to_drop = column_info.get("columns_dropped_before_modeling", [])

        if columns_to_drop:
            features = features.drop(columns=columns_to_drop, errors="ignore")

        if features.empty or len(features.columns) == 0:
            raise PreprocessingError(
                "No feature columns left after preprocessing column drops."
            )

        return features, target

    except PreprocessingError:
        raise
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        logger.exception("Feature-target split failed.")
        raise PreprocessingError(
            "Feature-target split failed.",
            error_detail=str(error),
        ) from error


def can_use_stratify(
    target: pd.Series,
    test_size: float,
) -> bool:
    """Decide if stratified split is safe."""
    if target is None or target.empty:
        logger.warning("Stratified split skipped because target is empty.")
        return False

    class_counts = value_counts_safely(target, dropna=False)
    number_of_classes = nunique_safely(target, dropna=False)

    expected_test_size = int(math.ceil(len(target) * test_size))
    expected_train_size = len(target) - expected_test_size

    if number_of_classes < 2:
        logger.warning(
            "Stratified split skipped because target has fewer than 2 classes.",
        )
        return False

    if class_counts.min() < 2:
        logger.warning(
            "Stratified split skipped because at least one class has less than 2 samples.",
        )
        return False

    if expected_test_size < number_of_classes:
        logger.warning(
            "Stratified split skipped because test set is smaller than number of classes.",
        )
        return False

    if expected_train_size < number_of_classes:
        logger.warning(
            "Stratified split skipped because train set is smaller than number of classes.",
        )
        return False

    return True


def create_train_test_split(
    df: pd.DataFrame,
    target_column: str,
    problem_type: str,
    test_size: float | None = None,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create train-test split."""
    try:
        logger.info("Starting train-test split")

        if problem_type is None or not str(problem_type).strip():
            raise PreprocessingError("Problem type is required.")

        test_size_value = (
            float(test_size)
            if test_size is not None
            else get_float_config_value(
                "modeling.test_size",
                0.2,
                minimum=0.01,
                maximum=0.99,
            )
        )
        random_state_value = (
            int(random_state)
            if random_state is not None
            else get_int_config_value("modeling.random_state", 42)
        )

        if not 0 < test_size_value < 1:
            raise PreprocessingError("test_size must be between 0 and 1.")

        features, target = split_features_target(
            df=df,
            target_column=target_column,
            drop_missing_target=True,
        )

        if len(features) < 2:
            raise PreprocessingError(
                "At least 2 valid rows are required for train-test split."
            )

        normalized_problem_type = problem_type.lower().strip()
        stratify = None

        if normalized_problem_type in CLASSIFICATION_TYPES and can_use_stratify(
            target,
            test_size_value,
        ):
            stratify = target

        train_features, test_features, train_target, test_target = train_test_split(
            features,
            target,
            test_size=test_size_value,
            random_state=random_state_value,
            stratify=stratify,
        )

        logger.info(
            "Train-test split completed successfully: train=%s test=%s stratified=%s",
            train_features.shape,
            test_features.shape,
            stratify is not None,
        )
        return train_features, test_features, train_target, test_target

    except PreprocessingError:
        raise
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        logger.exception("Train-test split failed.")
        raise PreprocessingError(
            "Train-test split failed.",
            error_detail=str(error),
        ) from error


if __name__ == "__main__":
    sample_df = pd.DataFrame(
        {
            "student_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "Age": [18, 19, 20, None, 22, 23, 24, 25, 26, 27],
            "Gender": ["M", "F", "M", "F", None, "M", "F", "M", "F", "M"],
            "StudyHours": [2.5, 3.0, np.inf, 4.0, 5.0, 2.0, 3.5, 4.5, 1.5, 5.5],
            "ExamDate": pd.date_range("2024-01-01", periods=10),
            "Grade": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
        },
    )

    pipeline_info = build_preprocessing_pipeline(
        df=sample_df,
        target_column="Grade",
    )

    print(
        {
            "numeric_columns": pipeline_info["numeric_columns"],
            "categorical_columns": pipeline_info["categorical_columns"],
            "datetime_columns": pipeline_info["datetime_columns"],
            "columns_dropped_before_modeling": pipeline_info[
                "columns_dropped_before_modeling"
            ],
            "warnings": pipeline_info["warnings"],
            "message": pipeline_info["message"],
        },
    )

    x_train, x_test, y_train, y_test = create_train_test_split(
        df=sample_df,
        target_column="Grade",
        problem_type="binary_classification",
    )

    print("X_train shape:", x_train.shape)
    print("X_test shape:", x_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
