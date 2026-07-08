from __future__ import annotations

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


def as_bool(value: Any) -> bool:
    """
    Convert config values safely into boolean.
    """
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.lower().strip() in {"true", "1", "yes", "y"}

    return bool(value)


def validate_preprocessing_inputs(df: pd.DataFrame, target_column: str) -> None:
    """
    Validate preprocessing inputs.
    """
    if df is None or df.empty:
        raise PreprocessingError("Input dataframe is empty.")

    if target_column is None or str(target_column).strip() == "":
        raise PreprocessingError("Target column is required.")

    if target_column not in df.columns:
        raise PreprocessingError(f"Target column not found: {target_column}")

    if len(df.columns) <= 1:
        raise PreprocessingError("Dataset must contain at least one feature column.")


def clean_infinite_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace np.inf and -np.inf with NaN before imputation/modeling.

    This keeps the sklearn pipeline simple and avoids custom transformers that
    can break MLflow/skops model serialization.
    """
    cleaned = df.copy()
    numeric_columns = cleaned.select_dtypes(include=["number"]).columns

    if len(numeric_columns) > 0:
        cleaned[numeric_columns] = cleaned[numeric_columns].replace(
            [np.inf, -np.inf],
            np.nan,
        )

    return cleaned


def detect_id_like_columns(features: pd.DataFrame) -> list[str]:
    """
    Detect ID-like columns that should usually be dropped before one-hot encoding.
    """
    id_unique_percent_threshold = float(
        get_config_value("audit.id_unique_percent_threshold", 95)
    )

    id_keywords = {
        "id",
        "uuid",
        "guid",
        "identifier",
        "serial",
        "roll",
        "roll_no",
        "zipcode",
        "zip",
        "email",
        "phone",
        "mobile",
    }

    id_like_columns: list[str] = []

    for column in features.columns:
        column_lower = str(column).lower().strip()
        unique_count = int(features[column].nunique(dropna=True))
        unique_percent = (unique_count / len(features)) * 100 if len(features) else 0

        name_suggests_id = any(keyword in column_lower for keyword in id_keywords)
        uniqueness_suggests_id = unique_percent >= id_unique_percent_threshold

        if name_suggests_id or uniqueness_suggests_id:
            id_like_columns.append(column)

    return id_like_columns


def detect_high_cardinality_columns(features: pd.DataFrame) -> list[str]:
    """
    Detect high-cardinality categorical columns that can explode one-hot encoding.
    """
    high_cardinality_threshold = int(
        get_config_value("audit.high_cardinality_threshold", 50)
    )

    categorical_columns = features.select_dtypes(
        include=["object", "category", "string"]
    ).columns

    high_cardinality_columns: list[str] = []

    for column in categorical_columns:
        unique_count = int(features[column].nunique(dropna=True))

        if unique_count >= high_cardinality_threshold:
            high_cardinality_columns.append(column)

    return high_cardinality_columns


def get_feature_columns(df: pd.DataFrame, target_column: str) -> dict[str, list[str]]:
    """
    Identify feature column groups.

    High-cardinality and ID-like columns can optionally be dropped by config.
    """
    features = df.drop(columns=[target_column]).copy()

    drop_id_like = as_bool(get_config_value("preprocessing.drop_id_like_columns", True))
    drop_high_cardinality = as_bool(
        get_config_value("preprocessing.drop_high_cardinality_columns", False)
    )

    id_like_columns = detect_id_like_columns(features) if drop_id_like else []
    high_cardinality_columns = (
        detect_high_cardinality_columns(features) if drop_high_cardinality else []
    )

    columns_to_drop = sorted(set(id_like_columns + high_cardinality_columns))

    features_for_model = features.drop(columns=columns_to_drop, errors="ignore")

    datetime_columns = features_for_model.select_dtypes(
        include=["datetime64", "datetimetz"]
    ).columns.tolist()

    numeric_columns = features_for_model.select_dtypes(include=["number"]).columns.tolist()

    categorical_columns = features_for_model.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()

    known_columns = set(numeric_columns + categorical_columns + datetime_columns)

    unsupported_columns = [
        column for column in features_for_model.columns if column not in known_columns
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
    """
    Convert datetime columns into numeric calendar features.
    """
    output = pd.DataFrame(index=datetime_df.index)

    for column in datetime_df.columns:
        series = pd.to_datetime(datetime_df[column], errors="coerce")

        output[f"{column}_year"] = series.dt.year
        output[f"{column}_month"] = series.dt.month
        output[f"{column}_day"] = series.dt.day
        output[f"{column}_dayofweek"] = series.dt.dayofweek
        output[f"{column}_is_month_start"] = series.dt.is_month_start.astype("float")
        output[f"{column}_is_month_end"] = series.dt.is_month_end.astype("float")

    return output


def build_numeric_pipeline() -> Pipeline:
    """
    Build numeric preprocessing pipeline.
    """
    numeric_strategy = str(
        get_config_value("preprocessing.numeric_imputer_strategy", "median")
    )
    scale_numeric = as_bool(get_config_value("preprocessing.scale_numeric", True))

    steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy=numeric_strategy)),
    ]

    if scale_numeric:
        steps.append(("scaler", StandardScaler()))

    return Pipeline(steps=steps)


def build_categorical_pipeline() -> Pipeline:
    """
    Build categorical preprocessing pipeline.
    """
    categorical_strategy = str(
        get_config_value("preprocessing.categorical_imputer_strategy", "most_frequent")
    )

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=categorical_strategy)),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )


def build_datetime_pipeline() -> Pipeline:
    """
    Build datetime preprocessing pipeline.
    """
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
            ("scaler", StandardScaler()),
        ]
    )


def build_preprocessing_pipeline(
    df: pd.DataFrame,
    target_column: str,
) -> dict[str, Any]:
    """
    Build reusable sklearn preprocessing pipeline.
    """
    try:
        logger.info("Starting preprocessing pipeline creation")

        validate_preprocessing_inputs(df, target_column)

        working_df = clean_infinite_values(df)
        column_info = get_feature_columns(working_df, target_column)

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
                ("categorical", build_categorical_pipeline(), categorical_columns)
            )

        if datetime_columns:
            transformers.append(("datetime", build_datetime_pipeline(), datetime_columns))

        if not transformers:
            raise PreprocessingError(
                "No supported feature columns found for preprocessing."
            )

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
        )

        warnings: list[str] = []

        if unsupported_columns:
            warnings.append(
                "Unsupported feature columns will be dropped during preprocessing."
            )

        if columns_dropped_before_modeling:
            warnings.append(
                "Some ID-like/high-cardinality columns were dropped before modeling."
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
                len(numeric_columns) + len(categorical_columns) + len(datetime_columns)
            ),
            "warnings": warnings,
            "message": "Preprocessing pipeline created successfully.",
        }

        logger.info("Preprocessing pipeline created successfully")
        return result

    except PreprocessingError:
        raise

    except Exception as error:
        logger.exception("Preprocessing pipeline creation failed.")
        raise PreprocessingError(
            "Preprocessing pipeline creation failed.",
            error_detail=str(error),
        ) from error


def clean_target_rows(
    df: pd.DataFrame,
    target_column: str,
) -> pd.DataFrame:
    """
    Remove rows where target is missing.
    """
    before_rows = len(df)
    cleaned_df = df.dropna(subset=[target_column]).copy()
    dropped_rows = before_rows - len(cleaned_df)

    if dropped_rows > 0:
        logger.warning("Dropped %s rows with missing target values.", dropped_rows)

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

    Also cleans infinite values before modeling.
    """
    try:
        logger.info("Splitting features and target")

        validate_preprocessing_inputs(df, target_column)

        working_df = clean_target_rows(df, target_column) if drop_missing_target else df.copy()
        working_df = clean_infinite_values(working_df)

        features = working_df.drop(columns=[target_column])
        target = working_df[target_column]

        column_info = get_feature_columns(working_df, target_column)
        columns_to_drop = column_info.get("columns_dropped_before_modeling", [])

        if columns_to_drop:
            features = features.drop(columns=columns_to_drop, errors="ignore")

        return features, target

    except PreprocessingError:
        raise

    except Exception as error:
        logger.exception("Feature-target split failed.")
        raise PreprocessingError(
            "Feature-target split failed.",
            error_detail=str(error),
        ) from error


def can_use_stratify(
    target: pd.Series,
    test_size: float,
) -> bool:
    """
    Decide if stratified split is safe.
    """
    class_counts = target.value_counts()
    number_of_classes = int(target.nunique(dropna=True))

    expected_test_size = int(len(target) * test_size)
    expected_train_size = len(target) - expected_test_size

    if number_of_classes < 2:
        logger.warning("Stratified split skipped because target has fewer than 2 classes.")
        return False

    if class_counts.min() < 2:
        logger.warning(
            "Stratified split skipped because at least one class has less than 2 samples."
        )
        return False

    if expected_test_size < number_of_classes:
        logger.warning(
            "Stratified split skipped because test set is smaller than number of classes."
        )
        return False

    if expected_train_size < number_of_classes:
        logger.warning(
            "Stratified split skipped because train set is smaller than number of classes."
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
    """
    Create train-test split.
    """
    try:
        logger.info("Starting train-test split")

        if problem_type is None or str(problem_type).strip() == "":
            raise PreprocessingError("Problem type is required.")

        test_size_value = float(
            test_size
            if test_size is not None
            else get_config_value("modeling.test_size", 0.2)
        )

        random_state_value = int(
            random_state
            if random_state is not None
            else get_config_value("modeling.random_state", 42)
        )

        if not 0 < test_size_value < 1:
            raise PreprocessingError("test_size must be between 0 and 1.")

        features, target = split_features_target(
            df=df,
            target_column=target_column,
            drop_missing_target=True,
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

        logger.info("Train-test split completed successfully")

        return train_features, test_features, train_target, test_target

    except PreprocessingError:
        raise

    except Exception as error:
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
        }
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
        }
    )

    X_train, X_test, y_train, y_test = create_train_test_split(
        df=sample_df,
        target_column="Grade",
        problem_type="binary_classification",
    )

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
