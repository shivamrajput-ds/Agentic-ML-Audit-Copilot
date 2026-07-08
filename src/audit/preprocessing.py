from __future__ import annotations

from typing import Any

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


def validate_preprocessing_inputs(df: pd.DataFrame, target_column: str) -> None:
    if df is None or df.empty:
        raise PreprocessingError("Input dataframe is empty.")

    if target_column is None or str(target_column).strip() == "":
        raise PreprocessingError("Target column is required.")

    if target_column not in df.columns:
        raise PreprocessingError(f"Target column not found: {target_column}")

    if len(df.columns) <= 1:
        raise PreprocessingError("Dataset must contain at least one feature column.")


def get_feature_columns(df: pd.DataFrame, target_column: str) -> dict[str, list[str]]:
    features = df.drop(columns=[target_column])

    datetime_columns = features.select_dtypes(
        include=["datetime64", "datetimetz"]
    ).columns.tolist()

    numeric_columns = features.select_dtypes(include=["number"]).columns.tolist()

    categorical_columns = features.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()

    known_columns = set(numeric_columns + categorical_columns + datetime_columns)

    unsupported_columns = [
        column for column in features.columns if column not in known_columns
    ]

    return {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": datetime_columns,
        "unsupported_columns": unsupported_columns,
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

        column_info = get_feature_columns(df, target_column)

        numeric_columns = column_info["numeric_columns"]
        categorical_columns = column_info["categorical_columns"]
        datetime_columns = column_info["datetime_columns"]
        unsupported_columns = column_info["unsupported_columns"]

        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )

        datetime_pipeline = Pipeline(
            steps=[
                (
                    "datetime_features",
                    FunctionTransformer(
                        extract_datetime_features,
                        validate=False,
                    ),
                ),
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        transformers = []

        if numeric_columns:
            transformers.append(("numeric", numeric_pipeline, numeric_columns))

        if categorical_columns:
            transformers.append(("categorical", categorical_pipeline, categorical_columns))

        if datetime_columns:
            transformers.append(("datetime", datetime_pipeline, datetime_columns))

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

        result: dict[str, Any] = {
            "preprocessor": preprocessor,
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "datetime_columns": datetime_columns,
            "unsupported_columns_dropped": unsupported_columns,
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
        logger.error(f"Preprocessing pipeline creation failed: {error}")
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
        logger.warning(f"Dropped {dropped_rows} rows with missing target values.")

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
    """
    try:
        logger.info("Splitting features and target")

        validate_preprocessing_inputs(df, target_column)

        working_df = clean_target_rows(df, target_column) if drop_missing_target else df

        features = working_df.drop(columns=[target_column])
        target = working_df[target_column]

        return features, target

    except PreprocessingError:
        raise

    except Exception as error:
        logger.error(f"Feature-target split failed: {error}")
        raise PreprocessingError(
            "Feature-target split failed.",
            error_detail=str(error),
        ) from error


def can_use_stratify(
    target: pd.Series,
    test_size: float,
) -> bool:
    class_counts = target.value_counts()
    number_of_classes = target.nunique(dropna=True)

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

        test_size = float(
            test_size
            if test_size is not None
            else get_config_value("training.test_size", 0.2)
        )

        random_state = int(
            random_state
            if random_state is not None
            else get_config_value("training.random_state", 42)
        )

        if not 0 < test_size < 1:
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
            test_size,
        ):
            stratify = target

        train_features, test_features, train_target, test_target = train_test_split(
            features,
            target,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )

        logger.info("Train-test split completed successfully")

        return train_features, test_features, train_target, test_target

    except PreprocessingError:
        raise

    except Exception as error:
        logger.error(f"Train-test split failed: {error}")
        raise PreprocessingError(
            "Train-test split failed.",
            error_detail=str(error),
        ) from error


if __name__ == "__main__":
    sample_df = pd.DataFrame(
        {
            "Age": [18, 19, 20, None, 22, 23, 24, 25, 26, 27],
            "Gender": ["M", "F", "M", "F", None, "M", "F", "M", "F", "M"],
            "StudyHours": [2.5, 3.0, None, 4.0, 5.0, 2.0, 3.5, 4.5, 1.5, 5.5],
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
            "unsupported_columns_dropped": pipeline_info["unsupported_columns_dropped"],
            "total_features_before_encoding": pipeline_info[
                "total_features_before_encoding"
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