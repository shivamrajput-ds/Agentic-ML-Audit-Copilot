from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.exceptions import AuditCopilotException, PreprocessingError
from src.utils.logger import get_logger


logger = get_logger(__name__)


def validate_preprocessing_inputs(
    df: pd.DataFrame,
    target_column: str,
) -> None:
    """
    Validate dataframe and target column before preprocessing.
    """
    if df is None or df.empty:
        raise PreprocessingError("Input dataframe is empty.")

    if not target_column:
        raise PreprocessingError("Target column is required.")

    if target_column not in df.columns:
        raise PreprocessingError(f"Target column not found: {target_column}")


def build_preprocessing_pipeline(
    df: pd.DataFrame,
    target_column: str,
) -> dict[str, Any]:
    """
    Build a reusable sklearn preprocessing pipeline.

    Numeric features:
    - median imputation
    - standard scaling

    Categorical features:
    - most-frequent imputation
    - one-hot encoding
    """
    try:
        logger.info("Starting preprocessing pipeline creation")

        validate_preprocessing_inputs(df, target_column)

        features = df.drop(columns=[target_column])
        datetime_columns = features.select_dtypes(
    include=["datetime64", "datetimetz"]
).columns.tolist()

        numeric_columns = features.select_dtypes(
            include=["number"]
        ).columns.tolist()

        categorical_columns = features.select_dtypes(
            include=["object", "category", "bool", "string"]
        ).columns.tolist()

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

        transformers = []

        if numeric_columns:
            transformers.append(("numeric", numeric_pipeline, numeric_columns))

        if categorical_columns:
            transformers.append(
                ("categorical", categorical_pipeline, categorical_columns)
            )

        if not transformers:
            raise PreprocessingError(
                "No supported feature columns found for preprocessing."
            )

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
        )

        result = {
    "preprocessor": preprocessor,
    "numeric_columns": numeric_columns,
    "categorical_columns": categorical_columns,
    "datetime_columns_dropped": datetime_columns,
    "total_features_before_encoding": (
        len(numeric_columns) + len(categorical_columns)
    ),
    "warning": (
        "Datetime columns are currently dropped by preprocessing. "
        "Future version can extract year/month/day features."
        if datetime_columns
        else None
    ),
    "message": "Preprocessing pipeline created successfully.",
}


        logger.info("Preprocessing pipeline created successfully")
        return result

    except PreprocessingError:
        raise

    except Exception as error:
        logger.error(f"Preprocessing pipeline creation failed: {error}")
        raise PreprocessingError(
            "Preprocessing pipeline creation failed",
            error_detail=str(error),
        ) from error


def split_features_target(
    df: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split dataframe into feature matrix X and target vector y.
    """
    try:
        logger.info("Splitting features and target")

        validate_preprocessing_inputs(df, target_column)

        features = df.drop(columns=[target_column])
        target = df[target_column]

        return features, target

    except PreprocessingError:
        raise

    except Exception as error:
        logger.error(f"Feature-target split failed: {error}")
        raise PreprocessingError(
            "Feature-target split failed",
            error_detail=str(error),
        ) from error


def create_train_test_split(
    df: pd.DataFrame,
    target_column: str,
    problem_type: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Create train-test split.

    For classification problems, stratified split is used when possible.
    For regression problems, regular random split is used.
    """
    try:
        logger.info("Starting train-test split")

        if not problem_type:
            raise PreprocessingError("Problem type is required.")

        features, target = split_features_target(df, target_column)

        normalized_problem_type = problem_type.lower().strip()
        stratify = None

        if normalized_problem_type in {
            "binary_classification",
            "multiclass_classification",
        }:
            class_counts = target.value_counts()
            number_of_classes = target.nunique()
            expected_test_size = int(len(target) * test_size)

            if class_counts.min() < 2:
                logger.warning(
                    "Stratified split skipped because at least one class has "
                    "less than 2 samples."
                )
            elif expected_test_size < number_of_classes:
                logger.warning(
                    "Stratified split skipped because test set is smaller than "
                    "number of classes."
                )
            else:
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
            "Train-test split failed",
            error_detail=str(error),
        ) from error


if __name__ == "__main__":
    sample_df = pd.DataFrame(
        {
            "Age": [18, 19, 20, None, 22, 23, 24, 25, 26, 27],
            "Gender": ["M", "F", "M", "F", None, "M", "F", "M", "F", "M"],
            "StudyHours": [2.5, 3.0, None, 4.0, 5.0, 2.0, 3.5, 4.5, 1.5, 5.5],
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
            "total_features_before_encoding": pipeline_info[
                "total_features_before_encoding"
            ],
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