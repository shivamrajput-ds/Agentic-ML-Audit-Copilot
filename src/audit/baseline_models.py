from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.audit.preprocessing import build_preprocessing_pipeline, create_train_test_split
from src.utils.config import get_config_value
from src.utils.exceptions import ModelTrainingError
from src.utils.logger import get_logger


logger = get_logger(__name__)

CLASSIFICATION_TYPES = {"binary_classification", "multiclass_classification"}


def train_baseline_models(
    df: pd.DataFrame,
    target_column: str,
    problem_type: str,
) -> dict[str, Any]:
    """
    Train and evaluate baseline models.

    These are sanity-check baselines, not final optimized models.
    """
    try:
        logger.info("Starting baseline model training")

        validate_inputs(df, target_column, problem_type)
        problem_type = problem_type.lower().strip()

        random_state = int(get_config_value("modeling.random_state", 42))
        test_size = float(get_config_value("modeling.test_size", 0.2))

        if problem_type == "regression":
            validate_regression_target(df[target_column])

        preprocessing_info = build_preprocessing_pipeline(
            df=df,
            target_column=target_column,
        )

        preprocessor = preprocessing_info["preprocessor"]

        X_train, X_test, y_train, y_test = create_train_test_split(
            df=df,
            target_column=target_column,
            problem_type=problem_type,
            test_size=test_size,
            random_state=random_state,
        )

        label_encoder = None

        if problem_type in CLASSIFICATION_TYPES:
            label_encoder = LabelEncoder()
            y_train_model = label_encoder.fit_transform(y_train.astype(str))
            y_test_model = label_encoder.transform(y_test.astype(str))
            class_labels = label_encoder.classes_.tolist()
        else:
            y_train_model = y_train
            y_test_model = y_test
            class_labels = []

        models = get_baseline_models(problem_type, random_state)

        results: dict[str, Any] = {}

        for model_name, model in models.items():
            logger.info(f"Training baseline model: {model_name}")

            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", model),
                ]
            )

            pipeline.fit(X_train, y_train_model)
            y_pred = pipeline.predict(X_test)

            if problem_type in CLASSIFICATION_TYPES:
                metrics = evaluate_classification(
                    model=pipeline,
                    X_test=X_test,
                    y_test=y_test_model,
                    y_pred=y_pred,
                    problem_type=problem_type,
                )

                confusion = confusion_matrix(y_test_model, y_pred).tolist()

            elif problem_type == "regression":
                metrics = evaluate_regression(
                    y_test=y_test_model,
                    y_pred=y_pred,
                )
                confusion = None

            else:
                raise ModelTrainingError(f"Unsupported problem type: {problem_type}")

            results[model_name] = {
                "metrics": metrics,
                "model_object": pipeline,
                "confusion_matrix": confusion,
            }

            logger.info(f"Completed model: {model_name}")

        best_model = select_best_model(results, problem_type)

        output: dict[str, Any] = {
            "problem_type": problem_type,
            "target_column": target_column,
            "models_trained": list(models.keys()),
            "results": {
                model_name: model_result["metrics"]
                for model_name, model_result in results.items()
            },
            "trained_model_objects": {
                model_name: model_result["model_object"]
                for model_name, model_result in results.items()
            },
            "best_model": best_model,
            "preprocessing_summary": {
                "numeric_columns": preprocessing_info.get("numeric_columns", []),
                "categorical_columns": preprocessing_info.get("categorical_columns", []),
                "datetime_columns": preprocessing_info.get("datetime_columns", []),
                "unsupported_columns_dropped": preprocessing_info.get(
                    "unsupported_columns_dropped",
                    [],
                ),
                "total_features_before_encoding": preprocessing_info.get(
                    "total_features_before_encoding",
                    0,
                ),
            },
            "evaluation_details": {
                "test_size": test_size,
                "random_state": random_state,
                "class_labels": class_labels,
                "confusion_matrices": {
                    model_name: model_result["confusion_matrix"]
                    for model_name, model_result in results.items()
                    if model_result["confusion_matrix"] is not None
                },
            },
            "message": "Baseline model training completed successfully.",
            "note": "These are baseline sanity-check models, not final optimized models.",
        }

        logger.info("Baseline model training completed successfully")
        return output

    except ModelTrainingError:
        raise

    except Exception as error:
        logger.error(f"Baseline model training failed: {error}")
        raise ModelTrainingError(
            "Baseline model training failed.",
            error_detail=str(error),
        ) from error


def validate_inputs(df: pd.DataFrame, target_column: str, problem_type: str) -> None:
    if df is None or df.empty:
        raise ModelTrainingError("Input dataframe is empty.")

    if target_column is None or str(target_column).strip() == "":
        raise ModelTrainingError("Target column is required.")

    if target_column not in df.columns:
        raise ModelTrainingError(f"Target column not found: {target_column}")

    if problem_type is None or str(problem_type).strip() == "":
        raise ModelTrainingError("Problem type is required.")

    if df[target_column].dropna().nunique() < 2:
        raise ModelTrainingError("Target column must contain at least 2 unique values.")


def validate_regression_target(target: pd.Series) -> None:
    if not pd.api.types.is_numeric_dtype(target):
        raise ModelTrainingError("Regression target must be numeric.")


def get_baseline_models(problem_type: str, random_state: int) -> dict[str, Any]:
    if problem_type in CLASSIFICATION_TYPES:
        return {
            "Logistic Regression": LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=random_state,
            ),
            "Random Forest Classifier": RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            ),
        }

    if problem_type == "regression":
        return {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(
                n_estimators=200,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1,
            ),
        }

    raise ModelTrainingError(f"Unsupported problem type: {problem_type}")


def evaluate_classification(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    problem_type: str,
) -> dict[str, float]:
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision_weighted": round(
            float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            4,
        ),
        "recall_weighted": round(
            float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            4,
        ),
        "f1_score": round(
            float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            4,
        ),
        "f1_macro": round(
            float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            4,
        ),
    }

    try:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)

            if problem_type == "binary_classification" and y_proba.shape[1] == 2:
                metrics["roc_auc"] = round(
                    float(roc_auc_score(y_test, y_proba[:, 1])),
                    4,
                )

            elif problem_type == "multiclass_classification" and y_proba.shape[1] > 2:
                metrics["roc_auc_ovr_weighted"] = round(
                    float(
                        roc_auc_score(
                            y_test,
                            y_proba,
                            multi_class="ovr",
                            average="weighted",
                        )
                    ),
                    4,
                )

    except Exception as error:
        logger.warning(f"Probability-based metrics skipped: {error}")

    return metrics


def evaluate_regression(y_test: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))

    metrics = {
        "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
        "rmse": round(rmse, 4),
        "r2_score": round(float(r2_score(y_test, y_pred)), 4),
    }

    try:
        y_test_array = np.asarray(y_test)
        y_pred_array = np.asarray(y_pred)
        non_zero_mask = y_test_array != 0

        if non_zero_mask.sum() > 0:
            mape = (
                np.mean(
                    np.abs(
                        (y_test_array[non_zero_mask] - y_pred_array[non_zero_mask])
                        / y_test_array[non_zero_mask]
                    )
                )
                * 100
            )
            metrics["mape"] = round(float(mape), 4)

    except Exception as error:
        logger.warning(f"MAPE skipped: {error}")

    return metrics


def select_best_model(results: dict[str, Any], problem_type: str) -> dict[str, Any]:
    if problem_type in CLASSIFICATION_TYPES:
        primary_metric = "f1_score"
        higher_is_better = True
    else:
        primary_metric = "rmse"
        higher_is_better = False

    best_model_name = None
    best_score = None

    for model_name, model_result in results.items():
        score = model_result["metrics"].get(primary_metric)

        if score is None:
            continue

        if best_score is None:
            best_model_name = model_name
            best_score = score
            continue

        if higher_is_better and score > best_score:
            best_model_name = model_name
            best_score = score

        if not higher_is_better and score < best_score:
            best_model_name = model_name
            best_score = score

    if best_model_name is None:
        raise ModelTrainingError("Could not select best baseline model.")

    return {
        "model_name": best_model_name,
        "selection_metric": primary_metric,
        "score": float(best_score),
        "higher_is_better": higher_is_better,
    }


if __name__ == "__main__":
    from src.audit.profiler import load_dataset
    from src.audit.problem_detector import detect_problem_type

    dataset_path = "data/sample/student_mark.csv"
    target_column = "Grade"

    df = load_dataset(dataset_path)
    problem_info = detect_problem_type(df, target_column)

    output = train_baseline_models(
        df=df,
        target_column=target_column,
        problem_type=problem_info["problem_type"],
    )

    printable_output = {
        key: value
        for key, value in output.items()
        if key != "trained_model_objects"
    }

    print(printable_output)