from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.audit.preprocessing import build_preprocessing_pipeline, create_train_test_split
from src.utils.config import get_config_value
from src.utils.exceptions import ModelTrainingError
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


def train_baseline_models(
    df: pd.DataFrame,
    target_column: str,
    problem_type: str,
) -> dict[str, Any]:
    """
    Train and evaluate baseline models.

    These are sanity-check baselines, not final optimized models.

    Important:
    - trained_model_objects is intentionally returned for MLflow/explainability.
    - runtime_objects is intentionally returned for explainability.
    - API/UI layers must strip these non-serializable/heavy objects before responses.
    """
    try:
        logger.info("Starting baseline model training")

        validate_inputs(df, target_column, problem_type)
        normalized_problem_type = problem_type.lower().strip()

        random_state = int(get_config_value("modeling.random_state", 42))
        test_size = float(get_config_value("modeling.test_size", 0.2))
        enable_cross_validation = as_bool(
            get_config_value("modeling.enable_cross_validation", False)
        )
        requested_cv_folds = int(get_config_value("modeling.cv_folds", 5))

        if normalized_problem_type == "regression":
            validate_regression_target(df[target_column])

        preprocessing_info = build_preprocessing_pipeline(
            df=df,
            target_column=target_column,
        )

        preprocessor = preprocessing_info["preprocessor"]

        X_train, X_test, y_train, y_test = create_train_test_split(
            df=df,
            target_column=target_column,
            problem_type=normalized_problem_type,
            test_size=test_size,
            random_state=random_state,
        )

        label_encoder: LabelEncoder | None = None
        class_labels: list[str] = []

        if normalized_problem_type in CLASSIFICATION_TYPES:
            label_encoder = LabelEncoder()
            y_train_model = np.asarray(label_encoder.fit_transform(y_train.astype(str)))
            y_test_model = np.asarray(label_encoder.transform(y_test.astype(str)))
            class_labels = [str(label) for label in label_encoder.classes_]
        else:
            y_train_model = np.asarray(y_train, dtype=float)
            y_test_model = np.asarray(y_test, dtype=float)

        safe_cv = get_safe_cv_splitter(
            y=y_train_model,
            problem_type=normalized_problem_type,
            requested_folds=requested_cv_folds,
            random_state=random_state,
        )

        cv_warning = None

        if enable_cross_validation and safe_cv is None:
            cv_warning = (
                "Cross-validation skipped because there are not enough samples "
                "per class/fold for a reliable split."
            )
            logger.warning(cv_warning)

        models = get_baseline_models(normalized_problem_type, random_state)

        results: dict[str, Any] = {}

        for model_name, model in models.items():
            logger.info("Training baseline model: %s", model_name)

            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", model),
                ]
            )

            pipeline.fit(X_train, y_train_model)
            y_pred = np.asarray(pipeline.predict(X_test))

            if normalized_problem_type in CLASSIFICATION_TYPES:
                metrics = evaluate_classification(
                    model=pipeline,
                    X_test=X_test,
                    y_test=y_test_model,
                    y_pred=y_pred,
                    problem_type=normalized_problem_type,
                )

                confusion = confusion_matrix(y_test_model, y_pred).tolist()

            elif normalized_problem_type == "regression":
                metrics = evaluate_regression(
                    y_test=y_test_model,
                    y_pred=y_pred,
                )
                confusion = None

            else:
                raise ModelTrainingError(
                    f"Unsupported problem type: {normalized_problem_type}"
                )

            if enable_cross_validation and safe_cv is not None:
                cv_results = run_cross_validation(
                    pipeline=pipeline,
                    X=X_train,
                    y=y_train_model,
                    problem_type=normalized_problem_type,
                    cv=safe_cv,
                )
                metrics.update(cv_results)

            results[model_name] = {
                "metrics": metrics,
                "model_object": pipeline,
                "confusion_matrix": confusion,
            }

            logger.info("Completed model: %s", model_name)

        best_model = select_best_model(results, normalized_problem_type)

        output: dict[str, Any] = {
            "problem_type": normalized_problem_type,
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
                "warnings": preprocessing_info.get("warnings", []),
            },
            "evaluation_details": {
                "test_size": test_size,
                "random_state": random_state,
                "cross_validation_enabled": bool(
                    enable_cross_validation and safe_cv is not None
                ),
                "requested_cv_folds": requested_cv_folds,
                "actual_cv_folds": getattr(safe_cv, "n_splits", None)
                if enable_cross_validation and safe_cv is not None
                else None,
                "cv_warning": cv_warning,
                "class_labels": class_labels,
                "label_encoder_used": label_encoder is not None,
                "train_shape": tuple(X_train.shape),
                "test_shape": tuple(X_test.shape),
                "confusion_matrices": {
                    model_name: model_result["confusion_matrix"]
                    for model_name, model_result in results.items()
                    if model_result["confusion_matrix"] is not None
                },
            },
            "runtime_objects": {
                "sample_features": X_test.copy(),
                "sample_target": y_test.copy(),
                "train_features": X_train.copy(),
                "test_features": X_test.copy(),
                "label_encoder": label_encoder,
            },
            "message": "Baseline model training completed successfully.",
            "note": "These are baseline sanity-check models, not final optimized models.",
        }

        logger.info("Baseline model training completed successfully")
        return output

    except ModelTrainingError:
        raise

    except Exception as error:
        logger.exception("Baseline model training failed.")
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

    normalized_problem_type = problem_type.lower().strip()

    if (
        normalized_problem_type not in CLASSIFICATION_TYPES
        and normalized_problem_type != "regression"
    ):
        raise ModelTrainingError(f"Unsupported problem type: {normalized_problem_type}")

    if df[target_column].dropna().nunique() < 2:
        raise ModelTrainingError("Target column must contain at least 2 unique values.")


def validate_regression_target(target: pd.Series) -> None:
    if not pd.api.types.is_numeric_dtype(target):
        raise ModelTrainingError("Regression target must be numeric.")


def get_baseline_models(problem_type: str, random_state: int) -> dict[str, Any]:
    """
    Return small, reliable baseline models.
    """
    parallel_jobs = int(get_config_value("performance.parallel_jobs", -1))

    rf_estimators = int(get_config_value("modeling.random_forest_estimators", 200))
    rf_min_samples_leaf = int(
        get_config_value("modeling.random_forest_min_samples_leaf", 2)
    )
    logistic_max_iter = int(get_config_value("modeling.logistic_max_iter", 1000))

    if problem_type in CLASSIFICATION_TYPES:
        return {
            "Logistic Regression": LogisticRegression(
                max_iter=logistic_max_iter,
                class_weight="balanced",
                random_state=random_state,
            ),
            "Random Forest Classifier": RandomForestClassifier(
                n_estimators=rf_estimators,
                min_samples_leaf=rf_min_samples_leaf,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=parallel_jobs,
            ),
        }

    if problem_type == "regression":
        return {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(
                n_estimators=rf_estimators,
                min_samples_leaf=rf_min_samples_leaf,
                random_state=random_state,
                n_jobs=parallel_jobs,
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
    """
    Evaluate classification baseline.
    """
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_test, y_pred)), 4),
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

    if problem_type == "binary_classification":
        metrics["precision_binary"] = round(
            float(precision_score(y_test, y_pred, average="binary", zero_division=0)),
            4,
        )
        metrics["recall_binary"] = round(
            float(recall_score(y_test, y_pred, average="binary", zero_division=0)),
            4,
        )
        metrics["f1_binary"] = round(
            float(f1_score(y_test, y_pred, average="binary", zero_division=0)),
            4,
        )

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
        logger.warning("Probability-based metrics skipped: %s", error)

    return metrics


def evaluate_regression(y_test: Any, y_pred: np.ndarray) -> dict[str, float]:
    """
    Evaluate regression baseline.
    """
    y_test_array = np.asarray(y_test, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)

    mse = mean_squared_error(y_test_array, y_pred_array)
    rmse = float(np.sqrt(mse))

    metrics = {
        "mae": round(float(mean_absolute_error(y_test_array, y_pred_array)), 4),
        "mse": round(float(mse), 4),
        "rmse": round(rmse, 4),
        "r2_score": round(float(r2_score(y_test_array, y_pred_array)), 4),
    }

    try:
        non_zero_mask = y_test_array != 0

        if int(non_zero_mask.sum()) > 0:
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
        logger.warning("MAPE skipped: %s", error)

    return metrics


def get_safe_cv_splitter(
    y: Any,
    problem_type: str,
    requested_folds: int,
    random_state: int,
) -> KFold | StratifiedKFold | None:
    """
    Build a safe CV splitter.

    Classification folds are capped by minority-class count to avoid
    invalid or unstable StratifiedKFold splits.
    """
    if requested_folds < 2:
        return None

    if problem_type in CLASSIFICATION_TYPES:
        y_series = pd.Series(np.asarray(y))
        class_counts = y_series.value_counts(dropna=False)

        if class_counts.empty:
            return None

        min_class_count = int(class_counts.min())
        safe_folds = min(int(requested_folds), min_class_count)

        if safe_folds < 2:
            return None

        return StratifiedKFold(
            n_splits=safe_folds,
            shuffle=True,
            random_state=random_state,
        )

    return KFold(
        n_splits=int(requested_folds),
        shuffle=True,
        random_state=random_state,
    )


def run_cross_validation(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: Any,
    problem_type: str,
    cv: KFold | StratifiedKFold,
) -> dict[str, float]:
    """
    Run optional cross-validation on training data.
    """
    try:
        if problem_type in CLASSIFICATION_TYPES:
            scoring = str(
                get_config_value("metrics.classification_default", "f1_weighted")
            )
        else:
            scoring = "neg_root_mean_squared_error"

        scores = cross_val_score(
            pipeline,
            X,
            np.asarray(y),
            cv=cv,
            scoring=scoring,
            n_jobs=int(get_config_value("performance.parallel_jobs", -1)),
        )

        if problem_type == "regression":
            scores = -scores

        return {
            "cv_mean": round(float(scores.mean()), 4),
            "cv_std": round(float(scores.std()), 4),
        }

    except Exception as error:
        logger.warning("Cross-validation skipped: %s", error)
        return {}


def select_best_model(results: dict[str, Any], problem_type: str) -> dict[str, Any]:
    """
    Select best baseline model using config-driven metric.
    """
    if problem_type in CLASSIFICATION_TYPES:
        primary_metric = str(
            get_config_value("metrics.classification_default", "f1_score")
        )

        if primary_metric == "f1_weighted":
            primary_metric = "f1_score"

        higher_is_better = True
    else:
        primary_metric = str(get_config_value("metrics.regression_default", "rmse"))
        higher_is_better = primary_metric not in {"mae", "mse", "rmse", "mape"}

        if primary_metric not in {"mae", "mse", "rmse", "mape", "r2_score"}:
            primary_metric = "rmse"
            higher_is_better = False

    best_model_name: str | None = None
    best_score: float | None = None

    for model_name, model_result in results.items():
        raw_score = model_result["metrics"].get(primary_metric)

        if raw_score is None:
            continue

        score = float(raw_score)

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

    if best_model_name is None or best_score is None:
        raise ModelTrainingError("Could not select best baseline model.")

    return {
        "model_name": best_model_name,
        "selection_metric": primary_metric,
        "score": float(best_score),
        "higher_is_better": bool(higher_is_better),
    }


def get_sample_features_for_explainability(
    baseline_results: dict[str, Any],
) -> pd.DataFrame | None:
    """
    Extract sample features from baseline output for explainability.
    """
    runtime_objects = baseline_results.get("runtime_objects", {})

    if not isinstance(runtime_objects, dict):
        return None

    sample_features = runtime_objects.get("sample_features")

    if isinstance(sample_features, pd.DataFrame):
        return sample_features

    return None


def strip_runtime_objects(baseline_results: dict[str, Any]) -> dict[str, Any]:
    """
    Return JSON-safe baseline results for API/UI/downloads.

    This removes trained sklearn model objects and runtime sample data.
    """
    cleaned = dict(baseline_results)
    cleaned.pop("trained_model_objects", None)
    cleaned.pop("runtime_objects", None)
    return cleaned


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

    printable_output = strip_runtime_objects(output)

    print(printable_output)
