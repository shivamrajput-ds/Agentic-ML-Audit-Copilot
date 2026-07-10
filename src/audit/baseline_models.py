from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
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

from src.audit.preprocessing import (
    build_preprocessing_pipeline,
    create_train_test_split,
)
from src.utils.config import get_config_value
from src.utils.exceptions import ModelTrainingError
from src.utils.logger import get_logger

logger = get_logger(__name__)

CLASSIFICATION_TYPES = {"binary_classification", "multiclass_classification"}
SUPPORTED_PROBLEM_TYPES = CLASSIFICATION_TYPES | {"regression"}
TRUE_VALUES = {"true", "1", "yes", "y", "on"}
FALSE_VALUES = {"false", "0", "no", "n", "off"}

CLASSIFICATION_SELECTION_ALIASES = {
    "f1": "f1_binary",
    "f1_binary": "f1_binary",
    "f1_score": "f1_score",
    "f1_weighted": "f1_score",
    "weighted_f1": "f1_score",
    "f1_macro": "f1_macro",
    "macro_f1": "f1_macro",
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision": "precision_binary",
    "precision_binary": "precision_binary",
    "precision_weighted": "precision_weighted",
    "recall": "recall_binary",
    "recall_binary": "recall_binary",
    "recall_weighted": "recall_weighted",
    "roc_auc": "roc_auc",
    "roc_auc_ovr_weighted": "roc_auc_ovr_weighted",
}

SKLEARN_CLASSIFICATION_SCORING_ALIASES = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "f1": "f1",
    "f1_binary": "f1",
    "f1_score": "f1_weighted",
    "f1_weighted": "f1_weighted",
    "weighted_f1": "f1_weighted",
    "f1_macro": "f1_macro",
    "macro_f1": "f1_macro",
    "precision": "precision",
    "precision_binary": "precision",
    "precision_weighted": "precision_weighted",
    "recall": "recall",
    "recall_binary": "recall",
    "recall_weighted": "recall_weighted",
    "roc_auc": "roc_auc",
    "roc_auc_ovr_weighted": "roc_auc_ovr_weighted",
}

REGRESSION_SELECTION_METRICS = {"mae", "mse", "rmse", "mape", "r2_score"}
LOWER_IS_BETTER_REGRESSION = {"mae", "mse", "rmse", "mape"}


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


def get_int_config(path: str, default: int, minimum: int | None = None) -> int:
    """Read integer config values with safe fallback and optional lower bound."""
    try:
        value = int(get_config_value(path, default))
    except (KeyError, TypeError, ValueError, RuntimeError):
        value = int(default)

    if minimum is not None:
        value = max(minimum, value)

    return value


def get_float_config(
    path: str,
    default: float,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Read float config values with safe fallback and optional bounds."""
    try:
        value = float(get_config_value(path, default))
    except (KeyError, TypeError, ValueError, RuntimeError):
        value = float(default)

    if not np.isfinite(value):
        value = float(default)

    if minimum is not None:
        value = max(minimum, value)

    if maximum is not None:
        value = min(maximum, value)

    return value


def normalize_problem_type(problem_type: str) -> str:
    """Normalize and validate problem type."""
    normalized = str(problem_type).lower().strip()

    if normalized not in SUPPORTED_PROBLEM_TYPES:
        raise ModelTrainingError(f"Unsupported problem type: {normalized}")

    return normalized


def normalize_target_column(df: pd.DataFrame, target_column: str) -> str:
    """Resolve target column while preserving backward-compatible exact matching."""
    requested = str(target_column).strip()

    if requested in df.columns:
        return requested

    case_matches = [
        column for column in df.columns if str(column).lower() == requested.lower()
    ]
    if len(case_matches) == 1:
        resolved = str(case_matches[0])
        logger.warning(
            "Target column resolved by case-insensitive match: %s -> %s",
            requested,
            resolved,
        )
        return resolved

    close_matches = [
        str(column)
        for column in df.columns
        if requested.lower() in str(column).lower()
        or str(column).lower() in requested.lower()
    ][:5]

    suggestion = f" Did you mean one of: {close_matches}?" if close_matches else ""
    raise ModelTrainingError(f"Target column not found: {requested}.{suggestion}")


def validate_inputs(df: pd.DataFrame, target_column: str, problem_type: str) -> str:
    """Validate baseline-model training inputs and return resolved target column."""
    if df is None or df.empty:
        raise ModelTrainingError("Input dataframe is empty.")

    if target_column is None or not str(target_column).strip():
        raise ModelTrainingError("Target column is required.")

    if problem_type is None or not str(problem_type).strip():
        raise ModelTrainingError("Problem type is required.")

    duplicate_columns = df.columns[df.columns.duplicated()].tolist()
    if duplicate_columns:
        raise ModelTrainingError(f"Duplicate column names found: {duplicate_columns}")

    resolved_target = normalize_target_column(df, target_column)
    normalize_problem_type(problem_type)

    non_null_target = df[resolved_target].replace([np.inf, -np.inf], np.nan).dropna()
    if non_null_target.nunique(dropna=True) < 2:
        raise ModelTrainingError("Target column must contain at least 2 unique values.")

    if len(df.columns) <= 1:
        raise ModelTrainingError("Dataset must contain at least one feature column.")

    return resolved_target


def validate_regression_target(target: pd.Series) -> None:
    """Validate that regression target is numeric and has usable finite values."""
    numeric_target = pd.to_numeric(target, errors="coerce").replace(
        [np.inf, -np.inf],
        np.nan,
    )

    if numeric_target.dropna().empty:
        raise ModelTrainingError("Regression target must contain numeric values.")

    if numeric_target.dropna().nunique() < 2:
        raise ModelTrainingError(
            "Regression target must contain at least 2 unique numeric values.",
        )

    non_numeric_count = int(target.notna().sum() - numeric_target.notna().sum())
    if non_numeric_count > 0:
        raise ModelTrainingError(
            "Regression target contains non-numeric values and cannot be modeled safely.",
        )


def validate_training_split(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    problem_type: str,
) -> list[str]:
    """Validate split output and return non-fatal warnings."""
    warnings: list[str] = []

    if x_train.empty or x_test.empty:
        raise ModelTrainingError("Train-test split produced an empty feature split.")

    if len(y_train) == 0 or len(y_test) == 0:
        raise ModelTrainingError("Train-test split produced an empty target split.")

    if problem_type in CLASSIFICATION_TYPES:
        train_classes = set(pd.Series(y_train).astype(str).unique())
        test_classes = set(pd.Series(y_test).astype(str).unique())
        unseen_test_classes = sorted(test_classes - train_classes)

        if unseen_test_classes:
            warnings.append(
                "Some classes appear only in the test split and not in training: "
                f"{unseen_test_classes}. Metrics may be unstable.",
            )

        if len(train_classes) < 2:
            raise ModelTrainingError(
                "Training split contains fewer than 2 target classes.",
            )

    return warnings


def get_baseline_models(problem_type: str, random_state: int) -> dict[str, Any]:
    """Return small, reliable baseline models."""
    parallel_jobs = get_int_config("performance.parallel_jobs", -1)
    rf_estimators = get_int_config(
        "modeling.random_forest_estimators",
        200,
        minimum=10,
    )
    rf_min_samples_leaf = get_int_config(
        "modeling.random_forest_min_samples_leaf",
        2,
        minimum=1,
    )
    logistic_max_iter = get_int_config(
        "modeling.logistic_max_iter",
        1_000,
        minimum=100,
    )

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


def encode_classification_targets(
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[np.ndarray, np.ndarray, LabelEncoder, list[str], list[str]]:
    """Encode classification targets using the full split label set."""
    label_encoder = LabelEncoder()
    combined = pd.concat([y_train, y_test], axis=0).astype(str)
    label_encoder.fit(combined)

    y_train_encoded = np.asarray(label_encoder.transform(y_train.astype(str)))
    y_test_encoded = np.asarray(label_encoder.transform(y_test.astype(str)))
    class_labels = [str(label) for label in label_encoder.classes_]

    warnings: list[str] = []
    train_labels = set(y_train.astype(str).unique())
    test_labels = set(y_test.astype(str).unique())
    labels_missing_from_train = sorted(test_labels - train_labels)

    if labels_missing_from_train:
        warnings.append(
            "One or more labels are present only in the test set: "
            f"{labels_missing_from_train}. The model cannot learn these labels.",
        )

    return y_train_encoded, y_test_encoded, label_encoder, class_labels, warnings


def prepare_regression_targets(
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert regression targets to finite float arrays."""
    train = pd.to_numeric(y_train, errors="coerce").replace([np.inf, -np.inf], np.nan)
    test = pd.to_numeric(y_test, errors="coerce").replace([np.inf, -np.inf], np.nan)

    if train.isna().any() or test.isna().any():
        raise ModelTrainingError(
            "Regression target contains missing, non-finite, or non-numeric values "
            "after preprocessing.",
        )

    return np.asarray(train, dtype=float), np.asarray(test, dtype=float)


def safe_metric(value: Any, digits: int = 4) -> float | None:
    """Convert metric output to finite rounded float."""
    try:
        metric = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(metric):
        return None

    return round(metric, digits)


def add_metric(metrics: dict[str, float], name: str, value: Any) -> None:
    """Add metric only if it is finite."""
    metric = safe_metric(value)
    if metric is not None:
        metrics[name] = metric


def evaluate_classification(
    model: Pipeline,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    problem_type: str,
) -> dict[str, float]:
    """Evaluate classification baseline."""
    metrics: dict[str, float] = {}

    add_metric(metrics, "accuracy", accuracy_score(y_test, y_pred))
    add_metric(metrics, "balanced_accuracy", balanced_accuracy_score(y_test, y_pred))
    add_metric(
        metrics,
        "precision_weighted",
        precision_score(y_test, y_pred, average="weighted", zero_division=0),
    )
    add_metric(
        metrics,
        "recall_weighted",
        recall_score(y_test, y_pred, average="weighted", zero_division=0),
    )
    add_metric(
        metrics,
        "f1_score",
        f1_score(y_test, y_pred, average="weighted", zero_division=0),
    )
    add_metric(
        metrics,
        "f1_macro",
        f1_score(y_test, y_pred, average="macro", zero_division=0),
    )

    if problem_type == "binary_classification" and len(np.unique(y_test)) <= 2:
        add_metric(
            metrics,
            "precision_binary",
            precision_score(y_test, y_pred, average="binary", zero_division=0),
        )
        add_metric(
            metrics,
            "recall_binary",
            recall_score(y_test, y_pred, average="binary", zero_division=0),
        )
        add_metric(
            metrics,
            "f1_binary",
            f1_score(y_test, y_pred, average="binary", zero_division=0),
        )

    try:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(x_test)

            if problem_type == "binary_classification" and y_proba.shape[1] == 2:
                add_metric(metrics, "roc_auc", roc_auc_score(y_test, y_proba[:, 1]))

            elif problem_type == "multiclass_classification" and y_proba.shape[1] > 2:
                add_metric(
                    metrics,
                    "roc_auc_ovr_weighted",
                    roc_auc_score(
                        y_test,
                        y_proba,
                        multi_class="ovr",
                        average="weighted",
                    ),
                )

    except (AttributeError, IndexError, TypeError, ValueError) as error:
        logger.warning("Probability-based metrics skipped: %s", error)

    return metrics


def evaluate_regression(y_test: Any, y_pred: np.ndarray) -> dict[str, float]:
    """Evaluate regression baseline."""
    y_test_array = np.asarray(y_test, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)

    if y_test_array.size == 0 or y_pred_array.size == 0:
        raise ModelTrainingError("Regression evaluation received empty arrays.")

    if not np.isfinite(y_test_array).all() or not np.isfinite(y_pred_array).all():
        raise ModelTrainingError("Regression evaluation received non-finite values.")

    mse = mean_squared_error(y_test_array, y_pred_array)
    rmse = float(np.sqrt(mse))

    metrics: dict[str, float] = {}
    add_metric(metrics, "mae", mean_absolute_error(y_test_array, y_pred_array))
    add_metric(metrics, "mse", mse)
    add_metric(metrics, "rmse", rmse)
    add_metric(metrics, "r2_score", r2_score(y_test_array, y_pred_array))

    try:
        non_zero_mask = y_test_array != 0

        if int(non_zero_mask.sum()) > 0:
            mape = (
                np.mean(
                    np.abs(
                        (y_test_array[non_zero_mask] - y_pred_array[non_zero_mask])
                        / y_test_array[non_zero_mask],
                    ),
                )
                * 100
            )
            add_metric(metrics, "mape", mape)

    except (FloatingPointError, TypeError, ValueError) as error:
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

    Classification folds are capped by minority-class count to avoid invalid or
    unstable StratifiedKFold splits.
    """
    if requested_folds < 2:
        return None

    y_array = np.asarray(y)

    if len(y_array) < requested_folds:
        requested_folds = len(y_array)

    if requested_folds < 2:
        return None

    if problem_type in CLASSIFICATION_TYPES:
        y_series = pd.Series(y_array)
        class_counts = y_series.value_counts(dropna=False)

        if class_counts.empty or len(class_counts) < 2:
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


def get_cv_scoring(problem_type: str) -> str:
    """Resolve sklearn-compatible cross-validation scoring name."""
    if problem_type in CLASSIFICATION_TYPES:
        configured = str(
            get_config_value("metrics.classification_default", "f1_weighted"),
        ).strip()
        return SKLEARN_CLASSIFICATION_SCORING_ALIASES.get(configured, "f1_weighted")

    return "neg_root_mean_squared_error"


def run_cross_validation(
    pipeline: Pipeline,
    x: pd.DataFrame,
    y: Any,
    problem_type: str,
    cv: KFold | StratifiedKFold,
) -> dict[str, float]:
    """Run optional cross-validation on training data."""
    try:
        scoring = get_cv_scoring(problem_type)
        scores = cross_val_score(
            pipeline,
            x,
            np.asarray(y),
            cv=cv,
            scoring=scoring,
            n_jobs=get_int_config("performance.parallel_jobs", -1),
            error_score=np.nan,
        )

        scores = np.asarray(scores, dtype=float)
        scores = scores[np.isfinite(scores)]

        if scores.size == 0:
            logger.warning("Cross-validation produced no finite scores.")
            return {}

        if problem_type == "regression":
            scores = -scores

        return {
            "cv_mean": round(float(scores.mean()), 4),
            "cv_std": round(float(scores.std()), 4),
        }

    except (AttributeError, TypeError, ValueError, RuntimeError) as error:
        logger.warning("Cross-validation skipped: %s", error)
        return {}


def normalize_selection_metric(problem_type: str) -> tuple[str, bool]:
    """Resolve selection metric and comparison direction."""
    if problem_type in CLASSIFICATION_TYPES:
        configured = str(
            get_config_value("metrics.classification_default", "f1_score"),
        ).strip()
        primary_metric = CLASSIFICATION_SELECTION_ALIASES.get(configured, "f1_score")
        return primary_metric, True

    configured = str(get_config_value("metrics.regression_default", "rmse")).strip()
    primary_metric = (
        configured if configured in REGRESSION_SELECTION_METRICS else "rmse"
    )
    higher_is_better = primary_metric not in LOWER_IS_BETTER_REGRESSION
    return primary_metric, higher_is_better


def select_best_model(results: dict[str, Any], problem_type: str) -> dict[str, Any]:
    """Select best baseline model using config-driven metric."""
    primary_metric, higher_is_better = normalize_selection_metric(problem_type)

    best_model_name: str | None = None
    best_score: float | None = None

    for model_name, model_result in results.items():
        if not isinstance(model_result, dict):
            continue

        metrics = model_result.get("metrics", {})
        if not isinstance(metrics, dict):
            continue

        raw_score = metrics.get(primary_metric)

        if raw_score is None and problem_type in CLASSIFICATION_TYPES:
            raw_score = metrics.get("f1_score")

        if raw_score is None:
            continue

        score = safe_metric(raw_score)
        if score is None:
            continue

        if best_score is None:
            best_model_name = str(model_name)
            best_score = score
            continue

        if higher_is_better and score > best_score:
            best_model_name = str(model_name)
            best_score = score

        if not higher_is_better and score < best_score:
            best_model_name = str(model_name)
            best_score = score

    if best_model_name is None or best_score is None:
        raise ModelTrainingError("Could not select best baseline model.")

    return {
        "model_name": best_model_name,
        "selection_metric": primary_metric,
        "score": float(best_score),
        "higher_is_better": bool(higher_is_better),
    }


def build_model_failure_record(model_name: str, error: Exception) -> dict[str, Any]:
    """Build JSON-safe model failure record."""
    return {
        "model_name": model_name,
        "error_type": error.__class__.__name__,
        "message": str(error),
    }


def train_single_baseline_model(
    model_name: str,
    model: Any,
    preprocessor_template: Any,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train_model: np.ndarray,
    y_test_model: np.ndarray,
    problem_type: str,
    enable_cross_validation: bool,
    safe_cv: KFold | StratifiedKFold | None,
) -> dict[str, Any]:
    """Train and evaluate one baseline model."""
    pipeline = Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor_template)),
            ("model", clone(model)),
        ],
    )

    pipeline.fit(x_train, y_train_model)
    y_pred = np.asarray(pipeline.predict(x_test))

    if problem_type in CLASSIFICATION_TYPES:
        metrics = evaluate_classification(
            model=pipeline,
            x_test=x_test,
            y_test=y_test_model,
            y_pred=y_pred,
            problem_type=problem_type,
        )
        confusion = confusion_matrix(y_test_model, y_pred).tolist()
    elif problem_type == "regression":
        metrics = evaluate_regression(y_test=y_test_model, y_pred=y_pred)
        confusion = None
    else:
        raise ModelTrainingError(f"Unsupported problem type: {problem_type}")

    if enable_cross_validation and safe_cv is not None:
        cv_results = run_cross_validation(
            pipeline=pipeline,
            x=x_train,
            y=y_train_model,
            problem_type=problem_type,
            cv=safe_cv,
        )
        metrics.update(cv_results)

    if not metrics:
        raise ModelTrainingError(f"No valid metrics were produced for {model_name}.")

    return {
        "metrics": metrics,
        "model_object": pipeline,
        "confusion_matrix": confusion,
    }


def build_preprocessing_summary(
    preprocessing_info: Mapping[str, Any],
) -> dict[str, Any]:
    """Build JSON-safe preprocessing summary."""
    return {
        "numeric_columns": preprocessing_info.get("numeric_columns", []),
        "categorical_columns": preprocessing_info.get("categorical_columns", []),
        "datetime_columns": preprocessing_info.get("datetime_columns", []),
        "unsupported_columns_dropped": preprocessing_info.get(
            "unsupported_columns_dropped",
            [],
        ),
        "id_like_columns_dropped": preprocessing_info.get(
            "id_like_columns_dropped", []
        ),
        "high_cardinality_columns_dropped": preprocessing_info.get(
            "high_cardinality_columns_dropped",
            [],
        ),
        "columns_dropped_before_modeling": preprocessing_info.get(
            "columns_dropped_before_modeling",
            [],
        ),
        "total_features_before_encoding": int(
            preprocessing_info.get("total_features_before_encoding", 0) or 0,
        ),
        "warnings": preprocessing_info.get("warnings", []),
    }


def build_evaluation_details(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    test_size: float,
    random_state: int,
    enable_cross_validation: bool,
    requested_cv_folds: int,
    safe_cv: KFold | StratifiedKFold | None,
    cv_warning: str | None,
    class_labels: list[str],
    label_encoder: LabelEncoder | None,
    results: dict[str, Any],
    split_warnings: list[str],
    model_failures: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build JSON-safe evaluation metadata."""
    return {
        "test_size": float(test_size),
        "random_state": int(random_state),
        "cross_validation_enabled": bool(
            enable_cross_validation and safe_cv is not None
        ),
        "requested_cv_folds": int(requested_cv_folds),
        "actual_cv_folds": (
            int(getattr(safe_cv, "n_splits", 0))
            if enable_cross_validation and safe_cv is not None
            else None
        ),
        "cv_warning": cv_warning,
        "class_labels": class_labels,
        "label_encoder_used": label_encoder is not None,
        "train_shape": [int(x_train.shape[0]), int(x_train.shape[1])],
        "test_shape": [int(x_test.shape[0]), int(x_test.shape[1])],
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "split_warnings": split_warnings,
        "model_failures": model_failures,
        "confusion_matrices": {
            model_name: model_result["confusion_matrix"]
            for model_name, model_result in results.items()
            if model_result.get("confusion_matrix") is not None
        },
    }


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

        resolved_target_column = validate_inputs(df, target_column, problem_type)
        normalized_problem_type = normalize_problem_type(problem_type)

        random_state = get_int_config("modeling.random_state", 42)
        test_size = get_float_config(
            "modeling.test_size",
            0.2,
            minimum=0.05,
            maximum=0.5,
        )
        enable_cross_validation = as_bool(
            get_config_value("modeling.enable_cross_validation", False),
        )
        requested_cv_folds = get_int_config("modeling.cv_folds", 5, minimum=2)

        if normalized_problem_type == "regression":
            validate_regression_target(df[resolved_target_column])

        preprocessing_info = build_preprocessing_pipeline(
            df=df,
            target_column=resolved_target_column,
        )
        preprocessor_template = preprocessing_info["preprocessor"]

        x_train, x_test, y_train, y_test = create_train_test_split(
            df=df,
            target_column=resolved_target_column,
            problem_type=normalized_problem_type,
            test_size=test_size,
            random_state=random_state,
        )

        split_warnings = validate_training_split(
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            problem_type=normalized_problem_type,
        )

        label_encoder: LabelEncoder | None = None
        class_labels: list[str] = []

        if normalized_problem_type in CLASSIFICATION_TYPES:
            (
                y_train_model,
                y_test_model,
                label_encoder,
                class_labels,
                encoding_warnings,
            ) = encode_classification_targets(y_train=y_train, y_test=y_test)
            split_warnings.extend(encoding_warnings)
        else:
            y_train_model, y_test_model = prepare_regression_targets(
                y_train=y_train,
                y_test=y_test,
            )

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
        model_failures: list[dict[str, Any]] = []

        for model_name, model in models.items():
            logger.info("Training baseline model: %s", model_name)

            try:
                results[model_name] = train_single_baseline_model(
                    model_name=model_name,
                    model=model,
                    preprocessor_template=preprocessor_template,
                    x_train=x_train,
                    x_test=x_test,
                    y_train_model=y_train_model,
                    y_test_model=y_test_model,
                    problem_type=normalized_problem_type,
                    enable_cross_validation=enable_cross_validation,
                    safe_cv=safe_cv,
                )
                logger.info("Completed model: %s", model_name)

            except (AttributeError, TypeError, ValueError, RuntimeError) as error:
                logger.warning("Baseline model failed: %s error=%s", model_name, error)
                model_failures.append(build_model_failure_record(model_name, error))

        if not results:
            raise ModelTrainingError(
                "All baseline models failed. Check preprocessing and target values.",
            )

        best_model = select_best_model(results, normalized_problem_type)

        output: dict[str, Any] = {
            "problem_type": normalized_problem_type,
            "target_column": resolved_target_column,
            "models_trained": list(results.keys()),
            "models": list(results.keys()),
            "models_attempted": list(models.keys()),
            "results": {
                model_name: model_result["metrics"]
                for model_name, model_result in results.items()
            },
            "trained_model_objects": {
                model_name: model_result["model_object"]
                for model_name, model_result in results.items()
            },
            "best_model": best_model,
            "preprocessing_summary": build_preprocessing_summary(preprocessing_info),
            "evaluation_details": build_evaluation_details(
                x_train=x_train,
                x_test=x_test,
                y_train=y_train,
                y_test=y_test,
                test_size=test_size,
                random_state=random_state,
                enable_cross_validation=enable_cross_validation,
                requested_cv_folds=requested_cv_folds,
                safe_cv=safe_cv,
                cv_warning=cv_warning,
                class_labels=class_labels,
                label_encoder=label_encoder,
                results=results,
                split_warnings=split_warnings,
                model_failures=model_failures,
            ),
            "runtime_objects": {
                "sample_features": x_test.copy(),
                "sample_target": y_test.copy(),
                "train_features": x_train.copy(),
                "test_features": x_test.copy(),
                "label_encoder": label_encoder,
            },
            "model_failures": model_failures,
            "warnings": split_warnings
            + ([cv_warning] if cv_warning else [])
            + [failure["message"] for failure in model_failures],
            "message": "Baseline model training completed successfully.",
            "note": "These are baseline sanity-check models, not final optimized models.",
        }

        logger.info("Baseline model training completed successfully")
        return output

    except ModelTrainingError:
        raise
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
    ) as error:
        logger.exception("Baseline model training failed.")
        raise ModelTrainingError(
            "Baseline model training failed.",
            error_detail=str(error),
        ) from error


def get_sample_features_for_explainability(
    baseline_results: dict[str, Any],
) -> pd.DataFrame | None:
    """Extract sample features from baseline output for explainability."""
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
    if not isinstance(baseline_results, dict):
        return {}

    cleaned = dict(baseline_results)
    cleaned.pop("trained_model_objects", None)
    cleaned.pop("runtime_objects", None)

    results = cleaned.get("results")
    if isinstance(results, dict):
        cleaned["results"] = {
            str(model_name): dict(metrics) if isinstance(metrics, dict) else metrics
            for model_name, metrics in results.items()
        }

    return cleaned


if __name__ == "__main__":
    from src.audit.problem_detector import detect_problem_type
    from src.audit.profiler import load_dataset

    dataset_path = "data/sample/student_mark.csv"
    target_column = "Grade"

    dataframe = load_dataset(dataset_path)
    problem_info = detect_problem_type(dataframe, target_column)

    output = train_baseline_models(
        df=dataframe,
        target_column=target_column,
        problem_type=problem_info["problem_type"],
    )

    printable_output = strip_runtime_objects(output)
    print(printable_output)
