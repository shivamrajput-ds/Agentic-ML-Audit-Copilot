# pyright: reportPrivateImportUsage=false
from __future__ import annotations

import json
import math
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import get_config_value
from src.utils.exceptions import MLflowTrackingError
from src.utils.logger import get_logger

logger = get_logger(__name__)

TRUE_VALUES = {"true", "1", "yes", "y", "on"}
BLOCKED_ARTIFACT_KEYS = {
    "trained_model_objects",
    "runtime_objects",
    "model_object",
    "sample_features",
    "sample_target",
    "train_features",
    "test_features",
    "label_encoder",
    "df",
}
MAX_PARAM_VALUE_LENGTH = 500
MAX_MLFLOW_KEY_LENGTH = 250

_MLFLOW_MODULE: Any | None = None
_MLFLOW_SKLEARN_MODULE: Any | None = None
_INFER_SIGNATURE_FUNC: Any | None = None


def as_bool(value: Any) -> bool:
    """Convert config values safely into boolean."""
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() in TRUE_VALUES

    return bool(value)


def get_int_config(path: str, default: int) -> int:
    """Read integer config values with safe fallback."""
    try:
        return int(get_config_value(path, default))
    except (TypeError, ValueError):
        return default


def get_float_config(path: str, default: float) -> float:
    """Read float config values with safe fallback."""
    try:
        value = float(get_config_value(path, default))
    except (TypeError, ValueError):
        return default

    return value if math.isfinite(value) else default


def get_mlflow_modules() -> tuple[Any, Any, Any]:
    """
    Import MLflow lazily.

    This keeps the project importable when MLflow tracking is disabled or when a
    lightweight test environment does not have MLflow installed.
    """
    global _INFER_SIGNATURE_FUNC
    global _MLFLOW_MODULE
    global _MLFLOW_SKLEARN_MODULE

    if (
        _MLFLOW_MODULE is not None
        and _MLFLOW_SKLEARN_MODULE is not None
        and _INFER_SIGNATURE_FUNC is not None
    ):
        return _MLFLOW_MODULE, _MLFLOW_SKLEARN_MODULE, _INFER_SIGNATURE_FUNC

    try:
        import mlflow  # type: ignore[import-untyped]
        import mlflow.sklearn  # type: ignore[import-untyped,reportPrivateImportUsage]
        from mlflow.models import infer_signature  # type: ignore[import-untyped]
    except ImportError as error:
        raise MLflowTrackingError(
            "MLflow is not installed. Install mlflow or disable mlflow.enabled.",
            error_detail=str(error),
        ) from error

    _MLFLOW_MODULE = mlflow
    _MLFLOW_SKLEARN_MODULE = mlflow.sklearn
    _INFER_SIGNATURE_FUNC = infer_signature

    return _MLFLOW_MODULE, _MLFLOW_SKLEARN_MODULE, _INFER_SIGNATURE_FUNC


def safe_metric_name(name: str) -> str:
    """Convert metric names to MLflow-safe keys."""
    cleaned = (
        str(name)
        .lower()
        .strip()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "_", cleaned).strip("_")
    return (cleaned or "metric")[:MAX_MLFLOW_KEY_LENGTH]


def safe_param_name(name: str) -> str:
    """Convert parameter names to MLflow-safe keys."""
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(name).strip())
    return (cleaned.strip("_") or "param")[:MAX_MLFLOW_KEY_LENGTH]


def safe_param_value(value: Any) -> str | int | float | bool:
    """Convert values into MLflow-safe parameter values."""
    if isinstance(value, bool):
        return value

    if is_number(value):
        numeric_value = float(value)
        return int(numeric_value) if numeric_value.is_integer() else numeric_value

    text = str(value)
    if len(text) > MAX_PARAM_VALUE_LENGTH:
        return text[: MAX_PARAM_VALUE_LENGTH - 15] + "...[truncated]"

    return text


def safe_artifact_path(path: str) -> str:
    """Normalize MLflow artifact path and avoid absolute/path-traversal values."""
    normalized = str(path).strip().replace("\\", "/").strip("/")
    parts = [part for part in normalized.split("/") if part and part not in {".", ".."}]
    return "/".join(parts) or "baseline_model"


def is_number(value: Any) -> bool:
    """Return True for finite numeric values that MLflow can log as metrics."""
    if isinstance(value, bool) or value is None:
        return False

    try:
        number = float(value)
    except (TypeError, ValueError):
        return False

    return math.isfinite(number)


def json_safe_scalar(value: Any) -> Any:
    """Convert scalar values into JSON-safe primitives."""
    if value is None or isinstance(value, str | bool):
        return value

    if is_number(value):
        number = float(value)
        return int(number) if number.is_integer() else number

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    return str(value)


def remove_unserializable_objects(data: Any) -> Any:
    """Remove runtime/model objects before JSON artifact logging."""
    if isinstance(data, dict):
        cleaned: dict[str, Any] = {}

        for key, value in data.items():
            if str(key) in BLOCKED_ARTIFACT_KEYS:
                continue

            cleaned[str(key)] = remove_unserializable_objects(value)

        return cleaned

    if isinstance(data, list):
        return [remove_unserializable_objects(item) for item in data]

    if isinstance(data, tuple | set):
        return [remove_unserializable_objects(item) for item in data]

    if isinstance(data, pd.DataFrame):
        return {
            "type": "DataFrame",
            "shape": [int(data.shape[0]), int(data.shape[1])],
            "columns": [str(column) for column in data.columns],
        }

    if isinstance(data, pd.Series):
        return {
            "type": "Series",
            "shape": [int(data.shape[0])],
            "name": str(data.name),
            "dtype": str(data.dtype),
        }

    if isinstance(data, Path):
        return str(data)

    return json_safe_scalar(data)


def log_json_artifact(data: dict[str, Any], filename: str) -> None:
    """Log JSON artifact safely to MLflow."""
    mlflow, _mlflow_sklearn, _infer_signature = get_mlflow_modules()
    safe_filename = Path(filename).name or "artifact.json"
    safe_data = remove_unserializable_objects(data)

    with tempfile.TemporaryDirectory(prefix="agentic_mlflow_") as temp_dir:
        file_path = Path(temp_dir) / safe_filename
        file_path.write_text(
            json.dumps(safe_data, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        mlflow.log_artifact(str(file_path))


def validate_baseline_results(baseline_results: dict[str, Any]) -> None:
    """Validate baseline results before MLflow tracking."""
    if not baseline_results:
        raise MLflowTrackingError("Baseline results are required for MLflow tracking.")

    if not isinstance(baseline_results, dict):
        raise MLflowTrackingError("Baseline results must be a dictionary.")

    if baseline_results.get("skipped"):
        raise MLflowTrackingError(
            "Baseline modeling was skipped, so there are no metrics to track.",
        )

    results = baseline_results.get("results")

    if not results:
        raise MLflowTrackingError("No model results found for MLflow tracking.")

    if not isinstance(results, dict):
        raise MLflowTrackingError("Baseline results must contain a valid results dict.")


def setup_mlflow(experiment_name: str) -> Any:
    """Configure MLflow tracking URI and experiment."""
    mlflow, _mlflow_sklearn, _infer_signature = get_mlflow_modules()
    clean_experiment_name = str(experiment_name).strip()

    if not clean_experiment_name:
        raise MLflowTrackingError("MLflow experiment name cannot be empty.")

    tracking_uri = get_config_value("mlflow.tracking_uri", None)

    if tracking_uri:
        mlflow.set_tracking_uri(str(tracking_uri))

    return mlflow.set_experiment(clean_experiment_name)


def log_param_safely(name: str, value: Any) -> None:
    """Log one MLflow param without crashing on invalid/oversized values."""
    mlflow, _mlflow_sklearn, _infer_signature = get_mlflow_modules()

    try:
        mlflow.log_param(safe_param_name(name), safe_param_value(value))
    except Exception as error:  # noqa: BLE001 - MLflow can raise provider-specific errors.
        logger.warning("Skipping MLflow param %s: %s", name, error)


def log_metric_safely(name: str, value: Any) -> None:
    """Log one MLflow metric without crashing on invalid values."""
    mlflow, _mlflow_sklearn, _infer_signature = get_mlflow_modules()

    if not is_number(value):
        return

    try:
        mlflow.log_metric(safe_metric_name(name), float(value))
    except Exception as error:  # noqa: BLE001 - MLflow can raise provider-specific errors.
        logger.warning("Skipping MLflow metric %s: %s", name, error)


def log_common_parent_params(
    baseline_results: dict[str, Any],
    problem_type: str,
    target_column: str,
    models_trained: list[Any],
    results: dict[str, Any],
    best_model: dict[str, Any],
) -> None:
    """Log parent run parameters."""
    log_param_safely("run_type", "baseline_experiment_parent")
    log_param_safely("problem_type", problem_type)
    log_param_safely("target_column", target_column)
    log_param_safely("models_count", len(results))
    log_param_safely("models_trained", ", ".join(map(str, models_trained)))
    log_param_safely("timestamp", datetime.now(UTC).isoformat())

    evaluation_details = baseline_results.get("evaluation_details", {})
    if isinstance(evaluation_details, dict):
        log_param_safely(
            "cross_validation_enabled",
            evaluation_details.get("cross_validation_enabled", False),
        )
        log_param_safely(
            "requested_cv_folds",
            evaluation_details.get("requested_cv_folds", "N/A"),
        )
        log_param_safely(
            "actual_cv_folds",
            evaluation_details.get("actual_cv_folds", "N/A"),
        )
        log_param_safely("test_size", evaluation_details.get("test_size", "N/A"))
        log_param_safely(
            "random_state",
            evaluation_details.get("random_state", "N/A"),
        )

    if best_model:
        log_param_safely("best_model_name", best_model.get("model_name", "N/A"))
        log_param_safely("selection_metric", best_model.get("selection_metric", "N/A"))

        if is_number(best_model.get("score")):
            log_metric_safely("best_model_score", best_model["score"])


def log_preprocessing_params(preprocessing_summary: dict[str, Any]) -> None:
    """Log preprocessing summary parameters for each child run."""
    log_param_safely(
        "numeric_columns_count",
        len(preprocessing_summary.get("numeric_columns", [])),
    )
    log_param_safely(
        "categorical_columns_count",
        len(preprocessing_summary.get("categorical_columns", [])),
    )
    log_param_safely(
        "datetime_columns_count",
        len(preprocessing_summary.get("datetime_columns", [])),
    )
    log_param_safely(
        "unsupported_columns_dropped_count",
        len(preprocessing_summary.get("unsupported_columns_dropped", [])),
    )
    log_param_safely(
        "id_like_columns_dropped_count",
        len(preprocessing_summary.get("id_like_columns_dropped", [])),
    )
    log_param_safely(
        "high_cardinality_columns_dropped_count",
        len(preprocessing_summary.get("high_cardinality_columns_dropped", [])),
    )
    log_param_safely(
        "total_features_before_encoding",
        preprocessing_summary.get("total_features_before_encoding", "N/A"),
    )


def log_metrics(metrics: dict[str, Any]) -> None:
    """Log numeric metrics safely."""
    for metric_name, metric_value in metrics.items():
        log_metric_safely(metric_name, metric_value)


def _infer_model_signature(
    model_object: Any,
    sample_input: pd.DataFrame,
) -> tuple[Any | None, pd.DataFrame | None]:
    """Best-effort model signature inference for MLflow model logging."""
    _mlflow, _mlflow_sklearn, infer_signature = get_mlflow_modules()

    try:
        input_example = sample_input.head(5).copy()
        predictions_sample = model_object.predict(input_example)
        signature = infer_signature(input_example, predictions_sample)
        return signature, input_example
    except Exception as error:  # noqa: BLE001 - signature inference is optional.
        logger.warning(
            "Could not infer model signature. Logging model without signature: %s",
            error,
        )
        return None, None


def try_log_best_model(
    model_name: str,
    model_object: Any,
    artifact_path: str,
    sample_input: pd.DataFrame | None,
) -> str | None:
    """
    Try to log best model artifact.

    Failure should not fail the entire audit; metrics are more important than
    model artifact persistence.
    """
    mlflow, mlflow_sklearn, _infer_signature = get_mlflow_modules()

    if model_object is None:
        logger.warning("Best model object not found. Skipping model logging.")
        return None

    try:
        normalized_artifact_path = safe_artifact_path(artifact_path)
        log_model_kwargs: dict[str, Any] = {
            "sk_model": model_object,
            "artifact_path": normalized_artifact_path,
        }

        if sample_input is not None and isinstance(sample_input, pd.DataFrame):
            if not sample_input.empty:
                signature, input_example = _infer_model_signature(
                    model_object,
                    sample_input,
                )

                if signature is not None:
                    log_model_kwargs["signature"] = signature

                if input_example is not None:
                    log_model_kwargs["input_example"] = input_example

        mlflow_sklearn.log_model(**log_model_kwargs)
        active_run = mlflow.active_run()

        if active_run is None:
            return None

        logged_model_uri = f"runs:/{active_run.info.run_id}/{normalized_artifact_path}"
        logger.info("Best model logged to MLflow: %s", logged_model_uri)
        return logged_model_uri

    except Exception as error:  # noqa: BLE001 - model logging is best-effort.
        logger.warning(
            "Model artifact logging skipped for %s. Metrics were still logged. Error: %s",
            model_name,
            error,
        )
        return None


def _normalize_baseline_sections(
    baseline_results: dict[str, Any],
) -> tuple[list[Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Normalize optional baseline result sections into expected types."""
    results = baseline_results.get("results", {})
    best_model = baseline_results.get("best_model", {})
    models_trained = baseline_results.get("models_trained", [])
    preprocessing_summary = baseline_results.get("preprocessing_summary", {})
    trained_model_objects = baseline_results.get("trained_model_objects", {})

    if not isinstance(results, dict):
        raise MLflowTrackingError("Baseline results must contain a valid results dict.")

    if not isinstance(best_model, dict):
        best_model = {}

    if not isinstance(models_trained, list):
        models_trained = list(results.keys())

    if not isinstance(preprocessing_summary, dict):
        preprocessing_summary = {}

    if not isinstance(trained_model_objects, dict):
        trained_model_objects = {}

    return (
        models_trained,
        results,
        best_model,
        preprocessing_summary,
        trained_model_objects,
    )


def _log_model_child_run(
    model_name: str,
    metrics: Any,
    problem_type: str,
    target_column: str,
    best_model: dict[str, Any],
    preprocessing_summary: dict[str, Any],
    log_models: bool,
    artifact_path: str,
    trained_model_objects: dict[str, Any],
    sample_input: pd.DataFrame | None,
) -> tuple[str, str | None, bool]:
    """Log a nested MLflow child run for a single baseline model."""
    mlflow, _mlflow_sklearn, _infer_signature = get_mlflow_modules()
    logged_model_uri: str | None = None
    model_artifact_logged = False

    with mlflow.start_run(run_name=str(model_name), nested=True) as child_run:
        is_best_model = model_name == best_model.get("model_name")

        log_param_safely("run_type", "baseline_model")
        log_param_safely("problem_type", problem_type)
        log_param_safely("target_column", target_column)
        log_param_safely("model_name", model_name)
        log_param_safely("is_best_model", is_best_model)
        log_param_safely("selection_metric", best_model.get("selection_metric", "N/A"))

        log_preprocessing_params(preprocessing_summary)

        if isinstance(metrics, dict):
            log_metrics(metrics)
        else:
            logger.warning("Metrics for model %s are not a dict. Skipping.", model_name)

        if is_best_model and is_number(best_model.get("score")):
            log_metric_safely("best_model_score", best_model["score"])

        if log_models and is_best_model:
            model_object = trained_model_objects.get(model_name)
            logged_model_uri = try_log_best_model(
                model_name=str(model_name),
                model_object=model_object,
                artifact_path=artifact_path,
                sample_input=sample_input,
            )
            model_artifact_logged = logged_model_uri is not None

        return child_run.info.run_id, logged_model_uri, model_artifact_logged


def track_baseline_experiment(
    baseline_results: dict[str, Any],
    experiment_name: str | None = None,
    sample_input: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Track baseline model metrics, parameters, artifacts and best model in MLflow.

    MLflow model artifact logging is best-effort. If artifact logging fails due
    to environment/skops/security issues, metrics and params are still logged.
    """
    try:
        logger.info("Starting MLflow tracking")

        validate_baseline_results(baseline_results)
        mlflow, _mlflow_sklearn, _infer_signature = get_mlflow_modules()

        experiment_name = experiment_name or str(
            get_config_value("mlflow.experiment_name", "agentic_ml_audit_baselines"),
        )

        log_models = as_bool(get_config_value("mlflow.log_models", True))
        log_artifacts = as_bool(get_config_value("mlflow.log_artifacts", True))
        artifact_path = safe_artifact_path(
            str(get_config_value("mlflow.artifact_path", "baseline_model")),
        )

        experiment = setup_mlflow(experiment_name)

        problem_type = str(baseline_results.get("problem_type", "N/A"))
        target_column = str(baseline_results.get("target_column", "N/A"))

        (
            models_trained,
            results,
            best_model,
            preprocessing_summary,
            trained_model_objects,
        ) = _normalize_baseline_sections(baseline_results)

        run_ids: dict[str, str] = {}
        logged_model_uri: str | None = None
        model_artifact_logged = False
        parent_run_id: str | None = None

        parent_run_name = (
            f"audit_baseline_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
        )
        parent_nested = mlflow.active_run() is not None

        with mlflow.start_run(
            run_name=parent_run_name, nested=parent_nested
        ) as parent_run:
            parent_run_id = parent_run.info.run_id

            log_common_parent_params(
                baseline_results=baseline_results,
                problem_type=problem_type,
                target_column=target_column,
                models_trained=models_trained,
                results=results,
                best_model=best_model,
            )

            if log_artifacts:
                log_json_artifact(baseline_results, "baseline_results.json")

            for model_name, metrics in results.items():
                logger.info("Logging MLflow child run for model: %s", model_name)

                child_run_id, child_model_uri, child_model_logged = (
                    _log_model_child_run(
                        model_name=str(model_name),
                        metrics=metrics,
                        problem_type=problem_type,
                        target_column=target_column,
                        best_model=best_model,
                        preprocessing_summary=preprocessing_summary,
                        log_models=log_models,
                        artifact_path=artifact_path,
                        trained_model_objects=trained_model_objects,
                        sample_input=sample_input,
                    )
                )

                run_ids[str(model_name)] = child_run_id

                if child_model_uri is not None:
                    logged_model_uri = child_model_uri

                model_artifact_logged = model_artifact_logged or child_model_logged

        output: dict[str, Any] = {
            "enabled": True,
            "experiment_name": experiment_name,
            "experiment_id": getattr(experiment, "experiment_id", None),
            "tracking_uri": mlflow.get_tracking_uri(),
            "parent_run_id": parent_run_id,
            "models_logged": [str(model_name) for model_name in results.keys()],
            "run_ids": run_ids,
            "best_model": best_model,
            "model_logging_enabled": log_models,
            "model_logged": model_artifact_logged,
            "logged_model_uri": logged_model_uri,
            "metrics_logged": True,
            "artifacts_logged": log_artifacts,
            "artifact_path": artifact_path,
            "message": "MLflow tracking completed successfully.",
        }

        logger.info("MLflow tracking completed successfully")
        return output

    except MLflowTrackingError:
        raise
    except Exception as error:  # noqa: BLE001 - normalize provider-specific MLflow errors.
        logger.exception("MLflow tracking failed.")
        raise MLflowTrackingError(
            "MLflow tracking failed.",
            error_detail=str(error),
        ) from error


if __name__ == "__main__":
    sample_baseline_results = {
        "problem_type": "binary_classification",
        "target_column": "Grade",
        "models_trained": ["Logistic Regression", "Random Forest Classifier"],
        "preprocessing_summary": {
            "numeric_columns": ["Age", "StudyHours"],
            "categorical_columns": ["Gender"],
            "datetime_columns": [],
            "total_features_before_encoding": 3,
        },
        "results": {
            "Logistic Regression": {
                "accuracy": 0.5,
                "precision_weighted": 0.25,
                "recall_weighted": 0.5,
                "f1_score": 0.3333,
                "roc_auc": 1.0,
            },
            "Random Forest Classifier": {
                "accuracy": 0.0,
                "precision_weighted": 0.0,
                "recall_weighted": 0.0,
                "f1_score": 0.0,
                "roc_auc": 0.0,
            },
        },
        "best_model": {
            "model_name": "Logistic Regression",
            "selection_metric": "f1_score",
            "score": 0.3333,
        },
        "trained_model_objects": {},
    }

    output = track_baseline_experiment(sample_baseline_results)
    print(output)
