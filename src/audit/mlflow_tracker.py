# pyright: reportPrivateImportUsage=false
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn  # type: ignore[reportPrivateImportUsage]
import pandas as pd
from mlflow.models import infer_signature

from src.utils.config import get_config_value
from src.utils.exceptions import MLflowTrackingError
from src.utils.logger import get_logger


logger = get_logger(__name__)


def as_bool(value: Any) -> bool:
    """
    Convert config values safely into boolean.
    """
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.lower().strip() in {"true", "1", "yes", "y"}

    return bool(value)


def safe_metric_name(name: str) -> str:
    """
    Convert metric names to MLflow-safe keys.
    """
    return (
        str(name)
        .lower()
        .strip()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )


def is_number(value: Any) -> bool:
    """
    Return True for numeric values that MLflow can log as metrics.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def remove_unserializable_objects(data: Any) -> Any:
    """
    Remove model objects, DataFrames, and unserializable runtime objects before JSON logging.
    """
    blocked_keys = {
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

    if isinstance(data, dict):
        cleaned: dict[str, Any] = {}

        for key, value in data.items():
            if key in blocked_keys:
                continue

            cleaned[str(key)] = remove_unserializable_objects(value)

        return cleaned

    if isinstance(data, list):
        return [remove_unserializable_objects(item) for item in data]

    if isinstance(data, tuple):
        return [remove_unserializable_objects(item) for item in data]

    if isinstance(data, pd.DataFrame):
        return {
            "type": "DataFrame",
            "shape": list(data.shape),
            "columns": [str(column) for column in data.columns],
        }

    if isinstance(data, pd.Series):
        return {
            "type": "Series",
            "shape": list(data.shape),
            "name": str(data.name),
        }

    if isinstance(data, (str, int, float, bool)) or data is None:
        return data

    return str(data)


def log_json_artifact(data: dict[str, Any], filename: str) -> None:
    """
    Log JSON artifact safely to MLflow.
    """
    temp_dir = Path("artifacts/mlflow_temp")
    temp_dir.mkdir(parents=True, exist_ok=True)

    file_path = temp_dir / filename
    safe_data = remove_unserializable_objects(data)

    with file_path.open("w", encoding="utf-8") as file:
        json.dump(safe_data, file, indent=2, default=str)

    mlflow.log_artifact(str(file_path))


def validate_baseline_results(baseline_results: dict[str, Any]) -> None:
    """
    Validate baseline results before MLflow tracking.
    """
    if not baseline_results:
        raise MLflowTrackingError(
            "Baseline results are required for MLflow tracking."
        )

    results = baseline_results.get("results")

    if not results:
        raise MLflowTrackingError("No model results found for MLflow tracking.")

    if not isinstance(results, dict):
        raise MLflowTrackingError("Baseline results must contain a valid results dict.")


def setup_mlflow(experiment_name: str) -> None:
    """
    Configure MLflow tracking URI and experiment.
    """
    tracking_uri = get_config_value("mlflow.tracking_uri", None)

    if tracking_uri:
        mlflow.set_tracking_uri(str(tracking_uri))

    mlflow.set_experiment(experiment_name)


def log_common_parent_params(
    baseline_results: dict[str, Any],
    problem_type: str,
    target_column: str,
    models_trained: list[Any],
    results: dict[str, Any],
    best_model: dict[str, Any],
) -> None:
    """
    Log parent run parameters.
    """
    mlflow.log_param("run_type", "baseline_experiment_parent")
    mlflow.log_param("problem_type", problem_type)
    mlflow.log_param("target_column", target_column)
    mlflow.log_param("models_count", len(results))
    mlflow.log_param("models_trained", ", ".join(map(str, models_trained)))
    mlflow.log_param("timestamp", datetime.now().isoformat())

    evaluation_details = baseline_results.get("evaluation_details", {})
    if isinstance(evaluation_details, dict):
        mlflow.log_param(
            "cross_validation_enabled",
            evaluation_details.get("cross_validation_enabled", False),
        )
        mlflow.log_param(
            "requested_cv_folds",
            evaluation_details.get("requested_cv_folds", "N/A"),
        )
        mlflow.log_param(
            "actual_cv_folds",
            evaluation_details.get("actual_cv_folds", "N/A"),
        )

    if best_model:
        mlflow.log_param("best_model_name", best_model.get("model_name", "N/A"))
        mlflow.log_param(
            "selection_metric",
            best_model.get("selection_metric", "N/A"),
        )

        if is_number(best_model.get("score")):
            mlflow.log_metric("best_model_score", float(best_model["score"]))


def log_preprocessing_params(preprocessing_summary: dict[str, Any]) -> None:
    """
    Log preprocessing summary parameters for each child run.
    """
    mlflow.log_param(
        "numeric_columns_count",
        len(preprocessing_summary.get("numeric_columns", [])),
    )
    mlflow.log_param(
        "categorical_columns_count",
        len(preprocessing_summary.get("categorical_columns", [])),
    )
    mlflow.log_param(
        "datetime_columns_count",
        len(preprocessing_summary.get("datetime_columns", [])),
    )
    mlflow.log_param(
        "unsupported_columns_dropped_count",
        len(preprocessing_summary.get("unsupported_columns_dropped", [])),
    )
    mlflow.log_param(
        "total_features_before_encoding",
        preprocessing_summary.get("total_features_before_encoding", "N/A"),
    )


def log_metrics(metrics: dict[str, Any]) -> None:
    """
    Log numeric metrics safely.
    """
    for metric_name, metric_value in metrics.items():
        if is_number(metric_value):
            mlflow.log_metric(
                safe_metric_name(metric_name),
                float(metric_value),
            )


def try_log_best_model(
    model_name: str,
    model_object: Any,
    artifact_path: str,
    sample_input: pd.DataFrame | None,
) -> str | None:
    """
    Try to log best model artifact.

    Failure should not fail the entire audit; metrics are more important than model artifact.
    """
    if model_object is None:
        logger.warning("Best model object not found. Skipping model logging.")
        return None

    try:
        log_model_kwargs: dict[str, Any] = {
            "sk_model": model_object,
            "artifact_path": artifact_path,
        }

        if sample_input is not None and not sample_input.empty:
            try:
                input_example = sample_input.head(5)
                predictions_sample = model_object.predict(input_example)

                log_model_kwargs["signature"] = infer_signature(
                    input_example,
                    predictions_sample,
                )
                log_model_kwargs["input_example"] = input_example

            except Exception as signature_error:
                logger.warning(
                    "Could not infer model signature. Logging model without signature: %s",
                    signature_error,
                )

        mlflow.sklearn.log_model(**log_model_kwargs)
        active_run = mlflow.active_run()

        if active_run is None:
            return None

        logged_model_uri = f"runs:/{active_run.info.run_id}/{artifact_path}"
        logger.info("Best model logged to MLflow: %s", logged_model_uri)
        return logged_model_uri

    except Exception as error:
        logger.warning(
            "Model artifact logging skipped for %s. Metrics were still logged. Error: %s",
            model_name,
            error,
        )
        return None


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

        experiment_name = experiment_name or str(
            get_config_value("mlflow.experiment_name", "agentic_ml_audit_baselines")
        )

        log_models = as_bool(get_config_value("mlflow.log_models", True))
        log_artifacts = as_bool(get_config_value("mlflow.log_artifacts", True))
        artifact_path = str(get_config_value("mlflow.artifact_path", "baseline_model"))

        setup_mlflow(experiment_name)

        problem_type = str(baseline_results.get("problem_type", "N/A"))
        target_column = str(baseline_results.get("target_column", "N/A"))
        models_trained = baseline_results.get("models_trained", [])
        results = baseline_results.get("results", {})
        best_model = baseline_results.get("best_model", {})
        preprocessing_summary = baseline_results.get("preprocessing_summary", {})
        trained_model_objects = baseline_results.get("trained_model_objects", {})

        if not isinstance(models_trained, list):
            models_trained = list(results.keys())

        if not isinstance(results, dict):
            raise MLflowTrackingError("Baseline results must contain a valid results dict.")

        if not isinstance(preprocessing_summary, dict):
            preprocessing_summary = {}

        if not isinstance(trained_model_objects, dict):
            trained_model_objects = {}

        run_ids: dict[str, str] = {}
        logged_model_uri: str | None = None
        model_artifact_logged = False

        parent_run_name = f"audit_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with mlflow.start_run(run_name=parent_run_name) as parent_run:
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

                with mlflow.start_run(run_name=str(model_name), nested=True) as child_run:
                    is_best_model = model_name == best_model.get("model_name")

                    mlflow.log_param("run_type", "baseline_model")
                    mlflow.log_param("problem_type", problem_type)
                    mlflow.log_param("target_column", target_column)
                    mlflow.log_param("model_name", model_name)
                    mlflow.log_param("is_best_model", is_best_model)
                    mlflow.log_param(
                        "selection_metric",
                        best_model.get("selection_metric", "N/A"),
                    )

                    log_preprocessing_params(preprocessing_summary)

                    if isinstance(metrics, dict):
                        log_metrics(metrics)

                    if is_best_model and is_number(best_model.get("score")):
                        mlflow.log_metric("best_model_score", float(best_model["score"]))

                    if log_models and is_best_model:
                        model_object = trained_model_objects.get(model_name)
                        logged_model_uri = try_log_best_model(
                            model_name=str(model_name),
                            model_object=model_object,
                            artifact_path=artifact_path,
                            sample_input=sample_input,
                        )
                        model_artifact_logged = logged_model_uri is not None

                    run_ids[str(model_name)] = child_run.info.run_id

            parent_run_id = parent_run.info.run_id

        output = {
            "enabled": True,
            "experiment_name": experiment_name,
            "parent_run_id": parent_run_id,
            "models_logged": list(results.keys()),
            "run_ids": run_ids,
            "best_model": best_model,
            "model_logged": model_artifact_logged,
            "logged_model_uri": logged_model_uri,
            "metrics_logged": True,
            "artifacts_logged": log_artifacts,
            "message": "MLflow tracking completed successfully.",
        }

        logger.info("MLflow tracking completed successfully")
        return output

    except MLflowTrackingError:
        raise

    except Exception as error:
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
