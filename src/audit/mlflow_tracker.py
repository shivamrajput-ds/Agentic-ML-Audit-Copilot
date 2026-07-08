from datetime import datetime
from typing import Any, Optional

import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from src.utils.config import get_config_value
from src.utils.exceptions import MLflowTrackingError
from src.utils.logger import get_logger


logger = get_logger(__name__)


def _as_bool(value: Any) -> bool:
    """
    Convert config values safely to boolean.
    """
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.lower().strip() in {"true", "1", "yes", "y"}

    return bool(value)


def track_baseline_experiment(
    baseline_results: dict[str, Any],
    experiment_name: str | None = None,
    sample_input: Optional[pd.DataFrame] = None,
) -> dict[str, Any]:
    """
    Track baseline model metrics, parameters, and best model pipeline in MLflow.

    Args:
        baseline_results: Output of train_baseline_models().
        experiment_name: Optional override for the MLflow experiment name.
        sample_input: A small slice of the *raw* feature dataframe (e.g.
            X_test.head(3)) used only to log a model signature and input
            example in MLflow. This is purely metadata for the MLflow UI —
            it does not affect training or predictions. If not provided,
            the model is still logged, just without a signature.
    """
    try:
        logger.info("Starting MLflow tracking")

        if not baseline_results:
            raise MLflowTrackingError(
                "Baseline results are required for MLflow tracking."
            )

        experiment_name = experiment_name or get_config_value(
            "mlflow.experiment_name",
            "agentic_ml_audit_baselines",
        )

        log_models = _as_bool(
            get_config_value("mlflow.log_models", True)
        )

        artifact_path = get_config_value(
            "mlflow.artifact_path",
            "baseline_model",
        )

        mlflow.set_experiment(experiment_name)

        problem_type = baseline_results.get("problem_type", "N/A")
        target_column = baseline_results.get("target_column", "N/A")
        models_trained = baseline_results.get("models_trained", [])
        results = baseline_results.get("results", {})
        best_model = baseline_results.get("best_model", {})
        preprocessing_summary = baseline_results.get("preprocessing_summary", {})
        trained_model_objects = baseline_results.get("trained_model_objects", {})

        if not results:
            raise MLflowTrackingError("No model results found for MLflow tracking.")

        run_ids: dict[str, str] = {}
        logged_model_uri = None

        for model_name, metrics in results.items():
            logger.info(f"Logging MLflow run for model: {model_name}")

            with mlflow.start_run(run_name=model_name) as run:
                is_best_model = model_name == best_model.get("model_name")

                mlflow.log_param("problem_type", problem_type)
                mlflow.log_param("target_column", target_column)
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("models_trained", ", ".join(models_trained))
                mlflow.log_param("is_best_model", is_best_model)
                mlflow.log_param(
                    "selection_metric",
                    best_model.get("selection_metric", "N/A"),
                )
                mlflow.log_param("timestamp", datetime.now().isoformat())

                mlflow.log_param(
                    "numeric_columns_count",
                    len(preprocessing_summary.get("numeric_columns", [])),
                )
                mlflow.log_param(
                    "categorical_columns_count",
                    len(preprocessing_summary.get("categorical_columns", [])),
                )
                mlflow.log_param(
                    "total_features_before_encoding",
                    preprocessing_summary.get(
                        "total_features_before_encoding",
                        "N/A",
                    ),
                )

                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)):
                        mlflow.log_metric(metric_name, float(metric_value))

                if is_best_model and isinstance(best_model.get("score"), (int, float)):
                    mlflow.log_metric("best_model_score", float(best_model["score"]))

                if log_models and is_best_model:
                    model_object = trained_model_objects.get(model_name)

                    if model_object is not None:
                        signature = None
                        input_example = None

                        # Build a model signature/input example when the
                        # caller provides a sample of the raw feature
                        # dataframe. This is optional metadata: without
                        # it, the model still logs fine, but MLflow's UI
                        # and mlflow.models.predict() cannot validate
                        # input shape/types for you.
                        if sample_input is not None and not sample_input.empty:
                            try:
                                predictions_sample = model_object.predict(
                                    sample_input
                                )
                                signature = infer_signature(
                                    sample_input,
                                    predictions_sample,
                                )
                                input_example = sample_input
                            except Exception as signature_error:
                                logger.warning(
                                    "Could not infer model signature, "
                                    f"logging without one: {signature_error}"
                                )

                        mlflow.sklearn.log_model(
                            sk_model=model_object,
                            artifact_path=artifact_path,
                            signature=signature,
                            input_example=input_example,
                        )
                        logged_model_uri = f"runs:/{run.info.run_id}/{artifact_path}"
                        logger.info(
                            f"Best model logged to MLflow: {logged_model_uri}"
                        )
                    else:
                        logger.warning(
                            "Best model object not found. "
                            "Skipping MLflow model logging."
                        )

                run_ids[model_name] = run.info.run_id

        output = {
            "experiment_name": experiment_name,
            "models_logged": list(results.keys()),
            "run_ids": run_ids,
            "best_model": best_model,
            "model_logged": logged_model_uri is not None,
            "logged_model_uri": logged_model_uri,
            "message": "MLflow tracking completed successfully.",
        }

        logger.info("MLflow tracking completed successfully")
        return output

    except MLflowTrackingError:
        raise

    except Exception as error:
        logger.error(f"MLflow tracking failed: {error}")
        raise MLflowTrackingError(
            "MLflow tracking failed",
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
            "total_features_before_encoding": 3,
        },
        "results": {
            "Logistic Regression": {
                "accuracy": 0.5,
                "precision": 0.25,
                "recall": 0.5,
                "f1_score": 0.3333,
                "roc_auc": 1.0,
            },
            "Random Forest Classifier": {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
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