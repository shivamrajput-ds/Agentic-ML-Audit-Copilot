from __future__ import annotations

import base64
import io
from typing import Any, cast

import numpy as np
import pandas as pd

from src.utils.config import get_config_value
from src.utils.exceptions import AuditCopilotException
from src.utils.logger import get_logger


logger = get_logger(__name__)


class ExplainabilityError(AuditCopilotException):
    """Raised when model explainability generation fails."""


TREE_MODEL_KEYWORDS = [
    "randomforest",
    "gradientboosting",
    "xgb",
    "xgboost",
    "lightgbm",
    "lgbm",
    "catboost",
    "decisiontree",
    "extratrees",
]

LINEAR_MODEL_KEYWORDS = [
    "logisticregression",
    "linearregression",
    "ridge",
    "lasso",
    "elasticnet",
    "sgd",
]


def as_bool(value: Any) -> bool:
    """
    Convert config values safely into boolean.
    """
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.lower().strip() in {"true", "1", "yes", "y"}

    return bool(value)


def records_from_dataframe(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Convert pandas records to Pylance-friendly list[dict[str, Any]].

    pandas returns list[dict[Hashable, Any]], which triggers strict Pylance.
    """
    raw_records = df.to_dict(orient="records")
    return [
        {str(key): value for key, value in row.items()}
        for row in raw_records
    ]


def get_explainability_config() -> dict[str, Any]:
    """
    Read explainability config from config.yaml.
    """
    return {
        "enabled": as_bool(get_config_value("explainability.enabled", False)),
        "run_shap": as_bool(get_config_value("explainability.run_shap", False)),
        "max_samples": int(get_config_value("explainability.max_samples", 200)),
        "top_n_features": int(get_config_value("explainability.top_n_features", 20)),
        "random_state": int(get_config_value("random_seed", 42)),
        "generate_plots": as_bool(
            get_config_value("explainability.generate_plots", True)
        ),
        "plot_max_features": int(
            get_config_value("explainability.plot_max_features", 20)
        ),
    }


def validate_inputs(
    baseline_results: dict[str, Any],
    sample_features: pd.DataFrame | None,
) -> None:
    """
    Validate explainability inputs.
    """
    if not baseline_results:
        raise ExplainabilityError("Baseline results are required for explainability.")

    best_model = baseline_results.get("best_model", {})
    best_model_name = best_model.get("model_name")

    if not best_model_name:
        raise ExplainabilityError("Best model name not found in baseline results.")

    trained_models = baseline_results.get("trained_model_objects", {})

    if not isinstance(trained_models, dict) or not trained_models:
        raise ExplainabilityError(
            "Trained model objects are missing. Explainability requires fitted models."
        )

    if best_model_name not in trained_models:
        raise ExplainabilityError(
            f"Best model object not found for model: {best_model_name}"
        )

    if sample_features is not None and not isinstance(sample_features, pd.DataFrame):
        raise ExplainabilityError("sample_features must be a pandas DataFrame.")


def get_best_model_pipeline(baseline_results: dict[str, Any]) -> Any:
    """
    Return fitted sklearn Pipeline for the selected best model.
    """
    best_model_name = baseline_results.get("best_model", {}).get("model_name")
    trained_models = baseline_results.get("trained_model_objects", {})

    if not isinstance(trained_models, dict):
        return None

    return trained_models.get(best_model_name)


def get_pipeline_parts(model_pipeline: Any) -> tuple[Any | None, Any]:
    """
    Extract preprocessor and estimator from a sklearn Pipeline.

    Expected pipeline:
    Pipeline([
        ("preprocessor", ...),
        ("model", ...)
    ])
    """
    if hasattr(model_pipeline, "named_steps"):
        preprocessor = model_pipeline.named_steps.get("preprocessor")
        estimator = model_pipeline.named_steps.get("model")

        if estimator is not None:
            return preprocessor, estimator

    return None, model_pipeline


def get_estimator_name(estimator: Any) -> str:
    """
    Return estimator class name.
    """
    return estimator.__class__.__name__


def normalize_model_name(name: str) -> str:
    """
    Normalize model name for keyword matching.
    """
    return (
        str(name)
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )


def is_tree_model(estimator: Any) -> bool:
    """
    Detect tree-based models.
    """
    normalized = normalize_model_name(get_estimator_name(estimator))
    return any(keyword in normalized for keyword in TREE_MODEL_KEYWORDS)


def is_linear_model(estimator: Any) -> bool:
    """
    Detect linear/coefficient-based models.
    """
    normalized = normalize_model_name(get_estimator_name(estimator))
    return any(keyword in normalized for keyword in LINEAR_MODEL_KEYWORDS)


def sample_dataframe(
    df: pd.DataFrame,
    max_samples: int,
    random_state: int,
) -> pd.DataFrame:
    """
    Sample dataframe for explainability to keep SHAP fast.
    """
    if df.empty:
        return df.copy()

    if len(df) <= max_samples:
        return df.copy()

    return df.sample(n=max_samples, random_state=random_state).copy()


def transform_features(
    preprocessor: Any | None,
    features: pd.DataFrame,
) -> Any:
    """
    Transform raw features using fitted preprocessor.
    """
    if preprocessor is None:
        return features

    return preprocessor.transform(features)


def get_feature_names(
    preprocessor: Any | None,
    raw_features: pd.DataFrame,
    transformed_features: Any,
) -> list[str]:
    """
    Get feature names after preprocessing.
    """
    if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
        try:
            names = preprocessor.get_feature_names_out()
            return [str(name) for name in names]
        except Exception as error:
            logger.warning("Could not get transformed feature names: %s", error)

    if isinstance(transformed_features, pd.DataFrame):
        return [str(col) for col in transformed_features.columns]

    shape = getattr(transformed_features, "shape", None)

    if shape is not None and len(shape) == 2:
        return [f"feature_{idx}" for idx in range(shape[1])]

    return [str(col) for col in raw_features.columns]


def to_numpy_array(values: Any) -> np.ndarray:
    """
    Convert dense/sparse/dataframe values to numpy array.
    """
    if hasattr(values, "toarray"):
        return np.asarray(values.toarray())

    if isinstance(values, pd.DataFrame):
        return values.to_numpy()

    return np.asarray(values)


def build_importance_dataframe(
    feature_names: list[str],
    importances: np.ndarray,
    top_n: int,
    importance_column: str = "importance",
) -> pd.DataFrame:
    """
    Build sorted feature importance dataframe.
    """
    importances = np.asarray(importances)

    if importances.ndim > 1:
        importances = np.mean(np.abs(importances), axis=0)

    if len(feature_names) != len(importances):
        min_len = min(len(feature_names), len(importances))
        feature_names = feature_names[:min_len]
        importances = importances[:min_len]

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            importance_column: importances.astype(float),
        }
    )

    importance_df["absolute_importance"] = importance_df[importance_column].abs()
    importance_df = importance_df.sort_values("absolute_importance", ascending=False)

    return importance_df.head(top_n).reset_index(drop=True)


def extract_tree_feature_importance(
    estimator: Any,
    feature_names: list[str],
    top_n: int,
) -> list[dict[str, Any]]:
    """
    Extract feature_importances_ from tree-based estimators.
    """
    if not hasattr(estimator, "feature_importances_"):
        return []

    importance_df = build_importance_dataframe(
        feature_names=feature_names,
        importances=np.asarray(estimator.feature_importances_),
        top_n=top_n,
        importance_column="importance",
    )

    importance_df["importance_type"] = "tree_feature_importance"
    return records_from_dataframe(importance_df)


def extract_linear_feature_importance(
    estimator: Any,
    feature_names: list[str],
    top_n: int,
) -> list[dict[str, Any]]:
    """
    Extract coefficient-based importance from linear models.
    """
    if not hasattr(estimator, "coef_"):
        return []

    coefficients = np.asarray(estimator.coef_)

    if coefficients.ndim == 2:
        if coefficients.shape[0] == 1:
            coefficients = coefficients[0]
        else:
            coefficients = np.mean(np.abs(coefficients), axis=0)

    importance_df = build_importance_dataframe(
        feature_names=feature_names,
        importances=coefficients,
        top_n=top_n,
        importance_column="importance",
    )

    importance_df["importance_type"] = "linear_coefficient"
    return records_from_dataframe(importance_df)


def get_builtin_feature_importance(
    estimator: Any,
    feature_names: list[str],
    top_n: int,
) -> dict[str, Any]:
    """
    Extract built-in model explainability from fitted estimator.
    """
    estimator_name = get_estimator_name(estimator)

    tree_importance = extract_tree_feature_importance(
        estimator=estimator,
        feature_names=feature_names,
        top_n=top_n,
    )

    if tree_importance:
        return {
            "available": True,
            "method": "tree_feature_importance",
            "model_type": estimator_name,
            "top_features": tree_importance,
            "message": "Feature importance extracted from estimator.feature_importances_.",
        }

    linear_importance = extract_linear_feature_importance(
        estimator=estimator,
        feature_names=feature_names,
        top_n=top_n,
    )

    if linear_importance:
        return {
            "available": True,
            "method": "linear_coefficients",
            "model_type": estimator_name,
            "top_features": linear_importance,
            "message": "Feature importance extracted from estimator coefficients.",
        }

    return {
        "available": False,
        "method": None,
        "model_type": estimator_name,
        "top_features": [],
        "message": (
            "Built-in feature importance is not available for this estimator. "
            "Use SHAP if enabled and supported."
        ),
    }


def import_shap_module() -> Any | None:
    """
    Import SHAP lazily so the project still works when shap is not installed.
    """
    try:
        import shap  # type: ignore

        return shap
    except Exception as error:
        logger.warning("SHAP is not available: %s", error)
        return None


def normalize_shap_values(shap_values: Any) -> np.ndarray:
    """
    Normalize SHAP values into array shape:
    - regression/binary single output: (n_samples, n_features)
    - multiclass/multi-output summarized: (n_samples, n_features)

    Supports:
    - shap.Explanation
    - list returned by older SHAP APIs
    - numpy arrays
    """
    values = shap_values

    if hasattr(values, "values"):
        values = values.values

    if isinstance(values, list):
        arrays = [np.asarray(item) for item in values]
        if not arrays:
            return np.asarray([])
        values = np.mean([np.abs(arr) for arr in arrays], axis=0)

    values_array = np.asarray(values)

    if values_array.ndim == 3:
        # Common shape: (n_samples, n_features, n_outputs)
        # We summarize outputs by mean absolute SHAP.
        values_array = np.mean(np.abs(values_array), axis=2)

    if values_array.ndim == 2:
        return values_array

    if values_array.ndim == 1:
        return values_array.reshape(1, -1)

    return np.asarray([])


def get_base_value(shap_values: Any) -> Any:
    """
    Extract SHAP expected/base value in JSON-safe form.
    """
    try:
        if hasattr(shap_values, "base_values"):
            base_values = np.asarray(shap_values.base_values)
            if base_values.ndim == 0:
                return float(base_values)
            if base_values.size == 1:
                return float(base_values.ravel()[0])
            return [float(value) for value in base_values.ravel()[:10]]

        if hasattr(shap_values, "expected_value"):
            value = shap_values.expected_value
            if isinstance(value, (list, tuple, np.ndarray)):
                arr = np.asarray(value).ravel()
                return [float(item) for item in arr[:10]]
            return float(value)
    except Exception:
        return None

    return None


def summarize_shap_values(
    shap_values_array: np.ndarray,
    feature_names: list[str],
    feature_values_array: np.ndarray,
    sample_index_values: list[Any],
    top_n: int,
) -> dict[str, Any]:
    """
    Build global and local SHAP summaries.

    Global:
    - mean_abs_shap
    - mean_shap
    - positive/negative direction

    Local:
    - top positive/negative contributors for first few sampled rows
    """
    if shap_values_array.size == 0:
        return {
            "global_importance": [],
            "positive_contributors": [],
            "negative_contributors": [],
            "local_explanations": [],
        }

    if len(feature_names) != shap_values_array.shape[1]:
        min_len = min(len(feature_names), shap_values_array.shape[1])
        feature_names = feature_names[:min_len]
        shap_values_array = shap_values_array[:, :min_len]
        feature_values_array = feature_values_array[:, :min_len]

    mean_abs = np.mean(np.abs(shap_values_array), axis=0)
    mean_signed = np.mean(shap_values_array, axis=0)

    global_df = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": mean_abs.astype(float),
            "mean_shap": mean_signed.astype(float),
        }
    )
    global_df["direction"] = np.where(
        global_df["mean_shap"] >= 0,
        "positive",
        "negative",
    )
    global_df = global_df.sort_values("mean_abs_shap", ascending=False)

    positive_df = global_df[global_df["mean_shap"] > 0].sort_values(
        "mean_shap",
        ascending=False,
    )
    negative_df = global_df[global_df["mean_shap"] < 0].sort_values(
        "mean_shap",
        ascending=True,
    )

    local_explanations: list[dict[str, Any]] = []
    rows_to_explain = min(5, shap_values_array.shape[0])

    for row_idx in range(rows_to_explain):
        row_values = shap_values_array[row_idx]
        row_feature_values = feature_values_array[row_idx]

        row_df = pd.DataFrame(
            {
                "feature": feature_names,
                "shap_value": row_values.astype(float),
                "feature_value": row_feature_values,
            }
        )
        row_df["abs_shap_value"] = row_df["shap_value"].abs()

        local_explanations.append(
            {
                "sample_position": int(row_idx),
                "sample_index": str(sample_index_values[row_idx])
                if row_idx < len(sample_index_values)
                else str(row_idx),
                "top_contributors": records_from_dataframe(
                    row_df.sort_values("abs_shap_value", ascending=False).head(top_n)
                ),
                "top_positive": records_from_dataframe(
                    row_df[row_df["shap_value"] > 0]
                    .sort_values("shap_value", ascending=False)
                    .head(top_n)
                ),
                "top_negative": records_from_dataframe(
                    row_df[row_df["shap_value"] < 0]
                    .sort_values("shap_value", ascending=True)
                    .head(top_n)
                ),
            }
        )

    return {
        "global_importance": records_from_dataframe(global_df.head(top_n)),
        "positive_contributors": records_from_dataframe(positive_df.head(top_n)),
        "negative_contributors": records_from_dataframe(negative_df.head(top_n)),
        "local_explanations": local_explanations,
    }


def generate_shap_bar_plot_base64(
    shap_module: Any,
    shap_values_array: np.ndarray,
    feature_names: list[str],
    max_features: int,
) -> str | None:
    """
    Generate SHAP summary bar plot as base64 PNG.

    This avoids returning matplotlib figure objects in workflow output.
    """
    try:
        import matplotlib.pyplot as plt

        if shap_values_array.size == 0:
            return None

        plt.figure()
        shap_module.summary_plot(
            shap_values_array,
            features=None,
            feature_names=feature_names,
            plot_type="bar",
            max_display=max_features,
            show=False,
        )

        buffer = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buffer, format="png", bbox_inches="tight", dpi=140)
        plt.close()

        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")

    except Exception as error:
        logger.warning("Could not generate SHAP bar plot: %s", error)
        return None


def generate_shap_beeswarm_plot_base64(
    shap_module: Any,
    shap_values_array: np.ndarray,
    feature_values_array: np.ndarray,
    feature_names: list[str],
    max_features: int,
) -> str | None:
    """
    Generate SHAP beeswarm/summary plot as base64 PNG.
    """
    try:
        import matplotlib.pyplot as plt

        if shap_values_array.size == 0:
            return None

        plt.figure()
        shap_module.summary_plot(
            shap_values_array,
            features=feature_values_array,
            feature_names=feature_names,
            max_display=max_features,
            show=False,
        )

        buffer = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buffer, format="png", bbox_inches="tight", dpi=140)
        plt.close()

        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")

    except Exception as error:
        logger.warning("Could not generate SHAP beeswarm plot: %s", error)
        return None


def run_shap_explainability(
    model_pipeline: Any,
    estimator: Any,
    raw_sample_features: pd.DataFrame,
    transformed_sample_features: Any,
    feature_names: list[str],
    top_n: int,
    generate_plots: bool,
    plot_max_features: int,
) -> dict[str, Any]:
    """
    Run real SHAP explainability safely.

    Output includes:
    - global mean absolute SHAP importance
    - positive/negative global contributors
    - local explanations for first few samples
    - optional base64 PNG plots for Streamlit display
    """
    shap = import_shap_module()

    if shap is None:
        return empty_shap_result(
            message="SHAP is not installed. Install it with: pip install shap"
        )

    try:
        transformed_array = to_numpy_array(transformed_sample_features)
        estimator_name = get_estimator_name(estimator)

        if transformed_array.size == 0:
            return empty_shap_result(
                message="No transformed features available for SHAP."
            )

        if is_tree_model(estimator):
            explainer = shap.TreeExplainer(estimator)
            raw_shap_values = explainer.shap_values(transformed_array)
            shap_method = "tree_shap"

        elif is_linear_model(estimator):
            explainer = shap.LinearExplainer(estimator, transformed_array)
            raw_shap_values = explainer.shap_values(transformed_array)
            shap_method = "linear_shap"

        else:
            # Model-agnostic fallback. It is slower but works for many sklearn pipelines.
            background = raw_sample_features.head(min(50, len(raw_sample_features)))
            explainer = shap.Explainer(model_pipeline.predict, background)
            raw_shap_values = explainer(raw_sample_features)
            shap_method = "model_agnostic_shap"

        shap_values_array = normalize_shap_values(raw_shap_values)
        feature_values_array = transformed_array

        if shap_values_array.size == 0:
            return empty_shap_result(
                method=shap_method,
                message="SHAP values could not be summarized.",
            )

        summary = summarize_shap_values(
            shap_values_array=shap_values_array,
            feature_names=feature_names,
            feature_values_array=feature_values_array,
            sample_index_values=list(raw_sample_features.index),
            top_n=top_n,
        )

        global_importance = cast(
            list[dict[str, Any]],
            summary.get("global_importance", []),
        )

        plots: dict[str, Any] = {}

        if generate_plots:
            plots["bar_plot_base64"] = generate_shap_bar_plot_base64(
                shap_module=shap,
                shap_values_array=shap_values_array,
                feature_names=feature_names,
                max_features=plot_max_features,
            )
            plots["beeswarm_plot_base64"] = generate_shap_beeswarm_plot_base64(
                shap_module=shap,
                shap_values_array=shap_values_array,
                feature_values_array=feature_values_array,
                feature_names=feature_names,
                max_features=plot_max_features,
            )

        return {
            "available": True,
            "method": shap_method,
            "model_type": estimator_name,
            "base_value": get_base_value(raw_shap_values),
            "top_features": global_importance,
            "global_importance": global_importance,
            "positive_contributors": summary.get("positive_contributors", []),
            "negative_contributors": summary.get("negative_contributors", []),
            "local_explanations": summary.get("local_explanations", []),
            "shap_values_shape": list(shap_values_array.shape),
            "plots": plots,
            "message": "Real SHAP explainability generated successfully.",
        }

    except Exception as error:
        logger.warning("SHAP explainability skipped: %s", error)
        return empty_shap_result(message=f"SHAP explainability skipped: {error}")


def empty_shap_result(
    message: str,
    method: str = "shap",
) -> dict[str, Any]:
    """
    Return a consistent empty SHAP result schema.
    """
    return {
        "available": False,
        "method": method,
        "top_features": [],
        "global_importance": [],
        "positive_contributors": [],
        "negative_contributors": [],
        "local_explanations": [],
        "plots": {},
        "message": message,
    }


def generate_explainability_summary(
    builtin_importance: dict[str, Any],
    shap_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Build concise explainability summary for UI/report.
    """
    if shap_result.get("available"):
        source = "shap"
        top_features = (
            shap_result.get("global_importance", [])
            or shap_result.get("top_features", [])
        )
    elif builtin_importance.get("available"):
        source = "built_in"
        top_features = builtin_importance.get("top_features", [])
    else:
        source = "not_available"
        top_features = []

    top_feature = None

    if isinstance(top_features, list) and top_features:
        first = top_features[0]
        if isinstance(first, dict):
            top_feature = str(first.get("feature"))

    return {
        "preferred_source": source,
        "top_feature": top_feature,
        "num_features_reported": len(top_features) if isinstance(top_features, list) else 0,
        "message": get_summary_message(source, top_feature),
    }


def get_summary_message(source: str, top_feature: str | None) -> str:
    """
    Human-readable explainability message.
    """
    if source == "shap":
        return (
            f"Real SHAP explainability is available. The most influential feature by mean absolute SHAP is '{top_feature}'."
            if top_feature
            else "Real SHAP explainability is available."
        )

    if source == "built_in":
        return (
            f"Built-in feature importance is available. The most influential feature is '{top_feature}'."
            if top_feature
            else "Built-in feature importance is available."
        )

    return (
        "Explainability is not available for the selected model. "
        "This can happen when the model type is unsupported or SHAP is disabled/not installed."
    )


def run_model_explainability(
    baseline_results: dict[str, Any],
    sample_features: pd.DataFrame | None,
) -> dict[str, Any]:
    """
    Generate built-in feature importance and real optional SHAP explanations.

    Parameters
    ----------
    baseline_results:
        Output from train_baseline_models(). Must include trained_model_objects.
    sample_features:
        Raw feature dataframe, usually X_test or a small dataframe with the same
        feature columns used during training.

    Returns
    -------
    dict[str, Any]
        JSON-safe explainability report.
    """
    try:
        config = get_explainability_config()

        if not config["enabled"]:
            return {
                "enabled": False,
                "available": False,
                "message": "Explainability is disabled in config.yaml.",
            }

        validate_inputs(baseline_results, sample_features)

        if sample_features is None or sample_features.empty:
            return {
                "enabled": True,
                "available": False,
                "message": "Sample features are required for explainability.",
            }

        best_model = baseline_results.get("best_model", {})
        best_model_name = best_model.get("model_name", "N/A")
        model_pipeline = get_best_model_pipeline(baseline_results)

        preprocessor, estimator = get_pipeline_parts(model_pipeline)

        if estimator is None:
            raise ExplainabilityError("Could not extract estimator from model pipeline.")

        sample = sample_dataframe(
            df=sample_features,
            max_samples=int(config["max_samples"]),
            random_state=int(config["random_state"]),
        )

        transformed_sample = transform_features(preprocessor, sample)
        transformed_array = to_numpy_array(transformed_sample)
        feature_names = get_feature_names(preprocessor, sample, transformed_sample)

        builtin_importance = get_builtin_feature_importance(
            estimator=estimator,
            feature_names=feature_names,
            top_n=int(config["top_n_features"]),
        )

        if bool(config["run_shap"]):
            shap_result = run_shap_explainability(
                model_pipeline=model_pipeline,
                estimator=estimator,
                raw_sample_features=sample,
                transformed_sample_features=transformed_sample,
                feature_names=feature_names,
                top_n=int(config["top_n_features"]),
                generate_plots=bool(config["generate_plots"]),
                plot_max_features=int(config["plot_max_features"]),
            )
        else:
            shap_result = empty_shap_result(
                message="SHAP is disabled in config.yaml."
            )

        summary = generate_explainability_summary(
            builtin_importance=builtin_importance,
            shap_result=shap_result,
        )

        report = {
            "enabled": True,
            "available": bool(
                builtin_importance.get("available") or shap_result.get("available")
            ),
            "best_model_name": best_model_name,
            "best_model": best_model,
            "model_type": get_estimator_name(estimator),
            "sample_rows_used": int(len(sample)),
            "transformed_feature_count": int(len(feature_names)),
            "transformed_shape": list(transformed_array.shape)
            if hasattr(transformed_array, "shape")
            else None,
            "config": {
                "run_shap": bool(config["run_shap"]),
                "max_samples": int(config["max_samples"]),
                "top_n_features": int(config["top_n_features"]),
                "generate_plots": bool(config["generate_plots"]),
                "plot_max_features": int(config["plot_max_features"]),
            },
            "builtin_feature_importance": builtin_importance,
            "shap": shap_result,
            "summary": summary,
            "notes": [
                "Feature importance explains model behavior, not causal impact.",
                "Mean absolute SHAP shows average impact magnitude across sampled rows.",
                "Positive SHAP values push predictions higher; negative values push predictions lower.",
                "For classification, SHAP values may be summarized across classes.",
                "Baseline explanations should be reviewed before final model tuning.",
            ],
        }

        logger.info(
            "Explainability completed. Model=%s Available=%s SHAP=%s",
            best_model_name,
            report["available"],
            shap_result.get("available", False),
        )

        return report

    except ExplainabilityError:
        raise

    except Exception as error:
        logger.exception("Explainability generation failed.")
        raise ExplainabilityError(
            "Explainability generation failed.",
            error_detail=str(error),
        ) from error


if __name__ == "__main__":
    from src.audit.baseline_models import train_baseline_models
    from src.audit.preprocessing import split_features_target
    from src.audit.profiler import load_dataset
    from src.audit.problem_detector import detect_problem_type

    dataset_path = "data/sample/student_mark.csv"
    target_column = "Grade"

    df = load_dataset(dataset_path)
    problem_info = detect_problem_type(df, target_column)

    baseline = train_baseline_models(
        df=df,
        target_column=target_column,
        problem_type=problem_info["problem_type"],
    )

    X, _ = split_features_target(df, target_column)

    explanation = run_model_explainability(
        baseline_results=baseline,
        sample_features=X,
    )

    print(explanation)
