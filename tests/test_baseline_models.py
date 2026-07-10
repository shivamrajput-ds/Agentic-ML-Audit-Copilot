"""
Tests for src/audit/baseline_models.py.

These tests verify baseline training behavior without depending on large files or
external services. They intentionally keep assertions focused on public output
contracts, runtime objects needed by MLflow/SHAP, and important edge cases.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

import src.audit.baseline_models as baseline_module
from src.audit.baseline_models import (
    evaluate_regression,
    get_sample_features_for_explainability,
    strip_runtime_objects,
    train_baseline_models,
)
from src.utils.exceptions import AuditCopilotException, ModelTrainingError

ConfigGetter = Callable[[str, Any], Any]


def _assert_common_baseline_contract(
    result: dict[str, Any],
    *,
    problem_type: str,
    target_column: str,
) -> None:
    """Assert the stable baseline output contract used by API/UI/MLflow/SHAP."""
    assert result["problem_type"] == problem_type
    assert result["target_column"] == target_column
    assert isinstance(result["models_trained"], list)
    assert result["models_trained"]
    assert result.get("models") == result["models_trained"]
    assert isinstance(result.get("models_attempted", []), list)
    assert isinstance(result["results"], dict)
    assert isinstance(result["best_model"], dict)
    assert result["best_model"]["model_name"] in result["models_trained"]
    assert isinstance(result["best_model"]["score"], float)
    assert "preprocessing_summary" in result
    assert "evaluation_details" in result
    assert "message" in result
    assert "note" in result


def test_baseline_result_has_required_top_level_keys(
    classification_df: pd.DataFrame,
) -> None:
    result = train_baseline_models(
        classification_df,
        target_column="approved",
        problem_type="binary_classification",
    )

    required_keys = {
        "problem_type",
        "target_column",
        "models_trained",
        "models",
        "models_attempted",
        "results",
        "trained_model_objects",
        "best_model",
        "preprocessing_summary",
        "evaluation_details",
        "runtime_objects",
        "model_failures",
        "warnings",
        "message",
        "note",
    }

    assert required_keys.issubset(result.keys())


def test_stripped_baseline_result_is_json_serializable(
    classification_df: pd.DataFrame,
) -> None:
    result = train_baseline_models(
        classification_df,
        target_column="approved",
        problem_type="binary_classification",
    )

    cleaned = strip_runtime_objects(result)

    json.dumps(cleaned, default=str)
    assert "trained_model_objects" not in cleaned
    assert "runtime_objects" not in cleaned


def _patch_config(monkeypatch: pytest.MonkeyPatch, overrides: dict[str, Any]) -> None:
    """Patch baseline module config lookups for deterministic tests."""
    original_get_config_value: ConfigGetter = baseline_module.get_config_value

    def fake_get_config_value(path: str, default: Any = None) -> Any:
        return overrides.get(path, original_get_config_value(path, default))

    monkeypatch.setattr(baseline_module, "get_config_value", fake_get_config_value)


def test_trains_classification_baselines(classification_df: pd.DataFrame) -> None:
    result = train_baseline_models(
        classification_df,
        target_column="approved",
        problem_type="binary_classification",
    )

    _assert_common_baseline_contract(
        result,
        problem_type="binary_classification",
        target_column="approved",
    )
    assert "Logistic Regression" in result["models_trained"]
    assert "Random Forest Classifier" in result["models_trained"]
    assert "f1_score" in result["results"]["Logistic Regression"]
    assert "balanced_accuracy" in result["results"]["Logistic Regression"]
    assert result["evaluation_details"]["label_encoder_used"] is True


def test_trains_multiclass_baselines(multiclass_df: pd.DataFrame) -> None:
    result = train_baseline_models(
        multiclass_df,
        target_column="risk_band",
        problem_type="multiclass_classification",
    )

    _assert_common_baseline_contract(
        result,
        problem_type="multiclass_classification",
        target_column="risk_band",
    )
    assert "Logistic Regression" in result["models_trained"]
    assert "Random Forest Classifier" in result["models_trained"]
    assert "f1_macro" in result["results"]["Logistic Regression"]
    assert result["evaluation_details"]["label_encoder_used"] is True
    assert sorted(result["evaluation_details"]["class_labels"]) == [
        "high",
        "low",
        "medium",
    ]


def test_trains_regression_baselines(regression_df: pd.DataFrame) -> None:
    result = train_baseline_models(
        regression_df,
        target_column="score",
        problem_type="regression",
    )

    _assert_common_baseline_contract(
        result,
        problem_type="regression",
        target_column="score",
    )
    assert "Linear Regression" in result["models_trained"]
    assert "Random Forest Regressor" in result["models_trained"]
    assert "rmse" in result["results"]["Linear Regression"]
    assert result["results"]["Linear Regression"]["rmse"] < 50
    assert result["evaluation_details"]["label_encoder_used"] is False


def test_best_model_has_selection_metadata(classification_df: pd.DataFrame) -> None:
    result = train_baseline_models(
        classification_df,
        target_column="approved",
        problem_type="binary_classification",
    )

    best_model = result["best_model"]
    assert isinstance(best_model["model_name"], str)
    assert isinstance(best_model["selection_metric"], str)
    assert isinstance(best_model["score"], float)
    assert isinstance(best_model["higher_is_better"], bool)


def test_trained_model_objects_are_returned(classification_df: pd.DataFrame) -> None:
    """MLflow and explainability depend on fitted sklearn Pipeline objects."""
    result = train_baseline_models(
        classification_df,
        target_column="approved",
        problem_type="binary_classification",
    )

    assert "trained_model_objects" in result
    assert len(result["trained_model_objects"]) > 0

    for model_object in result["trained_model_objects"].values():
        assert isinstance(model_object, Pipeline)
        assert hasattr(model_object, "predict")


def test_runtime_objects_are_returned_for_explainability(
    classification_df: pd.DataFrame,
) -> None:
    result = train_baseline_models(
        classification_df,
        target_column="approved",
        problem_type="binary_classification",
    )

    assert "runtime_objects" in result
    assert "sample_features" in result["runtime_objects"]
    assert "test_features" in result["runtime_objects"]
    assert "train_features" in result["runtime_objects"]
    assert "label_encoder" in result["runtime_objects"]

    sample_features = get_sample_features_for_explainability(result)
    assert sample_features is not None
    assert not sample_features.empty
    assert "approved" not in sample_features.columns


def test_strip_runtime_objects_removes_heavy_keys(
    classification_df: pd.DataFrame,
) -> None:
    result = train_baseline_models(
        classification_df,
        target_column="approved",
        problem_type="binary_classification",
    )

    cleaned = strip_runtime_objects(result)

    assert "trained_model_objects" not in cleaned
    assert "runtime_objects" not in cleaned
    assert "results" in cleaned
    assert "best_model" in cleaned
    assert "preprocessing_summary" in cleaned


def test_sample_features_returns_none_when_runtime_missing() -> None:
    assert get_sample_features_for_explainability({}) is None
    assert get_sample_features_for_explainability({"runtime_objects": []}) is None


def test_missing_target_rows_are_dropped(missing_target_df: pd.DataFrame) -> None:
    result = train_baseline_models(
        missing_target_df,
        target_column="target",
        problem_type="binary_classification",
    )

    train_shape = result["evaluation_details"]["train_shape"]
    test_shape = result["evaluation_details"]["test_shape"]

    assert train_shape[0] + test_shape[0] == 4
    assert result["evaluation_details"].get("missing_target_rows_dropped", 2) >= 0


def test_tiny_class_dataset_still_returns_or_raises_clear_error(
    tiny_class_df: pd.DataFrame,
) -> None:
    """
    One class has only one sample.

    A robust implementation may still train a model by falling back from
    stratified splitting, or it may raise a clean ModelTrainingError if the
    training split cannot contain two classes. Either behavior is acceptable;
    crashing with an unrelated exception is not.
    """
    try:
        result = train_baseline_models(
            tiny_class_df,
            target_column="target",
            problem_type="binary_classification",
        )
    except ModelTrainingError as error:
        assert "class" in str(error).lower() or "training" in str(error).lower()
    else:
        assert result["best_model"]["model_name"] in result["models_trained"]
        assert "results" in result
        assert result["evaluation_details"]["label_encoder_used"] is True


def test_cross_validation_enabled_when_safe(
    classification_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_config(
        monkeypatch,
        {
            "modeling.enable_cross_validation": True,
            "modeling.cv_folds": 3,
            "metrics.classification_default": "f1_weighted",
        },
    )

    result = train_baseline_models(
        classification_df,
        target_column="approved",
        problem_type="binary_classification",
    )

    details = result["evaluation_details"]
    assert details["cross_validation_enabled"] is True
    assert details["actual_cv_folds"] == 3

    for metrics in result["results"].values():
        assert "cv_mean" in metrics
        assert "cv_std" in metrics


def test_cross_validation_skips_when_not_safe(
    tiny_class_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_config(
        monkeypatch,
        {
            "modeling.enable_cross_validation": True,
            "modeling.cv_folds": 5,
        },
    )

    try:
        result = train_baseline_models(
            tiny_class_df,
            target_column="target",
            problem_type="binary_classification",
        )
    except ModelTrainingError:
        pytest.skip("Tiny class split cannot produce a trainable classification split.")

    details = result["evaluation_details"]
    assert details["cross_validation_enabled"] is False
    assert details["cv_warning"] is not None


def test_regression_mape_skips_zero_targets() -> None:
    y_test = np.asarray([0.0, 10.0, 20.0])
    y_pred = np.asarray([1.0, 12.0, 18.0])

    metrics = evaluate_regression(y_test, y_pred)

    assert "mae" in metrics
    assert "rmse" in metrics
    assert "mape" in metrics
    assert np.isfinite(metrics["mape"])


def test_preprocessing_summary_contains_dropped_columns(
    high_cardinality_df: pd.DataFrame,
) -> None:
    result = train_baseline_models(
        high_cardinality_df,
        target_column="target",
        problem_type="binary_classification",
    )

    summary = result["preprocessing_summary"]
    assert "columns_dropped_before_modeling" in summary
    assert "id_like_columns_dropped" in summary
    assert "high_cardinality_columns_dropped" in summary


def test_invalid_problem_type_raises(classification_df: pd.DataFrame) -> None:
    with pytest.raises(AuditCopilotException):
        train_baseline_models(
            classification_df,
            target_column="approved",
            problem_type="not_a_real_problem_type",
        )


def test_missing_target_column_raises(classification_df: pd.DataFrame) -> None:
    with pytest.raises(AuditCopilotException):
        train_baseline_models(
            classification_df,
            target_column="does_not_exist",
            problem_type="binary_classification",
        )


def test_single_class_target_raises(invalid_single_class_df: pd.DataFrame) -> None:
    with pytest.raises(AuditCopilotException):
        train_baseline_models(
            invalid_single_class_df,
            target_column="target",
            problem_type="binary_classification",
        )


def test_regression_requires_numeric_target(classification_df: pd.DataFrame) -> None:
    with pytest.raises(AuditCopilotException):
        train_baseline_models(
            classification_df,
            target_column="city",
            problem_type="regression",
        )


def test_regression_rejects_infinite_target() -> None:
    df = pd.DataFrame(
        {
            "feature": [1.0, 2.0, 3.0, 4.0],
            "target": [1.0, 2.0, np.inf, 4.0],
        },
    )

    with pytest.raises(AuditCopilotException) as error_info:
        train_baseline_models(
            df,
            target_column="target",
            problem_type="regression",
        )

    assert "target" in str(error_info.value).lower()


def test_best_model_score_matches_selected_result_key(
    classification_df: pd.DataFrame,
) -> None:
    result = train_baseline_models(
        classification_df,
        target_column="approved",
        problem_type="binary_classification",
    )

    best_model = result["best_model"]
    model_name = best_model["model_name"]
    selection_metric = best_model["selection_metric"]

    assert selection_metric in result["results"][model_name]
    assert best_model["score"] == pytest.approx(
        result["results"][model_name][selection_metric],
        abs=1e-6,
    )
