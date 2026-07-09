"""
Tests for src/audit/baseline_models.py.
"""
from __future__ import annotations

import pytest
from sklearn.pipeline import Pipeline

from src.audit.baseline_models import (
    get_sample_features_for_explainability,
    strip_runtime_objects,
    train_baseline_models,
)
from src.utils.exceptions import AuditCopilotException


def test_trains_classification_baselines(classification_df):
    result = train_baseline_models(
        classification_df,
        target_column="approved",
        problem_type="binary_classification",
    )

    assert "Logistic Regression" in result["models_trained"]
    assert "Random Forest Classifier" in result["models_trained"]
    assert result["best_model"]["model_name"] in result["models_trained"]
    assert "f1_score" in result["results"]["Logistic Regression"]
    assert result["problem_type"] == "binary_classification"


def test_trains_multiclass_baselines(multiclass_df):
    result = train_baseline_models(
        multiclass_df,
        target_column="risk_band",
        problem_type="multiclass_classification",
    )

    assert "Logistic Regression" in result["models_trained"]
    assert "Random Forest Classifier" in result["models_trained"]
    assert result["best_model"]["model_name"] in result["models_trained"]
    assert "f1_macro" in result["results"]["Logistic Regression"]
    assert result["evaluation_details"]["label_encoder_used"] is True
    assert len(result["evaluation_details"]["class_labels"]) == 3


def test_trains_regression_baselines(regression_df):
    result = train_baseline_models(
        regression_df,
        target_column="score",
        problem_type="regression",
    )

    assert "Linear Regression" in result["models_trained"]
    assert "Random Forest Regressor" in result["models_trained"]
    assert "rmse" in result["results"]["Linear Regression"]
    assert result["results"]["Linear Regression"]["rmse"] < 50
    assert result["evaluation_details"]["label_encoder_used"] is False


def test_best_model_has_a_score(classification_df):
    result = train_baseline_models(
        classification_df,
        target_column="approved",
        problem_type="binary_classification",
    )

    assert isinstance(result["best_model"]["score"], float)


def test_trained_model_objects_are_returned(classification_df):
    """
    MLflow and explainability depend on fitted sklearn Pipeline objects.
    """
    result = train_baseline_models(
        classification_df,
        target_column="approved",
        problem_type="binary_classification",
    )

    assert "trained_model_objects" in result
    assert len(result["trained_model_objects"]) > 0

    for model_object in result["trained_model_objects"].values():
        assert isinstance(model_object, Pipeline)


def test_runtime_objects_are_returned_for_explainability(classification_df):
    result = train_baseline_models(
        classification_df,
        target_column="approved",
        problem_type="binary_classification",
    )

    assert "runtime_objects" in result
    assert "sample_features" in result["runtime_objects"]
    assert "test_features" in result["runtime_objects"]

    sample_features = get_sample_features_for_explainability(result)
    assert sample_features is not None
    assert not sample_features.empty


def test_strip_runtime_objects_removes_heavy_keys(classification_df):
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


def test_missing_target_rows_are_dropped(missing_target_df):
    result = train_baseline_models(
        missing_target_df,
        target_column="target",
        problem_type="binary_classification",
    )

    train_shape = result["evaluation_details"]["train_shape"]
    test_shape = result["evaluation_details"]["test_shape"]

    assert train_shape[0] + test_shape[0] == 4


def test_tiny_class_dataset_still_trains_without_stratify(tiny_class_df):
    """
    One class has only one sample. Stratified split/CV may be skipped, but
    the baseline training should still return a valid result.
    """
    result = train_baseline_models(
        tiny_class_df,
        target_column="target",
        problem_type="binary_classification",
    )

    assert result["best_model"]["model_name"] in result["models_trained"]
    assert "results" in result


def test_invalid_problem_type_raises(classification_df):
    with pytest.raises(AuditCopilotException):
        train_baseline_models(
            classification_df,
            target_column="approved",
            problem_type="not_a_real_problem_type",
        )


def test_missing_target_column_raises(classification_df):
    with pytest.raises(AuditCopilotException):
        train_baseline_models(
            classification_df,
            target_column="does_not_exist",
            problem_type="binary_classification",
        )


def test_single_class_target_raises(invalid_single_class_df):
    with pytest.raises(AuditCopilotException):
        train_baseline_models(
            invalid_single_class_df,
            target_column="target",
            problem_type="binary_classification",
        )


def test_regression_requires_numeric_target(classification_df):
    with pytest.raises(AuditCopilotException):
        train_baseline_models(
            classification_df,
            target_column="city",
            problem_type="regression",
        )
