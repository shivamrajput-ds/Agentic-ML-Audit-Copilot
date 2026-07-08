"""
Tests for src/audit/baseline_models.py
"""
from __future__ import annotations

import pytest

from src.audit.baseline_models import train_baseline_models
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


def test_trains_regression_baselines(regression_df):
    result = train_baseline_models(
        regression_df,
        target_column="score",
        problem_type="regression",
    )
    assert "Linear Regression" in result["models_trained"]
    assert "rmse" in result["results"]["Linear Regression"]
    # RMSE should be a small, finite positive number for this clean
    # near-linear synthetic dataset
    assert result["results"]["Linear Regression"]["rmse"] < 50


def test_best_model_has_a_score(classification_df):
    result = train_baseline_models(
        classification_df,
        target_column="approved",
        problem_type="binary_classification",
    )
    assert isinstance(result["best_model"]["score"], float)


def test_trained_model_objects_are_returned(classification_df):
    """
    mlflow_tracker.py and the API layer both depend on this key existing
    so the best model's fitted pipeline can be logged/stripped correctly.
    """
    result = train_baseline_models(
        classification_df,
        target_column="approved",
        problem_type="binary_classification",
    )
    assert "trained_model_objects" in result
    assert len(result["trained_model_objects"]) > 0


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