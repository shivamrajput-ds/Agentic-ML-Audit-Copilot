"""
Tests for src/audit/preprocessing.py
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.audit.preprocessing import (
    build_preprocessing_pipeline,
    create_train_test_split,
    get_feature_columns,
)
from src.utils.exceptions import AuditCopilotException


def test_feature_columns_split_correctly(classification_df):
    columns = get_feature_columns(classification_df, target_column="approved")
    assert "age" in columns["numeric_columns"]
    assert "income" in columns["numeric_columns"]
    assert "city" in columns["categorical_columns"]
    assert "approved" not in columns["numeric_columns"]
    assert "approved" not in columns["categorical_columns"]


def test_pipeline_builds_and_fits(classification_df):
    pipeline_info = build_preprocessing_pipeline(
        classification_df, target_column="approved"
    )
    preprocessor = pipeline_info["preprocessor"]
    features = classification_df.drop(columns=["approved"])

    transformed = preprocessor.fit_transform(features)
    # numeric (age, income) + one-hot encoded city (3 categories) = 5 columns
    assert transformed.shape[0] == len(features)
    assert transformed.shape[1] >= 4
    assert set(pipeline_info["numeric_columns"]) == {"age", "income"}
    assert pipeline_info["categorical_columns"] == ["city"]


def test_train_test_split_shapes(classification_df):
    x_train, x_test, y_train, y_test = create_train_test_split(
        classification_df,
        target_column="approved",
        problem_type="binary_classification",
        test_size=0.25,
        random_state=42,
    )
    assert len(x_train) + len(x_test) == len(classification_df)
    assert len(y_train) + len(y_test) == len(classification_df)
    assert len(x_test) == len(y_test)


def test_split_raises_on_missing_target(classification_df):
    with pytest.raises(AuditCopilotException):
        create_train_test_split(
            classification_df,
            target_column="does_not_exist",
            problem_type="binary_classification",
        )


def test_split_raises_on_invalid_test_size(classification_df):
    with pytest.raises(AuditCopilotException):
        create_train_test_split(
            classification_df,
            target_column="approved",
            problem_type="binary_classification",
            test_size=1.5,
        )