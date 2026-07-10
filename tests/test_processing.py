"""
Tests for src/audit/preprocessing.py.

These tests verify that preprocessing remains safe, deterministic, and compatible
with downstream baseline modeling. The focus is not model quality; it is correct
feature/target splitting, leakage-safe preprocessing, and robust edge-case
handling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.audit import preprocessing as preprocessing_module
from src.audit.preprocessing import (
    build_categorical_pipeline,
    build_datetime_pipeline,
    build_numeric_pipeline,
    build_preprocessing_pipeline,
    can_use_stratify,
    clean_infinite_values,
    clean_target_rows,
    column_name_suggests_id,
    create_train_test_split,
    detect_high_cardinality_columns,
    detect_id_like_columns,
    extract_datetime_features,
    get_feature_columns,
    infer_datetime_feature_columns,
    split_features_target,
)
from src.utils.exceptions import AuditCopilotException


def test_feature_columns_split_correctly(classification_df):
    columns = get_feature_columns(classification_df, target_column="approved")

    assert "age" in columns["numeric_columns"]
    assert "income" in columns["numeric_columns"]
    assert "city" in columns["categorical_columns"]
    assert "approved" not in columns["numeric_columns"]
    assert "approved" not in columns["categorical_columns"]
    assert columns["columns_dropped_before_modeling"] == []


def test_pipeline_builds_and_fits(classification_df):
    pipeline_info = build_preprocessing_pipeline(
        classification_df,
        target_column="approved",
    )

    preprocessor = pipeline_info["preprocessor"]
    features = classification_df.drop(columns=["approved"])

    transformed = preprocessor.fit_transform(features)

    assert transformed.shape[0] == len(features)
    assert transformed.shape[1] >= 4
    assert set(pipeline_info["numeric_columns"]) == {"age", "income"}
    assert pipeline_info["categorical_columns"] == ["city"]
    assert pipeline_info["total_features_before_encoding"] == 3
    assert pipeline_info["message"] == "Preprocessing pipeline created successfully."


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
    assert set(x_train.columns) == {"age", "income", "city"}


def test_train_test_split_supports_regression(regression_df):
    x_train, x_test, y_train, y_test = create_train_test_split(
        regression_df,
        target_column="score",
        problem_type="regression",
        test_size=0.2,
        random_state=7,
    )

    assert len(x_train) + len(x_test) == len(regression_df)
    assert len(y_train) + len(y_test) == len(regression_df)
    assert "score" not in x_train.columns


def test_split_features_target_drops_missing_target(missing_target_df):
    features, target = split_features_target(
        missing_target_df,
        target_column="target",
        drop_missing_target=True,
    )

    assert len(features) == 4
    assert len(target) == 4
    assert target.isna().sum() == 0


def test_split_features_target_rejects_missing_target_when_not_dropping(
    missing_target_df,
):
    with pytest.raises(AuditCopilotException):
        split_features_target(
            missing_target_df,
            target_column="target",
            drop_missing_target=False,
        )


def test_clean_target_rows_raises_if_all_targets_missing():
    df = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "target": [None, None, None],
        },
    )

    with pytest.raises(AuditCopilotException):
        clean_target_rows(df, target_column="target")


def test_clean_target_rows_drops_infinite_target_values():
    df = pd.DataFrame(
        {
            "feature": [1.0, 2.0, 3.0, 4.0],
            "target": [0.0, 1.0, np.inf, -np.inf],
        },
    )

    cleaned = clean_target_rows(df, target_column="target")

    assert len(cleaned) == 2
    assert np.isfinite(cleaned["target"]).all()


def test_clean_infinite_values_replaces_inf():
    df = pd.DataFrame(
        {
            "feature": [1.0, np.inf, -np.inf],
            "text": ["a", "b", "c"],
        },
    )

    cleaned = clean_infinite_values(df)

    assert cleaned["feature"].isna().sum() == 2
    assert cleaned["text"].tolist() == ["a", "b", "c"]


def test_detect_id_like_columns_by_name_and_uniqueness():
    df = pd.DataFrame(
        {
            "customer_id": ["a", "b", "c", "d"],
            "normal_feature": [1, 1, 2, 2],
        },
    )

    id_columns = detect_id_like_columns(df)

    assert "customer_id" in id_columns
    assert "normal_feature" not in id_columns


def test_numeric_high_uniqueness_feature_is_not_dropped_as_id():
    df = pd.DataFrame(
        {
            "income": list(range(100)),
            "sales": np.linspace(10.0, 1000.0, 100),
        },
    )

    id_columns = detect_id_like_columns(df)

    assert "income" not in id_columns
    assert "sales" not in id_columns


def test_column_name_suggests_id_avoids_substring_false_positive():
    assert column_name_suggests_id("customer_id") is True
    assert column_name_suggests_id("id") is True
    assert column_name_suggests_id("candidate_score") is False
    assert column_name_suggests_id("income") is False


def test_detect_high_cardinality_columns():
    df = pd.DataFrame(
        {
            "category": [f"cat_{idx}" for idx in range(60)],
            "small_category": ["a", "b"] * 30,
        },
    )

    high_cardinality = detect_high_cardinality_columns(df)

    assert "category" in high_cardinality
    assert "small_category" not in high_cardinality


def test_get_feature_columns_drops_id_like_column(leaky_df):
    columns = get_feature_columns(leaky_df, target_column="result")

    assert "student_id" in columns["id_like_columns_dropped"]
    assert "student_id" in columns["columns_dropped_before_modeling"]
    assert "student_id" not in columns["categorical_columns"]


def test_get_feature_columns_can_drop_high_cardinality_when_enabled(monkeypatch):
    df = pd.DataFrame(
        {
            "category": [f"cat_{idx}" for idx in range(20)],
            "target": [0, 1] * 10,
        },
    )

    def fake_get_config_value(key_path, default=None):
        overrides = {
            "preprocessing.drop_id_like_columns": False,
            "preprocessing.drop_high_cardinality_columns": True,
            "audit.high_cardinality_threshold": 10,
        }
        return overrides.get(key_path, default)

    monkeypatch.setattr(preprocessing_module, "get_config_value", fake_get_config_value)

    columns = get_feature_columns(df, target_column="target")

    assert "category" in columns["high_cardinality_columns_dropped"]
    assert "category" in columns["columns_dropped_before_modeling"]


def test_datetime_features_are_extracted():
    df = pd.DataFrame(
        {
            "event_date": pd.date_range("2024-01-01", periods=3),
        },
    )

    extracted = extract_datetime_features(df)

    expected_columns = {
        "event_date_year",
        "event_date_month",
        "event_date_day",
        "event_date_dayofweek",
        "event_date_quarter",
        "event_date_is_month_start",
        "event_date_is_month_end",
    }

    assert expected_columns.issubset(set(extracted.columns))
    assert len(extracted) == 3


def test_pipeline_handles_datetime_columns():
    df = pd.DataFrame(
        {
            "age": [20, 21, 22, 23, 24, 25],
            "event_date": pd.date_range("2024-01-01", periods=6),
            "target": [0, 1, 0, 1, 0, 1],
        },
    )

    pipeline_info = build_preprocessing_pipeline(df, target_column="target")
    features = df.drop(columns=["target"])

    transformed = pipeline_info["preprocessor"].fit_transform(features)

    assert transformed.shape[0] == len(df)
    assert "event_date" in pipeline_info["datetime_columns"]
    assert any("Datetime columns" in warning for warning in pipeline_info["warnings"])


def test_infers_datetime_string_columns(monkeypatch):
    df = pd.DataFrame(
        {
            "created_date": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
                "2024-01-06",
            ],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )

    def fake_get_config_value(key_path, default=None):
        overrides = {
            "preprocessing.drop_id_like_columns": False,
            "preprocessing.infer_datetime_columns": True,
        }
        return overrides.get(key_path, default)

    monkeypatch.setattr(preprocessing_module, "get_config_value", fake_get_config_value)

    columns = get_feature_columns(df, target_column="target")

    assert "created_date" in columns["datetime_columns"]
    assert "created_date" not in columns["numeric_columns"]
    assert "target" not in columns["numeric_columns"]
    assert "target" not in columns["categorical_columns"]


def test_infer_datetime_feature_columns_ignores_plain_categories():
    df = pd.DataFrame(
        {
            "city": ["Delhi", "Mumbai", "Pune", "Bengaluru"],
            "segment": ["A", "B", "A", "C"],
        },
    )

    assert infer_datetime_feature_columns(df) == []


def test_build_individual_pipelines():
    numeric_pipeline = build_numeric_pipeline()
    categorical_pipeline = build_categorical_pipeline()
    datetime_pipeline = build_datetime_pipeline()

    assert numeric_pipeline.steps
    assert categorical_pipeline.steps
    assert datetime_pipeline.steps


def test_can_use_stratify_true_for_balanced_target():
    target = pd.Series([0, 1] * 10)

    assert can_use_stratify(target, test_size=0.2) is True


def test_can_use_stratify_false_for_tiny_class():
    target = pd.Series([1, 1, 1, 1, 0])

    assert can_use_stratify(target, test_size=0.2) is False


def test_can_use_stratify_false_when_test_set_too_small():
    target = pd.Series([0, 1, 2] * 4)

    assert can_use_stratify(target, test_size=0.1) is False


def test_target_column_whitespace_is_accepted(classification_df):
    columns = get_feature_columns(classification_df, target_column=" approved ")

    assert "approved" not in columns["numeric_columns"]
    assert "age" in columns["numeric_columns"]


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


@pytest.mark.parametrize("bad_test_size", [0, -0.1, 1.0])
def test_split_rejects_boundary_test_sizes(classification_df, bad_test_size):
    with pytest.raises(AuditCopilotException):
        create_train_test_split(
            classification_df,
            target_column="approved",
            problem_type="binary_classification",
            test_size=bad_test_size,
        )


def test_split_raises_on_blank_problem_type(classification_df):
    with pytest.raises(AuditCopilotException):
        create_train_test_split(
            classification_df,
            target_column="approved",
            problem_type="",
        )


def test_build_pipeline_raises_for_missing_target(classification_df):
    with pytest.raises(AuditCopilotException):
        build_preprocessing_pipeline(
            classification_df,
            target_column="missing",
        )


def test_single_column_dataframe_raises():
    df = pd.DataFrame({"target": [0, 1, 0, 1]})

    with pytest.raises(AuditCopilotException):
        build_preprocessing_pipeline(df, target_column="target")


def test_duplicate_column_names_raise():
    df = pd.DataFrame(
        [[1, 10, 0], [2, 20, 1], [3, 30, 0]],
        columns=["feature", "feature", "target"],
    )

    with pytest.raises(AuditCopilotException):
        build_preprocessing_pipeline(df, target_column="target")


def test_no_features_left_after_id_drop_raises():
    df = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3", "u4"],
            "target": [0, 1, 0, 1],
        },
    )

    with pytest.raises(AuditCopilotException):
        build_preprocessing_pipeline(df, target_column="target")


def test_split_raises_when_no_features_left_after_drops():
    df = pd.DataFrame(
        {
            "record_id": ["r1", "r2", "r3", "r4"],
            "target": [0, 1, 0, 1],
        },
    )

    with pytest.raises(AuditCopilotException):
        split_features_target(df, target_column="target")


def test_split_features_target_drops_configured_id_columns(leaky_df):
    features, target = split_features_target(leaky_df, target_column="result")

    assert "student_id" not in features.columns
    assert "result" not in features.columns
    assert len(features) == len(target)
