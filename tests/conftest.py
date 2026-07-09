"""
Shared pytest fixtures for the audit test suite.

The fixtures use small deterministic in-memory datasets. This keeps tests fast,
isolated, and independent from files under data/sample/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def classification_df() -> pd.DataFrame:
    """
    Clean binary classification dataset.

    40 rows with mild imbalance: 25 positive and 15 negative labels.
    """
    rng = np.random.default_rng(42)
    n = 40

    return pd.DataFrame(
        {
            "age": rng.integers(18, 60, size=n),
            "income": rng.integers(20_000, 90_000, size=n),
            "city": rng.choice(["Delhi", "Mumbai", "Pune"], size=n),
            "approved": [1] * 25 + [0] * 15,
        }
    )


@pytest.fixture
def multiclass_df() -> pd.DataFrame:
    """
    Clean multiclass classification dataset.

    Useful for problem detection, metric recommendation, imbalance checks,
    and baseline model tests.
    """
    rng = np.random.default_rng(123)
    n = 60

    return pd.DataFrame(
        {
            "age": rng.integers(18, 65, size=n),
            "income": rng.integers(25_000, 120_000, size=n),
            "city": rng.choice(["Delhi", "Mumbai", "Pune", "Bengaluru"], size=n),
            "risk_band": ["low"] * 25 + ["medium"] * 20 + ["high"] * 15,
        }
    )


@pytest.fixture
def imbalanced_classification_df() -> pd.DataFrame:
    """
    Strongly imbalanced binary classification dataset.

    Useful for testing class imbalance severity and metric recommendations.
    """
    rng = np.random.default_rng(99)
    n = 50

    return pd.DataFrame(
        {
            "age": rng.integers(18, 60, size=n),
            "income": rng.integers(20_000, 90_000, size=n),
            "city": rng.choice(["Delhi", "Mumbai", "Pune"], size=n),
            "approved": [1] * 47 + [0] * 3,
        }
    )


@pytest.fixture
def regression_df() -> pd.DataFrame:
    """
    Clean regression dataset.

    Target is a noisy linear function of hours_studied, so baseline models
    should produce reasonable metrics.
    """
    rng = np.random.default_rng(7)
    n = 50

    hours_studied = rng.uniform(0, 10, size=n)
    noise = rng.normal(0, 2, size=n)

    return pd.DataFrame(
        {
            "hours_studied": hours_studied,
            "attendance_percent": rng.uniform(50, 100, size=n),
            "score": hours_studied * 8 + noise + 20,
        }
    )


@pytest.fixture
def leaky_df() -> pd.DataFrame:
    """
    Dataset with obvious leakage signals.

    - total is a strong proxy for the result
    - target_copy directly duplicates the target column
    - student_id is identifier-like
    """
    rng = np.random.default_rng(1)
    n = 30

    math = rng.integers(40, 100, size=n)
    science = rng.integers(40, 100, size=n)
    total = math + science

    df = pd.DataFrame(
        {
            "student_id": [f"S{i:03d}" for i in range(n)],
            "math": math,
            "science": science,
            "total": total,
            "result": ["Pass" if value >= 100 else "Fail" for value in total],
        }
    )

    df["target_copy"] = df["result"]
    return df


@pytest.fixture
def messy_df() -> pd.DataFrame:
    """
    Dataset with common data quality issues.

    Includes missing values, constant column, possible ID-like column,
    and an exact duplicate row.
    """
    df = pd.DataFrame(
        {
            "record_id": [f"R{i:03d}" for i in range(10)],
            "feature_a": [1, 2, None, 4, 5, 1, 2, None, 4, 5],
            "feature_b": [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
            "feature_c": list(range(10)),
            "label": [0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
        }
    )

    return pd.concat([df, df.iloc[[0]]], ignore_index=True)


@pytest.fixture
def missing_target_df() -> pd.DataFrame:
    """
    Dataset with missing target values.

    Used to verify target cleaning and validation behavior.
    """
    return pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4, 5, 6],
            "feature_b": ["a", "b", "a", "b", "a", "b"],
            "target": [1, 0, None, 1, None, 0],
        }
    )


@pytest.fixture
def invalid_single_class_df() -> pd.DataFrame:
    """
    Invalid supervised-learning dataset.

    Target has only one class, so model training/problem detection should fail.
    """
    return pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4],
            "feature_b": ["a", "b", "c", "d"],
            "target": [1, 1, 1, 1],
        }
    )


@pytest.fixture
def tiny_class_df() -> pd.DataFrame:
    """
    Dataset where one class has only one row.

    Useful for testing stratified split and CV fallback behavior.
    """
    return pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4, 5],
            "feature_b": ["a", "b", "a", "b", "a"],
            "target": [1, 1, 1, 1, 0],
        }
    )


@pytest.fixture
def temp_classification_csv(
    tmp_path: Path,
    classification_df: pd.DataFrame,
) -> Path:
    """
    Temporary CSV file for workflow/API-style tests.
    """
    file_path = tmp_path / "classification_sample.csv"
    classification_df.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def temp_regression_csv(
    tmp_path: Path,
    regression_df: pd.DataFrame,
) -> Path:
    """
    Temporary CSV file for regression workflow tests.
    """
    file_path = tmp_path / "regression_sample.csv"
    regression_df.to_csv(file_path, index=False)
    return file_path
