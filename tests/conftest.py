"""
Shared pytest fixtures for the audit test suite.

The fixtures use small deterministic in-memory datasets. This keeps tests fast,
isolated, and independent from files under data/sample/.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import pytest

RANDOM_SEED: Final[int] = 42
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _shuffle_dataframe(
    df: pd.DataFrame,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Return a deterministically shuffled dataframe with a clean integer index."""
    return df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def _write_temp_csv(tmp_path: Path, df: pd.DataFrame, filename: str) -> Path:
    """Write a dataframe to a temporary CSV file and return its path."""
    file_path = tmp_path / filename
    df.to_csv(file_path, index=False)
    return file_path


# ---------------------------------------------------------------------------
# Global lightweight fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def project_root() -> Path:
    """Return repository root for tests that need stable project-relative paths."""
    return PROJECT_ROOT


@pytest.fixture
def random_seed() -> int:
    """Return the deterministic test random seed."""
    return RANDOM_SEED


@pytest.fixture
def sample_human_review_decision() -> dict[str, object]:
    """Reusable positive human-review decision for HITL workflow tests."""
    return {
        "approved": True,
        "reviewer": "pytest",
        "notes": "Approved for deterministic test continuation.",
        "decisions": {
            "possible_leakage_risks_reviewed": True,
            "data_quality_reviewed": True,
            "target_definition_reviewed": True,
        },
    }


# ---------------------------------------------------------------------------
# Core supervised-learning datasets
# ---------------------------------------------------------------------------
@pytest.fixture
def classification_df() -> pd.DataFrame:
    """
    Clean binary classification dataset.

    40 rows with mild imbalance: 25 positive and 15 negative labels.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    n_rows = 40

    df = pd.DataFrame(
        {
            "age": rng.integers(18, 60, size=n_rows),
            "income": rng.integers(20_000, 90_000, size=n_rows),
            "city": rng.choice(["Delhi", "Mumbai", "Pune"], size=n_rows),
            "approved": [1] * 25 + [0] * 15,
        },
    )

    return _shuffle_dataframe(df)


@pytest.fixture
def multiclass_df() -> pd.DataFrame:
    """
    Clean multiclass classification dataset.

    Useful for problem detection, metric recommendation, imbalance checks,
    and baseline model tests.
    """
    rng = np.random.default_rng(123)
    n_rows = 60

    df = pd.DataFrame(
        {
            "age": rng.integers(18, 65, size=n_rows),
            "income": rng.integers(25_000, 120_000, size=n_rows),
            "city": rng.choice(
                ["Delhi", "Mumbai", "Pune", "Bengaluru"],
                size=n_rows,
            ),
            "risk_band": ["low"] * 25 + ["medium"] * 20 + ["high"] * 15,
        },
    )

    return _shuffle_dataframe(df, random_state=123)


@pytest.fixture
def imbalanced_classification_df() -> pd.DataFrame:
    """
    Strongly imbalanced binary classification dataset.

    Useful for testing class imbalance severity and metric recommendations.
    """
    rng = np.random.default_rng(99)
    n_rows = 50

    df = pd.DataFrame(
        {
            "age": rng.integers(18, 60, size=n_rows),
            "income": rng.integers(20_000, 90_000, size=n_rows),
            "city": rng.choice(["Delhi", "Mumbai", "Pune"], size=n_rows),
            "approved": [1] * 47 + [0] * 3,
        },
    )

    return _shuffle_dataframe(df, random_state=99)


@pytest.fixture
def regression_df() -> pd.DataFrame:
    """
    Clean regression dataset.

    Target is a noisy linear function of hours_studied, so baseline models
    should produce reasonable metrics.
    """
    rng = np.random.default_rng(7)
    n_rows = 50

    hours_studied = rng.uniform(0, 10, size=n_rows)
    noise = rng.normal(0, 2, size=n_rows)

    return pd.DataFrame(
        {
            "hours_studied": hours_studied,
            "attendance_percent": rng.uniform(50, 100, size=n_rows),
            "score": hours_studied * 8 + noise + 20,
        },
    )


# ---------------------------------------------------------------------------
# Risk / data-quality datasets
# ---------------------------------------------------------------------------
@pytest.fixture
def leaky_df() -> pd.DataFrame:
    """
    Dataset with obvious leakage signals.

    - total is a strong proxy for the result
    - target_copy directly duplicates the target column
    - student_id is identifier-like
    """
    rng = np.random.default_rng(1)
    n_rows = 30

    math_scores = rng.integers(40, 100, size=n_rows)
    science_scores = rng.integers(40, 100, size=n_rows)
    total_scores = math_scores + science_scores

    df = pd.DataFrame(
        {
            "student_id": [f"S{i:03d}" for i in range(n_rows)],
            "math": math_scores,
            "science": science_scores,
            "total": total_scores,
            "result": ["Pass" if value >= 100 else "Fail" for value in total_scores],
        },
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
        },
    )

    return pd.concat([df, df.iloc[[0]]], ignore_index=True)


@pytest.fixture
def high_cardinality_df() -> pd.DataFrame:
    """Dataset with a high-cardinality categorical feature."""
    n_rows = 80
    rng = np.random.default_rng(202)

    return pd.DataFrame(
        {
            "customer_code": [f"CUST_{i:04d}" for i in range(n_rows)],
            "age": rng.integers(18, 70, size=n_rows),
            "segment": rng.choice(["A", "B", "C"], size=n_rows),
            "target": [1] * 45 + [0] * 35,
        },
    )


@pytest.fixture
def datetime_df() -> pd.DataFrame:
    """Dataset with a datetime feature for preprocessing/profile tests."""
    rng = np.random.default_rng(303)
    n_rows = 36

    return pd.DataFrame(
        {
            "signup_date": pd.date_range("2024-01-01", periods=n_rows, freq="D"),
            "usage_minutes": rng.uniform(10, 300, size=n_rows),
            "plan": rng.choice(["free", "plus", "pro"], size=n_rows),
            "churned": [0] * 24 + [1] * 12,
        },
    )


@pytest.fixture
def infinite_values_df() -> pd.DataFrame:
    """Dataset containing positive and negative infinity values."""
    return pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, np.inf, 4.0, 5.0, 6.0],
            "feature_b": [10.0, -np.inf, 12.0, 13.0, 14.0, 15.0],
            "target": [0, 1, 0, 1, 0, 1],
        },
    )


# ---------------------------------------------------------------------------
# Invalid / edge-case datasets
# ---------------------------------------------------------------------------
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
        },
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
        },
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
        },
    )


@pytest.fixture
def empty_feature_df() -> pd.DataFrame:
    """Dataset with only a target column, useful for preprocessing validation tests."""
    return pd.DataFrame({"target": [0, 1, 0, 1]})


# ---------------------------------------------------------------------------
# Temporary CSV fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def temp_classification_csv(
    tmp_path: Path,
    classification_df: pd.DataFrame,
) -> Path:
    """Temporary CSV file for workflow/API-style classification tests."""
    return _write_temp_csv(tmp_path, classification_df, "classification_sample.csv")


@pytest.fixture
def temp_multiclass_csv(
    tmp_path: Path,
    multiclass_df: pd.DataFrame,
) -> Path:
    """Temporary CSV file for multiclass workflow tests."""
    return _write_temp_csv(tmp_path, multiclass_df, "multiclass_sample.csv")


@pytest.fixture
def temp_regression_csv(
    tmp_path: Path,
    regression_df: pd.DataFrame,
) -> Path:
    """Temporary CSV file for regression workflow tests."""
    return _write_temp_csv(tmp_path, regression_df, "regression_sample.csv")


@pytest.fixture
def temp_leaky_csv(tmp_path: Path, leaky_df: pd.DataFrame) -> Path:
    """Temporary CSV file for leakage workflow tests."""
    return _write_temp_csv(tmp_path, leaky_df, "leaky_sample.csv")


@pytest.fixture
def temp_messy_csv(tmp_path: Path, messy_df: pd.DataFrame) -> Path:
    """Temporary CSV file for data-quality workflow tests."""
    return _write_temp_csv(tmp_path, messy_df, "messy_sample.csv")
