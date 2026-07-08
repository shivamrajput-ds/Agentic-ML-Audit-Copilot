"""
Shared pytest fixtures for the audit test suite.

These fixtures build small, deterministic in-memory DataFrames instead of
reading CSV files from disk — this keeps tests fast, isolated, and
independent of the sample data files in data/sample/.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def classification_df() -> pd.DataFrame:
    """
    A clean binary classification dataset with no leakage and no
    missing values. 40 rows, mildly imbalanced (25 vs 15).
    """
    rng = np.random.default_rng(42)
    n = 40

    return pd.DataFrame(
        {
            "age": rng.integers(18, 60, size=n),
            "income": rng.integers(20000, 90000, size=n),
            "city": rng.choice(["Delhi", "Mumbai", "Pune"], size=n),
            "approved": [1] * 25 + [0] * 15,
        }
    )


@pytest.fixture
def regression_df() -> pd.DataFrame:
    """
    A clean regression dataset where target is a noisy linear function
    of one numeric feature — baseline models should fit this reasonably.
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
    A dataset where "total" is a near-perfect proxy for the target,
    and "target_copy" directly duplicates the target column — used to
    verify leakage detection actually fires.
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
            "result": ["Pass" if t >= 100 else "Fail" for t in total],
        }
    )
    df["target_copy"] = df["result"]
    return df


@pytest.fixture
def messy_df() -> pd.DataFrame:
    """
    A dataset with missing values, a constant column, and a duplicate
    row — used to verify data_quality checks.
    """
    df = pd.DataFrame(
        {
            "feature_a": [1, 2, None, 4, 5, 1, 2, None, 4, 5],
            "feature_b": [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
            "feature_c": list(range(10)),
            "label": [0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
        }
    )
    # duplicate the first row to create an exact duplicate row
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    return df