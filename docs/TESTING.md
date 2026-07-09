# Testing Guide

## Overview

Agentic ML Audit Copilot includes an automated test suite to verify the correctness, reliability, and stability of the core audit workflow.

The project follows a deterministic-first philosophy, ensuring that ML computations produce consistent and reproducible results across supported environments.

---

# Testing Framework

The project uses:

- pytest
- pytest-cov (optional)

Tests are located in:

```text
tests/
```

---

# Test Coverage

The current automated test suite covers:

| Module | Status |
|----------|:------:|
| Dataset Profiling | ✅ |
| Problem Type Detection | ✅ |
| Data Quality Audit | ✅ |
| Leakage Detection | ✅ |
| Class Imbalance | ✅ |
| Preprocessing Pipeline | ✅ |
| Baseline Models | ✅ |

---

# Running All Tests

Run the complete test suite:

```bash
uv run pytest
```

Verbose output:

```bash
uv run pytest -v
```

Expected output:

```text
==============================
96 passed
==============================
```

---

# Run a Single Test File

Example:

```bash
uv run pytest tests/test_data_quality.py -v
```

---

# Run a Specific Test

Example:

```bash
uv run pytest tests/test_data_quality.py::test_detects_missing_values -v
```

---

# Test Dataset Strategy

The test suite uses synthetic datasets created entirely in memory.

Benefits include:

- Fast execution
- Deterministic behavior
- No dependency on external CSV files
- Repeatable test results

---

# Tested Scenarios

## Dataset Profiling

Current tests include:

- Empty datasets
- Missing values
- Duplicate rows
- Invalid datasets
- Invalid target columns

---

## Problem Type Detection

Current tests include:

- Binary classification
- Multiclass classification
- Regression
- Constant targets
- Integer-like regression targets
- Missing targets

---

## Data Quality Audit

Current tests verify:

- Missing values
- Duplicate rows
- Constant columns
- Identifier columns
- Invalid datasets

---

## Leakage Detection

Current tests verify:

- Target-like column names
- Duplicate target columns
- Identifier-based leakage risks
- Correlation-based leakage checks
- False-positive protection

> **Note**
>
> The application reports **possible leakage risks** only.
> Leakage is never treated as confirmed automatically.

---

## Class Imbalance

Current tests verify:

- Balanced datasets
- Moderate imbalance
- Severe imbalance
- Regression datasets

---

## Preprocessing Pipeline

Current tests verify:

- Feature separation
- Train/test split
- Sklearn preprocessing pipeline
- Invalid parameters
- Missing values

---

## Baseline Models

Current tests verify:

- Classification baselines
- Regression baselines
- Missing target handling
- Invalid problem types
- Best model selection
- Metric calculation

---

# Test Screenshot

![Test Suite](../assets/screenshots/test_suite.png)

---

# Continuous Integration

Every push and pull request automatically triggers GitHub Actions.

The CI pipeline performs:

- Dependency installation
- Ruff formatting check
- Ruff linting
- Automated test execution

If any step fails, the workflow is marked as failed.

---

# Writing New Tests

Recommended structure:

```python
def test_example():
    ...
```

Each new feature should include:

- Success case
- Invalid input
- Edge case
- Regression test (when applicable)

---

# Best Practices

- Keep tests independent.
- Prefer synthetic datasets.
- Avoid network dependencies.
- Avoid external files unless necessary.
- Use deterministic assertions.
- Test both expected and edge-case behavior.

---

# Current Status

Current project status:

```text
✓ 96 Tests Passed
✓ Ruff Formatting
✓ Ruff Linting
✓ GitHub Actions CI
✓ Docker Build Verified
```

---

# Future Improvements

Planned testing enhancements:

- FastAPI integration tests
- Streamlit UI tests
- Docker validation tests
- End-to-end workflow tests
- Performance benchmarks
- Stress testing
- Model reproducibility tests

---

# Summary

The automated test suite validates the deterministic audit workflow and helps prevent regressions as the project evolves.

Testing focuses on correctness, reproducibility, maintainability, and production-oriented reliability.