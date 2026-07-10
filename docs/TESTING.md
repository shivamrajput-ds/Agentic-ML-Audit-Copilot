# Testing Guide

## Overview

**Agentic ML Audit Copilot** includes an automated test suite for validating the core audit workflow.

The project follows a deterministic-first testing approach. Tests are designed to verify that audit modules, preprocessing, baseline models, and workflow helpers behave consistently across runs.

The test suite focuses on correctness, reproducibility, and regression protection.

---

## Testing Framework

The project uses:

- pytest
- Ruff for linting and formatting
- GitHub Actions for CI

Tests are located in:

```text
tests/
```

Pytest configuration is stored in:

```text
pytest.ini
```

---

## Test Configuration

Current `pytest.ini`:

```ini
[pytest]
pythonpath = .
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    -q
    --strict-config
    --strict-markers
```

This configuration keeps test discovery simple and strict.

---

## Run All Tests

Run the complete test suite:

```bash
uv run pytest -q
```

Run with more output:

```bash
uv run pytest -v
```

---

## Run a Single Test File

Example:

```bash
uv run pytest tests/test_data_quality.py -q
```

Another example:

```bash
uv run pytest tests/test_baseline_models.py -q
```

---

## Run a Specific Test

Example:

```bash
uv run pytest tests/test_data_quality.py::test_detects_missing_values -q
```

---

## Run Code Quality Checks

Before committing changes, run:

```bash
uv run ruff check . --fix --unsafe-fixes
uv run ruff format .
uv run pytest -q
```

Recommended final check:

```bash
git status
```

---

## Current Test Scope

| Area | Covered |
| --- | :---: |
| Dataset profiling | Yes |
| Problem type detection | Yes |
| Data quality audit | Yes |
| Leakage risk detection | Yes |
| Class imbalance detection | Yes |
| Metric recommendation | Yes |
| Preprocessing pipeline | Yes |
| Baseline models | Yes |
| Workflow helper behavior | Yes |

---

## Test Dataset Strategy

The tests use small synthetic datasets created in memory.

Benefits:

- Fast execution
- No dependency on external files
- Deterministic behavior
- Easier edge-case testing
- Better regression protection

This approach is preferred for unit tests because test behavior does not depend on local CSV files or external services.

---

## Tested Scenarios

## Dataset Profiling

Tests may cover:

- Valid datasets
- Empty datasets
- Missing values
- Duplicate rows
- Invalid target columns
- Dataset shape and column summaries

---

## Problem Type Detection

Tests may cover:

- Binary classification
- Multiclass classification
- Regression
- Constant targets
- Numeric targets with few unique values
- High-cardinality string targets
- Missing or invalid targets

Ambiguous cases should be marked safely instead of forcing an incorrect problem type.

---

## Data Quality Audit

Tests may cover:

- Missing values
- Duplicate rows
- Constant columns
- Near-constant columns
- Identifier-like columns
- High-cardinality columns
- Infinite values
- Basic outlier checks
- Invalid input handling

---

## Leakage Risk Detection

Tests may cover:

- Target-like column names
- Outcome-like feature names
- Duplicate target-like columns
- Identifier-based risks
- Correlation-based risks
- Proxy feature patterns
- False-positive protection

Important rule:

The application reports possible leakage risks only. Leakage is not treated as confirmed automatically.

---

## Class Imbalance Detection

Tests may cover:

- Balanced classification datasets
- Moderate imbalance
- Severe imbalance
- Multiclass imbalance
- Regression datasets where imbalance is not applicable
- JSON-safe not-applicable responses

---

## Metric Recommendation

Tests may cover:

- Binary classification metrics
- Multiclass classification metrics
- Imbalanced classification metrics
- Regression metrics
- Invalid or unknown problem types
- Recommendation structure and JSON safety

---

## Preprocessing Pipeline

Tests may cover:

- Numeric feature handling
- Categorical feature handling
- Missing value handling
- Train-test split behavior
- scikit-learn pipeline creation
- Invalid parameters
- Target column separation

Preprocessing should stay inside scikit-learn pipelines to reduce train-test leakage risk.

---

## Baseline Models

Tests may cover:

- Classification baselines
- Regression baselines
- Invalid problem types
- Missing target handling
- Best model selection
- Metric calculation
- JSON-safe result formatting

Baseline tests should verify sanity-check behavior, not advanced model optimization.

---

## Workflow Tests

Workflow-related tests may cover:

- State preparation
- Risk aggregation
- Decision routing
- Human review status
- Paused workflow behavior
- Approved workflow continuation
- JSON-safe output handling

The Human Review Gate should make it clear when modeling is paused and why.

---

## Continuous Integration

GitHub Actions can be used to run checks automatically on push and pull request events.

A typical CI workflow should run:

- Dependency installation
- Ruff linting
- Ruff formatting check
- pytest test suite

If any step fails, the CI workflow should fail.

---

## Writing New Tests

Use clear test names:

```python
def test_detects_missing_values():
    ...
```

For each new feature, try to include:

- A normal success case
- An invalid input case
- An edge case
- A regression test for previous bugs, if relevant

---

## Best Practices

- Keep tests independent.
- Prefer small synthetic datasets.
- Avoid network calls.
- Avoid external files unless the test specifically needs them.
- Keep assertions deterministic.
- Test both success and failure behavior.
- Do not hide important warnings unless there is a clear reason.
- Avoid testing implementation details too tightly.
- Prefer testing expected behavior and output structure.

---

## Common Commands

Run tests:

```bash
uv run pytest -q
```

Run tests with verbose output:

```bash
uv run pytest -v
```

Run one file:

```bash
uv run pytest tests/test_problem_detector.py -q
```

Run lint:

```bash
uv run ruff check .
```

Fix lint issues where safe:

```bash
uv run ruff check . --fix --unsafe-fixes
```

Format code:

```bash
uv run ruff format .
```

Run full local validation:

```bash
uv run ruff check . --fix --unsafe-fixes
uv run ruff format .
uv run pytest -q
```

---

## Expected Result

A healthy test run should finish with all tests passing.

Example:

```text
all tests passed
```

The exact number of tests may change as the project grows.

---

## Future Testing Improvements

Planned improvements:

- FastAPI integration tests
- Human review API tests
- Streamlit workflow tests
- Docker validation tests
- End-to-end audit workflow tests
- Performance benchmarks
- Larger synthetic dataset tests
- Error-handling regression tests
- Report generation tests
- MLflow tracking tests

---

## Summary

The test suite helps keep Agentic ML Audit Copilot stable as the project evolves.

Testing focuses on deterministic audit behavior, safe failure handling, JSON-safe outputs, and regression protection across the core ML audit workflow.
