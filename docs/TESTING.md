# Testing Guide

## Overview

**Agentic ML Audit Copilot** includes an automated test suite for validating the core audit workflow.

The project follows a deterministic-first testing approach. Tests are designed to verify that audit modules, preprocessing, baseline models, API helpers, and workflow behavior remain consistent across runs.

The test suite focuses on:

- Correctness
- Reproducibility
- Regression protection
- JSON-safe outputs
- Safe handling of risky or invalid inputs

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
uv run python -m py_compile app/streamlit_app.py
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
| Risk aggregation | Yes |
| Decision routing | Yes |
| Human review helpers | Yes |
| FastAPI response helpers | Yes |
| JSON-safe serialization | Yes |

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
- JSON-safe profile output

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
- Safe recommendation structure

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

## Risk Aggregator and Decision Router

Tests may cover:

- Aggregating findings from data quality, leakage, and imbalance modules
- Generating workflow-level risk summaries
- Creating review items
- Routing safe datasets forward
- Routing risky datasets to the Human Review Gate
- Blocking or stopping when risks require data fixes
- Keeping routing output JSON-safe

---

## Human Review Workflow

Tests may cover:

- Review item structure
- Allowed decision values
- Final decision validation
- Approved workflow continuation
- Rejected workflow stopping behavior
- Needs-fix workflow stopping behavior
- Reviewer notes and comments
- Exportable reviewer decision payloads

The Human Review Gate should make it clear when modeling is paused and why.

---

## FastAPI Tests

FastAPI-related tests may cover:

- Root endpoint
- Health endpoint
- Metadata endpoint
- Workflow guide endpoint
- Audit modes endpoint
- Decision template endpoint
- Upload validation
- Target column validation
- JSON-safe response formatting
- Error response structure

Future integration tests should cover:

- `POST /audit`
- `POST /audit/summary`
- `POST /audit/review-gate`
- `POST /audit/after-human-approval`

---

## Streamlit Checks

Streamlit is harder to test with standard unit tests, but changes should still be manually validated.

Recommended manual flow:

```text
1. Start Streamlit
2. Upload CSV
3. Select target column
4. Run audit
5. Confirm dashboard tabs render
6. Confirm Human Review Gate works
7. Continue after approval
8. Confirm baseline results, MLflow status, explainability, reports, and downloads render
9. Confirm Audit Q&A does not crash
```

Important UI regression checks:

- Avoid nested expanders.
- Use stable widget keys inside loops.
- Do not hide important warnings.
- Confirm session state reset behavior is understandable.

---

## Docker Validation

After major dependency or startup changes, validate Docker.

Build:

```bash
docker build -t agentic-ml-audit-copilot .
```

Run:

```bash
docker run --rm \
  --name agentic-audit-test \
  -p 8501:8501 \
  -p 8000:8000 \
  -e GROQ_API_KEY="your_groq_api_key" \
  agentic-ml-audit-copilot:latest
```

Check:

```text
Streamlit Dashboard: http://localhost:8501
FastAPI Docs:       http://localhost:8000/docs
Health Check:       http://localhost:8000/health
```

If the container name is already used:

```bash
docker rm -f agentic-audit-test
```

---

## Continuous Integration

GitHub Actions can be used to run checks automatically on push and pull request events.

A typical CI workflow should run:

- Dependency installation
- Ruff linting
- Ruff formatting check
- pytest test suite

If any step fails, the CI workflow should fail.

Recommended CI commands:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

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

Good test names describe behavior:

```python
def test_review_gate_pauses_when_high_leakage_risk_exists():
    ...
```

Avoid vague names:

```python
def test_case_1():
    ...
```

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
- Keep LLM-dependent behavior mocked or optional.
- Do not require real API keys in automated tests.

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
uv run python -m py_compile app/streamlit_app.py
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

- Broader FastAPI integration tests
- Human review API tests
- Streamlit workflow tests
- Docker validation tests
- End-to-end audit workflow tests
- Performance benchmarks
- Larger synthetic dataset tests
- Error-handling regression tests
- Report generation tests
- MLflow tracking tests
- Snapshot tests for JSON-safe API responses
- Deployment smoke tests

---

## Summary

The test suite helps keep **Agentic ML Audit Copilot** stable as the project evolves.

Testing focuses on deterministic audit behavior, safe failure handling, JSON-safe outputs, human review routing, and regression protection across the core ML audit workflow.
