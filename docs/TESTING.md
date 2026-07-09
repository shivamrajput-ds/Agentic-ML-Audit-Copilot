# Testing Guide

## Overview

Agentic ML Audit Copilot includes an automated test suite to verify the correctness and reliability of the core audit modules.

The goal of testing is to ensure that each module behaves consistently across different datasets and common edge cases.

---

# Testing Framework

The project uses

- pytest
- pytest-cov (optional)

Tests are located inside

```
tests/
```

---

# Test Coverage

Current test coverage includes the following modules.

| Module | Status |
|----------|---------|
| Baseline Models | ✅ |
| Class Imbalance | ✅ |
| Data Quality | ✅ |
| Leakage Detection | ✅ |
| Problem Detection | ✅ |
| Preprocessing | ✅ |

---

# Running All Tests

Execute the complete test suite.

```bash
python -m pytest -v
```

Expected output

```
==========================
96 passed
==========================
```

---

# Running a Single Test File

Example

```bash
python -m pytest tests/test_data_quality.py -v
```

---

# Running a Specific Test

Example

```bash
python -m pytest tests/test_data_quality.py::test_detects_missing_values -v
```

---

# Test Dataset Strategy

The project uses synthetic datasets created entirely in memory.

This approach provides

- Fast execution
- Deterministic behavior
- No dependency on external CSV files
- Repeatable results

---

# Tested Scenarios

## Dataset Profiling

The profiler is tested for

- Empty datasets
- Invalid datasets
- Missing values
- Duplicate rows

---

## Problem Detection

Test cases include

- Binary classification
- Multiclass classification
- Regression
- Constant target
- Missing target

---

## Data Quality

Current tests verify

- Missing values
- Duplicate rows
- Constant columns
- Possible identifier columns

---

## Leakage Detection

Current checks include

- Duplicate target columns
- Target-like column names
- Leakage warnings
- False-positive protection

---

## Class Imbalance

Verified scenarios

- Balanced datasets
- Mild imbalance
- Severe imbalance
- Regression datasets

---

## Preprocessing

Tests verify

- Feature separation
- Pipeline creation
- Train/Test split
- Invalid parameters

---

## Baseline Models

Verified

- Classification models
- Regression models
- Missing target handling
- Invalid problem type
- Best model selection

---

# Test Screenshot

![](../assets/screenshots/test_suite.png)

---

# Continuous Integration

Every push triggers GitHub Actions.

The pipeline performs

- Dependency installation
- Ruff lint checks
- Automated test execution

If any test fails, the workflow fails automatically.

---

# Writing New Tests

Recommended structure

```python
def test_example():
    ...
```

Each new feature should include

- Success case
- Invalid input
- Edge case

---

# Best Practices

- Keep tests independent.
- Avoid relying on external files.
- Use synthetic datasets whenever possible.
- Prefer deterministic assertions.
- Test edge cases in addition to standard cases.

---

# Current Status

Current automated test suite

```
96 Tests Passed
```

The project uses testing as a quality check before new changes are merged into the main branch.

---

# Future Improvements

Potential testing enhancements include

- Integration tests
- API tests
- Streamlit UI tests
- Docker validation
- Performance benchmarks
- End-to-end workflow tests

---

# Summary

The automated test suite validates the core functionality of the audit pipeline and helps ensure that changes do not introduce regressions. The current coverage focuses on deterministic behavior, common edge cases, and the reliability of the primary audit modules.