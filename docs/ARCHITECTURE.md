# System Architecture

## Overview

**Agentic ML Audit Copilot** is a modular machine learning audit platform designed to evaluate tabular datasets before model development.

Instead of immediately training optimized models, the system first audits the dataset for data quality issues, possible leakage risks, class imbalance, metric suitability, baseline performance, explainability, and reproducibility.

The project follows a **deterministic-first architecture**:

- Python performs all ML computation.
- Scikit-learn handles preprocessing and baseline modeling.
- MLflow tracks experiments.
- LangGraph orchestrates the audit workflow.
- The LLM is used only for explanations and report generation.

---

# High-Level Architecture

```text
User
 ↓
Streamlit Dashboard
 ↓
FastAPI Service
 ↓
LangGraph Workflow
 ↓
Audit Modules
 ↓
MLflow + Explainability + Report
 ↓
Final Audit Output
```

---

# Core Components

## Streamlit Dashboard

The Streamlit dashboard provides the main user interface.

Responsibilities:

- CSV upload
- Target column selection
- Audit execution
- Interactive visualizations
- Human review dashboard
- Explainability display
- Report downloads
- Audit Q&A

The dashboard does not perform ML computation directly. It delegates execution to backend workflow modules.

---

## FastAPI Service

FastAPI exposes the audit workflow through REST endpoints.

Main endpoints:

- `/`
- `/health`
- `/audit`
- `/audit/summary`

This allows external systems to trigger audits programmatically.

---

## LangGraph Workflow

LangGraph orchestrates the complete audit pipeline.

Execution flow:

```text
Load Dataset
 ↓
Profile Dataset
 ↓
Problem Detection
 ↓
Data Quality Audit
 ↓
Possible Leakage Detection
 ↓
Class Imbalance Detection
 ↓
Metric Recommendation
 ↓
Baseline Models
 ↓
MLflow Tracking
 ↓
Explainability
 ↓
LLM Report
 ↓
Final Summary
```

---

# Audit Modules

## Dataset Profiler

Generates dataset-level information:

- Shape
- Column types
- Missing values
- Duplicate rows
- Memory usage
- Target summary

---

## Problem Detector

Detects the ML task type:

- Binary classification
- Multiclass classification
- Regression

Ambiguous cases are marked for human review.

---

## Data Quality Audit

Checks:

- Missing values
- Duplicate rows
- Constant columns
- Near-constant columns
- High-cardinality columns
- Identifier-like columns
- Infinite values
- Outliers

The module reports findings and recommendations without modifying the dataset.

---

## Leakage Detection

Detects **possible leakage risks**, including:

- Target-like column names
- Duplicate target-like columns
- Highly correlated features
- Proxy features
- Identifier-like columns

The system never claims confirmed leakage automatically. Human review is required.

---

## Class Imbalance Detection

For classification tasks, the module calculates:

- Class counts
- Class percentages
- Majority class
- Minority class
- Imbalance ratio
- Severity level
- Recommended actions

---

## Metric Recommendation

Recommends evaluation metrics based on:

- Problem type
- Imbalance severity

Examples:

Classification:

- F1 Score
- Macro F1
- Weighted F1
- Balanced Accuracy
- ROC-AUC
- PR-AUC

Regression:

- RMSE
- MAE
- R²
- Median Absolute Error

---

## Preprocessing Pipeline

Preprocessing is built using scikit-learn pipelines.

Numeric features:

- Median imputation
- Scaling

Categorical features:

- Most frequent imputation
- One-hot encoding

All preprocessing stays inside sklearn pipelines to avoid train-test leakage.

---

## Baseline Models

The project trains simple baseline models.

Classification:

- Logistic Regression
- Random Forest Classifier

Regression:

- Linear Regression
- Random Forest Regressor

These models are used for sanity-check benchmarking, not final optimization.

---

## MLflow Tracking

MLflow tracks:

- Experiment name
- Model parameters
- Metrics
- Runs
- Best model
- Optional artifacts

This improves experiment reproducibility.

---

## Explainability

Explainability includes:

- Built-in feature importance
- SHAP summaries when enabled

Explainability is separated from training so it can evolve independently.

---

## LLM Report Generation

The LLM generates:

- Executive summary
- Audit explanation
- Markdown report
- Audit Q&A answers

The LLM does not perform ML computation.

---

# Data Flow

```text
CSV Upload
 ↓
Dataset Loaded
 ↓
Audit Workflow State Created
 ↓
Each Module Adds Results to State
 ↓
Baseline Models Trained
 ↓
MLflow Logs Experiments
 ↓
Explainability Generated
 ↓
LLM Explains Deterministic Results
 ↓
Final JSON + Markdown Report
```

---

# Folder Responsibilities

```text
app/
```

Contains:

- FastAPI API
- Streamlit dashboard

```text
src/audit/
```

Contains:

- Profiling
- Problem detection
- Data quality audit
- Leakage detection
- Class imbalance
- Metric recommendation
- Preprocessing
- Baseline models
- MLflow tracking
- Explainability
- Report generation
- Workflow orchestration

```text
src/utils/
```

Contains:

- Configuration helpers
- Logger
- Custom exceptions

```text
tests/
```

Contains:

- Unit tests
- Regression tests
- Synthetic test datasets

```text
assets/
```

Contains:

- Diagrams
- Screenshots
- Branding assets
- Demo assets

```text
reports/
```

Stores generated audit reports.

```text
data/
```

Stores sample and uploaded datasets.

---

# Configuration Design

Most behavior is controlled by:

```text
config.yaml
```

Configurable areas include:

- Logging
- Upload limits
- Random seed
- MLflow
- Explainability
- LLM settings
- Modeling defaults
- Metric defaults
- Preprocessing behavior

This avoids hardcoded behavior inside modules.

---

# Logging

The project uses centralized logging with:

- Console logging
- Rotating file logs
- Configurable log level
- Duplicate handler prevention

Logs are stored in:

```text
logs/
```

---

# Error Handling

The project uses domain-specific exceptions, including:

- `InvalidDatasetError`
- `InvalidTargetColumnError`
- `DataQualityError`
- `LeakageDetectionError`
- `MetricRecommendationError`
- `ClassImbalanceError`
- `PreprocessingError`
- `ModelTrainingError`
- `MLflowTrackingError`
- `ReportGenerationError`
- `AgentWorkflowError`

This improves debugging and API/UI error handling.

---

# Human-in-the-Loop Design

The system flags issues that require human review.

Examples:

- Possible leakage
- Identifier-like columns
- Severe imbalance
- High missing values
- Ambiguous problem type

The system provides recommendations but does not make final modeling decisions automatically.

---

# Testing Strategy

The test suite uses synthetic datasets and covers:

- Problem detection
- Data quality audit
- Leakage detection
- Class imbalance
- Preprocessing
- Baseline models

The goal is deterministic, repeatable test behavior.

---

# Deployment Architecture

The project supports Docker deployment.

The Docker container runs:

- FastAPI on port `8000`
- Streamlit on port `8501`

Docker Hub image:

```text
shivamrajput130/agentic-ml-audit-copilot:latest
```

---

# Design Principles

## Deterministic First

All ML calculations are performed by Python.

## Human Review Required

Risk flags are treated as possible findings, not final decisions.

## Modular Architecture

Each audit stage is isolated and independently maintainable.

## Configuration Driven

Behavior is controlled through config files.

## Reproducibility

Random seeds, sklearn pipelines, and MLflow improve reproducibility.

## Separation of Concerns

UI, API, workflow, audit modules, reporting, and utilities remain separate.

---

# Current Limitations

Current version focuses on:

- CSV datasets
- Tabular ML
- Classification and regression
- Single-machine execution

Not currently included:

- Distributed processing
- Time-series auditing
- Data drift monitoring
- Authentication
- Multi-user workspaces
- Kubernetes deployment

---

# Future Architecture

Planned future modules:

```text
Authentication
 ↓
User Workspaces
 ↓
Dataset Versioning
 ↓
Data Drift Detection
 ↓
Fairness & Bias Analysis
 ↓
Hyperparameter Optimization
 ↓
Model Registry
 ↓
Cloud Deployment
```

---

# Summary

Agentic ML Audit Copilot uses a modular, deterministic-first architecture for auditing tabular ML datasets before model development.

The system combines data quality checks, leakage-risk detection, metric recommendation, baseline modeling, MLflow tracking, explainability, FastAPI, Streamlit, Docker, and LLM-assisted reporting while keeping final decisions human-in-the-loop.
