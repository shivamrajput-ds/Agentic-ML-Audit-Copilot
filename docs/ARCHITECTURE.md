# System Architecture

## Overview

Agentic ML Audit Copilot is a modular machine learning auditing platform designed to evaluate tabular datasets before model training.

Instead of immediately training machine learning models, the system first performs a complete audit of the uploaded dataset. It identifies common data quality problems, detects possible target leakage, recommends suitable evaluation metrics, builds preprocessing pipelines, trains baseline models, tracks experiments, and generates professional audit reports.

The application follows a deterministic-first design.

Machine learning logic is implemented using Python and Scikit-learn, while the LLM is only responsible for explanations and report generation.

---

# High-Level Architecture

```
                User
                  │
                  ▼
        Streamlit Dashboard
                  │
                  ▼
            FastAPI Service
                  │
                  ▼
        LangGraph Workflow Engine
                  │
      ┌───────────┼────────────┐
      ▼           ▼            ▼
 Dataset      ML Audit      Report
 Processing    Modules      Generation
      │           │            │
      └───────────┼────────────┘
                  ▼
           Final Audit Report
```

---

# Core Components

The platform is divided into several independent modules.

Each module has a single responsibility.

This makes the system easier to maintain, extend, and test.

---

## 1. Streamlit Dashboard

The Streamlit application provides the primary user interface.

Responsibilities include:

- Dataset upload
- Target column selection
- Running audits
- Displaying audit results
- Human Review Dashboard
- SHAP visualizations
- Report download

The dashboard does not perform machine learning computations directly.

All processing is delegated to backend modules.

---

## 2. FastAPI Service

FastAPI exposes the audit pipeline through REST APIs.

Available endpoints include:

- Root endpoint
- Health check
- Full audit
- Audit summary

This enables integration with external systems.

---

## 3. LangGraph Workflow

LangGraph orchestrates the execution order of the audit modules.

Instead of calling modules manually, the workflow executes them sequentially.

Current execution flow:

```
Load Dataset

↓

Profile Dataset

↓

Problem Detection

↓

Data Quality Audit

↓

Leakage Detection

↓

Metric Recommendation

↓

Class Imbalance

↓

Preprocessing

↓

Baseline Models

↓

MLflow Tracking

↓

SHAP Explainability

↓

LLM Report

↓

Save Report
```

The workflow ensures every audit follows the same execution order.

---

# Audit Modules

Each module performs one clearly defined task.

---

## Dataset Profiler

Responsibilities:

- Dataset shape
- Column information
- Missing values
- Duplicate rows
- Memory usage
- Data types

Output

```
Profile Summary
```

---

## Problem Detector

Automatically determines

- Binary Classification
- Multiclass Classification
- Regression

The detected problem type is used by downstream modules.

---

## Data Quality Audit

Checks include

- Missing values
- Duplicate rows
- Constant columns
- High-cardinality features
- Possible identifier columns

The module generates warnings instead of modifying the dataset.

---

## Leakage Detection

Identifies possible leakage risks.

Current checks include

- Target duplicate columns
- Target-like feature names
- Correlation-based leakage
- Encoded target leakage
- Proxy features

The module reports possible risks but never claims confirmed leakage.

Human validation is still required.

---

## Metric Recommendation

Automatically recommends suitable evaluation metrics.

Classification

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC
- Balanced Accuracy

Regression

- RMSE
- MAE
- R²
- Median Absolute Error

The recommendation depends on

- Problem type
- Class imbalance

---

## Class Imbalance

Calculates

- Majority class
- Minority class
- Imbalance ratio
- Severity level

The output helps users decide whether resampling techniques are required.

---

## Preprocessing

Builds reusable preprocessing pipelines.

Current preprocessing steps

Numeric

- Median Imputation
- Standard Scaling

Categorical

- Most Frequent Imputation
- One Hot Encoding

ColumnTransformer combines both pipelines into a single reusable preprocessing object.

---

## Baseline Models

Classification

- Logistic Regression
- Random Forest Classifier

Regression

- Linear Regression
- Random Forest Regressor

These models are intentionally simple.

Their purpose is benchmarking rather than achieving maximum accuracy.

---

## MLflow Tracking

Every experiment logs

- Parameters
- Metrics
- Artifacts
- Models

This provides experiment reproducibility.

---

## Explainability

The explainability module supports

- SHAP values
- Feature Importance

These visualizations help users understand model behavior.

---

## LLM Report

The LLM module generates

- Executive Summary
- Audit Explanation
- Professional Markdown Report

The LLM never performs machine learning computations.

It only explains deterministic outputs.

---
# Data Flow

The complete execution flow is shown below.

```
                 User Uploads CSV
                         │
                         ▼
               Streamlit Dashboard
                         │
                         ▼
                  FastAPI Endpoint
                         │
                         ▼
               LangGraph Workflow
                         │
                         ▼
                Dataset Profiler
                         │
                         ▼
              Problem Detection
                         │
                         ▼
             Data Quality Audit
                         │
                         ▼
             Leakage Detection
                         │
                         ▼
          Metric Recommendation
                         │
                         ▼
          Class Imbalance Analysis
                         │
                         ▼
          Preprocessing Pipeline
                         │
                         ▼
          Baseline Model Training
                         │
                         ▼
             MLflow Experiment
                         │
                         ▼
           SHAP Explainability
                         │
                         ▼
            LLM Report Generation
                         │
                         ▼
              Final Audit Report
```

---

# Folder Responsibilities

```
app/
```

Contains application entry points.

Responsibilities

- Streamlit UI
- FastAPI API

---

```
src/audit/
```

Contains every audit module.

Responsibilities

- Dataset profiling
- Leakage detection
- Model training
- SHAP
- MLflow
- Report generation

---

```
src/utils/
```

Shared utility components.

Responsibilities

- Configuration
- Logging
- Custom Exceptions

---

```
tests/
```

Automated test suite.

Responsibilities

- Unit tests
- Regression tests
- Synthetic datasets

---

```
assets/
```

Repository branding.

Contains

- Diagrams
- Screenshots
- Logos
- Banner
- Demo GIF

---

```
reports/
```

Stores generated reports.

Examples

- Markdown
- JSON

---

```
data/
```

Dataset storage.

Contains

- Sample datasets
- Uploaded datasets

---

# Configuration Design

Most application settings are stored inside **config.yaml**.

Examples include

- Upload limits
- Random seed
- Logging
- MLflow
- Explainability
- Preprocessing
- Metric defaults

This avoids hardcoded values inside Python modules.

---

# Logging

The project uses a centralized logger.

Features

- Console logging
- Rotating log files
- Configurable log level
- Duplicate handler protection

Logs are stored inside

```
logs/
```

---

# Error Handling

The application uses custom exception classes.

Examples

- InvalidDatasetError
- InvalidTargetColumnError
- DataQualityError
- LeakageDetectionError
- ModelTrainingError
- ReportGenerationError

Every module raises domain-specific exceptions rather than generic exceptions.

---

# Explainability Design

Model explainability is isolated from training.

Current implementation includes

- SHAP values
- Feature importance

This separation allows explainability to evolve independently from training logic.

---

# MLflow Design

MLflow is responsible only for experiment tracking.

Tracked items include

- Parameters
- Metrics
- Models
- Artifacts

The training module remains independent of MLflow.

---

# Human Review Dashboard

Some findings require manual inspection.

Examples

- Possible leakage
- Identifier columns
- Severe imbalance
- High missing values

The dashboard presents these findings for review before users proceed with downstream modeling.

The system highlights potential issues but leaves the final decision to the user.

---

# Testing Strategy

The project includes automated tests covering

- Profiling
- Problem Detection
- Data Quality
- Leakage
- Class Imbalance
- Preprocessing
- Baseline Models

The test suite uses synthetic datasets to ensure deterministic behavior.

---

# Design Principles

The project follows a few core engineering principles.

### Modular Design

Each module has one responsibility.

---

### Deterministic First

Machine learning logic is deterministic.

The LLM is used only for explanations.

---

### Configuration Driven

Behavior is controlled through configuration files instead of hardcoded values.

---

### Reproducibility

Random seeds and MLflow tracking improve reproducibility.

---

### Separation of Concerns

UI, workflow, audit modules, reporting, and utilities are independent.

---

# Scalability Considerations

The current implementation is designed for small to medium-sized tabular datasets.

Future improvements may include

- Dask support
- Polars support
- Distributed preprocessing
- Cloud object storage
- Parallel model training

---

# Future Architecture

Potential future modules

```
Authentication

↓

Role Based Access

↓

Dataset Versioning

↓

Data Drift Detection

↓

Bias & Fairness Analysis

↓

Hyperparameter Optimization

↓

Model Registry

↓

Cloud Deployment
```

---

# Architectural Strengths

- Modular project structure
- Clear separation of responsibilities
- Deterministic ML pipeline
- Configuration-driven behavior
- Experiment tracking with MLflow
- Explainability support
- Independent audit modules
- Automated testing
- REST API support
- Interactive dashboard

---

# Current Limitations

Current scope intentionally excludes

- Distributed computing
- Time-series specific auditing
- Data drift monitoring
- Hyperparameter optimization
- Multi-user authentication
- Kubernetes deployment

These can be added without major architectural changes due to the modular design.

---

# Summary

Agentic ML Audit Copilot follows a modular architecture where every audit stage is isolated into its own component.

The workflow is orchestrated using LangGraph, while machine learning logic remains deterministic and independently testable.

This design keeps the codebase maintainable, extensible, and suitable for future enhancements without requiring major structural changes.