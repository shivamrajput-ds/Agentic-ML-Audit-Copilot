# Known Limitations

## Overview

Agentic ML Audit Copilot is designed to audit tabular machine learning datasets before model training.

The current implementation intentionally focuses on the most common machine learning workflows while keeping the architecture modular and easy to understand.

This document describes the current limitations of the project.

---

# Supported Dataset Formats

Currently supported

- CSV (.csv)

Not currently supported

- Excel (.xlsx)
- Parquet
- JSON
- SQL Databases
- Data Warehouses

---

# Dataset Size

The application is optimized for small to medium-sized datasets.

Very large datasets may require significant memory because Pandas performs in-memory processing.

Future versions may include

- Polars
- Dask
- Chunk-based processing

---

# Machine Learning Tasks

Currently supported

- Binary Classification
- Multiclass Classification
- Regression

Not currently supported

- Time Series Forecasting
- Clustering
- Recommendation Systems
- Reinforcement Learning

---

# Leakage Detection

The leakage module detects possible leakage risks.

Examples

- Duplicate target columns
- Target-like column names
- High feature correlation
- Proxy features

The system does not guarantee that leakage is confirmed.

Final validation should always be performed by the user.

---

# Data Quality

Current checks include

- Missing values
- Duplicate rows
- Constant columns
- High-cardinality columns
- Identifier-like columns

Additional checks such as schema validation and data contracts are not yet implemented.

---

# Explainability

Explainability currently includes

- SHAP values
- Feature importance

Additional techniques such as

- Partial Dependence Plots
- ICE Plots
- LIME

are not included in the current version.

---

# Baseline Models

The project trains baseline models only.

Classification

- Logistic Regression
- Random Forest Classifier

Regression

- Linear Regression
- Random Forest Regressor

Hyperparameter optimization is outside the current scope.

---

# Distributed Computing

The current implementation runs on a single machine.

Distributed processing frameworks such as

- Spark
- Ray
- Dask

are not yet supported.

---

# Authentication

The application currently does not include

- User authentication
- Authorization
- Role-based access control

The dashboard is intended for local use and demonstrations.

---

# API Security

Current API implementation does not include

- API keys
- JWT authentication
- OAuth2
- Rate limiting

These features are planned for future versions.

---

# Streamlit Dashboard

The dashboard supports one audit session at a time.

Features such as

- User accounts
- Shared workspaces
- Multi-user collaboration

are not available.

---

# Experiment Tracking

MLflow tracks experiments locally.

Remote tracking servers are supported through configuration but are not configured by default.

---

# Reports

Current report formats

- Markdown
- JSON

PDF and HTML report generation are planned for future releases.

---

# Testing

The automated test suite focuses on core functionality.

Future improvements may include

- Integration tests
- API tests
- UI tests
- Docker validation
- Performance testing

---

# Production Deployment

The project includes Docker support.

Production deployment topics such as

- Kubernetes
- Monitoring
- Centralized logging
- Auto-scaling

are outside the current scope.

---

# Future Improvements

Potential enhancements include

- Polars support
- Dask support
- Excel support
- Data Drift Detection
- Feature Drift Detection
- Fairness Analysis
- Hyperparameter Optimization
- Cloud Storage Integration
- Authentication
- Team Collaboration

---

# Summary

The current version provides a complete workflow for auditing tabular machine learning datasets before model training.

While some advanced enterprise capabilities are intentionally outside the project's scope, the modular architecture allows these features to be added in future releases without major structural changes.