# Known Limitations

## Overview

**Agentic ML Audit Copilot** is designed to audit tabular machine learning datasets before model development.

The project intentionally focuses on deterministic, reproducible, and modular ML auditing rather than supporting every possible machine learning workflow.

This document describes the current limitations of the project and highlights areas planned for future development.

---

# Supported Dataset Formats

Currently supported:

- CSV (.csv)

Planned support:

- Excel (.xlsx)
- Parquet
- JSON
- SQL databases
- Cloud storage connectors

---

# Dataset Size

The application is optimized for **small to medium-sized tabular datasets**.

Since preprocessing is currently performed using Pandas, very large datasets may require substantial memory.

Future improvements may include:

- Polars
- Dask
- Chunk-based processing
- Out-of-core execution

---

# Supported Machine Learning Tasks

Current support:

- Binary Classification
- Multiclass Classification
- Regression

Not currently supported:

- Time Series Forecasting
- Clustering
- Survival Analysis
- Recommendation Systems
- Reinforcement Learning

---

# Leakage Detection

The leakage module reports **possible leakage risks** only.

Current checks include:

- Target-like column names
- Identifier columns
- Highly correlated features
- Proxy feature detection
- Duplicate target-like columns

The application intentionally **does not automatically confirm leakage**.

Human review is required before making modeling decisions.

---

# Data Quality Checks

Current checks include:

- Missing values
- Duplicate rows
- Constant columns
- High-cardinality columns
- Identifier-like columns
- Basic dataset statistics

Currently not included:

- Schema validation
- Data contracts
- Business rule validation
- Cross-dataset consistency checks

---

# Explainability

Current explainability features:

- SHAP values
- Feature importance

Future improvements may include:

- Partial Dependence Plots (PDP)
- ICE Plots
- LIME
- Counterfactual explanations

---

# Baseline Modeling

The application trains baseline models only.

Classification:

- Logistic Regression
- Random Forest Classifier

Regression:

- Linear Regression
- Random Forest Regressor

The objective is benchmarking rather than maximizing predictive performance.

Hyperparameter optimization is intentionally outside the current scope.

---

# Distributed Computing

The application currently executes on a single machine.

Distributed execution frameworks such as:

- Spark
- Ray
- Dask

are not yet supported.

---

# Authentication & User Management

The application currently does not include:

- User authentication
- Authorization
- Role-based access control
- Multi-user accounts

The dashboard is intended primarily for local usage, demonstrations, and portfolio purposes.

---

# API Security

The FastAPI application currently does not include:

- JWT authentication
- OAuth2
- API key authentication
- Rate limiting

These capabilities are planned for future releases.

---

# Streamlit Dashboard

Current limitations include:

- Single-user sessions
- No collaborative workspaces
- No persistent project history
- No user account management

---

# MLflow Tracking

MLflow experiments are tracked locally by default.

Remote MLflow servers are supported through configuration but are not configured automatically.

---

# Report Generation

Current report formats:

- Markdown
- JSON

Future formats may include:

- PDF
- HTML

---

# Testing

The automated test suite focuses primarily on deterministic audit modules.

Future improvements may include:

- FastAPI integration tests
- Streamlit UI tests
- Docker validation tests
- End-to-end workflow tests
- Performance benchmarks
- Load testing

---

# Production Deployment

The project includes Docker support.

However, the following production capabilities are intentionally outside the current scope:

- Kubernetes deployment
- Monitoring and alerting
- Centralized logging
- Auto-scaling
- Service mesh integration

---

# LLM Limitations

The LLM is used only for:

- Report generation
- Audit explanations
- Audit question answering

The LLM **does not perform machine learning computations**.

All deterministic ML operations are executed using Python and Scikit-learn.

---

# Future Roadmap

Planned enhancements include:

- Data Drift Detection
- Feature Drift Detection
- Fairness & Bias Analysis
- Hyperparameter Optimization
- Time-Series Auditing
- Excel & Parquet Support
- Polars Integration
- Dask Integration
- Cloud Storage Support
- Authentication & User Management
- Team Workspaces
- Kubernetes Deployment

---

# Summary

Agentic ML Audit Copilot provides a production-oriented, deterministic workflow for auditing tabular machine learning datasets before model development.

While several advanced enterprise capabilities are intentionally outside the current scope, the modular architecture allows these features to be integrated in future releases with minimal architectural changes.