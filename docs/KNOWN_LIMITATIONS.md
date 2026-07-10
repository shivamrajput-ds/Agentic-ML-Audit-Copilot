# Known Limitations

## Overview

**Agentic ML Audit Copilot** is designed to audit tabular machine learning datasets before baseline model training.

The project intentionally focuses on deterministic, reproducible, and modular ML auditing. It does not try to support every possible ML workflow in the first release.

This document explains the current limitations and future improvement areas.

---

## Dataset Formats

Currently supported:

- CSV

Not currently supported:

- Excel
- Parquet
- JSON
- SQL databases
- Cloud storage connectors

Planned improvements may include:

- Excel support
- Parquet support
- JSON support
- Cloud storage integration
- Database connectors

---

## Dataset Size

The application is currently optimized for small to medium-sized tabular datasets.

Current limitations:

- Pandas-based processing
- Single-machine execution
- Memory usage depends on dataset size and number of columns
- SHAP computation may be slow for larger datasets

Future improvements may include:

- Polars support
- Dask support
- Chunk-based processing
- Out-of-core execution
- Performance benchmarks

---

## Supported ML Tasks

Currently supported:

- Binary classification
- Multiclass classification
- Regression

Not currently supported:

- Time-series forecasting
- Clustering
- Survival analysis
- Recommendation systems
- Reinforcement learning
- Computer vision
- Audio or speech models
- Large language model evaluation

---

## Leakage Detection

The leakage module reports possible leakage risks only.

Current checks may include:

- Target-like column names
- Outcome-like feature names
- Identifier-like columns
- Highly correlated features
- Proxy feature patterns
- Duplicate or target-derived-looking columns

Important limitation:

The system does not automatically confirm leakage.

Leakage findings should be treated as review signals. A human reviewer should decide whether the flagged column is truly unsafe for modeling.

---

## Human Review Workflow

The Human Review Gate is designed to make risk decisions explicit.

Current limitations:

- Human review decisions are passed through the current request flow
- No persistent reviewer account system
- No approval history database
- No role-based approval workflow
- No multi-reviewer sign-off

Future improvements may include:

- Persistent review history
- Reviewer identity management
- Approval audit logs
- Team review workflows
- Role-based access control

---

## Data Quality Checks

Current checks include:

- Missing values
- Duplicate rows
- Constant columns
- Near-constant columns
- High-cardinality columns
- Identifier-like columns
- Basic outlier checks
- Basic dataset statistics

Not currently included:

- Formal schema validation
- Data contracts
- Business rule validation
- Cross-dataset consistency checks
- Data lineage checks
- Domain-specific validation rules

---

## Metric Recommendation

The metric recommendation module suggests suitable metrics based on the detected task and audit context.

Current limitations:

- Recommendations are rule-based
- Domain-specific business metrics are not inferred automatically
- Cost-sensitive metrics are not fully supported
- Ranking, survival, and time-series metrics are not supported

---

## Baseline Modeling

The application trains baseline models only.

Classification baselines may include:

- Logistic Regression
- Random Forest Classifier

Regression baselines may include:

- Linear Regression
- Random Forest Regressor

Current limitations:

- No advanced model tuning
- No hyperparameter optimization
- No neural network training
- No model registry promotion flow
- No production model serving workflow

The goal is sanity-check benchmarking, not final model optimization.

---

## Explainability

Current explainability features may include:

- Feature importance
- SHAP summaries when supported

Current limitations:

- SHAP can be slow for larger datasets
- Some model and data combinations may not support SHAP cleanly
- Explanations are baseline-focused
- No fairness-specific explanation layer

Future improvements may include:

- Partial Dependence Plots
- ICE plots
- LIME
- Counterfactual explanations
- Fairness-aware explanations

---

## MLflow Tracking

MLflow is used for experiment tracking.

Current limitations:

- Local tracking is the default setup
- Remote MLflow server configuration is not automated
- Model registry workflow is not fully implemented
- No automatic model promotion or approval lifecycle

Future improvements may include:

- Remote MLflow setup guide
- Model registry integration
- Experiment comparison dashboard
- Artifact storage configuration

---

## Streamlit Dashboard

The Streamlit dashboard is designed for local usage, demos, and portfolio presentation.

Current limitations:

- Single-user usage
- No login system
- No collaborative workspaces
- No persistent project history
- No multi-user review queue
- No cloud-native session storage

---

## FastAPI Security

The current FastAPI service is designed for local and demo usage.

Not currently included:

- JWT authentication
- OAuth2
- API key authentication
- Rate limiting
- Request throttling
- Role-based access control
- Tenant isolation

Production deployments should add authentication, authorization, monitoring, rate limiting, and secure secret management.

---

## Report Generation

Current report formats:

- Markdown
- JSON

Not currently included:

- PDF reports
- HTML reports
- Email delivery
- Scheduled reports
- Report version history

Future improvements may include:

- PDF export
- HTML export
- Shareable report links
- Report templates

---

## LLM Usage

The LLM is used only for:

- Audit explanations
- Markdown report generation
- Audit Q&A

The LLM does not:

- Train models
- Compute metrics
- Confirm leakage
- Approve risky datasets
- Replace human judgment

All deterministic ML operations are performed using Python and scikit-learn.

---

## Testing

The test suite focuses on deterministic audit modules and workflow behavior.

Current testing scope includes:

- Problem detection
- Data quality checks
- Leakage risk detection
- Class imbalance checks
- Preprocessing
- Baseline models
- Metric recommendation
- Profiler behavior

Future improvements may include:

- FastAPI integration tests
- Streamlit UI tests
- Docker validation tests
- End-to-end workflow tests
- Performance benchmarks
- Load testing

---

## Deployment

The project includes Docker support.

Current limitations:

- Single-container local deployment focus
- No Kubernetes manifests
- No Helm charts
- No cloud deployment templates
- No autoscaling setup
- No centralized monitoring setup
- No production-grade logging stack

Future improvements may include:

- Kubernetes deployment
- Cloud deployment templates
- Monitoring and alerting
- Centralized logging
- Health-based autoscaling

---

## Production Readiness

This project is production-oriented in structure, but it should not be treated as a complete enterprise production system without additional hardening.

Before production use, consider adding:

- Authentication
- Authorization
- Rate limiting
- Audit logging
- Secure secret management
- Monitoring and alerting
- Data governance controls
- Infrastructure hardening
- Backup and recovery strategy
- Compliance review

---

## Roadmap

Planned improvement areas:

- Data drift detection
- Feature drift detection
- Fairness and bias analysis
- Hyperparameter optimization
- PDF and HTML reports
- Excel and Parquet support
- Polars and Dask support
- Authentication and user management
- Team workspaces
- Persistent human review history
- Model registry support
- Kubernetes deployment
- Cloud deployment templates

---

## Summary

Agentic ML Audit Copilot provides a deterministic-first workflow for auditing tabular ML datasets before baseline modeling.

The current version focuses on CSV-based tabular classification and regression workflows, human review, baseline benchmarking, MLflow tracking, explainability, FastAPI, Streamlit, and report generation.

Advanced enterprise features are intentionally outside the current scope, but the modular architecture is designed so they can be added in future releases.
