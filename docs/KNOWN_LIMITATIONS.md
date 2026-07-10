# Known Limitations

## Overview

**Agentic ML Audit Copilot** is designed to audit tabular machine learning datasets before baseline model training.

The project intentionally focuses on deterministic, reproducible, and modular ML auditing. It does not try to support every possible ML workflow in the first release.

This document explains the current limitations and future improvement areas for the `v1.1.0` release.

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
- High-cardinality categorical features can increase preprocessing time
- SHAP computation may be slow for larger datasets
- Very large uploads may exceed configured limits or available memory

Future improvements may include:

- Polars support
- Dask support
- Chunk-based processing
- Out-of-core execution
- Performance benchmarks
- Background job execution for long audits

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

- Human review decisions are passed through the current request/session flow
- No persistent reviewer account system
- No approval history database
- No role-based approval workflow
- No multi-reviewer sign-off
- No enterprise approval audit trail
- Stateless API continuation requires resubmitting the dataset and reviewer decision payload

Future improvements may include:

- Persistent review history
- Reviewer identity management
- Approval audit logs
- Team review workflows
- Role-based access control
- Stored review sessions

---

## Data Quality Checks

Current checks include:

- Missing values
- Duplicate rows
- Constant columns
- Near-constant columns
- High-cardinality columns
- Identifier-like columns
- Infinite values
- Basic outlier checks
- Basic dataset statistics

Not currently included:

- Formal schema validation
- Data contracts
- Business rule validation
- Cross-dataset consistency checks
- Data lineage checks
- Domain-specific validation rules
- Automated data correction

The system reports data issues and recommendations, but it does not silently modify the dataset.

---

## Metric Recommendation

The metric recommendation module suggests suitable metrics based on the detected task and audit context.

Current limitations:

- Recommendations are rule-based
- Domain-specific business metrics are not inferred automatically
- Cost-sensitive metrics are not fully supported
- Ranking, survival, and time-series metrics are not supported
- Business impact trade-offs must still be decided by the user or reviewer

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
- No automated model selection for deployment
- No fairness-aware model optimization

The goal is sanity-check benchmarking, not final model optimization.

---

## Explainability

Current explainability features may include:

- Built-in feature importance
- SHAP summaries when supported
- Human-readable interpretation notes

Current limitations:

- SHAP can be slow for larger datasets
- Some model and data combinations may not support SHAP cleanly
- Explanations are baseline-focused
- No fairness-specific explanation layer
- No counterfactual or causal explanation support

Future improvements may include:

- Partial Dependence Plots
- ICE plots
- LIME
- Counterfactual explanations
- Fairness-aware explanations
- Better local explanation views

---

## MLflow Tracking

MLflow is used for experiment tracking.

Current limitations:

- Local tracking is the default setup
- Remote MLflow server configuration is not automated
- Model registry workflow is not fully implemented
- No automatic model promotion or approval lifecycle
- Docker container runs the app services, while MLflow UI is usually inspected separately in local development

Future improvements may include:

- Remote MLflow setup guide
- Model registry integration
- Experiment comparison dashboard
- Artifact storage configuration
- Model approval lifecycle

---

## Streamlit Dashboard

The Streamlit dashboard is designed for local usage, demos, Streamlit Cloud deployment, and portfolio presentation.

Current limitations:

- Single-user usage
- No login system
- No collaborative workspaces
- No persistent project history
- No multi-user review queue
- No cloud-native session storage
- Session state may reset after refresh or redeployment
- Large datasets may make the dashboard slower

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
- Persistent user identity
- Production-grade request audit logs

Production deployments should add authentication, authorization, monitoring, rate limiting, request logging, and secure secret management.

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
- Signed approval reports
- Shareable report links

Future improvements may include:

- PDF export
- HTML export
- Shareable report links
- Report templates
- Versioned audit reports

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
- Replace security or governance review

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
- Workflow helper behavior
- FastAPI behavior
- JSON-safe response helpers

Future improvements may include:

- Broader FastAPI integration tests
- Streamlit UI tests
- Docker validation tests
- End-to-end workflow tests
- Performance benchmarks
- Load testing
- Regression tests for larger datasets

---

## Deployment

The project includes Docker and Streamlit Cloud deployment support.

Current limitations:

- Single-container local Docker deployment focus
- No Kubernetes manifests
- No Helm charts
- No cloud deployment templates
- No autoscaling setup
- No centralized monitoring setup
- No production-grade logging stack
- Streamlit Cloud deployment is dashboard-focused and does not expose the full Docker runtime

Docker currently runs:

```text
FastAPI:   8000
Streamlit: 8501
```

Published Docker images:

```text
shivamrajput130/agentic-ml-audit-copilot:latest
shivamrajput130/agentic-ml-audit-copilot:v1.1.0
```

Future improvements may include:

- Kubernetes deployment
- Cloud deployment templates
- Monitoring and alerting
- Centralized logging
- Health-based autoscaling
- Background worker architecture

---

## Data Privacy

The project is intended for learning, local demos, and portfolio-style evaluation.

Current limitations:

- No enterprise data governance layer
- No dataset retention policy
- No automatic redaction of sensitive fields
- No PII detection module
- No encryption-at-rest configuration
- No multi-tenant isolation

Users should avoid uploading confidential, regulated, or sensitive datasets to public demo deployments.

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
- Vulnerability scanning
- Container image scanning
- Dependency security scanning

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

The current version focuses on CSV-based tabular classification and regression workflows, human review, baseline benchmarking, MLflow tracking, explainability, FastAPI, Streamlit, Docker, Streamlit Cloud deployment, and report generation.

Advanced enterprise features are intentionally outside the current scope, but the modular architecture is designed so they can be added in future releases.
