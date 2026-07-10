# Project Review

## Overview

**Agentic ML Audit Copilot** is a deterministic-first machine learning audit system for tabular datasets.

The project reviews a dataset before baseline model training. It checks data quality, possible leakage risks, class imbalance, metric suitability, workflow risk, baseline performance, explainability, and report readiness.

The goal is not to replace an ML engineer or act as an AutoML system. The goal is to behave like a practical ML reviewer that helps answer:

> Is this dataset ready for responsible baseline modeling?

The project combines ML engineering, workflow orchestration, experiment tracking, explainability, API design, dashboarding, Docker, testing, and LLM-assisted reporting while keeping all ML computation deterministic.

---

## Objectives

The main objectives are:

- Audit datasets before model training
- Detect data quality issues
- Surface possible leakage risks
- Detect class imbalance
- Recommend suitable evaluation metrics
- Build reproducible preprocessing pipelines
- Benchmark simple baseline models
- Track experiments with MLflow
- Explain baseline model behavior
- Pause risky workflows for human review
- Generate structured Markdown and JSON reports
- Provide both Streamlit and FastAPI interfaces

---

## Architecture Overview

The project follows a modular architecture where each audit stage has a focused responsibility.

Current workflow:

```text
User
  |
  v
Streamlit UI / FastAPI API
  |
  v
CSV Upload + Target Selection
  |
  v
LangGraph Audit Workflow
  |
  v
Dataset Profiler
  |
  v
Problem Type Detector
  |
  v
Parallel Audit Layer
  |-- Data Quality Audit
  |-- Leakage Detection
  |-- Class Imbalance Detection
  |
  v
Risk Aggregator
  |
  v
Decision Router
  |
  v
Human Review Gate
  |-- Stop / Fix Dataset
  |-- Human Approved
          |
          v
      Metric Recommender
          |
          v
      Preprocessing Pipeline
          |
          v
      Baseline Models
          |
          v
      MLflow Tracking
          |
          v
      Explainability / SHAP
          |
          v
      LLM Audit Report
          |
          v
      Audit Q&A
          |
          v
      Final Dashboard + JSON Report
```

This design makes the system easier to test, maintain, extend, and explain.

---

## Key Strengths

## 1. Deterministic-First Design

The project clearly separates deterministic computation from LLM-generated explanation.

Python performs:

- Dataset profiling
- Data quality checks
- Leakage risk detection
- Class imbalance analysis
- Metric recommendation logic
- Preprocessing
- Baseline model training
- Metric calculation
- MLflow tracking
- SHAP and feature importance generation

The LLM is used only for:

- Report generation
- Audit explanation
- Audit Q&A

This improves reproducibility and avoids relying on the LLM for core ML decisions.

---

## 2. Human-in-the-Loop Workflow

The project includes a Human Review Gate for risky datasets.

The system can pause when it finds issues such as:

- Possible target leakage
- Identifier-like features
- Severe class imbalance
- High missing values
- Ambiguous problem type
- Other workflow-level risks

Reviewer decisions include:

- Accept risk and continue
- Accept flag and fix later
- Mark false positive
- Needs data fix
- Reject modeling

This is one of the strongest parts of the project because it shows that the system does not blindly automate risky ML decisions.

---

## 3. Modular Architecture

The project is divided into clear modules:

- Profiler
- Problem detector
- Data quality audit
- Leakage detection
- Class imbalance detection
- Risk aggregation
- Decision routing
- Metric recommendation
- Preprocessing
- Baseline models
- MLflow tracking
- Explainability
- LLM report generation
- Workflow orchestration

This structure improves maintainability and makes future audit modules easier to add.

---

## 4. LangGraph Workflow Orchestration

LangGraph is used to organize the audit workflow.

This gives the project a clear agentic workflow structure without making the LLM responsible for deterministic computation.

The workflow is useful for:

- State management
- Ordered execution
- Conditional routing
- Human review pause points
- Future workflow expansion

---

## 5. FastAPI Backend

The project exposes a REST API for programmatic access.

Important API groups include:

System endpoints:

- `GET /`
- `GET /health`
- `GET /metadata`
- `GET /workflow-guide`

Audit endpoints:

- `POST /audit`
- `POST /audit/summary`
- `GET /audit/modes`

Human review endpoints:

- `POST /audit/review-gate`
- `GET /human-review/decision-template`
- `POST /audit/after-human-approval`

This makes the project more than a Streamlit-only demo.

---

## 6. Streamlit Dashboard

The Streamlit dashboard provides a usable interface for the complete audit workflow.

It includes:

- Dataset upload
- Target selection
- Executive dashboard
- Data quality view
- Leakage risk view
- Human Review Gate
- Baseline model results
- MLflow status
- Explainability output
- Report download
- Audit Q&A

The UI makes the project easier to understand for recruiters, reviewers, and users.

---

## 7. Baseline-First Modeling

The project focuses on baseline models rather than pretending to provide final optimized models.

Classification baselines may include:

- Logistic Regression
- Random Forest Classifier

Regression baselines may include:

- Linear Regression
- Random Forest Regressor

This is a practical engineering choice because baselines help validate whether the data is usable before advanced optimization.

---

## 8. MLflow Experiment Tracking

MLflow is integrated for experiment tracking.

Tracked information may include:

- Problem type
- Model names
- Parameters
- Metrics
- Best baseline model
- Runtime metadata

This improves experiment reproducibility and makes the project closer to a real ML engineering workflow.

---

## 9. Explainability

The project includes explainability support through:

- Feature importance
- SHAP summaries when supported

This helps users understand which features influence baseline model behavior.

Explainability is especially useful after human approval because it gives the reviewer more confidence in the modeling result.

---

## 10. Testing and Code Quality

The repository includes automated tests and quality checks.

Current quality practices include:

- pytest test suite
- Ruff linting
- Ruff formatting
- GitHub Actions CI
- Type hints
- Structured logging
- Custom exceptions
- Configuration-driven behavior

This improves reliability and makes the repository easier to maintain.

---

## Supported Scope

Current support:

- CSV datasets
- Tabular ML
- Binary classification
- Multiclass classification
- Regression
- Local execution
- Docker execution
- Streamlit dashboard
- FastAPI backend

---

## Engineering Decisions

## Configuration-Driven Design

Most behavior is controlled through:

```text
config.yaml
```

This reduces hardcoding and makes the system easier to modify.

---

## No Automatic Leakage Confirmation

The leakage module reports possible risks only.

This is intentional. Leakage often requires domain knowledge, so the system surfaces review signals instead of claiming final truth.

---

## Human Review Before Risky Modeling

If serious risks are found, the workflow can pause before downstream modeling.

This prevents the system from producing baseline results that may look valid but are based on unsafe assumptions.

---

## Preprocessing Inside Pipelines

Preprocessing is handled inside scikit-learn pipelines.

This helps reduce train-test leakage and keeps training behavior reproducible.

---

## Separate UI, API, and Workflow Logic

The project separates:

- Streamlit UI
- FastAPI API
- LangGraph workflow
- Audit modules
- Utilities
- Reports

This separation makes the project easier to debug and extend.

---

## Repository Quality Checklist

| Area | Status |
| --- | :---: |
| Modular architecture | Yes |
| Deterministic audit workflow | Yes |
| Human Review Gate | Yes |
| Risk Aggregator | Yes |
| Decision Router | Yes |
| FastAPI backend | Yes |
| Streamlit dashboard | Yes |
| LangGraph workflow | Yes |
| MLflow tracking | Yes |
| SHAP / feature importance | Yes |
| Docker support | Yes |
| GitHub Actions CI | Yes |
| pytest test suite | Yes |
| Ruff formatting and linting | Yes |
| Documentation | Yes |
| Security policy | Yes |
| Contribution guide | Yes |
| Changelog | Yes |

---

## Current Limitations

The current version intentionally does not include:

- Distributed processing
- Time-series auditing
- Data drift monitoring
- Feature drift detection
- Fairness and bias certification
- Hyperparameter optimization
- Production authentication
- Multi-user collaboration
- Persistent reviewer history
- Kubernetes deployment
- Cloud-native deployment templates

These are valid future improvements, but they are outside the current scope.

---

## Future Roadmap

Potential future improvements include:

- Data drift detection
- Feature drift detection
- Fairness and bias analysis
- Hyperparameter optimization
- Time-series audit support
- PDF reports
- HTML reports
- Excel and Parquet support
- Polars support
- Dask support
- Authentication
- Team workspaces
- Persistent human review history
- Model registry integration
- Kubernetes deployment
- Cloud deployment templates

---

## Intended Audience

This project is suitable for:

- Data science learners
- ML engineering learners
- Data scientists
- Machine learning engineers
- AI engineers
- Portfolio reviewers
- Internship and job applications
- Educational demonstrations

---

## Technologies Used

Core technologies:

- Python
- Pandas
- NumPy
- scikit-learn
- LangGraph
- FastAPI
- Streamlit
- MLflow
- SHAP
- Groq
- Docker
- GitHub Actions
- Ruff
- pytest

---

## Overall Review

Agentic ML Audit Copilot is a strong portfolio-level ML engineering project because it goes beyond basic model training.

It demonstrates:

- Data auditing before modeling
- Deterministic ML checks
- Human-in-the-loop workflow design
- FastAPI backend design
- Streamlit dashboarding
- MLflow tracking
- Explainability
- Docker deployment
- Automated testing
- Documentation discipline

The strongest part of the project is the combination of deterministic audit checks with a Human Review Gate. This makes the project more realistic than a simple ML dashboard or AutoML-style demo.

---

## Conclusion

Agentic ML Audit Copilot demonstrates a practical, production-oriented approach to auditing tabular ML datasets before baseline model training.

By combining deterministic audit modules, workflow orchestration, human review, experiment tracking, explainability, FastAPI, Streamlit, Docker, testing, and documentation, the project provides a strong foundation for reliable ML engineering workflows.

The current version is best positioned as a high-quality portfolio and learning project with a clear path toward more advanced enterprise features.
