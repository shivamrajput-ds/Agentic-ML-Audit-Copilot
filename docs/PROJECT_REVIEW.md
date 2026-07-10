# Project Review

## Overview

**Agentic ML Audit Copilot** is a deterministic-first machine learning audit system for tabular datasets.

The project reviews a dataset before baseline model training. It checks data quality, possible leakage risks, class imbalance, metric suitability, workflow risk, baseline performance, explainability, and report readiness.

The goal is not to replace an ML engineer or behave like a full AutoML platform. The goal is to act like a practical ML reviewer that helps answer:

> Is this dataset ready for responsible baseline modeling?

The project combines ML engineering, workflow orchestration, experiment tracking, explainability, API design, dashboarding, Docker, testing, and LLM-assisted reporting while keeping all core ML computation deterministic.

---

## Release Context

Current reviewed version:

```text
v1.1.0
```

Key release focus:

- Human-in-the-loop workflow
- Risk Aggregator
- Decision Router
- Continue-after-approval flow
- Improved Streamlit dashboard
- Improved FastAPI documentation
- Docker release
- Streamlit Cloud deployment
- GitHub-ready documentation and assets

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
- Keep the LLM limited to explanation, Q&A, and report writing

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
- Risk aggregation
- Decision routing
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

## 3. Risk Aggregator and Decision Router

The Risk Aggregator combines findings across audit modules into a workflow-level risk summary.

The Decision Router then decides whether the workflow should:

```text
Continue
Pause for Human Review
Stop / Fix Dataset
```

This makes the workflow more realistic than a simple checklist because it connects risk detection to an actual modeling decision.

---

## 4. Modular Architecture

The project is divided into clear modules:

- Profiler
- Problem detector
- Data quality audit
- Leakage detection
- Class imbalance detection
- Risk aggregation
- Decision routing
- Human review
- Metric recommendation
- Preprocessing
- Baseline models
- MLflow tracking
- Explainability
- LLM report generation
- Workflow orchestration

This structure improves maintainability and makes future audit modules easier to add.

---

## 5. LangGraph Workflow Orchestration

LangGraph is used to organize the audit workflow.

This gives the project a clear agentic workflow structure without making the LLM responsible for deterministic computation.

The workflow is useful for:

- State management
- Ordered execution
- Conditional routing
- Human review pause points
- Future workflow expansion

This is a good use of an agentic framework because the graph controls the workflow while deterministic Python modules do the actual ML work.

---

## 6. FastAPI Backend

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

## 7. Streamlit Dashboard

The Streamlit dashboard provides a usable interface for the complete audit workflow.

It includes:

- Dataset upload
- Target selection
- Executive dashboard
- Data quality view
- Leakage risk view
- Human Review Gate
- Reviewer decision controls
- Baseline model results
- MLflow status
- Explainability output
- Report download
- Audit Q&A

The UI makes the project easier to understand for recruiters, reviewers, and users.

---

## 8. Baseline-First Modeling

The project focuses on baseline models rather than pretending to provide final optimized models.

Classification baselines may include:

- Logistic Regression
- Random Forest Classifier

Regression baselines may include:

- Linear Regression
- Random Forest Regressor

This is a practical engineering choice because baselines help validate whether the data is usable before advanced optimization.

---

## 9. MLflow Experiment Tracking

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

## 10. Explainability

The project includes explainability support through:

- Feature importance
- SHAP summaries when supported
- Human-readable interpretation notes

This helps users understand which features influence baseline model behavior.

Explainability is especially useful after human approval because it gives the reviewer more confidence in the baseline modeling result.

---

## 11. Docker and Deployment Readiness

The project includes Docker support.

The Docker image runs:

```text
FastAPI:   8000
Streamlit: 8501
```

Published image tags:

```text
shivamrajput130/agentic-ml-audit-copilot:latest
shivamrajput130/agentic-ml-audit-copilot:v1.1.0
```

The Streamlit dashboard is also deployable through Streamlit Cloud from GitHub.

This gives the project two useful demo paths:

- Streamlit Cloud for quick public dashboard access
- Docker Hub for reproducible local/demo execution with both Streamlit and FastAPI

---

## 12. Testing and Code Quality

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
- JSON-safe API response handling

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
- Streamlit Cloud dashboard deployment
- Streamlit dashboard
- FastAPI backend
- Human-in-the-loop review workflow

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
- Assets and documentation

This separation makes the project easier to debug and extend.

---

## JSON-Safe API Design

The API avoids returning runtime-only Python objects such as:

- DataFrames
- Model objects
- Fitted preprocessors
- Raw training arrays
- Non-serializable NumPy objects

This is important for practical API reliability.

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
| Docker Hub release | Yes |
| Streamlit Cloud deployment | Yes |
| GitHub Actions CI | Yes |
| pytest test suite | Yes |
| Ruff formatting and linting | Yes |
| Documentation | Yes |
| Security policy | Yes |
| Contribution guide | Yes |
| Changelog | Yes |
| Asset structure | Yes |

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
- Full enterprise governance workflow

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

## Portfolio Positioning

This project is strongest when positioned as:

```text
A Human-in-the-Loop ML Audit Copilot for tabular datasets before baseline model training.
```

Best resume/project framing:

```text
Built a deterministic-first Agentic ML audit system that profiles tabular datasets, detects data quality, leakage, and imbalance risks, routes unsafe workflows through a Human Review Gate, benchmarks baseline models, logs experiments with MLflow, explains results with SHAP, and exposes the workflow through FastAPI, Streamlit, Docker, and Streamlit Cloud.
```

Avoid positioning it as:

```text
AutoML platform
Production governance platform
Final model optimization system
Enterprise security platform
```

The honest positioning is stronger because the project is focused, realistic, and defensible in interviews.

---

## Suggested Rating

For a student portfolio / internship-level ML engineering project:

| Area | Rating |
| --- | :---: |
| Idea quality | 9/10 |
| ML engineering depth | 8.5/10 |
| Agentic workflow usage | 8.5/10 |
| Human-in-the-loop design | 9/10 |
| API + UI completeness | 8.5/10 |
| Docker/deployment readiness | 8.5/10 |
| Documentation quality | 9/10 |
| Production readiness | 7/10 |
| Portfolio value | 9/10 |

Overall portfolio rating:

```text
8.8 / 10
```

With future drift detection, fairness checks, persistent review history, and stronger integration tests, this can move closer to:

```text
9.2 / 10
```

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
- Streamlit Cloud deployment
- Automated testing
- Documentation discipline

The strongest part of the project is the combination of deterministic audit checks with a Human Review Gate. This makes the project more realistic than a simple ML dashboard or AutoML-style demo.

---

## Conclusion

Agentic ML Audit Copilot demonstrates a practical, production-oriented approach to auditing tabular ML datasets before baseline model training.

By combining deterministic audit modules, workflow orchestration, human review, experiment tracking, explainability, FastAPI, Streamlit, Docker, testing, and documentation, the project provides a strong foundation for reliable ML engineering workflows.

The current version is best positioned as a high-quality portfolio and learning project with a clear path toward more advanced enterprise features.
