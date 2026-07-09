# Project Review

## Overview

**Agentic ML Audit Copilot** is an end-to-end machine learning audit platform that evaluates tabular datasets before model development.

Instead of optimizing models immediately, the system performs a deterministic audit of the dataset to identify potential risks that could negatively impact downstream machine learning performance.

The project combines traditional machine learning engineering practices with workflow orchestration, experiment tracking, explainability, and LLM-assisted reporting while keeping all ML computation deterministic.

---

# Objectives

The primary objectives of the project are to:

- Audit datasets before model training
- Improve dataset quality assessment
- Detect possible data leakage risks
- Recommend suitable evaluation metrics
- Build reproducible preprocessing pipelines
- Benchmark baseline models
- Generate structured audit reports
- Improve model transparency through explainability

---

# Architecture Overview

The project follows a modular architecture where every audit stage is implemented as an independent component.

Current workflow:

```
Dataset Upload

↓

Dataset Profiling

↓

Problem Type Detection

↓

Data Quality Audit

↓

Possible Leakage Detection

↓

Metric Recommendation

↓

Class Imbalance Detection

↓

Preprocessing Pipeline

↓

Baseline Models

↓

MLflow Tracking

↓

Explainability

↓

LLM Report Generation
```

Each module has a single responsibility, making the codebase easier to test, maintain, and extend.

---

# Key Strengths

## Modular Design

The project is divided into independent audit modules, including:

- Dataset Profiling
- Problem Type Detection
- Data Quality Audit
- Leakage Detection
- Metric Recommendation
- Class Imbalance Analysis
- Preprocessing
- Baseline Models
- Explainability
- Report Generation

This modular approach improves maintainability and extensibility.

---

## Deterministic-First Philosophy

The project intentionally separates deterministic computation from generative AI.

Python performs:

- Dataset profiling
- Data quality analysis
- Leakage checks
- Feature engineering
- Baseline training
- Metric calculation
- Explainability
- Experiment tracking

The LLM is responsible only for:

- Report generation
- Audit explanations
- Audit Q&A

This design improves reproducibility and reduces the risk of AI-generated inconsistencies.

---

## Workflow Orchestration

The complete audit process is orchestrated using LangGraph.

The workflow executes automatically while preserving deterministic execution order and modularity.

---

## Explainability

Model behavior can be interpreted using:

- SHAP values
- Feature importance

These components improve transparency by helping users understand which features influence predictions.

---

## Experiment Tracking

MLflow is integrated for experiment management.

Tracked information includes:

- Parameters
- Metrics
- Baseline models
- Artifacts

This improves reproducibility and simplifies experiment comparison.

---

## User Experience

The Streamlit dashboard provides:

- Dataset upload
- Audit execution
- Interactive visualizations
- Human review dashboard
- Audit report generation
- Report download

The interface is designed to make complex ML auditing accessible without requiring command-line interaction.

---

## API Support

FastAPI exposes the complete audit workflow through REST endpoints.

This allows the audit engine to be integrated into external systems and automated pipelines.

---

## Testing and Quality Assurance

The project includes automated tests covering core functionality.

Current test coverage includes:

- Dataset Profiling
- Problem Type Detection
- Data Quality Audit
- Leakage Detection
- Class Imbalance
- Preprocessing
- Baseline Models

The repository also includes:

- Ruff formatting
- Ruff linting
- GitHub Actions CI
- Docker validation

---

# Supported Problem Types

Current support includes:

- Binary Classification
- Multiclass Classification
- Regression

Input format:

- CSV datasets

---

# Engineering Decisions

Several implementation decisions were made intentionally.

## Configuration-Driven Design

Application behavior is controlled through `config.yaml`.

This avoids hardcoded values and improves flexibility.

---

## Human-in-the-Loop Review

Potential issues requiring manual validation are surfaced through the Human Review Dashboard.

Examples include:

- Possible target leakage
- Identifier columns
- High missing values
- Severe class imbalance

The application intentionally avoids making automatic business decisions.

---

## Baseline-First Modeling

Rather than immediately optimizing models, the project first establishes baseline performance.

This provides a reliable reference point before advanced experimentation.

---

## Production-Oriented Repository

The repository includes:

- Docker support
- GitHub Actions CI
- Automated tests
- Documentation
- Security policy
- Contribution guidelines
- Changelog
- Type hints
- Structured logging

These practices improve maintainability and reproducibility.

---

# Current Limitations

The current version intentionally excludes:

- Distributed processing
- Time-series auditing
- Data drift monitoring
- Feature drift detection
- Hyperparameter optimization
- Fairness analysis
- Authentication
- Multi-user collaboration
- Cloud-native deployment

These features are planned for future releases.

---

# Future Roadmap

Potential future enhancements include:

- Data Drift Detection
- Feature Drift Detection
- Fairness & Bias Analysis
- Hyperparameter Optimization
- Time-Series Support
- PDF Reports
- HTML Reports
- Polars Support
- Dask Support
- Authentication
- Team Workspaces
- Kubernetes Deployment
- Cloud Deployment Templates

---

# Intended Audience

The project is suitable for:

- Data Scientists
- Machine Learning Engineers
- AI Engineers
- Students
- Researchers
- Portfolio projects
- Educational demonstrations

---

# Technologies

Core technologies include:

- Python
- Pandas
- NumPy
- Scikit-learn
- LangGraph
- FastAPI
- Streamlit
- MLflow
- SHAP
- Docker
- GitHub Actions
- Ruff
- Pytest

---

# Repository Quality Checklist

| Feature | Status |
|----------|:------:|
| Modular Architecture | ✅ |
| Deterministic Workflow | ✅ |
| FastAPI | ✅ |
| Streamlit | ✅ |
| LangGraph | ✅ |
| MLflow | ✅ |
| SHAP | ✅ |
| Docker | ✅ |
| GitHub Actions CI | ✅ |
| Automated Tests | ✅ |
| Documentation | ✅ |

---

# Conclusion

Agentic ML Audit Copilot demonstrates a production-oriented approach to auditing tabular machine learning datasets before model development.

By combining deterministic machine learning, workflow orchestration, experiment tracking, explainability, containerization, automated testing, and LLM-assisted reporting, the project provides a structured foundation for building reliable machine learning workflows.

The modular architecture also allows additional audit capabilities to be integrated with minimal changes, making the project suitable for both learning and future expansion.