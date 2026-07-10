# Roadmap

This document outlines the planned development roadmap for **Agentic ML Audit Copilot**.

The roadmap is a guide, not a fixed promise. Priorities may change as the project evolves.

---

## Vision

Build a deterministic-first machine learning audit system that helps data scientists review tabular datasets before model training.

The long-term goal is to combine:

- Data quality auditing
- Possible leakage risk detection
- Human-in-the-loop review
- Baseline model benchmarking
- Explainability
- Experiment tracking
- Workflow orchestration
- API-first access
- Clean reporting
- Deployment-ready engineering practices

The project will continue to follow one important principle:

> Python performs ML computation. The LLM explains the results.

---

## Current Release

## Version 1.1.0 - Completed

Current release includes:

- Dataset profiling
- Problem type detection
- Data quality audit
- Possible leakage risk detection
- Class imbalance detection
- Risk Aggregator
- Decision Router
- Human Review Gate
- Reviewer decision workflow
- Continue-after-approval workflow
- Metric recommendation
- Scikit-learn preprocessing pipeline
- Baseline model training
- Feature importance support
- SHAP explainability support
- MLflow experiment tracking
- LangGraph workflow orchestration
- FastAPI REST API
- FastAPI human review endpoints
- Streamlit dashboard
- Streamlit Cloud deployment
- Audit Q&A
- Markdown report export
- JSON report export
- Docker support
- Docker Hub release
- GitHub Actions CI
- pytest test suite
- Ruff formatting and linting
- Repository assets and documentation

Release artifacts:

```text
Streamlit Dashboard
FastAPI API
Docker image
GitHub documentation
Architecture diagrams
Dashboard screenshots
Demo assets
```

Docker image tags:

```text
shivamrajput130/agentic-ml-audit-copilot:latest
shivamrajput130/agentic-ml-audit-copilot:v1.1.0
```

---

## Version 1.2 - Data Format and Usability Improvements

Planned improvements:

- Excel dataset support
- Parquet dataset support
- JSON dataset support
- Better missing-value visualizations
- Better target distribution visualizations
- Improved leakage risk summaries
- Configurable baseline model selection
- Improved report templates
- Better dashboard download experience
- Safer handling for large categorical columns
- Cleaner upload validation messages
- Better sample dataset examples

Status: Planned

---

## Version 1.3 - Explainability Improvements

Planned improvements:

- Partial Dependence Plots
- ICE plots
- LIME support
- Feature interaction visualization
- Model comparison dashboard
- Better SHAP summaries
- Clearer feature importance explanations
- Explainability export support
- Local explanation examples
- Improved interpretation notes for non-technical users

Status: Planned

---

## Version 1.4 - Advanced Data Auditing

Planned improvements:

- Data drift detection
- Feature drift detection
- Dataset comparison
- Schema validation
- Data contracts
- Feature statistics history
- Train-test distribution comparison
- Better outlier diagnostics
- Domain rule validation support
- Better duplicate and near-duplicate diagnostics
- Dataset readiness scoring improvements

Status: Planned

---

## Version 1.5 - Machine Learning Enhancements

Planned improvements:

- Hyperparameter optimization
- Cross-validation dashboard
- Feature selection suggestions
- Model calibration analysis
- Probability calibration plots
- Additional baseline models
- Configurable model registry hooks
- Better metric comparison views
- Better classification threshold analysis
- Better regression residual diagnostics

Status: Planned

---

## Version 1.6 - Human Review and Governance

Planned improvements:

- Persistent human review history
- Reviewer identity tracking
- Approval audit logs
- Review status dashboard
- Multi-reviewer support
- Role-based approval flow
- Dataset approval history
- Exportable review decision records
- Signed review summary export
- Review comments and evidence tracking

Status: Planned

---

## Version 2.0 - Scalability and Enterprise Readiness

Planned improvements:

- Polars support
- Dask support
- Chunk-based processing
- Out-of-core execution
- Large dataset support
- Background audit jobs
- Progress tracking
- API job status endpoint
- Remote MLflow setup guide
- Cloud deployment templates
- More robust artifact storage
- Production logging strategy

Status: Future

---

## Future Ideas

Possible future modules:

- Time-series auditing
- Fairness and bias analysis
- PDF reports
- HTML reports
- Email report delivery
- Dataset versioning
- Model registry integration
- Cloud storage connectors
- REST API versioning
- Monitoring dashboards
- Kubernetes deployment
- Team workspaces
- Authentication and authorization
- Workspace-level project history
- Report sharing links
- Scheduled audit runs

Status: Idea

---

## Documentation Roadmap

Planned documentation improvements:

- More API request examples
- More sample datasets
- End-to-end demo guide
- Deployment examples
- Troubleshooting guide
- MLflow usage guide
- Human review workflow examples
- Report interpretation guide
- Docker Hub usage guide
- Streamlit Cloud deployment guide
- Interview/project explanation guide

Status: Planned

---

## Testing Roadmap

Future testing goals:

- FastAPI integration tests
- Streamlit workflow tests
- Docker validation tests
- End-to-end audit workflow tests
- Human review API tests
- Performance benchmarks
- Larger synthetic dataset tests
- Error-handling regression tests
- Snapshot tests for JSON-safe responses
- Deployment smoke tests

Status: Planned

---

## Community Roadmap

Future community improvements:

- Good first issues
- Issue templates
- Pull request template
- More example workflows
- Contributor setup guide
- Project discussion board
- Project wiki
- Demo dataset contribution guide
- Documentation contribution guide

Status: Planned

---

## Guiding Principles

Future development should follow these principles:

- Keep ML computation deterministic
- Keep human review explicit
- Keep modules small and testable
- Avoid over-automation of risky decisions
- Keep API responses JSON-safe
- Keep preprocessing inside pipelines
- Keep documentation accurate
- Avoid exaggerated claims
- Prefer reliable baselines before advanced optimization
- Treat leakage as a possible risk, not automatic truth
- Keep the LLM limited to explanation, Q&A, and report writing

---

## What Should Not Be Prioritized Yet

The project should not rush into:

- Complex AutoML claims
- Production governance claims
- Full enterprise security claims
- Large-scale distributed architecture before core audit quality improves
- Too many models before stronger audit and validation modules
- Heavy UI features that do not improve review quality

The strongest direction is to improve audit reliability, explainability, review history, and deployment quality step by step.

---

## Status Legend

| Status | Meaning |
| --- | --- |
| Completed | Already implemented |
| Planned | Intended for a future release |
| Future | Larger architectural improvement |
| Idea | Possible future exploration |

---

## Summary

Agentic ML Audit Copilot is evolving toward a stronger ML audit and review platform.

The current version already supports deterministic tabular data auditing, human-in-the-loop review, baseline modeling, MLflow tracking, SHAP explainability, FastAPI, Streamlit, Streamlit Cloud, Docker, Docker Hub release, and documentation.

Future releases will focus on broader data format support, stronger explainability, drift detection, governance, scalability, and deployment readiness while preserving the core deterministic-first design.
