# Changelog

All notable changes to **Agentic ML Audit Copilot** are documented here.

This project follows a simple release format inspired by **Keep a Changelog** and uses **Semantic Versioning**.

---

## [1.1.0] - 2026-07-10

### Added

- Human-in-the-loop approval gate for risky datasets
- Risk Aggregator to combine data quality, leakage, imbalance, and workflow risks
- Decision Router to pause or continue the workflow based on audit risk
- Reviewer decision workflow with:
  - Accept risk and continue
  - Accept flag and fix later
  - Mark false positive
  - Needs data fix
  - Reject modeling
- Continue-after-approval workflow for baseline models, MLflow, explainability, and final reports
- Professional FastAPI documentation for human review workflows
- New API helper endpoints:
  - `GET /metadata`
  - `GET /workflow-guide`
  - `GET /human-review/decision-template`
  - `GET /audit/modes`
- Premium Streamlit dashboard layout
- Dedicated Human Review Gate dashboard tab
- Audit Q&A tab for asking questions about completed audit results
- Updated repository assets:
  - System architecture diagram
  - Human-in-the-loop workflow diagram
  - FastAPI workflow diagram
  - Dashboard screenshots
  - Repository banner

### Improved

- Streamlit UI design, layout, cards, dashboard sections, and navigation
- FastAPI endpoint descriptions and Swagger documentation
- API response safety by removing runtime objects from JSON responses
- Human review messaging when MLflow, SHAP, and reports are paused
- Workflow status visibility across Streamlit and FastAPI
- Test coverage for workflow modules, API behavior, and audit helpers
- Configuration fallbacks and safer error handling
- JSON serialization across API, reports, and downloads
- README-ready asset structure for a cleaner GitHub presentation

### Fixed

- Streamlit report tab crash caused by unsafe empty-dictionary comparison
- Missing Audit Q&A tab in the dashboard
- MLflow tab messaging when workflow is paused at the human gate
- Over-strict test assertions around optional response fields
- FastAPI human review decision validation
- API documentation clarity for stateless approval flow
- Several edge cases in audit modules and test fixtures

---

## [1.0.0] - 2026-07-09

### Added

- End-to-end ML audit workflow for tabular datasets
- Dataset profiling and statistical summary
- Automatic problem type detection
- Data quality assessment
- Possible data leakage risk detection
- Class imbalance analysis
- Evaluation metric recommendation
- Scikit-learn preprocessing pipeline
- Baseline model benchmarking
- Feature importance and SHAP explainability support
- MLflow experiment tracking
- LangGraph workflow orchestration
- FastAPI REST API
- Interactive Streamlit dashboard
- Markdown and JSON report export
- Docker support
- GitHub Actions CI pipeline
- Pytest test suite
- Ruff formatting and linting setup

### Improved

- Modular project architecture
- Configuration-driven behavior
- Structured logging
- Centralized exception handling
- Production-oriented repository structure
- Type hints and maintainability
- Developer workflow and setup commands

### Fixed

- Preprocessing pipeline initialization issues
- Explainability integration edge cases
- Docker startup configuration
- Dependency compatibility issues
- SHAP compatibility with NumPy
- Import resolution issues
- API serialization issues
- MLflow logging reliability
- Workflow stability and error handling

---

## Roadmap

Planned improvements:

- Data drift detection
- Feature drift detection
- Fairness and bias analysis
- Hyperparameter optimization
- PDF audit reports
- HTML audit reports
- Polars support
- Dask support
- Authentication and user management
- Team workspaces
- Kubernetes deployment
- Cloud deployment templates

---

## Versioning

This project follows Semantic Versioning.

- **Major**: Breaking architecture or API changes
- **Minor**: New features, audit modules, or workflow capabilities
- **Patch**: Bug fixes, documentation updates, performance improvements, and maintenance work