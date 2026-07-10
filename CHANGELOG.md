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
- Dedicated Human Review Gate dashboard tab
- Audit Q&A tab for asking questions about completed audit results
- New FastAPI helper endpoints:
  - `GET /metadata`
  - `GET /workflow-guide`
  - `GET /human-review/decision-template`
  - `GET /audit/modes`
- FastAPI human-review-first workflow:
  - `POST /audit/review-gate`
  - `POST /audit/after-human-approval`
- Premium Streamlit dashboard layout
- Streamlit deployment support for the updated v1.1.0 workflow
- Docker image release tags:
  - `shivamrajput130/agentic-ml-audit-copilot:latest`
  - `shivamrajput130/agentic-ml-audit-copilot:v1.1.0`
- Updated repository assets:
  - Repository banner
  - System architecture diagram
  - Human-in-the-loop workflow diagram
  - FastAPI workflow diagram
  - Streamlit dashboard screenshots
  - FastAPI docs screenshot
  - Demo GIF

### Improved

- Streamlit UI design, layout, cards, dashboard sections, and navigation
- Human Review Gate UX with reviewer decision controls and review export
- FastAPI endpoint descriptions and Swagger documentation
- API response safety by removing runtime objects from JSON responses
- Human review messaging when MLflow, SHAP, and reports are paused
- Workflow status visibility across Streamlit and FastAPI
- MLflow tab messaging for paused and completed workflow states
- SHAP and feature-importance display flow in the dashboard
- JSON serialization across API, reports, downloads, and session state
- Documentation for Docker, Streamlit Cloud, API usage, architecture, assets, and limitations
- README presentation for live demo, Docker Hub, architecture diagrams, screenshots, and v1.1.0 workflow
- CI and local quality-check instructions
- Configuration fallbacks and safer error handling
- README-ready asset structure for cleaner GitHub presentation

### Fixed

- Streamlit crash caused by nested expanders in the Human Review Gate
- Streamlit report tab crash caused by unsafe empty-dictionary comparison
- Missing Audit Q&A tab in the dashboard
- MLflow tab messaging when the workflow is paused at the human gate
- Over-strict test assertions around optional response fields
- FastAPI human review decision validation
- API documentation clarity for stateless approval flow
- Docker startup flow for running FastAPI and Streamlit together
- Docker build reliability notes for dependency download retries
- README Docker section that previously implied MLflow UI was started inside the Docker container
- Several edge cases in audit modules and test fixtures

### Deployment

- Verified Docker container health check
- Verified FastAPI on port `8000`
- Verified Streamlit on port `8501`
- Published Docker Hub `v1.1.0` image
- Verified Docker Hub pull for `v1.1.0`
- Verified Streamlit Cloud deployment for the updated dashboard

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
- **Minor**: New features, audit modules, workflow capabilities, or deployment improvements
- **Patch**: Bug fixes, documentation updates, performance improvements, and maintenance work
