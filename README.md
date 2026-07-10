<p align="center">
  <img src="assets/branding/repo_banner.png" width="100%" alt="Agentic ML Audit Copilot Banner">
</p>

<h1 align="center">Agentic ML Audit Copilot</h1>

<p align="center">
  <b>Human-in-the-loop ML audit system before model training</b>
</p>

<p align="center">
  A deterministic-first ML engineering project that audits tabular datasets before model development,
  surfaces data risks, pauses risky workflows for human review, benchmarks baseline models,
  tracks experiments, explains results, and generates grounded audit reports.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/FastAPI-API-009688?style=for-the-badge&logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/LangGraph-Workflow-6A5ACD?style=for-the-badge" alt="LangGraph">
  <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn" alt="scikit-learn">
  <img src="https://img.shields.io/badge/MLflow-Tracking-0194E2?style=for-the-badge" alt="MLflow">
  <img src="https://img.shields.io/badge/SHAP-Explainability-FFB000?style=for-the-badge" alt="SHAP">
  <img src="https://img.shields.io/badge/Docker-v1.1.0-2496ED?style=for-the-badge&logo=docker" alt="Docker">
  <img src="https://img.shields.io/badge/License-MIT-success?style=for-the-badge" alt="License">
</p>

<p align="center">
  <a href="https://shivamrajput-ds-agentic-ml-audit-copilo-appstreamlit-app-joxap5.streamlit.app">
    <img src="https://img.shields.io/badge/Live_Streamlit_App-success?style=for-the-badge" alt="Live Streamlit App">
  </a>
  <a href="https://youtu.be/kFzNam74QBc">
    <img src="https://img.shields.io/badge/YouTube_Demo-red?style=for-the-badge" alt="YouTube Demo">
  </a>
  <a href="https://hub.docker.com/r/shivamrajput130/agentic-ml-audit-copilot">
    <img src="https://img.shields.io/badge/Docker_Hub-2496ED?style=for-the-badge&logo=docker" alt="Docker Hub">
  </a>
</p>

---

## Overview

Most ML projects start by training models too early.

In real projects, poor data quality, possible target leakage, class imbalance, and unsuitable metrics can make a model look stronger than it really is.

**Agentic ML Audit Copilot** solves this by auditing a dataset before model training. It behaves like a junior ML reviewer: it profiles the data, detects common ML risks, routes risky cases through a human review gate, and only then continues to baseline modeling, tracking, explainability, and report generation.

This is not an AutoML tool. The goal is not to train the best possible model. The goal is to decide whether the dataset is ready for responsible baseline experimentation.

---

## Live Demo

### Streamlit Application

```text
https://shivamrajput-ds-agentic-ml-audit-copilo-appstreamlit-app-joxap5.streamlit.app/
```

### YouTube Walkthrough

```text
https://youtu.be/kFzNam74QBc
```

### Docker Hub

```text
https://hub.docker.com/r/shivamrajput130/agentic-ml-audit-copilot
```

Recommended Docker image:

```text
shivamrajput130/agentic-ml-audit-copilot:v1.1.0
```

---

## Demo Preview

<p align="center">
  <img src="assets/demo/demo_git.gif" width="100%" alt="Agentic ML Audit Copilot Demo">
</p>

---

## What It Checks

| Area | What the system does |
| --- | --- |
| Dataset Profiling | Rows, columns, data types, memory usage, missing values, duplicate rows, and target summary |
| Problem Detection | Detects classification or regression setup |
| Data Quality | Finds missing values, duplicates, constant columns, near-constant columns, ID-like columns, high-cardinality columns, infinite values, and outliers |
| Leakage Risk | Flags target-like columns, suspicious names, proxy features, and suspicious correlations |
| Class Imbalance | Measures class distribution, minority class, imbalance ratio, and severity |
| Risk Aggregation | Combines deterministic risk signals into review items |
| Decision Routing | Decides whether the workflow can continue or needs human review |
| Human Review | Allows reviewer decisions before modeling continues |
| Metric Recommendation | Suggests suitable metrics for the detected problem type |
| Preprocessing | Builds scikit-learn preprocessing pipelines |
| Baseline Models | Trains baseline models for comparison |
| MLflow Tracking | Logs baseline experiment metadata and metrics |
| Explainability | Generates built-in feature importance and SHAP-based summaries when available |
| Reports | Exports Markdown and JSON audit reports |
| API | Provides FastAPI endpoints for programmatic audit access |
| UI | Provides a Streamlit dashboard for interactive review |

---

## Core Design Principles

- **Deterministic-first:** Python performs ML computation, data checks, and risk detection.
- **Human-in-the-loop:** Risky datasets can be paused before modeling.
- **LLM is not the judge:** The LLM is used only for explanations, Q&A, and report writing.
- **Baseline-first:** The system trains simple baselines instead of pretending to be AutoML.
- **Transparent workflow:** Every major decision is visible in the dashboard and API response.
- **Audit before training:** The project focuses on responsible pre-training review.

---

## System Architecture

<p align="center">
  <img src="assets/architecture/01_system_architecture.png" width="95%" alt="System Architecture">
</p>

The architecture separates the Streamlit UI, FastAPI backend, LangGraph workflow, audit modules, risk routing, human review, modeling, tracking, explainability, and reporting layers.

---

## Human-in-the-Loop Workflow

<p align="center">
  <img src="assets/architecture/02_hitl_workflow.png" width="95%" alt="Human-in-the-Loop Workflow">
</p>

The workflow can pause at the Human Review Gate when important risks are detected.

Reviewer decisions include:

- Accept risk and continue
- Accept flag and fix later
- Mark false positive
- Needs data fix
- Reject modeling

If the final human decision approves modeling, the workflow continues to metric recommendation, preprocessing, baseline models, MLflow, explainability, and final report generation.

If the final human decision rejects modeling, the workflow stops so the dataset can be fixed first.

---

## FastAPI Workflow

<p align="center">
  <img src="assets/architecture/03_fastapi_workflow.png" width="95%" alt="FastAPI Workflow">
</p>

The API supports both direct audit runs and a human-review-first workflow.

---

## Dashboard Screenshots

### Streamlit Home

<p align="center">
  <img src="assets/screenshots/01_streamlit_home.png" width="95%" alt="Streamlit Home">
</p>

### Human Review Gate

<p align="center">
  <img src="assets/screenshots/02_human_review_gate.png" width="95%" alt="Human Review Gate">
</p>

### Executive Dashboard

<p align="center">
  <img src="assets/screenshots/03_executive_dashboard.png" width="95%" alt="Executive Dashboard">
</p>

### FastAPI Docs

<p align="center">
  <img src="assets/screenshots/04_fastapi_docs.png" width="95%" alt="FastAPI Docs">
</p>

---

## Workflow

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

---

## Technology Stack

| Layer | Tools |
| --- | --- |
| Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | scikit-learn |
| Workflow Orchestration | LangGraph |
| API | FastAPI |
| Dashboard | Streamlit |
| Experiment Tracking | MLflow |
| Explainability | SHAP |
| LLM Provider | Groq |
| Visualization | Plotly |
| Testing | pytest |
| Linting and Formatting | Ruff |
| Packaging | uv |
| Deployment | Docker, Streamlit Community Cloud |

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/shivamrajput-ds/Agentic-ML-Audit-Copilot.git
cd Agentic-ML-Audit-Copilot
```

### 2. Create a virtual environment

```bash
uv venv --python 3.12
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Windows Git Bash:

```bash
source .venv/Scripts/activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
uv pip install -r requirements.txt
uv pip install -e .
```

### 4. Configure environment variables

Create a `.env` file locally:

```text
GROQ_API_KEY=your_groq_api_key
```

Do not commit `.env`.

For deterministic audit-only usage, LLM features can be disabled if supported by the configuration:

```bash
export LLM_ENABLED=false
```

Windows PowerShell:

```powershell
$env:LLM_ENABLED="false"
```

### 5. Run Streamlit

```bash
uv run streamlit run app/streamlit_app.py --server.port 8501
```

Open:

```text
http://localhost:8501
```

### 6. Run FastAPI

Use a second terminal:

```bash
uv run uvicorn app.api:app --reload --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000/docs
```

---

## FastAPI Endpoints

### System

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/` | API information |
| GET | `/health` | Health check |
| GET | `/metadata` | Project and runtime metadata |
| GET | `/workflow-guide` | Human review workflow guide |

### Audit

| Method | Endpoint | Description |
| --- | --- | --- |
| POST | `/audit` | Run audit workflow |
| POST | `/audit/summary` | Run audit and return lightweight summary |
| GET | `/audit/modes` | Show available audit modes |

### Human Review

| Method | Endpoint | Description |
| --- | --- | --- |
| POST | `/audit/review-gate` | Run audit until human review gate |
| GET | `/human-review/decision-template` | Return reviewer decision JSON template |
| POST | `/audit/after-human-approval` | Continue workflow after reviewer approval |

### Recommended Human Review API Flow

```text
1. POST /audit/review-gate
2. Review human_review.review_items
3. GET /human-review/decision-template
4. Fill reviewer decision JSON
5. POST /audit/after-human-approval
6. Continue to metrics, baselines, MLflow, SHAP, and final report
```

---

## MLflow Tracking

The project logs baseline experiment information with MLflow.

Tracked information may include:

- Problem type
- Baseline model names
- Evaluation metrics
- Best baseline model
- Parameters
- Runtime metadata

Run MLflow UI locally:

```bash
uv run mlflow ui --backend-store-uri mlruns --host 127.0.0.1 --port 5000
```

Open:

```text
http://127.0.0.1:5000
```

Note: the Docker container runs FastAPI and Streamlit. MLflow tracking data is generated by the workflow, but the MLflow UI is usually inspected separately in local development.

---

## Testing

Run the full test suite:

```bash
uv run pytest -q
```

Run a specific test file:

```bash
uv run pytest tests/test_data_quality.py -q
```

---

## Code Quality

Run Ruff checks:

```bash
uv run ruff check . --fix --unsafe-fixes
```

Format the project:

```bash
uv run ruff format .
```

---

## Docker

The Docker image runs both services:

- Streamlit on port `8501`
- FastAPI on port `8000`

Pull the stable release:

```bash
docker pull shivamrajput130/agentic-ml-audit-copilot:v1.1.0
```

Run the stable release:

```bash
docker run --rm \
  --name agentic-audit-copilot \
  -p 8501:8501 \
  -p 8000:8000 \
  -e GROQ_API_KEY="your_groq_api_key" \
  shivamrajput130/agentic-ml-audit-copilot:v1.1.0
```

One-line Git Bash command:

```bash
docker run --rm --name agentic-audit-copilot -p 8501:8501 -p 8000:8000 -e GROQ_API_KEY="your_groq_api_key" shivamrajput130/agentic-ml-audit-copilot:v1.1.0
```

Open:

```text
Streamlit Dashboard: http://localhost:8501
FastAPI Docs:       http://localhost:8000/docs
Health Check:       http://localhost:8000/health
```

Build locally:

```bash
docker build -t agentic-ml-audit-copilot .
```

Run local image:

```bash
docker run --rm \
  --name agentic-audit-test \
  -p 8501:8501 \
  -p 8000:8000 \
  -e GROQ_API_KEY="your_groq_api_key" \
  agentic-ml-audit-copilot:latest
```

Tag and push:

```bash
docker tag agentic-ml-audit-copilot:latest shivamrajput130/agentic-ml-audit-copilot:latest
docker tag agentic-ml-audit-copilot:latest shivamrajput130/agentic-ml-audit-copilot:v1.1.0

docker push shivamrajput130/agentic-ml-audit-copilot:latest
docker push shivamrajput130/agentic-ml-audit-copilot:v1.1.0
```

More details:

```text
DOCKER.md
```

---

## Streamlit Community Cloud

Streamlit Community Cloud deploys from GitHub, not Docker Hub.

For Streamlit Cloud:

- Push the latest code to GitHub.
- Set `GROQ_API_KEY` in Streamlit Cloud secrets.
- Do not upload `.env`.
- Make sure `requirements.txt`, `app/streamlit_app.py`, `src/`, `config.yaml`, and project modules are pushed.

Example Streamlit secret:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

---

## Project Structure

```text
Agentic-ML-Audit-Copilot/
├── app/
│   ├── api.py
│   └── streamlit_app.py
├── src/
│   ├── audit/
│   └── utils/
├── tests/
├── docs/
├── assets/
│   ├── architecture/
│   ├── branding/
│   ├── demo/
│   └── screenshots/
├── data/
│   └── sample/
├── reports/
├── artifacts/
├── logs/
├── .github/
├── .streamlit/
├── config.yaml
├── pyproject.toml
├── requirements.txt
├── pytest.ini
├── Dockerfile
├── DOCKER.md
├── README.md
├── CHANGELOG.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── SECURITY.md
└── LICENSE.md
```

---

## Documentation

| Document | Purpose |
| --- | --- |
| `docs/ARCHITECTURE.md` | System architecture and workflow design |
| `docs/API.md` | FastAPI endpoint guide |
| `docs/USAGE.md` | Streamlit app and API usage guide |
| `docs/TESTING.md` | Testing strategy and commands |
| `docs/KNOWN_LIMITATIONS.md` | Known limitations and scope boundaries |
| `docs/ROADMAP.md` | Planned improvements |
| `docs/PROJECT_REVIEW.md` | Portfolio-level project review |
| `docs/ASSETS.md` | Asset and screenshot guide |
| `DOCKER.md` | Docker build, run, and publish guide |
| `CHANGELOG.md` | Release history |
| `CONTRIBUTING.md` | Contribution guidelines |
| `CODE_OF_CONDUCT.md` | Community rules |
| `SECURITY.md` | Security policy |
| `LICENSE.md` | License details |

---

## Engineering Highlights

- LangGraph-based audit workflow
- Parallel deterministic audit checks
- Risk Aggregator and Decision Router
- Human Review Gate for risky datasets
- Reviewer decision export as JSON
- FastAPI backend with Swagger documentation
- Streamlit dashboard with audit tabs and downloads
- Baseline model benchmarking
- MLflow experiment tracking
- SHAP and built-in feature importance support
- LLM-based audit report and Q&A
- JSON-safe API responses
- Configuration-driven behavior
- Centralized logging and exception handling
- pytest test suite
- Ruff linting and formatting
- Dockerized local deployment
- Streamlit Cloud deployment
- GitHub Actions CI support

---

## Current Limitations

This project currently focuses on:

- CSV datasets
- Tabular ML
- Classification and regression
- Single-machine execution
- Baseline model benchmarking
- Pre-training audit and review

It is not a replacement for:

- Full enterprise data governance
- Security review
- Production monitoring
- Fairness certification
- Model approval boards
- AutoML systems
- Production model serving platforms

---

## Roadmap

Planned improvements:

- Data drift detection
- Feature drift detection
- Fairness and bias analysis
- Hyperparameter optimization
- PDF reports
- HTML reports
- Polars support
- Dask support
- Authentication
- Team workspaces
- Kubernetes deployment
- Cloud deployment templates

---

## Contributing

Contributions are welcome.

Before opening a pull request:

```bash
uv run ruff check . --fix --unsafe-fixes
uv run ruff format .
uv run pytest -q
```

Please read:

```text
CONTRIBUTING.md
CODE_OF_CONDUCT.md
```

---

## Security

Do not commit secrets.

Use:

```text
.env.example
```

for documenting required variables.

Use environment variables or platform secrets for real API keys.

---

## License

This project is released under the MIT License.

See:

```text
LICENSE.md
```

---

## Author

**Shivam Rajput**

Data Science | Machine Learning | MLOps | Agentic AI

- Portfolio: `https://shivamrajput-ds.github.io/portfolio-website/`
- GitHub: `https://github.com/shivamrajput-ds`
- LinkedIn: `https://www.linkedin.com/in/shivam-rajput-ds/`
- Docker Hub: `https://hub.docker.com/r/shivamrajput130/agentic-ml-audit-copilot`
- YouTube: `https://youtu.be/kFzNam74QBc`
- Kaggle: `https://www.kaggle.com/shivamja`
- LeetCode: `https://leetcode.com/u/ShivamSynapse/`
- X: `https://x.com/ShivamR65014299`

---

<p align="center">
  <b>Agentic ML Audit Copilot</b>
</p>

<p align="center">
  Python • scikit-learn • FastAPI • Streamlit • LangGraph • MLflow • SHAP • Docker • Groq
</p>
