<p align="center">
  <img src="assets/branding/banner.png" width="100%" alt="Agentic ML Audit Copilot Banner">
</p>

<h1 align="center">
🤖 Agentic ML Audit Copilot
</h1>

<p align="center">
<b>Human-in-the-Loop • Deterministic-First • Machine Learning Audit Platform</b>
</p>

<p align="center">

A production-ready platform that audits tabular machine learning datasets before model development by combining deterministic machine learning, explainability, experiment tracking, and grounded LLM-assisted reporting.

</p>

---

<p align="center">

<img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python">

<img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi">

<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit">

<img src="https://img.shields.io/badge/LangGraph-Agentic%20Workflow-blue?style=for-the-badge">

<img src="https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2?style=for-the-badge">

<img src="https://img.shields.io/badge/SHAP-Explainability-orange?style=for-the-badge">

<img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker">

<img src="https://img.shields.io/badge/Tests-96%20Passed-success?style=for-the-badge">

<img src="https://img.shields.io/badge/License-MIT-success?style=for-the-badge">

</p>

---

<p align="center">

<a href="https://shivamrajput-ds-agentic-ml-audit-copilo-appstreamlit-app-joxap5.streamlit.app">

<img src="https://img.shields.io/badge/🚀_Live_Streamlit_App-success?style=for-the-badge">

</a>

<a href="https://youtu.be/kFzNam74QBc">

<img src="https://img.shields.io/badge/🎥_YouTube_Demo-red?style=for-the-badge">

</a>

<a href="https://hub.docker.com/r/shivamrajput130/agentic-ml-audit-copilot">

<img src="https://img.shields.io/badge/🐳_Docker_Hub-2496ED?style=for-the-badge">

</a>

<a href="https://github.com/shivamrajput-ds/Agentic-ML-Audit-Copilot">

<img src="https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github">

</a>

</p>

---

# 📌 Overview

Most machine learning projects start by directly training models.

However, poor data quality, hidden target leakage, severe class imbalance, and incorrect evaluation metrics often lead to misleading results.

**Agentic ML Audit Copilot** introduces a deterministic-first audit workflow that evaluates a dataset **before** model training.

Instead of acting like an AutoML system, the project behaves like a **Junior Machine Learning Reviewer**, performing automated checks while keeping the final decision with the user.

The workflow combines:

- Dataset Profiling
- Data Quality Assessment
- Possible Leakage Detection
- Class Imbalance Analysis
- Metric Recommendation
- Scikit-learn Baseline Models
- MLflow Experiment Tracking
- SHAP Explainability
- Human Review Dashboard
- LLM-powered Audit Explanation

---

# 🚀 Why This Project?

Real-world ML projects often fail because of problems in the data rather than the model itself.

Common issues include:

- Missing values
- Duplicate rows
- Constant features
- Identifier columns
- Hidden leakage
- Incorrect evaluation metrics
- Severe class imbalance
- Lack of experiment tracking
- Poor explainability

Instead of training increasingly complex models, this project first answers a more important question:

> **"Is this dataset actually ready for machine learning?"**

---

# 🎯 Key Highlights

✅ Deterministic-first workflow

✅ Human-in-the-loop review

✅ LangGraph workflow orchestration

✅ MLflow experiment tracking

✅ SHAP explainability

✅ FastAPI REST API

✅ Streamlit Dashboard

✅ Docker Deployment

✅ GitHub Actions CI

✅ Automated Testing

---

# 🎬 Live Demo

## 🌐 Streamlit Application

**Live Application**

https://shivamrajput-ds-agentic-ml-audit-copilo-appstreamlit-app-joxap5.streamlit.app/

---

## 🎥 YouTube Walkthrough

Watch the complete project demonstration.

https://youtu.be/kFzNam74QBc

---

## 📸 Demo Preview

<p align="center">

<img src="assets/demo/demo_gif.gif" width="100%">

</p>

---

# ⭐ Features

| Category | Features |
|------------|------------------------------------------------|
| Dataset Audit | Dataset profiling, schema inspection, missing values, duplicate detection |
| Problem Detection | Binary classification, multiclass classification, regression |
| Data Quality | Missing values, duplicates, constant features, identifiers, outliers |
| Leakage Detection | Target-like columns, proxy features, correlation-based checks |
| Metric Recommendation | Automatic metric recommendation based on task |
| Class Imbalance | Severity detection with recommendations |
| Preprocessing | Scikit-learn preprocessing pipelines |
| Baseline Models | Logistic Regression, Random Forest, Linear Regression |
| Explainability | SHAP summaries, Feature Importance |
| Experiment Tracking | MLflow integration |
| API | FastAPI REST API |
| Dashboard | Interactive Streamlit UI |
| Reports | Markdown & JSON |
| Deployment | Docker |
| Testing | pytest |
| Code Quality | Ruff |
| Workflow | LangGraph |

---

# 🏗 System Architecture

<p align="center">

<img src="assets/diagrams/architecture.png" width="95%">

</p>

The platform separates user interaction, workflow orchestration, audit modules, explainability, and reporting into independent layers.

This modular design makes the application easier to maintain, extend, and test.

---

# 🔄 Audit Workflow

```text
CSV Upload
      │
      ▼
Dataset Profiling
      │
      ▼
Problem Detection
      │
      ▼
Data Quality Audit
      │
      ▼
Possible Leakage Detection
      │
      ▼
Class Imbalance Analysis
      │
      ▼
Metric Recommendation
      │
      ▼
Preprocessing Pipeline
      │
      ▼
Baseline Models
      │
      ▼
MLflow Tracking
      │
      ▼
Explainability
      │
      ▼
LLM Audit Report
      │
      ▼
Human Review Dashboard
```

---

# 📷 Application Screenshots

## Streamlit Dashboard

![](assets/screenshots/streamlit_home.png)

---

## Dataset Upload

![](assets/screenshots/dataset_upload.png)

---

## Data Quality Dashboard

![](assets/screenshots/data_quality.png)

---

## Leakage Detection

![](assets/screenshots/leakage_risks.png)

---

## Baseline Models

![](assets/screenshots/baseline_models.png)

---

## SHAP Explainability

![](assets/screenshots/explainability_shap.png)

---

## MLflow Tracking

![](assets/screenshots/mlflow_tracking.png)

---

## Swagger API

![](assets/screenshots/swagger_ui.png)

---

# 🛠 Technology Stack

| Layer | Technologies |
|---------|------------------------------|
| Language | Python |
| Data | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Workflow | LangGraph |
| Explainability | SHAP |
| Experiment Tracking | MLflow |
| Dashboard | Streamlit |
| API | FastAPI |
| LLM | Groq |
| Visualization | Plotly |
| Testing | pytest |
| Linting | Ruff |
| Packaging | uv |
| Deployment | Docker |

---

# ⚡ Quick Start

## Clone Repository

```bash
git clone https://github.com/shivamrajput-ds/Agentic-ML-Audit-Copilot.git

cd Agentic-ML-Audit-Copilot
```

---

## Create Virtual Environment

```bash
uv venv
```

Windows

```bash
.venv\Scripts\activate
```

Linux / macOS

```bash
source .venv/bin/activate
```

---

## Install Dependencies

```bash
uv pip install -r requirements.txt
```

---

## Configure Environment Variables

Create

```
.env
```

Example

```text
GROQ_API_KEY=YOUR_GROQ_API_KEY
```

---

## Run Streamlit

```bash
uv run streamlit run app/streamlit_app.py
```

Open

```
http://localhost:8501
```

---

## Run FastAPI

```bash
uv run uvicorn app.api:app --reload
```

Open

```
http://localhost:8000/docs
```

---

# 🐳 Docker

The project is fully containerized.

Pull directly from Docker Hub:

```bash
docker pull shivamrajput130/agentic-ml-audit-copilot:latest
```

Or build locally:

```bash
docker build -t agentic-ml-audit-copilot .
```

➡️ **README Part 2** continues with:
- Docker Usage
- API
- MLflow
- Testing
- Project Structure
- Documentation
- Roadmap
- Design Philosophy
- Contributing
- License
- Author
- Footer

---

# 🐳 Docker Deployment

The application is fully containerized and can be executed without installing Python locally.

## Pull Image from Docker Hub

```bash
docker pull shivamrajput130/agentic-ml-audit-copilot:latest
```

## Run Container

```bash
docker run --rm \
-p 8501:8501 \
-p 8000:8000 \
-e GROQ_API_KEY="YOUR_GROQ_API_KEY" \
shivamrajput130/agentic-ml-audit-copilot:latest
```

Open:

```text
Streamlit
http://localhost:8501

FastAPI
http://localhost:8000/docs

Health Check
http://localhost:8000/health
```

## Build Locally

```bash
docker build -t agentic-ml-audit-copilot .
```

```bash
docker run --rm \
-p 8501:8501 \
-p 8000:8000 \
-e GROQ_API_KEY="YOUR_GROQ_API_KEY" \
agentic-ml-audit-copilot
```

---

# 🌐 REST API

The project exposes a FastAPI backend for programmatic access.

| Method | Endpoint | Description |
|---------|----------|-------------|
| GET | `/` | API Information |
| GET | `/health` | Health Check |
| POST | `/audit` | Run Complete Audit |
| POST | `/audit/summary` | Lightweight Summary |

Swagger UI

```
http://localhost:8000/docs
```

ReDoc

```
http://localhost:8000/redoc
```

---

# 📊 MLflow Experiment Tracking

Every audit automatically logs experiment metadata.

Tracked information includes

- Experiment Name
- Baseline Models
- Evaluation Metrics
- Parameters
- Best Model
- Optional Model Artifacts

Run MLflow locally

```bash
mlflow ui
```

Open

```
http://localhost:5000
```

---

# 🔍 Explainability

The explainability module provides transparent model interpretation.

Current capabilities include

- SHAP Summary
- Feature Importance
- Human-readable explanations

The goal is to help users understand **why** a baseline model produced a particular result rather than only reporting evaluation metrics.

---

# 🧪 Testing

The repository includes automated unit tests covering the core audit workflow.

Run all tests

```bash
uv run pytest
```

Run a specific test

```bash
uv run pytest tests/test_data_quality.py -v
```

Current status

```
✅ 96 Tests Passed
```

---

# ✨ Code Quality

Ruff is used for linting and formatting.

Run lint

```bash
uv run ruff check .
```

Format code

```bash
uv run ruff format .
```

GitHub Actions automatically runs

- Ruff
- Tests
- Formatting checks

on every push and pull request.

---

# 📂 Project Structure

```text
Agentic-ML-Audit-Copilot/
│
├── app/
│   ├── api.py
│   └── streamlit_app.py
│
├── src/
│   ├── audit/
│   ├── utils/
│   └── ...
│
├── assets/
│
├── data/
│
├── docs/
│
├── reports/
│
├── tests/
│
├── Dockerfile
├── config.yaml
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

# 📚 Documentation

Comprehensive documentation is available in the **docs/** directory.

| Document | Description |
|----------|-------------|
| ARCHITECTURE.md | Complete system architecture |
| API.md | FastAPI documentation |
| DOCKER.md | Docker guide |
| USAGE.md | Usage guide |
| TESTING.md | Testing documentation |
| ROADMAP.md | Future roadmap |
| PROJECT_REVIEW.md | Technical project review |
| KNOWN_LIMITATIONS.md | Current limitations |
| SECURITY.md | Security policy |
| ASSETS.md | Assets guide |

---

# 🎯 Engineering Highlights

The project demonstrates practical software engineering concepts including

- Modular Architecture
- Configuration-driven Development
- Human-in-the-loop Design
- Deterministic ML Pipelines
- LangGraph Workflow Orchestration
- MLflow Experiment Tracking
- SHAP Explainability
- Docker Deployment
- FastAPI Backend
- Streamlit Dashboard
- Automated Testing
- GitHub Actions CI

---

# 💡 Design Philosophy

The project follows five core principles.

### Deterministic First

Python performs all machine learning computation.

The LLM never performs prediction or model training.

---

### Human-in-the-Loop

Potential risks are surfaced to the user.

Final decisions remain with the ML practitioner.

---

### Modular Design

Every audit module has a single responsibility.

This makes the project easier to maintain and extend.

---

### Reproducibility

Scikit-learn pipelines, fixed random seeds, and MLflow improve reproducibility.

---

### Separation of Concerns

The UI, API, workflow engine, audit modules, explainability, reporting, and utilities remain independent.

---

# ⚠ Current Limitations

Current scope intentionally focuses on

- CSV datasets
- Tabular Machine Learning
- Classification
- Regression
- Single-machine execution

Future versions may include

- Data Drift Detection
- Feature Drift Detection
- Fairness Analysis
- Time-Series Auditing
- Authentication
- Team Workspaces
- Kubernetes Deployment

---

# 🛣 Roadmap

Planned improvements

- Excel Support
- Parquet Support
- Polars Integration
- Dask Integration
- Data Drift Detection
- Bias & Fairness Analysis
- Hyperparameter Optimization
- PDF Reports
- HTML Reports
- Cloud Storage
- Authentication
- Model Registry

---

# 🤝 Contributing

Contributions are welcome.

Before opening a Pull Request

```bash
uv run ruff format .
```

```bash
uv run ruff check .
```

```bash
uv run pytest
```

Please read

```
CONTRIBUTING.md
```

---

# 📜 License

This project is released under the **MIT License**.

See

```
LICENSE.md
```

for details.

---

# 👨‍💻 Author

## Shivam Rajput

Data Science | Machine Learning | MLOps | Agentic AI

### Connect with me

- 🌐 Portfolio  
  https://shivamrajput-ds.github.io/portfolio-website/

- 💻 GitHub  
  https://github.com/shivamrajput-ds

- 💼 LinkedIn  
  https://www.linkedin.com/in/shivam-rajput-ds/

- 🐳 Docker Hub  
  https://hub.docker.com/r/shivamrajput130/agentic-ml-audit-copilot

- 🎥 YouTube  
  https://youtu.be/kFzNam74QBc

- 📊 Kaggle  
  https://www.kaggle.com/shivamja

- 💻 LeetCode  
  https://leetcode.com/u/ShivamSynapse/

- 🐦 X (Twitter)  
  https://x.com/ShivamR65014299

---

# ⭐ Support

If you found this project useful,

please consider giving it a ⭐ on GitHub.

It helps increase the visibility of the project and motivates future improvements.

---

<p align="center">

Built with ❤️ using

Python • Scikit-learn • FastAPI • Streamlit • LangGraph • MLflow • SHAP • Docker • Groq

</p>

<p align="center">

<b>Agentic ML Audit Copilot • v1.0.0</b>

</p>