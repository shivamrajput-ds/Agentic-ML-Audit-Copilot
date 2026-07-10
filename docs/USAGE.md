# Usage Guide

## Overview

This guide explains how to install, configure, and use **Agentic ML Audit Copilot**.

The project can be used through:

- Streamlit dashboard
- FastAPI REST API
- Docker
- MLflow UI

The workflow follows a deterministic-first approach:

- Python performs ML computation and audit checks.
- The LLM is used only for explanations, audit Q&A, and report generation.
- Risky datasets can pause at the Human Review Gate before modeling continues.

---

## System Requirements

Recommended:

- Python 3.11 or 3.12
- Git
- uv
- Docker Desktop, optional

Supported operating systems:

- Windows
- Linux
- macOS

---

## Clone the Repository

```bash
git clone https://github.com/shivamrajput-ds/Agentic-ML-Audit-Copilot.git
cd Agentic-ML-Audit-Copilot
```

---

## Create Virtual Environment

## Using uv

Create the virtual environment:

```bash
uv venv
```

Activate it.

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
uv pip install -r requirements.txt
uv pip install -e .
```

---

## Using Python venv

Create the virtual environment:

```bash
python -m venv .venv
```

Activate it.

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

---

## Configure Environment Variables

Create a `.env` file in the project root.

Example:

```text
GROQ_API_KEY=your_groq_api_key
```

The repository includes:

```text
.env.example
```

Use `.env.example` only as a reference. Do not place real secrets inside it.

If you want to run deterministic audit checks without LLM usage, disable the LLM if supported by your configuration:

Linux/macOS:

```bash
export LLM_ENABLED=false
```

Windows PowerShell:

```powershell
$env:LLM_ENABLED="false"
```

---

## Project Configuration

Most application behavior is controlled through:

```text
config.yaml
```

Common configurable areas:

- Logging
- Upload limits
- Random seed
- MLflow
- Explainability
- LLM settings
- Modeling defaults
- Metric defaults
- Preprocessing behavior
- Report output paths

---

## Run the Streamlit Dashboard

Start the dashboard:

```bash
uv run streamlit run app/streamlit_app.py --server.port 8501
```

Open:

```text
http://localhost:8501
```

---

## Run FastAPI

Use a second terminal:

```bash
uv run uvicorn app.api:app --reload --host 127.0.0.1 --port 8000
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Health endpoint:

```text
http://127.0.0.1:8000/health
```

---

## Run MLflow UI

Start MLflow:

```bash
uv run mlflow ui
```

Open:

```text
http://localhost:5000
```

MLflow may track:

- Problem type
- Baseline model names
- Parameters
- Metrics
- Best baseline model
- Runtime metadata

---

## Run with Docker

## Pull from Docker Hub

```bash
docker pull shivamrajput130/agentic-ml-audit-copilot:latest
```

Run:

```bash
docker run --rm \
  -p 8501:8501 \
  -p 8000:8000 \
  -e GROQ_API_KEY="your_groq_api_key" \
  shivamrajput130/agentic-ml-audit-copilot:latest
```

Open:

```text
Streamlit: http://localhost:8501
FastAPI:   http://localhost:8000/docs
Health:    http://localhost:8000/health
```

---

## Build Locally

```bash
docker build -t agentic-ml-audit-copilot .
```

Run local image:

```bash
docker run --rm \
  -p 8501:8501 \
  -p 8000:8000 \
  -e GROQ_API_KEY="your_groq_api_key" \
  agentic-ml-audit-copilot
```

---

## Main Audit Workflow

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

## Using the Streamlit Dashboard

## Step 1: Start the App

```bash
uv run streamlit run app/streamlit_app.py --server.port 8501
```

Open:

```text
http://localhost:8501
```

---

## Step 2: Upload Dataset

Upload a CSV file.

Example datasets:

```text
student_mark.csv
housing.csv
churn.csv
```

---

## Step 3: Select Target Column

Choose the column that the model would predict.

Examples:

```text
Grade
price
Churn
target
label
```

---

## Step 4: Run Audit

Click the audit run button in the dashboard.

The system will run deterministic checks and show workflow status.

---

## Step 5: Review Results

The dashboard may show:

- Executive summary
- Dataset profile
- Data quality findings
- Possible leakage risks
- Class imbalance results
- Risk summary
- Workflow decision
- Human Review Gate
- Metric recommendation
- Baseline model comparison
- MLflow status
- Explainability output
- LLM audit report
- Audit Q&A
- JSON and Markdown downloads

---

## Human Review Gate

The Human Review Gate appears when the workflow finds risks that need human judgment.

Examples:

- Possible target leakage
- Identifier-like columns
- Severe class imbalance
- High missing values
- Ambiguous problem type

Reviewer decision options:

- Accept risk and continue
- Accept flag and fix later
- Mark false positive
- Needs data fix
- Reject modeling

Final human decision:

```text
approved
rejected
needs_fix
```

If approved, the workflow continues to:

```text
Metric Recommendation
Preprocessing
Baseline Models
MLflow
SHAP
Final Report
```

If rejected or marked as needing a data fix, the workflow stops so the dataset can be fixed first.

Important:

The system reports possible leakage risks only. It does not automatically confirm leakage.

---

## Using the FastAPI Workflow

## Simple Audit

```bash
curl -X POST "http://127.0.0.1:8000/audit" \
  -F "file=@data/sample/student_mark.csv" \
  -F "target_column=Grade"
```

---

## Lightweight Summary

```bash
curl -X POST "http://127.0.0.1:8000/audit/summary" \
  -F "file=@data/sample/student_mark.csv" \
  -F "target_column=Grade"
```

---

## Human Review API Flow

Recommended flow for explicit human approval:

```text
1. POST /audit/review-gate
2. Inspect human_review.review_items
3. GET /human-review/decision-template
4. Fill reviewer decision JSON
5. POST /audit/after-human-approval
6. Continue to baseline models, MLflow, SHAP, and final report
```

### 1. Run Until Human Review Gate

```bash
curl -X POST "http://127.0.0.1:8000/audit/review-gate" \
  -F "file=@data/sample/student_mark.csv" \
  -F "target_column=Grade"
```

### 2. Get Decision Template

```bash
curl -X GET "http://127.0.0.1:8000/human-review/decision-template"
```

### 3. Continue After Approval

```bash
curl -X POST "http://127.0.0.1:8000/audit/after-human-approval" \
  -F "file=@data/sample/student_mark.csv" \
  -F "target_column=Grade" \
  -F 'human_review_decision_json={
    "final_decision": "approved",
    "reviewer": "reviewer-name",
    "notes": "Reviewed risks and approved baseline modeling.",
    "decisions": [
      {
        "risk_id": "risk_001",
        "decision": "accept_risk_and_continue",
        "comment": "Accepted for baseline run."
      }
    ]
  }'
```

---

## Download Reports

Reports may be available in:

- Markdown
- JSON

Generated files may be stored in:

```text
reports/
```

Dashboard downloads may also be available after audit completion.

---

## Run Tests

Run the full test suite:

```bash
uv run pytest -q
```

Run one test file:

```bash
uv run pytest tests/test_data_quality.py -q
```

Run one specific test:

```bash
uv run pytest tests/test_data_quality.py::test_detects_missing_values -q
```

---

## Linting and Formatting

Run Ruff lint:

```bash
uv run ruff check .
```

Fix safe lint issues:

```bash
uv run ruff check . --fix --unsafe-fixes
```

Format code:

```bash
uv run ruff format .
```

Recommended before commit:

```bash
uv run ruff check . --fix --unsafe-fixes
uv run ruff format .
uv run pytest -q
```

---

## Troubleshooting

## No module named `src`

Install the project in editable mode:

```bash
uv pip install -e .
```

Then run again:

```bash
uv run streamlit run app/streamlit_app.py --server.port 8501
```

Alternative:

```bash
PYTHONPATH=. uv run streamlit run app/streamlit_app.py --server.port 8501
```

Windows PowerShell:

```powershell
$env:PYTHONPATH="."
uv run streamlit run app/streamlit_app.py --server.port 8501
```

---

## Invalid Target Column

Check that:

- The target column exists in the uploaded CSV
- The name matches exactly
- There are no extra spaces in the column name

---

## Missing GROQ API Key

Create a `.env` file:

```text
GROQ_API_KEY=your_groq_api_key
```

Or set it in the terminal.

Linux/macOS:

```bash
export GROQ_API_KEY="your_groq_api_key"
```

Windows PowerShell:

```powershell
$env:GROQ_API_KEY="your_groq_api_key"
```

---

## Docker Port Already in Use

Use different host ports:

```bash
docker run --rm \
  -p 8502:8501 \
  -p 8001:8000 \
  -e GROQ_API_KEY="your_groq_api_key" \
  agentic-ml-audit-copilot
```

Open:

```text
Streamlit: http://localhost:8502
FastAPI:   http://localhost:8001/docs
```

---

## FastAPI Port Already in Use

Use another port:

```bash
uv run uvicorn app.api:app --reload --host 127.0.0.1 --port 8001
```

Open:

```text
http://127.0.0.1:8001/docs
```

---

## MLflow UI Does Not Open

Try specifying the backend store path:

```bash
uv run mlflow ui --backend-store-uri mlruns --port 5000
```

Open:

```text
http://localhost:5000
```

---

## SHAP Takes Too Long

SHAP can be slower on larger datasets.

Possible actions:

- Use a smaller sample
- Disable SHAP if supported by config
- Run baseline audit first
- Avoid very large datasets in local demo mode

---

## Best Practices

- Verify the target column before running the audit.
- Review possible leakage warnings carefully.
- Treat leakage findings as possible risks, not confirmed facts.
- Use representative datasets.
- Inspect class imbalance before trusting accuracy.
- Review baseline metrics before advanced optimization.
- Review SHAP and feature importance outputs.
- Use MLflow to compare baseline runs.
- Never commit API keys or `.env` files.
- Keep screenshots and reports free from private information.

---

## Common Local Workflow

For development:

```bash
uv pip install -r requirements.txt
uv pip install -e .
uv run ruff check . --fix --unsafe-fixes
uv run ruff format .
uv run pytest -q
uv run streamlit run app/streamlit_app.py --server.port 8501
```

In a second terminal:

```bash
uv run uvicorn app.api:app --reload --host 127.0.0.1 --port 8000
```

Optional MLflow:

```bash
uv run mlflow ui
```

---

## Summary

Agentic ML Audit Copilot helps users audit tabular datasets before baseline model training.

A typical user flow is:

1. Upload a CSV dataset.
2. Select the target column.
3. Run deterministic audit checks.
4. Review risks at the Human Review Gate if needed.
5. Approve or stop the workflow.
6. Continue to metrics, baselines, MLflow, SHAP, and final report.
7. Download Markdown or JSON audit output.

The system is designed to support informed human decision-making, not blind automation.
