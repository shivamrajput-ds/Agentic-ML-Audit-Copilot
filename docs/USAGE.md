# Usage Guide

## Overview

This guide explains how to install, configure, and use **Agentic ML Audit Copilot**.

The project can be executed using:

- Streamlit Dashboard
- FastAPI REST API
- Docker
- MLflow

The audit workflow is deterministic-first. Python performs all ML computation, while the LLM is used only for explanations and report generation.

---

# System Requirements

Recommended:

- Python 3.11 or 3.12
- Git
- uv (recommended)
- Docker Desktop (optional)

Supported operating systems:

- Windows
- Linux
- macOS

---

# Clone the Repository

```bash
git clone https://github.com/shivamrajput-ds/Agentic-ML-Audit-Copilot.git

cd Agentic-ML-Audit-Copilot
```

---

# Create Virtual Environment

## Using uv (Recommended)

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
```

---

## Using Python

```bash
python -m venv .venv
```

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
```

---

# Configure Environment Variables

Create a `.env` file in the project root.

Example:

```text
GROQ_API_KEY=YOUR_GROQ_API_KEY
```

The project also includes:

```text
.env.example
```

---

# Project Configuration

Most application settings are controlled through:

```text
config.yaml
```

Examples include:

- Logging
- Upload limits
- MLflow
- Explainability
- Random seed
- Model defaults
- Metric defaults
- Preprocessing

---

# Run the Streamlit Dashboard

```bash
uv run streamlit run app/streamlit_app.py
```

Open:

```text
http://localhost:8501
```

---

# Run FastAPI

```bash
uv run uvicorn app.api:app --reload
```

Swagger UI:

```text
http://localhost:8000/docs
```

Health endpoint:

```text
http://localhost:8000/health
```

---

# Run MLflow

```bash
uv run mlflow ui
```

Open:

```text
http://localhost:5000
```

MLflow records:

- Parameters
- Metrics
- Model artifacts
- Best baseline model

---

# Run with Docker

Build:

```bash
docker build -t agentic-ml-audit-copilot .
```

Run:

```bash
docker run --rm \
-p 8000:8000 \
-p 8501:8501 \
-e GROQ_API_KEY="YOUR_GROQ_API_KEY" \
agentic-ml-audit-copilot
```

Or pull directly from Docker Hub:

```bash
docker pull shivamrajput130/agentic-ml-audit-copilot:latest
```

```bash
docker run --rm \
-p 8000:8000 \
-p 8501:8501 \
-e GROQ_API_KEY="YOUR_GROQ_API_KEY" \
shivamrajput130/agentic-ml-audit-copilot:latest
```

---

# Audit Workflow

The application executes the following deterministic workflow:

```
Upload CSV

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

---

# Using the Dashboard

## Step 1

Launch Streamlit.

## Step 2

Upload a CSV dataset.

Example:

```
Housing.csv
```

## Step 3

Choose the target column.

Examples:

```
Price

Target

Churn

Label
```

## Step 4

Click:

```
Run Audit
```

The workflow executes automatically.

---

# Audit Results

The dashboard provides:

- Dataset profile
- Missing value analysis
- Duplicate detection
- Data quality score
- Possible leakage risks
- Class imbalance analysis
- Metric recommendation
- Baseline model comparison
- Feature importance
- SHAP explainability
- MLflow tracking summary
- Human review checklist
- AI-generated audit report

---

# Download Reports

Reports are generated automatically.

Supported formats:

- Markdown
- JSON

Reports are stored in:

```text
reports/
```

---

# Human Review

The system highlights findings requiring manual review.

Examples:

- Possible target leakage
- Identifier columns
- Severe class imbalance
- High missing values

The application **never confirms leakage automatically**.

Human review is always recommended.

---

# Running Tests

Execute the complete test suite:

```bash
uv run pytest
```

Expected output:

```text
96 passed
```

---

# Linting

Check formatting:

```bash
uv run ruff format --check .
```

Run Ruff:

```bash
uv run ruff check .
```

Auto-format:

```bash
uv run ruff format .
```

---

# Troubleshooting

## No module named `src`

Run:

```bash
PYTHONPATH=. uv run streamlit run app/streamlit_app.py
```

---

## Invalid Target Column

Verify that the selected target column exists in the uploaded CSV.

---

## Missing GROQ API Key

Create:

```text
.env
```

Add:

```text
GROQ_API_KEY=YOUR_GROQ_API_KEY
```

---

## Docker Port Already in Use

Use different host ports.

Example:

```bash
docker run --rm \
-p 8001:8000 \
-p 8502:8501 \
-e GROQ_API_KEY="YOUR_GROQ_API_KEY" \
agentic-ml-audit-copilot
```

---

## Health Check

Open:

```text
http://localhost:8000/health
```

---

# Best Practices

- Review all possible leakage warnings before training models.
- Verify the selected target column.
- Use representative datasets.
- Inspect baseline model performance before optimization.
- Review SHAP explanations.
- Keep dependencies updated.
- Never commit API keys.

---

# Design Philosophy

Agentic ML Audit Copilot follows a deterministic-first approach.

- Python performs all ML computation.
- The LLM is used only for explanations.
- Possible leakage is never treated as confirmed.
- Human review is required before making modeling decisions.

---

# Summary

Agentic ML Audit Copilot simplifies dataset auditing before model development.

Users only need to:

1. Upload a dataset.
2. Select the target column.
3. Run the audit.

The platform performs deterministic analysis, generates explainability insights, tracks experiments with MLflow, and produces a structured audit report for informed human decision-making.