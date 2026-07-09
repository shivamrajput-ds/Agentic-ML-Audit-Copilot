# Usage Guide

## Overview

This guide explains how to set up, run, and use the Agentic ML Audit Copilot locally.

The project can be executed using either the Streamlit dashboard or the FastAPI REST API.

---

# System Requirements

Recommended

- Python 3.11 or 3.12
- Git
- uv (recommended)
- Docker Desktop (optional)

Supported Operating Systems

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

```bash
uv venv

uv sync
```

Activate

Windows

```bash
.venv\Scripts\activate
```

Linux/macOS

```bash
source .venv/bin/activate
```

---

## Using Python

```bash
python -m venv .venv
```

Activate

Windows

```bash
.venv\Scripts\activate
```

Linux/macOS

```bash
source .venv/bin/activate
```

Install packages

```bash
pip install -r requirements.txt
```

---

# Configure Environment Variables

Create a `.env` file in the project root.

Example

```text
GROQ_API_KEY=YOUR_GROQ_API_KEY
```

---

# Project Configuration

Most application settings are controlled through

```
config.yaml
```

Examples

- Upload limits
- Logging
- MLflow
- Explainability
- Random seed
- Preprocessing
- Metric defaults

---

# Running Streamlit

Start the dashboard

```bash
python -m streamlit run app/streamlit_app.py
```

Open

```
http://localhost:8501
```

---

# Running FastAPI

Start the API

```bash
uvicorn app.api:app --reload
```

Open Swagger

```
http://localhost:8000/docs
```

---

# Running with Docker

Build

```bash
docker build -t agentic-ml-audit-copilot .
```

Run

```bash
docker run \
-p 8501:8501 \
-p 8000:8000 \
-e GROQ_API_KEY=YOUR_GROQ_API_KEY \
agentic-ml-audit-copilot
```

Open

```
Streamlit

http://localhost:8501

FastAPI

http://localhost:8000/docs
```

---

# Using the Application

## Step 1

Open the Streamlit dashboard.

---

## Step 2

Upload a CSV dataset.

Example

```
customer_data.csv
```

---

## Step 3

Select the target column.

Example

```
Churn

OR

Target

OR

Price
```

---

## Step 4

Click

```
Run Audit
```

The workflow automatically executes.

```
Profile Dataset

↓

Problem Detection

↓

Data Quality Audit

↓

Leakage Detection

↓

Metric Recommendation

↓

Class Imbalance

↓

Preprocessing

↓

Baseline Models

↓

MLflow Tracking

↓

Explainability

↓

LLM Report
```

---

# Audit Results

The dashboard displays

- Dataset Summary
- Missing Values
- Duplicate Rows
- Leakage Risks
- Metric Recommendation
- Baseline Results
- SHAP
- Feature Importance
- MLflow
- Human Review
- Audit Report

---

# Download Reports

Reports are generated automatically.

Supported formats

- Markdown
- JSON

Reports are stored inside

```
reports/
```

---

# Human Review Dashboard

The dashboard highlights findings that may require manual validation.

Examples

- Possible leakage
- Identifier columns
- Severe imbalance
- High missing values

The application assists the user by surfacing these findings but does not automatically approve or reject them.

---

# Running Tests

Execute the complete test suite

```bash
python -m pytest -v
```

Expected output

```
96 passed
```

---

# Linting

Run Ruff

```bash
ruff check .
```

Auto-fix

```bash
ruff check . --fix
```

---

# MLflow

Run the MLflow UI

```bash
mlflow ui
```

Open

```
http://localhost:5000
```

The UI displays

- Parameters
- Metrics
- Models
- Artifacts

---

# Sample Workflow

```
Upload CSV

↓

Select Target

↓

Run Audit

↓

Review Results

↓

Download Report
```

---

# Troubleshooting

## ModuleNotFoundError

Solution

```bash
pip install -r requirements.txt
```

or

```bash
uv sync
```

---

## Invalid Target Column

Ensure the selected target column exists in the uploaded dataset.

---

## Missing GROQ API Key

Create

```
.env
```

and add

```text
GROQ_API_KEY=YOUR_GROQ_API_KEY
```

---

## Docker Port Already in Use

Run the container on different ports.

Example

```bash
docker run \
-p 8502:8501 \
-p 8001:8000 \
-e GROQ_API_KEY=YOUR_GROQ_API_KEY \
agentic-ml-audit-copilot
```

---

# Best Practices

- Review leakage warnings before training models.
- Use representative datasets.
- Verify the selected target column.
- Inspect baseline model performance before experimentation.
- Keep dependencies updated.

---

# Summary

The application is designed to provide a simple workflow for auditing tabular machine learning datasets.

Users only need to upload a dataset, select the target column, and run the audit. The platform performs the remaining analysis automatically and generates a structured report with recommendations and supporting insights.