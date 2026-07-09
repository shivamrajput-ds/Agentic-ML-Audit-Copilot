# Docker Guide

This guide explains how to build, run, test, and publish the Docker image for **Agentic ML Audit Copilot**.

Agentic ML Audit Copilot runs two services inside one container:

- **FastAPI** backend on port `8000`
- **Streamlit** dashboard on port `8501`

Docker Hub image:

```text
shivamrajput130/agentic-ml-audit-copilot:latest
```

Docker Hub repository:

```text
https://hub.docker.com/r/shivamrajput130/agentic-ml-audit-copilot
```

---

## 1. Requirements

Install Docker Desktop:

```text
https://www.docker.com/products/docker-desktop/
```

Verify Docker:

```bash
docker --version
docker ps
```

---

## 2. Pull Image from Docker Hub

```bash
docker pull shivamrajput130/agentic-ml-audit-copilot:latest
```

---

## 3. Run Docker Image

### Linux / macOS / Git Bash

```bash
docker run --rm \
  --name agentic-audit-copilot \
  -p 8000:8000 \
  -p 8501:8501 \
  -e GROQ_API_KEY="your_groq_api_key" \
  shivamrajput130/agentic-ml-audit-copilot:latest
```

### Windows PowerShell

```powershell
docker run --rm `
  --name agentic-audit-copilot `
  -p 8000:8000 `
  -p 8501:8501 `
  -e GROQ_API_KEY="your_groq_api_key" `
  shivamrajput130/agentic-ml-audit-copilot:latest
```

If you do not want to use LLM report generation, you can run without `GROQ_API_KEY`, but LLM-based report generation and audit Q&A may not work.

---

## 4. Open the Application

Streamlit dashboard:

```text
http://localhost:8501
```

FastAPI docs:

```text
http://localhost:8000/docs
```

Health check:

```text
http://localhost:8000/health
```

Expected health response:

```json
{
  "status": "healthy",
  "service": "agentic-ml-audit-copilot",
  "version": "1.0.0"
}
```

---

## 5. Build Image Locally

From the project root:

```bash
docker build -t agentic-ml-audit-copilot .
```

Run the local image:

```bash
docker run --rm \
  --name agentic-audit-test \
  -p 8000:8000 \
  -p 8501:8501 \
  -e GROQ_API_KEY="your_groq_api_key" \
  agentic-ml-audit-copilot
```

---

## 6. Tag and Push to Docker Hub

Login:

```bash
docker login
```

Tag image:

```bash
docker tag agentic-ml-audit-copilot:latest shivamrajput130/agentic-ml-audit-copilot:latest
```

Push image:

```bash
docker push shivamrajput130/agentic-ml-audit-copilot:latest
```

---

## 7. Environment Variables

| Variable | Required | Default | Description |
|---|---:|---|---|
| `GROQ_API_KEY` | Optional | None | Required for LLM report generation and audit Q&A |
| `API_HOST` | No | `0.0.0.0` | FastAPI host |
| `API_PORT` | No | `8000` | FastAPI port |
| `API_WORKERS` | No | `1` | FastAPI worker count |
| `STREAMLIT_HOST` | No | `0.0.0.0` | Streamlit host |
| `STREAMLIT_PORT` | No | `8501` | Streamlit port |

Example with custom ports:

```bash
docker run --rm \
  --name agentic-audit-copilot \
  -p 8001:8000 \
  -p 8502:8501 \
  -e API_PORT=8000 \
  -e STREAMLIT_PORT=8501 \
  -e GROQ_API_KEY="your_groq_api_key" \
  shivamrajput130/agentic-ml-audit-copilot:latest
```

Open:

```text
http://localhost:8001/docs
http://localhost:8502
```

---

## 8. Project Services

### FastAPI

FastAPI exposes API endpoints for health checks and audit execution.

Useful endpoints:

```text
GET  /health
GET  /docs
POST /audit
POST /audit/summary
```

### Streamlit

Streamlit provides the visual dashboard for:

- CSV upload
- Target column selection
- Data quality audit
- Possible leakage risks
- Metric recommendation
- Class imbalance analysis
- Baseline model results
- Explainability
- MLflow summary
- Audit report
- Downloads

### MLflow

MLflow tracking is used inside the workflow to log:

- Experiment name
- Model runs
- Metrics
- Parameters
- Best model information
- Optional model artifacts

To inspect MLflow locally outside Docker, run:

```bash
uv run mlflow ui --host 127.0.0.1 --port 5000
```

Then open:

```text
http://127.0.0.1:5000
```

---

## 9. Dockerfile Summary

The Docker image uses:

```dockerfile
FROM python:3.12-slim
```

It installs:

- Python dependencies from `requirements.txt`
- System dependencies required by pandas, scikit-learn, MLflow, SHAP, and health checks
- FastAPI and Streamlit runtime services

The container runs:

```bash
./start.sh
```

`start.sh` starts both services:

- FastAPI with `uvicorn`
- Streamlit with `streamlit run`

---

## 10. Runtime Directories

The container creates these directories:

```text
data/uploads/
logs/
reports/
artifacts/mlflow_temp/
.cache/
```

Purpose:

| Directory | Purpose |
|---|---|
| `data/uploads/` | Temporary uploaded CSV files |
| `logs/` | Runtime logs |
| `reports/` | Generated audit reports |
| `artifacts/` | Temporary artifacts and caches |
| `.cache/` | Runtime cache directory |

---

## 11. Security Notes

- Do not hardcode API keys in code.
- Pass secrets using environment variables.
- Do not commit `.env` files.
- Use `.env.example` for documenting required environment variables.
- The application uses a non-root Docker user.
- Uploaded files should be treated as untrusted input.
- LLM output is used only for explanations and reports, not ML computation.

---

## 12. Troubleshooting

### Container name already exists

```bash
docker rm -f agentic-audit-copilot
```

Then run again.

---

### Port already in use

Use different host ports:

```bash
docker run --rm \
  --name agentic-audit-copilot \
  -p 8001:8000 \
  -p 8502:8501 \
  -e GROQ_API_KEY="your_groq_api_key" \
  shivamrajput130/agentic-ml-audit-copilot:latest
```

Open:

```text
http://localhost:8001/docs
http://localhost:8502
```

---

### Docker Hub login fails

Try:

```bash
docker logout
docker login
```

If DNS/network errors occur, restart Docker Desktop and run:

```bash
wsl --shutdown
```

Then reopen Docker Desktop.

---

### Streamlit says `No module named src`

Run the app from the project root or ensure `PYTHONPATH=/app` inside Docker.

For local development:

```bash
PYTHONPATH=. uv run streamlit run app/streamlit_app.py
```

For Windows PowerShell:

```powershell
$env:PYTHONPATH="."
uv run streamlit run app/streamlit_app.py
```

---

### Health check fails

Check container logs:

```bash
docker logs agentic-audit-copilot
```

Verify FastAPI is running:

```text
http://localhost:8000/health
```

---

## 13. Clean Docker Resources

Remove stopped containers:

```bash
docker container prune
```

Remove unused images:

```bash
docker image prune
```

Remove build cache:

```bash
docker builder prune
```

---

## 14. Recommended Demo Command

For README, Docker Hub, and YouTube description:

```bash
docker run --rm \
  -p 8000:8000 \
  -p 8501:8501 \
  -e GROQ_API_KEY="your_groq_api_key" \
  shivamrajput130/agentic-ml-audit-copilot:latest
```

Then open:

```text
Streamlit Dashboard: http://localhost:8501
FastAPI Docs:       http://localhost:8000/docs
Health Check:       http://localhost:8000/health
```

---

## 15. Design Principle

Agentic ML Audit Copilot follows a deterministic-first design:

- Python performs all ML computation.
- The LLM only explains completed audit results.
- Possible leakage is never treated as confirmed automatically.
- Human review is required for final decisions.
