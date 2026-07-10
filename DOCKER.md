# Docker Guide

This guide explains how to build, run, test, and publish the Docker image for **Agentic ML Audit Copilot**.

The Docker container starts both services in one image:

- FastAPI backend on port `8000`
- Streamlit dashboard on port `8501`

Docker Hub repository:

```text
https://hub.docker.com/r/shivamrajput130/agentic-ml-audit-copilot
```

Recommended Docker Hub images:

```text
shivamrajput130/agentic-ml-audit-copilot:latest
shivamrajput130/agentic-ml-audit-copilot:v1.1.0
```

Use `latest` for the newest release and `v1.1.0` for a stable/reproducible demo.

---

## Requirements

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

## Pull Image from Docker Hub

Recommended stable version:

```bash
docker pull shivamrajput130/agentic-ml-audit-copilot:v1.1.0
```

Latest version:

```bash
docker pull shivamrajput130/agentic-ml-audit-copilot:latest
```

---

## Run the Docker Image

### Linux / macOS / Git Bash

Recommended stable run:

```bash
docker run --rm \
  --name agentic-audit-copilot \
  -p 8501:8501 \
  -p 8000:8000 \
  -e GROQ_API_KEY="your_groq_api_key" \
  shivamrajput130/agentic-ml-audit-copilot:v1.1.0
```

Latest run:

```bash
docker run --rm \
  --name agentic-audit-copilot \
  -p 8501:8501 \
  -p 8000:8000 \
  -e GROQ_API_KEY="your_groq_api_key" \
  shivamrajput130/agentic-ml-audit-copilot:latest
```

One-line Git Bash command:

```bash
docker run --rm --name agentic-audit-copilot -p 8501:8501 -p 8000:8000 -e GROQ_API_KEY="your_groq_api_key" shivamrajput130/agentic-ml-audit-copilot:v1.1.0
```

Important for Git Bash: if you use `\`, it must be the final character on the line. Do not paste `\ --name` on the same line.

### Windows PowerShell

```powershell
docker run --rm `
  --name agentic-audit-copilot `
  -p 8501:8501 `
  -p 8000:8000 `
  -e GROQ_API_KEY="your_groq_api_key" `
  shivamrajput130/agentic-ml-audit-copilot:v1.1.0
```

If `GROQ_API_KEY` is not provided, deterministic audit checks can still run if the application configuration allows LLM-disabled mode. LLM report generation and Audit Q&A may not work without the key.

---

## Safer Secret Usage

Avoid pasting real API keys directly into long terminal commands or documentation.

Git Bash:

```bash
export GROQ_API_KEY="your_groq_api_key"

docker run --rm \
  --name agentic-audit-copilot \
  -p 8501:8501 \
  -p 8000:8000 \
  -e GROQ_API_KEY="$GROQ_API_KEY" \
  shivamrajput130/agentic-ml-audit-copilot:v1.1.0
```

Windows PowerShell:

```powershell
$env:GROQ_API_KEY="your_groq_api_key"

docker run --rm `
  --name agentic-audit-copilot `
  -p 8501:8501 `
  -p 8000:8000 `
  -e GROQ_API_KEY="$env:GROQ_API_KEY" `
  shivamrajput130/agentic-ml-audit-copilot:v1.1.0
```

Do not commit `.env` files or hardcode API keys in source code, Dockerfile, README, or Docker Hub description.

---

## Open the Application

Streamlit dashboard:

```text
http://localhost:8501
```

FastAPI Swagger docs:

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
  "version": "1.1.0"
}
```

Command-line health check:

```bash
curl http://localhost:8000/health
```

---

## Build Image Locally

From the project root:

```bash
docker build -t agentic-ml-audit-copilot .
```

If dependency downloads are unstable, build with plain logs:

```bash
docker build --progress=plain -t agentic-ml-audit-copilot .
```

Run the local image:

```bash
docker run --rm \
  --name agentic-audit-test \
  -p 8501:8501 \
  -p 8000:8000 \
  -e GROQ_API_KEY="your_groq_api_key" \
  agentic-ml-audit-copilot:latest
```

Check running container:

```bash
docker ps
```

A healthy container should show both port mappings:

```text
0.0.0.0:8000->8000/tcp
0.0.0.0:8501->8501/tcp
```

---

## Tag and Push to Docker Hub

Login:

```bash
docker login
```

Tag the current local image as both `latest` and `v1.1.0`:

```bash
docker tag agentic-ml-audit-copilot:latest shivamrajput130/agentic-ml-audit-copilot:latest
docker tag agentic-ml-audit-copilot:latest shivamrajput130/agentic-ml-audit-copilot:v1.1.0
```

Verify tags point to the same image ID:

```bash
docker images | grep agentic
```

Expected pattern:

```text
agentic-ml-audit-copilot                          latest    <same-image-id>
shivamrajput130/agentic-ml-audit-copilot          latest    <same-image-id>
shivamrajput130/agentic-ml-audit-copilot          v1.1.0    <same-image-id>
```

Push both tags:

```bash
docker push shivamrajput130/agentic-ml-audit-copilot:latest
docker push shivamrajput130/agentic-ml-audit-copilot:v1.1.0
```

Verify pull:

```bash
docker pull shivamrajput130/agentic-ml-audit-copilot:v1.1.0
```

Run the pushed image:

```bash
docker run --rm \
  --name agentic-audit-test \
  -p 8501:8501 \
  -p 8000:8000 \
  shivamrajput130/agentic-ml-audit-copilot:v1.1.0
```

---

## Environment Variables

| Variable | Required | Default | Description |
| --- | :---: | --- | --- |
| `GROQ_API_KEY` | Optional | None | Required for LLM report generation and Audit Q&A |
| `API_HOST` | No | `0.0.0.0` | FastAPI host inside container |
| `API_PORT` | No | `8000` | FastAPI port inside container |
| `API_WORKERS` | No | `1` | FastAPI worker count |
| `STREAMLIT_HOST` | No | `0.0.0.0` | Streamlit host inside container |
| `STREAMLIT_PORT` | No | `8501` | Streamlit port inside container |
| `LLM_ENABLED` | No | Config-based | Disable LLM features if supported by config |

Example with custom host ports:

```bash
docker run --rm \
  --name agentic-audit-copilot \
  -p 8502:8501 \
  -p 8001:8000 \
  -e GROQ_API_KEY="your_groq_api_key" \
  shivamrajput130/agentic-ml-audit-copilot:v1.1.0
```

Open:

```text
Streamlit: http://localhost:8502
FastAPI:   http://localhost:8001/docs
Health:    http://localhost:8001/health
```

---

## Services Inside the Container

### FastAPI

FastAPI exposes the audit workflow through REST endpoints.

System endpoints:

```text
GET / 
GET /health
GET /metadata
GET /workflow-guide
```

Audit endpoints:

```text
POST /audit
POST /audit/summary
GET  /audit/modes
```

Human Review endpoints:

```text
POST /audit/review-gate
GET  /human-review/decision-template
POST /audit/after-human-approval
```

---

### Streamlit

Streamlit provides the visual dashboard for:

- CSV upload
- Target column selection
- Executive dashboard
- Data quality audit
- Possible leakage risks
- Class imbalance analysis
- Human Review Gate
- Metric recommendation
- Baseline model results
- MLflow status
- SHAP and feature importance
- Audit report
- Audit Q&A
- Markdown and JSON downloads

---

## MLflow

MLflow tracking is used by the workflow to log experiment information.

Tracked information may include:

- Experiment name
- Problem type
- Baseline model runs
- Metrics
- Parameters
- Best baseline model
- Runtime metadata

To inspect MLflow locally outside Docker, run:

```bash
uv run mlflow ui --backend-store-uri mlruns --host 127.0.0.1 --port 5000
```

Open:

```text
http://127.0.0.1:5000
```

---

## Dockerfile Summary

The Docker image uses:

```dockerfile
FROM python:3.12-slim
```

The image installs:

- Python dependencies from `requirements.txt`
- Runtime dependencies for pandas, scikit-learn, MLflow, SHAP, FastAPI, and Streamlit
- Health check utilities if required by the Dockerfile

The container starts through:

```bash
./start.sh
```

`start.sh` starts both services:

- FastAPI with `uvicorn`
- Streamlit with `streamlit run`

---

## Runtime Directories

The container may create or use these directories:

```text
data/uploads/
logs/
reports/
artifacts/
mlruns/
.cache/
```

| Directory | Purpose |
| --- | --- |
| `data/uploads/` | Uploaded CSV files or temporary upload storage |
| `logs/` | Runtime logs |
| `reports/` | Generated audit reports |
| `artifacts/` | Runtime artifacts and temporary outputs |
| `mlruns/` | Local MLflow tracking data |
| `.cache/` | Runtime cache directory |

If you want these outputs to persist after the container stops, mount volumes.

Example:

```bash
docker run --rm \
  --name agentic-audit-copilot \
  -p 8501:8501 \
  -p 8000:8000 \
  -e GROQ_API_KEY="your_groq_api_key" \
  -v "$(pwd)/reports:/app/reports" \
  -v "$(pwd)/mlruns:/app/mlruns" \
  shivamrajput130/agentic-ml-audit-copilot:v1.1.0
```

Windows PowerShell volume example:

```powershell
docker run --rm `
  --name agentic-audit-copilot `
  -p 8501:8501 `
  -p 8000:8000 `
  -e GROQ_API_KEY="your_groq_api_key" `
  -v "${PWD}/reports:/app/reports" `
  -v "${PWD}/mlruns:/app/mlruns" `
  shivamrajput130/agentic-ml-audit-copilot:v1.1.0
```

---

## Human Review API Flow in Docker

After the container starts, use the API docs:

```text
http://localhost:8000/docs
```

Recommended HITL flow:

```text
1. POST /audit/review-gate
2. Inspect human_review.review_items
3. GET /human-review/decision-template
4. Fill reviewer decision JSON
5. POST /audit/after-human-approval
6. Continue to metrics, baselines, MLflow, SHAP, and final report
```

Example:

```bash
curl -X POST "http://localhost:8000/audit/review-gate" \
  -F "file=@data/sample/student_mark.csv" \
  -F "target_column=Grade"
```

---

## Streamlit Cloud Note

Streamlit Community Cloud deploys from GitHub, not Docker Hub.

For Streamlit Cloud:

- Push the latest code to GitHub.
- Set `GROQ_API_KEY` in Streamlit Cloud secrets.
- Do not upload `.env`.
- Make sure `requirements.txt`, `app/streamlit_app.py`, `src/`, `config/`, and project modules are pushed.

Example Streamlit secret:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

---

## Security Notes

- Do not hardcode API keys in the Dockerfile or source code.
- Pass secrets through environment variables.
- Do not commit `.env` files.
- Use `.env.example` only for documenting required variables.
- Uploaded files should be treated as untrusted input.
- The application uses a non-root Docker user if configured in the Dockerfile.
- LLM output is used only for explanations, reports, and Q&A.
- Python performs deterministic ML computation and audit checks.

Production deployments should add:

- Authentication
- Authorization
- Rate limiting
- Secure secret management
- Monitoring and alerting
- Network controls
- Centralized logging

---

## Troubleshooting

### Container Name Already Exists

Remove the old container:

```bash
docker rm -f agentic-audit-copilot
```

If you used the test container name:

```bash
docker rm -f agentic-audit-test
```

Then run again.

---

### Port Already in Use

Use different host ports:

```bash
docker run --rm \
  --name agentic-audit-copilot \
  -p 8502:8501 \
  -p 8001:8000 \
  -e GROQ_API_KEY="your_groq_api_key" \
  shivamrajput130/agentic-ml-audit-copilot:v1.1.0
```

Open:

```text
Streamlit: http://localhost:8502
FastAPI:   http://localhost:8001/docs
```

---

### Docker Hub Login Fails

Try:

```bash
docker logout
docker login
```

If Docker Desktop or WSL has networking issues on Windows:

```bash
wsl --shutdown
```

Then reopen Docker Desktop.

---

### Docker Push Replaces the Existing Image Tag

When you push a tag that already exists, Docker Hub updates that tag to the new image.

Example:

```bash
docker push shivamrajput130/agentic-ml-audit-copilot:latest
```

This updates the `latest` tag to the current pushed image.

For stable demos, always push a versioned tag too:

```bash
docker push shivamrajput130/agentic-ml-audit-copilot:v1.1.0
```

### Streamlit Shows `No module named src`

For Docker, check that the image was built from the project root and that the application path is correct.

For local development, install the project in editable mode:

```bash
uv pip install -e .
```

Or run with `PYTHONPATH`.

Linux/macOS:

```bash
PYTHONPATH=. uv run streamlit run app/streamlit_app.py --server.port 8501
```

Windows PowerShell:

```powershell
$env:PYTHONPATH="."
uv run streamlit run app/streamlit_app.py --server.port 8501
```

---

### Health Check Fails

Check container logs:

```bash
docker logs agentic-audit-copilot
```

Verify the health endpoint:

```text
http://localhost:8000/health
```

If using custom host ports, use the mapped host port.

---

### LLM Report Does Not Generate

Check that `GROQ_API_KEY` is set:

```bash
docker logs agentic-audit-copilot
```

Run with the key:

```bash
docker run --rm \
  -p 8501:8501 \
  -p 8000:8000 \
  -e GROQ_API_KEY="your_groq_api_key" \
  shivamrajput130/agentic-ml-audit-copilot:v1.1.0
```

---

### SHAP Is Slow

SHAP can be slower for larger datasets.

Possible actions:

- Use a smaller demo dataset
- Disable SHAP if supported by configuration
- Run deterministic audit checks first
- Avoid very large files in local demo mode

---

## Clean Docker Resources

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

Remove unused containers, networks, images, and cache:

```bash
docker system prune
```

Use prune commands carefully because they remove unused Docker resources.

---

## Recommended Demo Command

Use this command for README, Docker Hub, and demo videos:

```bash
docker run --rm \
  -p 8501:8501 \
  -p 8000:8000 \
  -e GROQ_API_KEY="your_groq_api_key" \
  shivamrajput130/agentic-ml-audit-copilot:v1.1.0
```

Then open:

```text
Streamlit Dashboard: http://localhost:8501
FastAPI Docs:       http://localhost:8000/docs
Health Check:       http://localhost:8000/health
```

---

## Design Principle

Agentic ML Audit Copilot follows a deterministic-first design:

- Python performs ML computation and audit checks.
- The LLM only explains completed audit results.
- Possible leakage is treated as a risk, not confirmed truth.
- Human review is required for risky modeling decisions.

---

## Summary

The Docker setup provides a simple way to run both the Streamlit dashboard and FastAPI backend in one container.

It is suitable for local demos, portfolio review, and reproducible project presentation. Production deployments should add authentication, monitoring, secure secret handling, and infrastructure hardening.
