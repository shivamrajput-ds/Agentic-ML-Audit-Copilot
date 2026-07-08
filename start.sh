#!/bin/bash

set -e

echo "Starting Agentic ML Audit Copilot..."

mkdir -p logs reports uploads artifacts

mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns &

# BUG FIX: this previously pointed to "app.api.main:app", but there is no
# app/api/main.py in this project — the FastAPI app instance lives
# directly in app/api.py as `app = FastAPI(...)`. The old path made
# uvicorn crash on startup with "ModuleNotFoundError: No module named
# 'app.api.main'". Corrected to "app.api:app".
uvicorn app.api:app \
  --host 0.0.0.0 \
  --port 8000 &

streamlit run app/streamlit_app.py \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  --server.headless true