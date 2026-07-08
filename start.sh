#!/bin/bash

set -e

echo "Starting Agentic ML Audit Copilot..."

mkdir -p logs reports uploads artifacts

mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns &

uvicorn app.api.main:app \
  --host 0.0.0.0 \
  --port 8000 &

streamlit run app/streamlit_app.py \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  --server.headless true
