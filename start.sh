#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/uploads logs reports artifacts/mlflow_temp .cache

API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8000}"
STREAMLIT_HOST="${STREAMLIT_HOST:-0.0.0.0}"
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"

echo "Starting Agentic ML Audit Copilot"
echo "FastAPI:    http://${API_HOST}:${API_PORT}"
echo "Streamlit:  http://${STREAMLIT_HOST}:${STREAMLIT_PORT}"

uvicorn app.api:app \
  --host "${API_HOST}" \
  --port "${API_PORT}" \
  --workers "${API_WORKERS:-1}" &

API_PID=$!

streamlit run app/streamlit_app.py \
  --server.address "${STREAMLIT_HOST}" \
  --server.port "${STREAMLIT_PORT}" \
  --server.headless true &

STREAMLIT_PID=$!

shutdown() {
  echo "Stopping services..."
  kill "${API_PID}" "${STREAMLIT_PID}" 2>/dev/null || true
  wait "${API_PID}" "${STREAMLIT_PID}" 2>/dev/null || true
}

trap shutdown INT TERM

wait -n "${API_PID}" "${STREAMLIT_PID}"
shutdown
