#!/usr/bin/env bash
set -euo pipefail

mkdir -p   data/uploads   logs   reports   artifacts   mlruns   .cache/matplotlib

API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8000}"
API_WORKERS="${API_WORKERS:-1}"

STREAMLIT_HOST="${STREAMLIT_HOST:-0.0.0.0}"
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"

echo "Starting Agentic ML Audit Copilot"
echo "FastAPI:   http://${API_HOST}:${API_PORT}"
echo "Streamlit: http://${STREAMLIT_HOST}:${STREAMLIT_PORT}"

python -m uvicorn app.api:app   --host "${API_HOST}"   --port "${API_PORT}"   --workers "${API_WORKERS}" &

API_PID=$!

python -m streamlit run app/streamlit_app.py   --server.address "${STREAMLIT_HOST}"   --server.port "${STREAMLIT_PORT}"   --server.headless true   --browser.gatherUsageStats false &

STREAMLIT_PID=$!

shutdown() {
  echo "Stopping services..."
  kill "${API_PID}" "${STREAMLIT_PID}" 2>/dev/null || true
  wait "${API_PID}" "${STREAMLIT_PID}" 2>/dev/null || true
}

trap shutdown INT TERM

set +e
wait -n "${API_PID}" "${STREAMLIT_PID}"
EXIT_CODE=$?
set -e

shutdown
exit "${EXIT_CODE}"