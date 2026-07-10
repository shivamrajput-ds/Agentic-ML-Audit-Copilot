# syntax=docker/dockerfile:1

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    HOME=/app \
    XDG_CACHE_HOME=/app/.cache \
    MPLCONFIGDIR=/app/.cache/matplotlib \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install \
        --no-cache-dir \
        --timeout 180 \
        --retries 10 \
        --prefer-binary \
        --index-url https://pypi.org/simple \
        -r requirements.txt

COPY . .

RUN sed -i 's/\r$//' start.sh \
    && chmod +x start.sh \
    && mkdir -p \
        data/uploads \
        logs \
        reports \
        artifacts \
        mlruns \
        .cache/matplotlib \
    && groupadd --system appgroup \
    && useradd --system --gid appgroup --home /app --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["./start.sh"]
