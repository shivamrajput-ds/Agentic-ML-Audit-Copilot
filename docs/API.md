# API Documentation

## Overview

**Agentic ML Audit Copilot** exposes a REST API built with **FastAPI**.

The API allows clients to upload tabular datasets, execute the complete deterministic audit workflow, and retrieve structured audit results.

Interactive API documentation is automatically generated through Swagger UI and ReDoc.

---

# Base URL

Local development:

```text
http://localhost:8000
```

Swagger UI:

```text
http://localhost:8000/docs
```

ReDoc:

```text
http://localhost:8000/redoc
```

Health endpoint:

```text
http://localhost:8000/health
```

---

# Authentication

Current version:

- No authentication required

Planned future support:

- API Keys
- JWT Authentication
- OAuth2
- Role-Based Access Control

---

# Supported Dataset Format

Currently supported:

```text
.csv
```

Future support:

- Excel (.xlsx)
- Parquet
- JSON

Maximum upload size is configurable through:

```text
config.yaml
```

---

# API Endpoints

| Method | Endpoint | Description |
|----------|-----------|--------------------------------|
| GET | `/` | API information |
| GET | `/health` | Service health check |
| POST | `/audit` | Run complete ML audit |
| POST | `/audit/summary` | Return lightweight audit summary |

---

# GET /

Returns basic API metadata.

### Request

```http
GET /
```

### Example Response

```json
{
  "message": "Agentic ML Audit Copilot API is running.",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/health",
  "human_in_the_loop": true
}
```

---

# GET /health

Returns service status.

### Request

```http
GET /health
```

### Example Response

```json
{
  "status": "healthy",
  "service": "agentic-ml-audit-copilot",
  "version": "1.0.0"
}
```

---

# POST /audit

Runs the complete deterministic audit workflow.

### Request

**Content-Type**

```text
multipart/form-data
```

### Parameters

| Field | Type | Required | Description |
|--------|------|:--------:|-------------|
| file | CSV | ✅ | Dataset |
| target_column | String | ✅ | Target column name |

---

### Example

```bash
curl -X POST http://localhost:8000/audit \
  -F "file=@Housing.csv" \
  -F "target_column=price"
```

---

### Successful Response

Example:

```json
{
  "message": "Audit completed successfully.",
  "problem_type": "regression",
  "audit_score": 84.05,
  "best_model": "Linear Regression"
}
```

The complete response may also include:

- Dataset profile
- Problem type
- Data quality
- Leakage analysis
- Metric recommendation
- Class imbalance
- Preprocessing summary
- Baseline model comparison
- Explainability
- MLflow tracking
- Audit report

---

# POST /audit/summary

Returns a lightweight summary without the complete audit payload.

Useful for dashboards and integrations requiring only key metrics.

### Request

```bash
curl -X POST http://localhost:8000/audit/summary \
  -F "file=@Housing.csv" \
  -F "target_column=price"
```

### Example Response

```json
{
  "problem_type": "regression",
  "audit_score": 84.05,
  "best_model": "Linear Regression"
}
```

---

# Response Structure

The full audit response may contain:

```text
Profile

Problem Type Detection

Data Quality Audit

Possible Leakage Detection

Metric Recommendation

Class Imbalance

Preprocessing

Baseline Models

Explainability

MLflow Tracking

Audit Report
```

---

# HTTP Status Codes

| Status | Description |
|---------|-------------|
| 200 | Request completed successfully |
| 400 | Invalid request |
| 413 | Uploaded file exceeds configured size |
| 422 | Validation error |
| 500 | Internal server error |

---

# Common Error Responses

## Missing Target Column

```json
{
  "detail": "Target column is required."
}
```

---

## Unsupported File Type

```json
{
  "detail": "Unsupported file type."
}
```

---

## File Too Large

```json
{
  "detail": "File too large."
}
```

---

## Internal Server Error

```json
{
  "detail": "Unexpected server error during audit execution."
}
```

---

# Swagger UI

FastAPI automatically generates interactive API documentation.

Open:

```text
http://localhost:8000/docs
```

Features:

- Interactive endpoint testing
- Request validation
- Response schemas
- Automatic OpenAPI documentation

Example:

![](../assets/screenshots/swagger_ui.png)

---

# Performance Notes

Current implementation:

- CSV datasets
- Synchronous audit execution
- Threadpool execution for CPU-intensive tasks
- Configurable upload limits
- Deterministic execution order

Performance depends on:

- Dataset size
- Number of features
- SHAP computation
- Baseline model training

---

# Security Notes

Current implementation:

- Input validation
- File type validation
- Configurable upload limits
- Environment-variable-based secrets

Future improvements:

- JWT authentication
- API keys
- OAuth2
- Rate limiting
- Request throttling

---

# API Design Principles

The API follows a deterministic-first approach.

- Python performs all ML computation.
- The LLM is used only for explanations and report generation.
- Possible leakage is never treated as confirmed automatically.
- Human review is required before making downstream modeling decisions.

---

# Future Enhancements

Planned API improvements:

- Authentication
- Rate limiting
- Async background jobs
- Progress tracking
- Batch audit endpoints
- Dataset versioning
- Cloud storage integration
- WebSocket progress updates

---

# Summary

The FastAPI service exposes the complete deterministic audit workflow through a small, well-defined set of REST endpoints.

Automatic OpenAPI documentation, structured responses, and a modular architecture make the API straightforward to integrate into ML engineering workflows, dashboards, and automation pipelines.