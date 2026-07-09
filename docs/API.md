# API Documentation

## Overview

Agentic ML Audit Copilot exposes a REST API using FastAPI.

The API allows applications to upload datasets, execute the audit workflow, and retrieve structured audit results.

Interactive API documentation is available through Swagger UI.

---

# Base URL

Local

```
http://localhost:8000
```

Swagger UI

```
http://localhost:8000/docs
```

ReDoc

```
http://localhost:8000/redoc
```

---

# Authentication

The current version does not require authentication.

Future versions may support

- API Keys
- JWT Authentication
- OAuth2

---

# Supported File Types

Currently supported

```
.csv
```

Maximum upload size is configurable through

```
config.yaml
```

---

# API Endpoints

| Method | Endpoint | Description |
|----------|-----------|---------------------------|
| GET | `/` | Root endpoint |
| GET | `/health` | Health check |
| POST | `/audit` | Run complete audit |
| POST | `/audit/summary` | Lightweight audit summary |

---

# GET /

Returns basic information about the API.

### Request

```http
GET /
```

### Example Response

```json
{
  "message": "Agentic ML Audit Copilot API is running.",
  "docs": "/docs",
  "health": "/health",
  "human_in_the_loop": true
}
```

---

# GET /health

Used for service monitoring.

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

Runs the complete audit workflow.

### Request

Multipart Form Data

| Field | Type | Required |
|--------|------|----------|
| file | CSV File | Yes |
| target_column | String | Yes |

---

### Example

```bash
curl -X POST http://localhost:8000/audit \
-F "file=@dataset.csv" \
-F "target_column=target"
```

---

### Successful Response

The response contains

- Profile
- Problem Detection
- Data Quality
- Leakage Detection
- Metric Recommendation
- Class Imbalance
- Baseline Models
- Explainability
- MLflow Results
- Audit Report

Example

```json
{
  "message":"Audit completed successfully.",
  "problem_type":"binary_classification",
  "audit_score":92
}
```

---

# POST /audit/summary

Returns a lightweight audit summary.

Useful when the client does not require the complete report.

---

### Request

Multipart Form Data

| Field | Type |
|--------|------|
| file | CSV |
| target_column | String |

---

### Example

```bash
curl -X POST http://localhost:8000/audit/summary \
-F "file=@dataset.csv" \
-F "target_column=target"
```

---

### Example Response

```json
{
  "problem_type":"binary_classification",
  "audit_score":92,
  "best_model":"Random Forest Classifier"
}
```

---

# Response Structure

The complete audit response may include

```
Profile

Problem Detection

Data Quality

Leakage Detection

Metric Recommendation

Class Imbalance

Preprocessing

Baseline Results

Explainability

MLflow Results

Audit Report
```

---

# Error Codes

| Status | Meaning |
|----------|----------------|
| 200 | Success |
| 400 | Invalid request |
| 413 | File too large |
| 422 | Validation error |
| 500 | Internal server error |

---

# Common Errors

### Missing Target Column

```json
{
  "detail":"Target column is required."
}
```

---

### Invalid File Type

```json
{
  "detail":"Unsupported file type."
}
```

---

### File Too Large

```json
{
  "detail":"File too large."
}
```

---

### Internal Error

```json
{
  "detail":"Unexpected server error during audit execution."
}
```

---

# Swagger UI

The project includes automatic interactive documentation.

Open

```
http://localhost:8000/docs
```

Features

- Interactive requests
- Response schemas
- Request validation
- API testing

Example

![](../assets/screenshots/swagger_ui.png)

---

# Performance Notes

Current implementation supports

- CSV datasets
- Synchronous audit execution
- Configurable upload limits
- Threadpool execution for CPU-intensive tasks

---

# Future Improvements

Possible future enhancements

- Authentication
- Rate limiting
- Dataset versioning
- Async background jobs
- WebSocket progress updates
- Batch audit endpoints

---

# Summary

The API is intentionally lightweight and focuses on exposing the complete machine learning audit workflow through a small number of well-defined endpoints.

FastAPI automatically generates interactive documentation, making integration and testing straightforward.