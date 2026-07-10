# Security Policy

## Overview

Security is an important part of **Agentic ML Audit Copilot**.

This document explains how security issues should be reported and describes the current security posture of the project.

Agentic ML Audit Copilot is a deterministic-first, human-in-the-loop ML audit application for tabular datasets. It is suitable for local demos, portfolio review, and ML engineering practice. Production use requires additional security controls.

---

## Supported Versions

Security updates are provided for the latest stable release.

| Version | Supported |
| --- | :---: |
| `1.x` | Yes |
| `< 1.0` | No |

Current stable release:

```text
v1.1.0
```

---

## Reporting a Security Vulnerability

If you find a potential security issue, please do not open a public GitHub issue immediately.

Report it privately to the project maintainer and include:

- A clear description of the issue
- Steps to reproduce
- Potential impact
- Environment details
- Logs, screenshots, or proof of concept if helpful
- Suggested mitigation, if available

Responsible disclosure helps prevent unnecessary public exposure before the issue is reviewed and fixed.

---

## Security Scope

Examples of security-related issues include:

- Sensitive information exposure
- Secrets or API key leakage
- Unsafe file uploads
- Arbitrary file access
- Path traversal vulnerabilities
- Remote code execution
- Dependency vulnerabilities
- Container security issues
- Authentication or authorization bypass in future releases
- Unsafe handling of uploaded files
- Unsafe logging of user data or secrets

---

## Current Security Measures

The project currently includes:

- File type validation for uploaded datasets
- Upload size limits
- Safe filename handling
- Project-local upload path resolution
- Environment-variable-based secret configuration
- `.env.example` for documenting required environment variables
- `.gitignore` and `.dockerignore` rules for local secrets and runtime outputs
- Configuration-driven application behavior
- Structured exception handling
- JSON-safe API responses and downloads
- Basic CSV formula-injection protection for CSV downloads
- Dependency pinning through `requirements.txt` and `pyproject.toml`
- Dockerized deployment support
- Non-root Docker container execution, if configured in the Dockerfile
- CI-based testing and linting

---

## Secrets Management

Use environment variables for secrets.

Linux/macOS/Git Bash:

```bash
export GROQ_API_KEY="your_api_key"
```

Windows PowerShell:

```powershell
$env:GROQ_API_KEY="your_api_key"
```

Docker:

```bash
docker run --rm \
  -p 8501:8501 \
  -p 8000:8000 \
  -e GROQ_API_KEY="your_api_key" \
  shivamrajput130/agentic-ml-audit-copilot:v1.1.0
```

Never:

- Commit API keys
- Commit `.env` files
- Hardcode credentials in source code
- Put secrets inside Dockerfiles
- Put secrets inside README, docs, screenshots, or demo videos
- Share secrets in GitHub issues, pull requests, screenshots, logs, or terminal output

The repository includes:

```text
.env.example
```

Use this file only to document required environment variables. Do not place real secrets inside it.

If a secret is accidentally exposed, rotate or revoke it immediately.

---

## File Upload Security

The application accepts user-uploaded tabular datasets.

Uploaded files should always be treated as untrusted input.

Current upload-related controls may include:

- CSV-only upload mode
- File extension validation
- File size limit
- Safe filename extraction
- Project-local upload directory
- Optional cleanup of uploaded files
- Error handling for malformed or unreadable CSV files

Users should:

- Upload trusted datasets only
- Avoid uploading sensitive or confidential information
- Validate data sources before using them for model training
- Review audit results before acting on recommendations
- Avoid uploading private production data to public demo deployments

The audit workflow is designed to identify possible data and ML risks, but it does not replace a full security review or data governance process.

---

## API Security

The FastAPI backend exposes audit endpoints for uploaded files and review workflows.

Current project-level API endpoints are intended mainly for local demos and portfolio evaluation.

Before exposing the API publicly, add:

- Authentication
- Authorization
- Rate limiting
- Request size limits at infrastructure level
- Secure CORS configuration
- Abuse monitoring
- Centralized logging
- Secret management through platform-level secrets

Do not expose a public API endpoint that accepts arbitrary uploaded files without additional security controls.

---

## Streamlit Cloud Security

When deploying on Streamlit Community Cloud:

- Store `GROQ_API_KEY` in Streamlit secrets.
- Do not commit `.env`.
- Do not print secrets in the app.
- Do not upload sensitive datasets to public demo apps.
- Keep demo datasets small and non-confidential.
- Reboot or redeploy the app after changing secrets or dependency files.

Example Streamlit secret:

```toml
GROQ_API_KEY = "your_api_key"
```

---

## Dependency Security

Dependencies are managed through:

- `requirements.txt`
- `pyproject.toml`
- `uv.lock`, if present

Recommended practices:

- Update dependencies regularly
- Review dependency changelogs
- Monitor security advisories
- Rebuild Docker images periodically
- Avoid adding unnecessary dependencies
- Prefer pinned versions for reproducible demos
- Re-run tests after dependency updates

Recommended checks before release:

```bash
uv run ruff check .
uv run pytest -q
docker build -t agentic-ml-audit-copilot .
```

---

## Docker Security

The Docker setup is designed for safer local and demo deployment.

It includes or should include:

- Non-root runtime user
- Environment-variable-based configuration
- Isolated application environment
- Explicit exposed ports
- `.dockerignore` rules for secrets, virtual environments, caches, logs, reports, and local runtime data

When deploying Docker images:

- Use trusted registries
- Keep Docker updated
- Limit exposed ports
- Avoid running containers with unnecessary privileges
- Use secure secret management
- Rebuild images after dependency updates
- Avoid mounting sensitive host directories
- Avoid shipping `.env`, local datasets, logs, or MLflow runs inside the image

The published image runs:

```text
FastAPI:   8000
Streamlit: 8501
```

---

## Third-Party Services

This project may interact with third-party services such as:

- Groq
- MLflow
- Streamlit Community Cloud
- Docker Hub

Users are responsible for:

- Securing API credentials
- Managing deployed infrastructure
- Monitoring third-party service usage
- Following the security guidance of each provider
- Understanding where uploaded data and logs may be processed or stored

---

## Security Design Principles

Agentic ML Audit Copilot follows these principles:

- Deterministic-first execution
- Human-in-the-loop review for risky datasets
- No automatic confirmation of data leakage
- Configuration-driven behavior
- Least-privilege handling of secrets
- Clear separation between ML computation and LLM explanation
- Transparent warnings instead of silent risk suppression

Python performs ML computations and deterministic audit checks.

The LLM is used only for:

- Explanations
- Audit Q&A
- Report generation

The LLM should not be treated as the final authority for security, data readiness, leakage confirmation, or production approval.

---

## Data Privacy Notes

The project is intended for demonstration and ML audit experimentation.

Avoid uploading:

- Personally identifiable information
- Financial records
- Health records
- Private customer data
- Confidential business data
- Production datasets without permission

For real organizational use, add formal data governance, access control, audit logging, and retention policies.

---

## Production Limitations

This project is currently designed for:

- Learning
- Research
- Portfolio demonstration
- ML engineering practice
- Local demos

Before using it in a production environment, consider adding:

- Authentication
- Authorization
- Role-based access control
- Audit logging
- Rate limiting
- Monitoring and alerting
- Secure secret management
- Network security controls
- Input sandboxing
- Centralized logging
- Cloud infrastructure hardening
- Dataset retention policies
- Vulnerability scanning
- Container image scanning
- Dependency security scanning

---

## Known Non-Goals

This project does not currently provide:

- Enterprise-grade identity management
- Multi-user access control
- Production model serving security
- Full data governance certification
- Fairness certification
- Formal compliance approval
- Secure multi-tenant isolation

---

## Contact

If you believe you have found a security issue, please report it responsibly to the project maintainer.

Thank you for helping improve the security of **Agentic ML Audit Copilot**.
