# Security Policy

## Overview

Security is an important part of **Agentic ML Audit Copilot**.

This document explains how security issues should be reported and describes the current security posture of the project.

---

## Supported Versions

Security updates are provided for the latest stable release.

| Version | Supported |
| --- | :---: |
| 1.x | Yes |
| < 1.0 | No |

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

---

## Current Security Measures

The project currently includes:

- File type validation for uploaded datasets
- Upload size limits
- Safe filename handling
- Environment-variable-based secret configuration
- `.env.example` for documenting required environment variables
- Configuration-driven application behavior
- Structured exception handling
- JSON-safe API responses
- Dependency pinning through `requirements.txt` and `pyproject.toml`
- Dockerized deployment support
- Non-root Docker container execution
- CI-based testing and linting

---

## Secrets Management

Use environment variables for secrets.

Linux/macOS:

```bash
export GROQ_API_KEY="your_api_key"
```

Windows PowerShell:

```powershell
$env:GROQ_API_KEY="your_api_key"
```

Never:

- Commit API keys
- Commit `.env` files
- Hardcode credentials in source code
- Share secrets in GitHub issues, pull requests, screenshots, or logs

The repository includes:

```text
.env.example
```

Use this file only to document required environment variables. Do not place real secrets inside it.

---

## File Upload Security

The application accepts user-uploaded tabular datasets.

Uploaded files should always be treated as untrusted input.

Users should:

- Upload trusted datasets only
- Avoid uploading sensitive or confidential information
- Validate data sources before using them for model training
- Review audit results before acting on recommendations

The audit workflow is designed to identify possible data risks, but it does not replace a full security review or data governance process.

---

## Dependency Security

Dependencies are managed through:

- `requirements.txt`
- `pyproject.toml`
- `uv.lock`

Recommended practices:

- Update dependencies regularly
- Review dependency changelogs
- Monitor security advisories
- Rebuild Docker images periodically
- Avoid adding unnecessary dependencies

---

## Docker Security

The Docker setup is designed for safer local and demo deployment.

It includes:

- Non-root runtime user
- Environment-variable-based configuration
- Isolated application environment
- Explicit exposed ports

When deploying Docker images:

- Use trusted registries
- Keep Docker updated
- Limit exposed ports
- Avoid running containers with unnecessary privileges
- Use secure secret management
- Rebuild images after dependency updates

---

## Third-Party Services

This project may interact with third-party services such as:

- Groq
- MLflow

Users are responsible for:

- Securing API credentials
- Managing deployed infrastructure
- Monitoring third-party service usage
- Following the security guidance of each provider

---

## Security Design Principles

Agentic ML Audit Copilot follows these principles:

- Deterministic-first execution
- Human-in-the-loop review for risky datasets
- No automatic confirmation of data leakage
- Configuration-driven behavior
- Least-privilege handling of secrets
- Clear separation between ML computation and LLM explanation

Python performs ML computations and deterministic audit checks.

The LLM is used only for:

- Explanations
- Audit Q&A
- Report generation

---

## Production Limitations

This project is currently designed for:

- Learning
- Research
- Portfolio demonstration
- ML engineering practice

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

---

## Contact

If you believe you have found a security issue, please report it responsibly to the project maintainer.

Thank you for helping improve the security of **Agentic ML Audit Copilot**.
