# Security Policy

## Overview

Security is an important aspect of every software project.

This document describes how security-related issues should be reported and outlines the current security posture of **Agentic ML Audit Copilot**.

---

## Supported Versions

Security updates are provided only for the latest stable release.

| Version | Supported |
|----------|:---------:|
| 1.x | ✅ |
| < 1.0 | ❌ |

---

## Reporting a Security Vulnerability

If you discover a potential security issue, please **do not open a public GitHub issue immediately**.

Instead, report the issue privately to the project maintainer and include:

- Description of the issue
- Steps to reproduce
- Potential impact
- Environment details
- Suggested mitigation (if available)

Responsible disclosure helps prevent unnecessary public exposure before the issue has been reviewed and addressed.

---

## Security Scope

Examples of security-related issues include:

- Sensitive information exposure
- Secrets or API key leakage
- Arbitrary file access
- Path traversal vulnerabilities
- Unsafe file uploads
- Remote code execution
- Dependency vulnerabilities
- Container security issues
- Authentication or authorization bypass (future releases)

---

## Current Security Measures

The project currently includes:

- File type validation
- Upload size limits
- Safe filename handling
- Environment variable support for secrets
- Configuration-driven behavior
- Structured exception handling
- Dependency version pinning
- Dockerized deployment support
- Non-root Docker container execution
- CI-based automated testing and linting

---

## Secrets Management

### Recommended

Store secrets using environment variables:

```bash
export GROQ_API_KEY="your_api_key"
```

Windows PowerShell:

```powershell
$env:GROQ_API_KEY="your_api_key"
```

### Avoid

Never:

- Commit API keys
- Commit `.env` files
- Hardcode credentials in source code
- Share secrets in GitHub issues or pull requests

The repository includes:

```text
.env.example
```

for documenting required environment variables.

---

## Dependency Security

Dependencies are managed through:

- `requirements.txt`
- `pyproject.toml`

Recommended practices:

- Update dependencies regularly
- Review dependency changelogs
- Monitor security advisories
- Rebuild Docker images periodically

---

## File Upload Security

The application accepts user-uploaded datasets.

Users should:

- Upload trusted datasets only
- Review audit results before acting on them
- Avoid uploading sensitive or confidential information
- Validate data sources before model training

Uploaded files should always be considered untrusted input.

---

## Docker Security

The Docker image includes:

- Non-root runtime user
- Environment-variable-based configuration
- Minimal runtime dependencies
- Isolated application environment

When deploying:

- Use trusted Docker registries
- Keep Docker updated
- Limit exposed ports
- Use secure secrets management

---

## Third-Party Services

This project may interact with:

- Groq
- MLflow

Users are responsible for:

- Securing API credentials
- Managing infrastructure security
- Monitoring deployed environments
- Following the security recommendations of third-party providers

---

## Security Design Principles

Agentic ML Audit Copilot follows these principles:

- Deterministic-first execution
- Human-in-the-loop review
- No automatic confirmation of data leakage
- Configuration-driven behavior
- Least-privilege secret handling
- Separation of ML computation and LLM explanations

Python performs all ML computation.

The LLM is used only for:

- Explanations
- Report generation
- Audit Q&A

---

## Limitations

This project is designed primarily for:

- Education
- Research
- Portfolio demonstration
- Learning ML engineering practices

Before deploying in a production environment, consider implementing:

- Authentication
- Authorization
- Audit logging
- Monitoring and alerting
- Rate limiting
- Secret management solutions
- Infrastructure hardening
- Network security controls

---

## Contact

If you believe you have identified a security issue, please report it responsibly through GitHub or directly to the project maintainer.

Thank you for helping improve the security of **Agentic ML Audit Copilot**.