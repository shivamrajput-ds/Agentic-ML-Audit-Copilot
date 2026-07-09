# Security Policy

## Overview

Security is important for every software project, regardless of its size.

This document explains how security-related issues should be reported for Agentic ML Audit Copilot.

---

# Supported Versions

The latest version of the project receives security updates.

Older versions may not receive fixes.

| Version | Supported |
|----------|:---------:|
| 1.x | ✅ |
| < 1.0 | ❌ |

---

# Reporting a Security Issue

If you discover a potential security issue, please do not open a public GitHub issue immediately.

Instead, describe the issue privately and include the following information:

- Description of the issue
- Steps to reproduce
- Potential impact
- Suggested mitigation (if available)

This helps prevent unnecessary public exposure until the issue has been reviewed.

---

# Scope

Examples of security-related issues include

- Sensitive information exposure
- Authentication bypass (future versions)
- Arbitrary file access
- Unsafe file uploads
- Remote code execution
- Dependency vulnerabilities

---

# Current Security Measures

The project currently includes the following safeguards.

- File type validation
- Upload size limits
- Configurable application settings
- Custom exception handling
- Safe filename handling
- Environment variable support for API keys
- Docker support
- Dependency pinning

---

# Best Practices

When using this project

- Never commit `.env` files.
- Store API keys in environment variables.
- Keep dependencies updated.
- Validate uploaded datasets before use.
- Review audit findings before model training.

---

# Dependencies

Dependencies are managed through

- `requirements.txt`
- `pyproject.toml`

Keep packages updated regularly to receive security fixes.

---

# Third-Party Services

The project may interact with third-party services such as

- Groq
- MLflow

Users are responsible for securing their own API keys and deployment environments.

---

# Disclaimer

This project is intended for educational, research, and portfolio purposes.

Before deploying in a production environment, additional security measures such as authentication, authorization, monitoring, and infrastructure hardening should be implemented.

---

# Contact

If you identify a security issue, please report it responsibly through GitHub or the project maintainer.

Thank you for helping improve the security of the project.