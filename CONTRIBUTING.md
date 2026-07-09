# Contributing

Thank you for your interest in contributing to Agentic ML Audit Copilot.

Contributions that improve the project, fix bugs, enhance documentation, or add useful features are always welcome.

---

# Before You Start

Please take a few minutes to understand the project structure before making changes.

Useful documentation

- README.md
- docs/ARCHITECTURE.md
- docs/API.md
- docs/USAGE.md

---

# Ways to Contribute

You can contribute by

- Fixing bugs
- Improving documentation
- Writing tests
- Improving code quality
- Optimizing performance
- Adding new audit modules
- Improving Streamlit UI
- Improving FastAPI endpoints

---

# Development Setup

Clone the repository

```bash
git clone https://github.com/shivamrajput-ds/Agentic-ML-Audit-Copilot.git

cd Agentic-ML-Audit-Copilot
```

Create virtual environment

Using uv

```bash
uv venv

uv sync
```

Or using Python

```bash
python -m venv .venv
```

Activate

Windows

```bash
.venv\Scripts\activate
```

Linux/macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Create a Branch

Create a new branch before making changes.

```bash
git checkout -b feature/your-feature-name
```

Examples

```
feature/data-drift
```

```
feature/improve-ui
```

```
bugfix/leakage-check
```

---

# Coding Guidelines

Please follow these guidelines

- Write clear and readable code.
- Keep functions focused on one responsibility.
- Avoid unnecessary complexity.
- Prefer descriptive variable names.
- Use type hints where appropriate.
- Add comments only when they improve understanding.

---

# Testing

Run the complete test suite before creating a pull request.

```bash
python -m pytest -v
```

Ensure that all tests pass successfully.

---

# Linting

Run Ruff before submitting changes.

```bash
ruff check .
```

Auto-fix

```bash
ruff check . --fix
```

---

# Pull Request Checklist

Before submitting a Pull Request, verify that

- Code builds successfully
- Tests pass
- Documentation is updated (if required)
- No unnecessary files are included
- New functionality is explained clearly

---

# Reporting Issues

If you find a bug

Please include

- Description
- Steps to reproduce
- Expected behavior
- Actual behavior
- Environment information
  - Operating System
  - Python Version
  - Package Version

---

# Feature Requests

Feature requests are welcome.

Please describe

- The problem
- Proposed solution
- Expected benefit
- Example use case

---

# Documentation

Documentation improvements are always appreciated.

Examples

- Fix spelling mistakes
- Improve explanations
- Add examples
- Improve diagrams

---

# Code Style

The project follows

- PEP 8
- Meaningful function names
- Modular design
- Configuration-driven development

---

# Commit Messages

Use short and descriptive commit messages.

Examples

```text
Add SHAP summary visualization
```

```text
Improve leakage detection
```

```text
Fix preprocessing pipeline
```

Avoid

```text
update
```

```text
changes
```

```text
final final latest
```

---

# Community

Please be respectful when discussing issues or reviewing pull requests.

Constructive feedback is encouraged.

---

# Thank You

Thank you for taking the time to contribute.

Every contribution, whether it is code, documentation, testing, or feedback, helps improve the project.