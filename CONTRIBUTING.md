# Contributing

Thank you for your interest in contributing to **Agentic ML Audit Copilot**.

Contributions that improve the project, fix bugs, enhance documentation, add tests, or introduce useful audit capabilities are welcome.

---

## Before You Start

Please review the main project documentation:

- `README.md`
- `docs/ARCHITECTURE.md`
- `docs/API.md`
- `docs/USAGE.md`

---

## Ways to Contribute

You can contribute by:

- Fixing bugs
- Improving documentation
- Adding tests
- Improving code quality
- Optimizing performance
- Adding new audit modules
- Improving the Streamlit dashboard
- Improving FastAPI endpoints
- Enhancing MLflow or explainability support

---

## Development Setup

Clone the repository:

```bash
git clone https://github.com/shivamrajput-ds/Agentic-ML-Audit-Copilot.git
cd Agentic-ML-Audit-Copilot
```

Create and activate a virtual environment:

```bash
uv venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
uv pip install -r requirements.txt
```

---

## Run Quality Checks

Before creating a pull request, run:

```bash
uv run ruff format .
uv run ruff check .
uv run pytest
```

---

## Coding Guidelines

Please follow these guidelines:

- Keep code simple, readable, and practical.
- Keep functions focused on one responsibility.
- Use descriptive variable and function names.
- Use type hints where appropriate.
- Avoid unnecessary over-engineering.
- Keep preprocessing inside sklearn pipelines.
- Do not hardcode secrets or API keys.
- Do not claim confirmed leakage automatically.
- Report leakage only as a possible risk requiring human review.

---

## Branch Naming

Create a new branch before making changes:

```bash
git checkout -b feature/your-feature-name
```

Examples:

```text
feature/data-drift
feature/improve-ui
bugfix/leakage-check
docs/update-readme
```

---

## Pull Request Checklist

Before submitting a pull request, verify that:

- Tests pass
- Ruff linting passes
- Ruff formatting passes
- Documentation is updated if required
- No unnecessary files are committed
- No secrets are included
- New functionality is explained clearly

---

## Reporting Issues

When reporting a bug, include:

- Description
- Steps to reproduce
- Expected behavior
- Actual behavior
- Operating system
- Python version
- Relevant package versions
- Screenshots or logs if helpful

---

## Feature Requests

Feature requests are welcome.

Please describe:

- The problem
- Proposed solution
- Expected benefit
- Example use case

---

## Commit Messages

Use short and descriptive commit messages.

Good examples:

```text
Add SHAP summary visualization
Improve leakage detection
Fix preprocessing pipeline
Update Docker setup
```

Avoid:

```text
update
changes
final final latest
```

---

## Community Guidelines

Please be respectful when discussing issues or reviewing pull requests.

Constructive feedback is encouraged.

---

## Thank You

Thank you for helping improve **Agentic ML Audit Copilot**.