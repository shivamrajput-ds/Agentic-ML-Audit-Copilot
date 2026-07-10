# Contributing

Thank you for your interest in contributing to **Agentic ML Audit Copilot**.

Contributions are welcome if they improve reliability, documentation, testing, user experience, or audit capabilities.

---

## Before You Start

Please review the main documentation:

- `README.md`
- `docs/ARCHITECTURE.md`
- `docs/API.md`
- `docs/USAGE.md`
- `CHANGELOG.md`
- `CODE_OF_CONDUCT.md`

---

## Ways to Contribute

You can contribute by:

- Fixing bugs
- Improving documentation
- Adding tests
- Improving code quality
- Optimizing performance
- Improving the Streamlit dashboard
- Improving FastAPI endpoints
- Adding new audit modules
- Improving MLflow tracking
- Improving explainability support
- Improving human-in-the-loop review workflows

---

## Development Setup

Clone the repository:

```bash
git clone https://github.com/shivamrajput-ds/Agentic-ML-Audit-Copilot.git
cd Agentic-ML-Audit-Copilot
```

Create a virtual environment:

```bash
uv venv
```

Activate the environment.

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
uv pip install -e .
```

---

## Run the Application

Run the Streamlit dashboard:

```bash
uv run streamlit run app/streamlit_app.py --server.port 8501
```

Run the FastAPI backend:

```bash
uv run uvicorn app.api:app --reload --host 127.0.0.1 --port 8000
```

Open the API docs:

```text
http://127.0.0.1:8000/docs
```

---

## Run Quality Checks

Before opening a pull request, run:

```bash
uv run ruff check . --fix --unsafe-fixes
uv run ruff format .
uv run pytest -q
```

A pull request should not introduce failing tests or formatting issues.

---

## Coding Guidelines

Please follow these guidelines:

- Keep code simple, readable, and practical.
- Keep functions focused on one responsibility.
- Use descriptive variable and function names.
- Use type hints where appropriate.
- Avoid unnecessary over-engineering.
- Keep preprocessing inside scikit-learn pipelines.
- Do not hardcode secrets, API keys, tokens, or private paths.
- Do not commit large generated files unless they are required assets.
- Keep API responses JSON-safe.
- Keep user-facing errors clear and actionable.

---

## ML Audit Rules

This project follows a deterministic-first audit philosophy.

Please keep these rules in mind:

- Python performs all ML computations and risk checks.
- The LLM is used only for explanations, Q&A, and report writing.
- Do not claim confirmed leakage automatically.
- Report leakage as a possible risk requiring human review.
- Risky datasets should pass through the Human Review Gate.
- Baseline modeling should continue only after the workflow allows it.
- Avoid hiding important warnings from the user.

---

## Human-in-the-Loop Workflow Rules

When changing the human review workflow, make sure the user can clearly understand:

- Why the workflow is paused
- Which risk items need review
- What each reviewer decision means
- Whether modeling is allowed to continue
- Whether MLflow, SHAP, and reports are paused or completed

The main HITL decisions are:

- Accept risk and continue
- Accept flag and fix later
- Mark false positive
- Needs data fix
- Reject modeling

---

## Branch Naming

Create a new branch before making changes:

```bash
git checkout -b feature/your-feature-name
```

Examples:

```text
feature/data-drift
feature/improve-human-gate
feature/api-docs
bugfix/leakage-check
bugfix/streamlit-dashboard
docs/update-readme
tests/add-api-tests
```

---

## Commit Messages

Use short and descriptive commit messages.

Good examples:

```text
Add human review decision template endpoint
Improve leakage risk summary
Fix preprocessing pipeline validation
Update Streamlit dashboard assets
Add tests for metric recommendation
```

Avoid vague messages:

```text
update
changes
final
final final
latest
```

---

## Pull Request Checklist

Before submitting a pull request, verify that:

- Tests pass
- Ruff linting passes
- Ruff formatting passes
- Documentation is updated if needed
- New behavior is explained clearly
- No secrets are included
- No unnecessary files are committed
- API changes are reflected in docs
- UI changes include screenshots if helpful
- Human review behavior remains clear and safe

---

## Reporting Bugs

When reporting a bug, include:

- Short description
- Steps to reproduce
- Expected behavior
- Actual behavior
- Operating system
- Python version
- Relevant package versions
- Full error logs or traceback
- Screenshots if the issue is UI-related

---

## Feature Requests

Feature requests are welcome.

Please describe:

- The problem
- The proposed solution
- Expected benefit
- Example use case
- Any risks or trade-offs

Good feature areas include:

- Data drift detection
- Feature drift detection
- Fairness and bias analysis
- Report export formats
- API usability
- Dashboard clarity
- Test coverage
- Deployment support

---

## Documentation Updates

Documentation should be:

- Clear
- Simple
- Accurate
- Easy to follow
- Updated when behavior changes

Avoid exaggerated claims. Keep documentation honest and useful.

---

## Community Guidelines

Please be respectful in issues, pull requests, discussions, and reviews.

Constructive feedback is encouraged. Personal attacks, harassment, or disrespectful behavior are not acceptable.

See `CODE_OF_CONDUCT.md` for details.

---

## Thank You

Thank you for helping improve **Agentic ML Audit Copilot**.