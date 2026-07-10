# Contributing

Thank you for your interest in contributing to **Agentic ML Audit Copilot**.

Contributions are welcome if they improve reliability, documentation, testing, user experience, deployment quality, or ML audit capabilities.

---

## Before You Start

Please review the main documentation:

- `README.md`
- `docs/ARCHITECTURE.md`
- `docs/API.md`
- `docs/USAGE.md`
- `DOCKER.md`
- `CHANGELOG.md`
- `CODE_OF_CONDUCT.md`
- `SECURITY.md`

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
- Improving Docker and deployment support
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
uv venv --python 3.12
```

Activate the environment.

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Windows Git Bash:

```bash
source .venv/Scripts/activate
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

## Environment Variables

Create a local `.env` file only on your machine:

```text
GROQ_API_KEY=your_groq_api_key
```

Do not commit `.env` or real API keys.

For deterministic audit-only usage, LLM features can be disabled if supported by the configuration:

```bash
export LLM_ENABLED=false
```

Windows PowerShell:

```powershell
$env:LLM_ENABLED="false"
```

---

## Run the Application

Run the Streamlit dashboard:

```bash
uv run streamlit run app/streamlit_app.py --server.port 8501
```

Open:

```text
http://localhost:8501
```

Run the FastAPI backend in a second terminal:

```bash
uv run uvicorn app.api:app --reload --host 127.0.0.1 --port 8000
```

Open the API docs:

```text
http://127.0.0.1:8000/docs
```

Optional MLflow UI:

```bash
uv run mlflow ui --backend-store-uri mlruns --host 127.0.0.1 --port 5000
```

Open:

```text
http://127.0.0.1:5000
```

---

## Run with Docker

Build locally:

```bash
docker build -t agentic-ml-audit-copilot .
```

Run both Streamlit and FastAPI:

```bash
docker run --rm \
  --name agentic-audit-test \
  -p 8501:8501 \
  -p 8000:8000 \
  -e GROQ_API_KEY="your_groq_api_key" \
  agentic-ml-audit-copilot:latest
```

Open:

```text
Streamlit Dashboard: http://localhost:8501
FastAPI Docs:       http://localhost:8000/docs
Health Check:       http://localhost:8000/health
```

For more details, read:

```text
DOCKER.md
```

---

## Run Quality Checks

Before opening a pull request, run:

```bash
uv run python -m py_compile app/streamlit_app.py
uv run ruff check . --fix --unsafe-fixes
uv run ruff format .
uv run pytest -q
```

A pull request should not introduce failing tests, formatting issues, or broken imports.

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
- Prefer deterministic checks over LLM-based judgment.
- Keep Streamlit UI state predictable and easy to reset.

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
- Do not present baseline models as production-ready final models.

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

Changes to the human review workflow should be reflected in:

- `README.md`
- `docs/ARCHITECTURE.md`
- `docs/API.md`
- `docs/USAGE.md`
- Streamlit UI copy
- FastAPI response examples, if behavior changes

---

## API Guidelines

When changing FastAPI endpoints:

- Keep request and response schemas clear.
- Keep error messages actionable.
- Avoid returning non-serializable Python objects.
- Document new endpoints in `docs/API.md`.
- Update README endpoint tables if needed.
- Test upload-based endpoints when possible.

---

## Streamlit Guidelines

When changing the dashboard:

- Avoid nested expanders.
- Use stable Streamlit keys for widgets inside loops.
- Keep file upload and session state behavior clear.
- Keep the Human Gate easy to understand.
- Do not hide important warnings.
- Test the full flow: upload CSV → select target → run audit → review gate → continue modeling → report/downloads.

---

## Testing Guidelines

Add or update tests when you change:

- Data quality checks
- Leakage detection
- Metric recommendation
- Class imbalance logic
- Baseline modeling behavior
- Human review routing
- API response behavior
- JSON-safe serialization

Recommended command:

```bash
uv run pytest -q
```

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
- Docker behavior is still correct if runtime dependencies changed

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
- Whether the issue happens locally, in Docker, or on Streamlit Cloud

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

## Security Guidelines

Please read:

```text
SECURITY.md
```

Do not include:

- Real API keys
- `.env` files
- Private datasets
- Access tokens
- Credentials
- Personal private paths

Use placeholders such as:

```text
your_groq_api_key
```

---

## Community Guidelines

Please be respectful in issues, pull requests, discussions, and reviews.

Constructive feedback is encouraged. Personal attacks, harassment, or disrespectful behavior are not acceptable.

See:

```text
CODE_OF_CONDUCT.md
```

---

## Thank You

Thank you for helping improve **Agentic ML Audit Copilot**.
