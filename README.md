<p align="center">
  <img src="assets/branding/banner.png" alt="Agentic ML Audit Copilot" width="100%">
</p>

# 🤖 Agentic ML Audit Copilot

**A deterministic-first Agentic AI application that audits tabular datasets for ML risks — before you waste a single training run.**

Most "AI + ML" demo projects let an LLM run the pipeline. This one doesn't. Every risk detection, metric calculation, and model training step here is plain deterministic Python. The LLM is only allowed to *explain* what the Python already found — never to compute it.

---

## Why this project exists

Before training any model, a competent ML engineer checks:
- Is my target column even usable?
- Is any feature leaking the answer?
- Is my data imbalanced enough to make accuracy meaningless?
- What baseline should I even be beating?

This tool automates that review — like a junior ML reviewer sitting next to you, flagging risks before you commit GPU hours to a model that was doomed from the start.

---

## Core Philosophy

> **Python performs every ML computation. The LLM never does.**

| LLM is allowed to | LLM is NEVER allowed to |
|---|---|
| Explain audit results in plain English | Detect leakage |
| Summarize risks | Calculate metrics |
| Recommend next steps | Train models |
| Answer follow-up questions about the audit | Choose the best model |
| Write the final Markdown report | Preprocess data or engineer features |

If the LLM (Groq) is unavailable or unconfigured, the app **falls back to a fully deterministic Markdown report** — the audit never depends on an API key to produce useful output.

---

## Architecture

<p align="center">
  <img src="assets/diagrams/architecture.png" alt="Architecture diagram" width="85%">
</p>

```
CSV Upload
    │
    ▼
Dataset Profiler ──────────► shape, dtypes, missing values, target summary
    │
    ▼
Problem Type Detection ────► binary / multiclass / regression
    │
    ▼
Data Quality Audit ────────► missing values, duplicates, constant/high-cardinality/ID columns
    │
    ▼
Leakage Detection ─────────► name-based, duplicate-target, correlation, classification-proxy risks
    │
    ▼
Class Imbalance Check ─────► ratio, severity, recommended metrics
    │
    ▼
Metric Recommendation ─────► primary + secondary metrics, reasoning
    │
    ▼
Preprocessing ──────────────► ColumnTransformer (impute + scale + one-hot), stratified split
    │
    ▼
Baseline Models ────────────► Logistic/Linear Regression + Random Forest, evaluated
    │
    ▼
MLflow Tracking ─────────────► params, metrics, best model artifact logged
    │
    ▼
LLM Report (Groq) ──────────► grounded explanation, or deterministic fallback
    │
    ▼
Audit Chat ──────────────────► ask follow-up questions, answered only from audit context
```

The whole pipeline is orchestrated as a **LangGraph state machine** (`src/audit/workflow.py`), where each stage is an independent, testable, single-responsibility node.

<p align="center">
  <img src="assets/diagrams/workflow_graph.png" alt="LangGraph workflow" width="70%">
</p>

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data / ML | Pandas, NumPy, scikit-learn |
| Orchestration | LangGraph |
| Experiment tracking | MLflow |
| LLM (report + chat) | Groq (`llama-3.3-70b-versatile`) |
| API | FastAPI |
| UI | Streamlit |
| Testing | pytest |
| Packaging | uv / pip, Docker |

---

## Features

- 📊 **Dataset profiling** — shape, column types, missing values, target distribution
- 🎯 **Automatic problem-type detection** — binary / multiclass / regression, with edge-case guards (constant targets, all-missing targets)
- 🧩 **Data quality audit** — missing values, duplicate rows, constant/near-constant columns, high-cardinality columns, possible ID columns
- 🚨 **Leakage risk detection** — name-based, duplicate-target, numeric correlation, and classification-proxy checks. Always reports *possible* risk, never confirmed leakage — final judgment stays with the human.
- ⚖️ **Class imbalance analysis** — ratio, severity level, rare-class detection, imbalance-aware metric suggestions
- 📌 **Metric recommendation** — problem-type and imbalance-aware, with plain-language reasoning
- 🤖 **Baseline model benchmarking** — Logistic/Linear Regression + Random Forest, evaluated on accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix (classification) or MAE, RMSE, R², MAPE (regression)
- 🧪 **MLflow experiment tracking** — every run logged with params, metrics, and the best model's signed artifact
- 📄 **LLM-generated audit report** — strictly grounded in the deterministic results, with a fully deterministic fallback if the LLM is unavailable
- 💬 **Audit chat** — ask questions about your completed audit; answers are grounded only in what was actually computed
- ✅ **34 passing pytest tests** covering every deterministic module

---

## Screenshots

<table>
<tr>
<td><img src="assets/screenshots/streamlit_home.png" width="400"><br><sub>Upload & target selection</sub></td>
<td><img src="assets/screenshots/data_quality.png" width="400"><br><sub>Data quality audit</sub></td>
</tr>
<tr>
<td><img src="assets/screenshots/leakage_risks.png" width="400"><br><sub>Possible leakage risks</sub></td>
<td><img src="assets/screenshots/baseline_models.png" width="400"><br><sub>Baseline model benchmark</sub></td>
</tr>
<tr>
<td><img src="assets/screenshots/mlflow_tracking.png" width="400"><br><sub>MLflow experiment tracking</sub></td>
<td><img src="assets/screenshots/audit_report.png" width="400"><br><sub>Generated AI audit report</sub></td>
</tr>
</table>

---

## Installation

### Prerequisites
- Python 3.11+
- A [Groq API key](https://console.groq.com) (optional — the app works without one, using the deterministic fallback report)

### Option A — using `uv` (recommended)

```bash
git clone https://github.com/shivamrajput-ds/Agentic-ML-Audit-Copilot.git
cd Agentic-ML-Audit-Copilot

uv sync
```

### Option B — using plain `pip`

```bash
git clone https://github.com/shivamrajput-ds/Agentic-ML-Audit-Copilot.git
cd Agentic-ML-Audit-Copilot

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

If this is omitted, the app still runs fully — the AI report and audit chat gracefully fall back to deterministic output instead of failing.

---

## Running the app

### Streamlit UI

```bash
uv run streamlit run app/streamlit_app.py
# or, with plain pip:
streamlit run app/streamlit_app.py
```

### FastAPI backend

```bash
uv run uvicorn app.api:app --reload
# or, with plain pip:
uvicorn app.api:app --reload
```

### Docker (runs MLflow UI + FastAPI + Streamlit together)

```bash
docker build -t agentic-ml-audit-copilot .
docker run -p 8501:8501 -p 8000:8000 -p 5000:5000 agentic-ml-audit-copilot
```

| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| FastAPI docs | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | API root / welcome message |
| `GET` | `/health` | Health check |
| `POST` | `/audit` | Upload a CSV + target column, get back the full audit result as JSON |

Example request:

```bash
curl -X POST "http://localhost:8000/audit" \
  -F "file=@data/sample/student_mark.csv" \
  -F "target_column=Grade"
```

---

## Running tests

```bash
uv run pytest tests/ -v
# or, with plain pip:
pytest tests/ -v
```

34 tests cover problem-type detection, data quality, leakage detection, class imbalance, preprocessing, and baseline model training — all against real in-memory datasets, no mocking.

---

## Project Structure

```
├── app/
│   ├── api.py                # FastAPI endpoints
│   └── streamlit_app.py      # Streamlit UI
├── src/
│   ├── audit/
│   │   ├── profiler.py               # Dataset profiling
│   │   ├── problem_detector.py       # Problem type detection
│   │   ├── data_quality.py           # Data quality checks
│   │   ├── leakage.py                # Leakage risk detection
│   │   ├── metric_recommender.py     # Metric recommendation
│   │   ├── class_imbalance.py        # Class imbalance analysis
│   │   ├── preprocessing.py          # Preprocessing pipeline
│   │   ├── baseline_models.py        # Baseline model training
│   │   ├── mlflow_tracker.py         # MLflow experiment tracking
│   │   ├── llm_report.py             # LLM report + audit chat
│   │   └── workflow.py               # LangGraph orchestration
│   └── utils/
│       ├── config.py          # Config loader
│       ├── logger.py          # Rotating file + console logger
│       └── exceptions.py      # Custom exception hierarchy
├── tests/                     # pytest suite (34 tests)
├── data/sample/                # Sample dataset for local testing
├── config.yaml                 # All thresholds — fully config-driven, no magic numbers
├── Dockerfile
├── start.sh
└── requirements.txt / pyproject.toml
```

---

## Sample Run

Using the included sample dataset (`data/sample/student_mark.csv`, target column `Grade`):

- **Problem type detected:** `multiclass_classification`
- **Leakage risks flagged:** 2 (`Total` and `Percentage` — both direct aggregates of the subject scores)
- **Class imbalance:** high severity
- **Best baseline model:** Logistic Regression
- **Report:** saved to `reports/audit_report.md`

---

## Roadmap

- [x] Deterministic audit pipeline (profiling → leakage → imbalance → baselines)
- [x] MLflow tracking with model signatures
- [x] LangGraph orchestration
- [x] LLM report + audit chat with strict anti-hallucination rules
- [x] pytest suite (34 tests)
- [x] Docker support
- [ ] Feature importance (RandomForest)
- [ ] SHAP explainability
- [ ] Cross-validation (config flag already present, not yet wired)
- [ ] PDF report export
- [ ] Dataset drift detection

---

## License

This project is open for learning and portfolio use. No formal license has been applied yet.

---

## Author

Built by [Shivam Rajput](https://github.com/shivamrajput-ds) as a production-inspired, interview-focused ML engineering project.