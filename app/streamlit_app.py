from pathlib import Path
import uuid

import pandas as pd
import streamlit as st

from src.audit.llm_report import ask_about_audit
from src.audit.workflow import run_audit_workflow
from src.utils.config import get_config_value


UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


st.set_page_config(
    page_title="Agentic ML Audit Copilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom theme (CSS)
# Streamlit's default theme is plain, so we inject a small CSS block to make
# the app look more like a polished product: a subtle color accent, rounded
# metric "cards", and a nicer header banner. Kept lightweight on purpose —
# no external libraries, just CSS variables, in line with the project's
# "simple over clever" coding standard.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --accent: #6C5CE7;
        --accent-soft: #A29BFE;
        --success: #00B894;
        --warning: #FDCB6E;
        --danger: #D63031;
        --bg-card: #F8F9FB;
    }

    .main-header {
        padding: 1.75rem 2rem;
        border-radius: 14px;
        background: linear-gradient(135deg, var(--accent) 0%, var(--accent-soft) 100%);
        color: white;
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        margin: 0;
        font-size: 1.9rem;
    }
    .main-header p {
        margin: 0.35rem 0 0 0;
        opacity: 0.9;
        font-size: 0.95rem;
    }

    /* Metric cards
       NOTE: the card background is light (--bg-card), so text inside it
       must be forced to a dark color explicitly. Without this, Streamlit's
       dark-mode default text color (near-white) becomes invisible on the
       light card — this was the "blank white box" bug. */
    div[data-testid="stMetric"] {
        background-color: var(--bg-card);
        border: 1px solid #E7E7F0;
        border-radius: 12px;
        padding: 0.9rem 0.9rem 0.6rem 0.9rem;
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] div,
    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricValue"] {
        color: #1A1A2E !important;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        font-weight: 600;
    }

    /* Chat answer box */
    .chat-answer {
        background-color: var(--bg-card);
        border-left: 4px solid var(--accent);
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-top: 0.5rem;
        color: #1A1A2E !important;
    }
    .chat-answer * {
        color: #1A1A2E !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header banner (replaces plain st.title/st.write)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>🤖 Agentic ML Audit Copilot</h1>
        <p>Audit your dataset for risks, imbalance, and leakage — before you waste a single training run.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar: quick project context for the user (helps recruiters/interviewers
# understand the deterministic-first philosophy at a glance).
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("ℹ️ How this works")
    st.markdown(
        """
        1. **Upload** a CSV dataset.
        2. **Pick** the target column.
        3. **Run Audit** — all ML logic (leakage, imbalance,
           metrics, baseline models) runs in deterministic Python.
        4. The **LLM only explains** the results — it never
           computes anything itself.
        """
    )
    st.divider()
    st.caption("Built as a Junior ML Reviewer: audits before you train, not instead of training.")


def remove_non_serializable_objects(result: dict) -> dict:
    """Remove sklearn pipeline objects before UI display and chat usage."""
    baseline_results = result.get("baseline_results", {}).copy()
    baseline_results.pop("trained_model_objects", None)
    result["baseline_results"] = baseline_results
    return result


def cleanup_file(file_path: Path) -> None:
    """Delete uploaded file if Streamlit cleanup is enabled."""
    cleanup_enabled = bool(get_config_value("streamlit.cleanup_uploaded_files", False))
    if cleanup_enabled and file_path.exists():
        file_path.unlink(missing_ok=True)


def show_dataset_overview(profile: dict) -> None:
    st.subheader("📊 Dataset Overview")
    shape = profile.get("shape", {})
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", shape.get("rows", "N/A"))
    with col2:
        st.metric("Columns", shape.get("columns", "N/A"))
    with col3:
        st.metric("Duplicate Rows", profile.get("duplicate_rows", 0))


def show_target_distribution(df: pd.DataFrame, target_column: str) -> None:
    st.subheader("🎯 Target Distribution")
    target_counts = df[target_column].value_counts(dropna=False)
    chart_df = pd.DataFrame(
        {"Class": target_counts.index.astype(str), "Count": target_counts.values}
    ).set_index("Class")
    st.bar_chart(chart_df)


def show_missing_values(data_quality: dict) -> None:
    st.subheader("🧩 Missing Values")
    missing_values = data_quality.get("missing_values", {})
    if not missing_values:
        st.success("No missing values found in feature columns.")
        return
    rows = [
        {
            "Column": column,
            "Missing Count": values.get("missing_count", 0),
            "Missing %": values.get("missing_percent", 0),
        }
        for column, values in missing_values.items()
    ]
    missing_df = pd.DataFrame(rows).sort_values("Missing %", ascending=False)
    st.dataframe(missing_df, use_container_width=True)
    st.bar_chart(missing_df.set_index("Column")["Missing %"])


def show_metric_recommendation(metric_result: dict) -> None:
    st.subheader("📌 Metric Recommendation")
    primary_metric = metric_result.get("primary_metric", "N/A")
    reason = metric_result.get("reason", "N/A")
    recommended_metrics = metric_result.get("recommended_metrics", [])
    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric("Primary Metric", primary_metric)
    with col2:
        st.info(reason)
    if recommended_metrics:
        st.table(pd.DataFrame({"Recommended Metrics": recommended_metrics}))


def show_class_imbalance(imbalance_result: dict) -> None:
    st.subheader("⚖️ Class Imbalance")
    if not imbalance_result.get("is_applicable", True):
        st.info(imbalance_result.get("message", "Not applicable."))
        return
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Imbalance Ratio", imbalance_result.get("imbalance_ratio", "N/A"))
    with col2:
        st.metric("Severity", imbalance_result.get("imbalance_severity", "N/A"))
    with col3:
        st.metric("Minority Class", imbalance_result.get("minority_class", "N/A"))
    class_counts = imbalance_result.get("class_counts", {})
    if class_counts:
        class_df = pd.DataFrame(
            {"Class": list(class_counts.keys()), "Count": list(class_counts.values())}
        ).set_index("Class")
        st.bar_chart(class_df)
    warning = imbalance_result.get("warning")
    if warning:
        st.warning(warning)


def show_leakage(leakage_result: dict) -> None:
    st.subheader("🚨 Possible Leakage Risks")
    total_risks = leakage_result.get("total_possible_leakage_risks", 0)
    st.metric("Total Possible Risks", total_risks)
    all_risks = leakage_result.get("all_risks", [])
    if not all_risks:
        st.success("No possible leakage risks detected.")
        return
    risk_df = pd.DataFrame(all_risks)
    if "risk_level" in risk_df.columns:
        st.bar_chart(risk_df["risk_level"].value_counts())
    for risk in all_risks:
        risk_level = risk.get("risk_level", "unknown").upper()
        column = risk.get("column", "N/A")
        reason = risk.get("reason", "N/A")
        with st.expander(f"{risk_level} risk: {column}"):
            st.write(reason)
            st.json(risk)


def show_baseline_results(baseline_result: dict) -> None:
    st.subheader("🤖 Baseline Model Benchmark")
    best_model = baseline_result.get("best_model", {})
    results = baseline_result.get("results", {})
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Best Model", best_model.get("model_name", "N/A"))
    with col2:
        st.metric("Selection Metric", best_model.get("selection_metric", "N/A"))
    with col3:
        st.metric("Best Score", best_model.get("score", "N/A"))
    if not results:
        st.info("No baseline results available.")
        return
    rows = []
    for model_name, metrics in results.items():
        row = {"Model": model_name}
        row.update(metrics)
        rows.append(row)
    result_df = pd.DataFrame(rows)
    st.dataframe(result_df, use_container_width=True)
    metric_columns = [column for column in result_df.columns if column != "Model"]
    if metric_columns:
        selected_metric = st.selectbox("Select metric for model comparison", options=metric_columns)
        chart_df = result_df[["Model", selected_metric]].set_index("Model")
        st.bar_chart(chart_df)


def show_mlflow_results(mlflow_results: dict) -> None:
    """Display MLflow tracking summary.

    BUG FIX: this previously read a key called "model_logged" (singular,
    boolean) which does not exist anywhere in the audit context — the
    workflow/report code always produces "models_logged" (a *list* of
    model names). That mismatch meant this panel silently always showed
    "False", even when models were logged successfully. Now it reads the
    correct key and displays the actual logged model names.
    """
    st.subheader("🧪 MLflow Tracking")
    models_logged = mlflow_results.get("models_logged", [])
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Experiment", mlflow_results.get("experiment_name", "N/A"))
    with col2:
        st.metric("Models Logged", len(models_logged))
    if models_logged:
        st.write(", ".join(models_logged))
    message = mlflow_results.get("message")
    if message:
        st.caption(message)
    logged_uri = mlflow_results.get("logged_model_uri")
    if logged_uri:
        st.code(logged_uri)


def show_ai_report(report_text: str) -> None:
    st.subheader("📄 Generated AI Audit Report")
    if not report_text:
        st.warning("Audit report not available.")
        return
    st.markdown(report_text)
    st.download_button(
        label="Download Audit Report",
        data=report_text,
        file_name="audit_report.md",
        mime="text/markdown",
    )


def answer_audit_question(audit_result: dict, user_question: str) -> str:
    """Route a user question to the LLM audit-chat helper.

    Falls back to a friendly message if the LLM is unavailable (e.g. no
    API key configured) instead of crashing the UI.
    """
    response = ask_about_audit(
        audit_context=audit_result,
        user_question=user_question,
    )

    if response:
        return response

    return "LLM response available nahi hai. Audit report ke basis par manually review karo."


# ---------------------------------------------------------------------------
# Main flow: upload -> preview -> run audit -> display results
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    file_path = UPLOAD_DIR / f"{uuid.uuid4()}_{Path(uploaded_file.name).name}"
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    max_upload_mb = float(get_config_value("streamlit.max_upload_mb", 25))
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    if file_size_mb > max_upload_mb:
        cleanup_file(file_path)
        st.error(f"File too large. Maximum allowed size is {max_upload_mb} MB.")
        st.stop()

    df_preview = pd.read_csv(file_path)
    st.subheader("Dataset Preview")
    st.dataframe(df_preview.head(), use_container_width=True)

    target_column = st.selectbox("Select target column", options=df_preview.columns.tolist())
    show_target_distribution(df_preview, target_column)

    if st.button("Run Audit", type="primary"):
        try:
            with st.spinner("Running agentic ML audit..."):
                result = run_audit_workflow(dataset_path=str(file_path), target_column=target_column)
            result = remove_non_serializable_objects(result)
            st.session_state["audit_result"] = result
            st.session_state["df_preview"] = df_preview
            st.session_state["target_column"] = target_column
            st.success("Audit completed successfully!")
        except Exception as error:
            st.error(f"Audit failed: {error}")
        finally:
            cleanup_file(file_path)


if "audit_result" in st.session_state:
    result = st.session_state["audit_result"]
    df_preview = st.session_state["df_preview"]
    target_column = st.session_state["target_column"]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Problem Type", result.get("problem_type", "N/A"))
    with col2:
        leakage_count = result.get("leakage", {}).get("total_possible_leakage_risks", 0)
        st.metric("Leakage Risks", leakage_count)
    with col3:
        best_model_name = result.get("baseline_results", {}).get("best_model", {}).get("model_name", "N/A")
        st.metric("Best Baseline", best_model_name)

    st.divider()

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
        ["Overview", "Data Quality", "Metrics", "Imbalance", "Leakage", "Models", "MLflow", "AI Report"]
    )

    with tab1:
        show_dataset_overview(result.get("profile", {}))
        show_target_distribution(df_preview, target_column)
    with tab2:
        show_missing_values(result.get("data_quality", {}))
    with tab3:
        show_metric_recommendation(result.get("metric_recommendation", {}))
    with tab4:
        show_class_imbalance(result.get("class_imbalance", {}))
    with tab5:
        show_leakage(result.get("leakage", {}))
    with tab6:
        show_baseline_results(result.get("baseline_results", {}))
    with tab7:
        show_mlflow_results(result.get("mlflow_results", {}))
    with tab8:
        show_ai_report(result.get("audit_report", ""))

    st.divider()
    st.subheader("💬 Ask AI About This Audit")
    user_question = st.text_input("Ask a question", placeholder="Example: Why is F1 Score recommended?")
    if st.button("Ask AI"):
        if not user_question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Generating answer..."):
                answer = answer_audit_question(
                    audit_result=result,
                    user_question=user_question.strip(),
                )
            st.markdown(f'<div class="chat-answer">{answer}</div>', unsafe_allow_html=True)