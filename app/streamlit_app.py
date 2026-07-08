from __future__ import annotations

from pathlib import Path
from typing import Any
import json
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
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --accent: #6C5CE7;
        --accent-soft: #A29BFE;
        --bg-card: #F8F9FB;
        --border-card: #E7E7F0;
        --text-dark: #1A1A2E;
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
        opacity: 0.92;
        font-size: 0.95rem;
    }

    div[data-testid="stMetric"] {
        background-color: var(--bg-card);
        border: 1px solid var(--border-card);
        border-radius: 12px;
        padding: 0.9rem 0.9rem 0.6rem 0.9rem;
    }

    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] div,
    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricValue"] {
        color: var(--text-dark) !important;
    }

    button[data-baseweb="tab"] {
        font-weight: 600;
    }

    .chat-answer {
        background-color: var(--bg-card);
        border-left: 4px solid var(--accent);
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-top: 0.5rem;
        color: var(--text-dark) !important;
    }

    .chat-answer * {
        color: var(--text-dark) !important;
    }

    .small-muted {
        color: #6B7280;
        font-size: 0.88rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="main-header">
        <h1>🤖 Agentic ML Audit Copilot</h1>
        <p>Audit your dataset for data quality, leakage, imbalance, metrics, and baseline model readiness before training.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def as_bool(value: Any) -> bool:
    """
    Convert config values safely into boolean.
    """
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.lower().strip() in {"true", "1", "yes", "y"}

    return bool(value)


def cleanup_file(file_path: Path) -> None:
    """
    Delete uploaded file if cleanup is enabled in config.yaml.
    """
    cleanup_enabled = as_bool(
        get_config_value("streamlit.cleanup_uploaded_files", False)
    )

    if cleanup_enabled and file_path.exists():
        file_path.unlink(missing_ok=True)


def remove_non_serializable_objects(result: dict[str, Any]) -> dict[str, Any]:
    """
    Remove sklearn pipeline objects before storing result in Streamlit state.
    """
    clean_result = result.copy()
    baseline_results = clean_result.get("baseline_results", {}).copy()
    baseline_results.pop("trained_model_objects", None)
    clean_result["baseline_results"] = baseline_results
    clean_result.pop("df", None)
    return clean_result


def reset_audit_state() -> None:
    """
    Clear previous audit state from Streamlit session.
    """
    keys_to_clear = [
        "audit_result",
        "df_preview",
        "target_column",
        "uploaded_filename",
        "chat_history",
    ]

    for key in keys_to_clear:
        st.session_state.pop(key, None)


def safe_dataframe_preview(file_path: Path) -> pd.DataFrame:
    """
    Read uploaded CSV for preview with a clear UI error on failure.
    """
    try:
        return pd.read_csv(file_path)

    except Exception as error:
        st.error(f"Failed to read CSV file: {error}")
        st.stop()


def to_json_download(data: dict[str, Any]) -> str:
    """
    Convert audit result to JSON string for download.
    """
    return json.dumps(data, indent=2, default=str)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("ℹ️ How this works")
    st.markdown(
        """
        1. **Upload** a CSV dataset.
        2. **Select** the target column.
        3. **Run Audit**.
        4. Python performs all deterministic ML checks.
        5. LLM only explains the completed audit.
        """
    )

    st.divider()

    st.markdown("### Deterministic-first rule")
    st.caption(
        "Leakage detection, metrics, imbalance checks, preprocessing, and model training are all done by Python. The LLM only writes explanations and answers audit questions."
    )

    st.divider()

    if st.button("Clear Current Audit"):
        reset_audit_state()
        st.rerun()


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
def show_dataset_overview(profile: dict[str, Any]) -> None:
    """
    Display dataset-level summary cards.
    """
    st.subheader("📊 Dataset Overview")

    shape = profile.get("shape", {})
    column_types = profile.get("column_types", {})

    numeric_columns = profile.get("numeric_columns") or column_types.get(
        "numeric_columns", []
    )
    categorical_columns = profile.get("categorical_columns") or column_types.get(
        "categorical_columns", []
    )
    datetime_columns = profile.get("datetime_columns") or column_types.get(
        "datetime_columns", []
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Rows", shape.get("rows", "N/A"))

    with col2:
        st.metric("Columns", shape.get("columns", "N/A"))

    with col3:
        st.metric("Duplicate Rows", profile.get("duplicate_rows", 0))

    with col4:
        st.metric("Datetime Columns", len(datetime_columns))

    type_df = pd.DataFrame(
        {
            "Type": ["Numeric", "Categorical", "Datetime"],
            "Count": [
                len(numeric_columns),
                len(categorical_columns),
                len(datetime_columns),
            ],
        }
    ).set_index("Type")

    st.bar_chart(type_df)


def show_target_distribution(df: pd.DataFrame, target_column: str) -> None:
    """
    Display target distribution chart.
    """
    st.subheader("🎯 Target Distribution")

    target_counts = df[target_column].value_counts(dropna=False)

    chart_df = pd.DataFrame(
        {
            "Class": target_counts.index.astype(str),
            "Count": target_counts.values,
        }
    ).set_index("Class")

    st.bar_chart(chart_df)

    with st.expander("View target counts"):
        st.dataframe(chart_df, use_container_width=True)


def show_data_quality(data_quality: dict[str, Any]) -> None:
    """
    Display data quality findings.
    """
    st.subheader("🧩 Data Quality Audit")

    target_quality = data_quality.get("target_quality", {})
    warnings = data_quality.get("warnings", [])

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Duplicate Rows", data_quality.get("duplicate_rows", 0))

    with col2:
        st.metric(
            "Target Missing %",
            target_quality.get("missing_percent", "N/A"),
        )

    with col3:
        st.metric("Warnings", len(warnings))

    if warnings:
        for warning in warnings:
            if warning == "No major basic data quality issues detected.":
                st.success(warning)
            else:
                st.warning(warning)

    missing_values = data_quality.get("missing_values", {})

    st.markdown("#### Missing Values")

    if not missing_values:
        st.success("No missing values found in feature columns.")
    else:
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

    show_quality_table(
        title="High Missing Columns",
        records=data_quality.get("high_missing_columns", []),
    )
    show_quality_table(
        title="Constant Columns",
        records=[{"column": col} for col in data_quality.get("constant_columns", [])],
    )
    show_quality_table(
        title="High Cardinality Columns",
        records=data_quality.get("high_cardinality_columns", []),
    )
    show_quality_table(
        title="Possible ID Columns",
        records=data_quality.get("possible_id_columns", []),
    )


def show_quality_table(title: str, records: list[dict[str, Any]]) -> None:
    """
    Show a quality section as an expandable table.
    """
    with st.expander(title):
        if records:
            st.dataframe(pd.DataFrame(records), use_container_width=True)
        else:
            st.success("None detected.")


def show_metric_recommendation(metric_result: dict[str, Any]) -> None:
    """
    Display metric recommendation.
    """
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


def show_class_imbalance(imbalance_result: dict[str, Any]) -> None:
    """
    Display class imbalance analysis.
    """
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
            {
                "Class": list(class_counts.keys()),
                "Count": list(class_counts.values()),
            }
        ).set_index("Class")
        st.bar_chart(class_df)
        st.dataframe(class_df, use_container_width=True)

    warning = imbalance_result.get("warning")
    if warning:
        st.warning(warning)


def show_leakage(leakage_result: dict[str, Any]) -> None:
    """
    Display possible leakage-risk analysis.
    """
    st.subheader("🚨 Possible Leakage Risks")

    total_risks = leakage_result.get("total_possible_leakage_risks", 0)
    all_risks = leakage_result.get("all_risks", [])

    st.metric("Total Possible Risks", total_risks)

    if not all_risks:
        st.success("No possible leakage risks detected.")
        st.caption(
            "This does not guarantee absence of leakage. It only means no risk was detected by current deterministic checks."
        )
        return

    risk_df = pd.DataFrame(all_risks)

    if "risk_level" in risk_df.columns:
        st.bar_chart(risk_df["risk_level"].value_counts())

    st.dataframe(risk_df, use_container_width=True)

    for risk in all_risks:
        risk_level = risk.get("risk_level", "unknown").upper()
        column = risk.get("column", "N/A")
        reason = risk.get("reason", "N/A")

        with st.expander(f"{risk_level} risk: {column}"):
            st.write(reason)
            st.json(risk)


def show_baseline_results(baseline_result: dict[str, Any]) -> None:
    """
    Display baseline model benchmark.
    """
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

    note = baseline_result.get("note")
    if note:
        st.info(note)

    if not results:
        st.warning("No baseline results available.")
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
        selected_metric = st.selectbox(
            "Select metric for model comparison",
            options=metric_columns,
        )
        chart_df = result_df[["Model", selected_metric]].set_index("Model")
        st.bar_chart(chart_df)


def show_mlflow_results(mlflow_results: dict[str, Any]) -> None:
    """
    Display MLflow tracking summary.
    """
    st.subheader("🧪 MLflow Tracking")

    models_logged = mlflow_results.get("models_logged", [])
    model_logged = mlflow_results.get("model_logged", False)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Experiment", mlflow_results.get("experiment_name", "N/A"))

    with col2:
        st.metric("Runs Logged", len(models_logged))

    with col3:
        st.metric("Best Model Logged", str(model_logged))

    if models_logged:
        st.write("Models logged:")
        st.write(", ".join(models_logged))

    logged_uri = mlflow_results.get("logged_model_uri")
    if logged_uri:
        st.write("Logged best model URI:")
        st.code(logged_uri)

    message = mlflow_results.get("message")
    if message:
        st.caption(message)


def show_ai_report(report_text: str) -> None:
    """
    Display generated Markdown audit report.
    """
    st.subheader("📄 Generated AI Audit Report")

    if not report_text:
        st.warning("Audit report not available.")
        return

    st.markdown(report_text)

    st.download_button(
        label="Download Markdown Report",
        data=report_text,
        file_name="audit_report.md",
        mime="text/markdown",
    )


def show_downloads(result: dict[str, Any]) -> None:
    """
    Display audit result download buttons.
    """
    st.subheader("⬇️ Downloads")

    report_text = result.get("audit_report", "")
    json_text = to_json_download(result)

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="Download Audit Report (.md)",
            data=report_text,
            file_name="audit_report.md",
            mime="text/markdown",
            disabled=not bool(report_text),
        )

    with col2:
        st.download_button(
            label="Download Audit Result (.json)",
            data=json_text,
            file_name="audit_result.json",
            mime="application/json",
        )


def answer_audit_question(audit_result: dict[str, Any], user_question: str) -> str:
    """
    Ask the LLM a question about the current audit result.
    """
    response = ask_about_audit(
        audit_context=audit_result,
        user_question=user_question,
    )

    if response:
        return response

    return (
        "LLM response available nahi hai. Audit report ke basis par manually "
        "review karo."
    )


def show_audit_chat(result: dict[str, Any]) -> None:
    """
    Display audit-specific chat assistant with conversation history.
    """
    st.subheader("💬 Ask AI About This Audit")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for message in st.session_state["chat_history"]:
        role = message.get("role", "assistant")
        content = message.get("content", "")

        with st.chat_message(role):
            st.markdown(content)

    user_question = st.chat_input(
        "Ask about this audit. Example: Why is F1 Score recommended?"
    )

    if user_question:
        st.session_state["chat_history"].append(
            {"role": "user", "content": user_question}
        )

        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Generating grounded answer..."):
                answer = answer_audit_question(
                    audit_result=result,
                    user_question=user_question.strip(),
                )
            st.markdown(answer)

        st.session_state["chat_history"].append(
            {"role": "assistant", "content": answer}
        )


# ---------------------------------------------------------------------------
# Upload and audit execution
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    current_filename = uploaded_file.name

    if st.session_state.get("uploaded_filename") != current_filename:
        reset_audit_state()
        st.session_state["uploaded_filename"] = current_filename

    file_path = UPLOAD_DIR / f"{uuid.uuid4()}_{Path(uploaded_file.name).name}"

    with file_path.open("wb") as file:
        file.write(uploaded_file.getbuffer())

    max_upload_mb = float(get_config_value("streamlit.max_upload_mb", 25))
    file_size_mb = file_path.stat().st_size / (1024 * 1024)

    if file_size_mb > max_upload_mb:
        cleanup_file(file_path)
        st.error(f"File too large. Maximum allowed size is {max_upload_mb} MB.")
        st.stop()

    df_preview = safe_dataframe_preview(file_path)

    st.subheader("Dataset Preview")
    st.dataframe(df_preview.head(), use_container_width=True)

    target_column = st.selectbox(
        "Select target column",
        options=df_preview.columns.tolist(),
    )

    show_target_distribution(df_preview, target_column)

    if st.button("Run Audit", type="primary"):
        try:
            with st.spinner("Running deterministic audit + LLM report..."):
                result = run_audit_workflow(
                    dataset_path=str(file_path),
                    target_column=target_column,
                )

            result = remove_non_serializable_objects(result)

            st.session_state["audit_result"] = result
            st.session_state["df_preview"] = df_preview
            st.session_state["target_column"] = target_column
            st.session_state["chat_history"] = []

            st.success("Audit completed successfully!")

        except Exception as error:
            st.error(f"Audit failed: {error}")

        finally:
            cleanup_file(file_path)


# ---------------------------------------------------------------------------
# Audit result display
# ---------------------------------------------------------------------------
if "audit_result" in st.session_state:
    result = st.session_state["audit_result"]
    df_preview = st.session_state["df_preview"]
    target_column = st.session_state["target_column"]

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Problem Type", result.get("problem_type", "N/A"))

    with col2:
        leakage_count = result.get("leakage", {}).get(
            "total_possible_leakage_risks",
            0,
        )
        st.metric("Leakage Risks", leakage_count)

    with col3:
        best_model_name = (
            result.get("baseline_results", {})
            .get("best_model", {})
            .get("model_name", "N/A")
        )
        st.metric("Best Baseline", best_model_name)

    st.divider()

    tabs = st.tabs(
        [
            "Overview",
            "Data Quality",
            "Metrics",
            "Imbalance",
            "Leakage",
            "Models",
            "MLflow",
            "AI Report",
            "Downloads",
        ]
    )

    with tabs[0]:
        show_dataset_overview(result.get("profile", {}))
        show_target_distribution(df_preview, target_column)

    with tabs[1]:
        show_data_quality(result.get("data_quality", {}))

    with tabs[2]:
        show_metric_recommendation(result.get("metric_recommendation", {}))

    with tabs[3]:
        show_class_imbalance(result.get("class_imbalance", {}))

    with tabs[4]:
        show_leakage(result.get("leakage", {}))

    with tabs[5]:
        show_baseline_results(result.get("baseline_results", {}))

    with tabs[6]:
        show_mlflow_results(result.get("mlflow_results", {}))

    with tabs[7]:
        show_ai_report(result.get("audit_report", ""))

    with tabs[8]:
        show_downloads(result)

    st.divider()
    show_audit_chat(result)
