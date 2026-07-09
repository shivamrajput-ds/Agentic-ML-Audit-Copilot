from __future__ import annotations

import base64
import json
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
# CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --accent: #6C5CE7;
        --accent-2: #00D4FF;
        --accent-soft: #A29BFE;
        --bg-card: #FFFFFF;
        --bg-soft: #F7F7FB;
        --border-card: #E7E7F0;
        --text-dark: #16162A;
        --text-muted: #6B7280;
        --success: #10B981;
        --warning: #F59E0B;
        --danger: #EF4444;
    }

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2.5rem;
    }

    .hero {
        padding: 2.1rem 2.25rem;
        border-radius: 22px;
        background:
          radial-gradient(circle at top right, rgba(0,212,255,0.28), transparent 30%),
          linear-gradient(135deg, #211C84 0%, #6C5CE7 52%, #00D4FF 100%);
        color: white;
        margin-bottom: 1.4rem;
        box-shadow: 0 18px 42px rgba(64, 57, 150, 0.22);
    }

    .hero h1 {
        margin: 0;
        font-size: 2.15rem;
        letter-spacing: -0.02em;
    }

    .hero p {
        margin: 0.55rem 0 0 0;
        opacity: 0.94;
        font-size: 1rem;
        max-width: 980px;
        line-height: 1.55;
    }

    .hero-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 1.15rem;
    }

    .badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.35rem 0.65rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.16);
        border: 1px solid rgba(255,255,255,0.28);
        color: white;
        font-size: 0.82rem;
        font-weight: 650;
    }

    .section-card {
        border: 1px solid var(--border-card);
        border-radius: 18px;
        padding: 1.1rem 1.15rem;
        background: var(--bg-card);
        box-shadow: 0 10px 28px rgba(22,22,42,0.045);
        margin-bottom: 1rem;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, #FFFFFF 0%, #F8F9FF 100%);
        border: 1px solid var(--border-card);
        border-radius: 16px;
        padding: 1rem 1rem 0.75rem 1rem;
        box-shadow: 0 8px 22px rgba(22,22,42,0.05);
    }

    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] div,
    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricValue"] {
        color: var(--text-dark) !important;
    }

    button[data-baseweb="tab"] {
        font-weight: 700;
    }

    .verdict {
        padding: 0.85rem 1rem;
        border-radius: 14px;
        font-weight: 700;
        border: 1px solid rgba(0,0,0,0.06);
        margin: 0.5rem 0 1rem 0;
    }

    .verdict-good {
        background: rgba(16,185,129,0.12);
        color: #065F46;
    }

    .verdict-review {
        background: rgba(245,158,11,0.14);
        color: #92400E;
    }

    .verdict-risk {
        background: rgba(239,68,68,0.13);
        color: #991B1B;
    }

    .footer {
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid var(--border-card);
        color: var(--text-muted);
        font-size: 0.85rem;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Enterprise visual polish
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(108, 92, 231, 0.10), transparent 28%),
            radial-gradient(circle at top right, rgba(0, 212, 255, 0.11), transparent 30%),
            linear-gradient(180deg, #F8FAFC 0%, #EEF2FF 52%, #F8FAFC 100%);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FFFFFF 0%, #F4F7FF 100%);
        border-right: 1px solid rgba(108, 92, 231, 0.14);
    }

    [data-testid="stFileUploader"] {
        border: 1.5px dashed rgba(108, 92, 231, 0.42);
        border-radius: 18px;
        padding: 0.85rem;
        background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(248,249,255,0.96));
    }

    div[data-testid="stAlert"] {
        border-radius: 14px;
    }

    .stage-pill {
        display: inline-block;
        margin: 0.15rem 0.2rem 0.15rem 0;
        padding: 0.36rem 0.62rem;
        border-radius: 999px;
        background: rgba(108, 92, 231, 0.10);
        color: #211C84;
        font-size: 0.78rem;
        font-weight: 750;
        border: 1px solid rgba(108, 92, 231, 0.16);
    }

    .mini-card {
        padding: 0.9rem 1rem;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.78);
        border: 1px solid rgba(108, 92, 231, 0.14);
        box-shadow: 0 10px 24px rgba(22,22,42,0.045);
        margin-bottom: 0.7rem;
    }

    .mini-card-title {
        font-size: 0.86rem;
        font-weight: 800;
        color: #211C84;
        margin-bottom: 0.2rem;
    }

    .mini-card-text {
        font-size: 0.82rem;
        color: #4B5563;
        line-height: 1.45;
    }

    .stDownloadButton > button {
        border-radius: 14px;
        font-weight: 800;
        border: 1px solid rgba(108, 92, 231, 0.22);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.lower().strip() in {"true", "1", "yes", "y"}

    return bool(value)


def get_project_version() -> str:
    return str(get_config_value("project.version", "1.0.0"))


def cleanup_file(file_path: Path) -> None:
    cleanup_enabled = as_bool(
        get_config_value("streamlit.cleanup_uploaded_files", False)
    )

    if cleanup_enabled and file_path.exists():
        file_path.unlink(missing_ok=True)


def reset_audit_state() -> None:
    keys_to_clear = [
        "audit_result",
        "df_preview",
        "target_column",
        "uploaded_filename",
        "chat_history",
        "last_runtime_seconds",
        "pending_question",
    ]

    for key in keys_to_clear:
        st.session_state.pop(key, None)


def safe_dataframe_preview(file_path: Path) -> pd.DataFrame:
    try:
        preview_rows = int(get_config_value("dataset.profiling_sample_rows", 100000))
        return pd.read_csv(file_path, nrows=preview_rows, low_memory=False)

    except Exception as error:
        st.error(f"Failed to read CSV file: {error}")
        st.stop()


def remove_non_serializable_objects(result: dict[str, Any]) -> dict[str, Any]:
    clean_result = dict(result)

    baseline_results = dict(clean_result.get("baseline_results", {}))
    baseline_results.pop("trained_model_objects", None)
    baseline_results.pop("runtime_objects", None)

    clean_result["baseline_results"] = baseline_results
    clean_result.pop("df", None)

    return clean_result


def sanitize_csv_cell(value: Any) -> Any:
    """
    Prevent CSV formula injection for spreadsheet downloads.
    """
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def sanitize_dataframe_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitize dataframe cells before CSV download.

    Uses DataFrame.map on newer pandas and falls back to applymap on older versions.
    """
    if hasattr(df, "map"):
        return df.map(sanitize_csv_cell)

    return df.applymap(sanitize_csv_cell)


def to_json_download(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, default=str)


def safe_number(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "N/A":
            return default
        return float(value)
    except Exception:
        return default


def make_bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    orientation: str = "v",
) -> go.Figure:
    fig = px.bar(
        df,
        x=x,
        y=y,
        title=title,
        orientation=orientation,
        text=y if orientation == "v" else x,
    )
    fig.update_layout(
        height=390,
        margin=dict(l=10, r=10, t=52, b=10),
        title_font_size=18,
        showlegend=False,
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    return fig


def make_donut_chart(labels: list[Any], values: list[Any], title: str) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Pie(
                labels=[str(label) for label in labels],
                values=values,
                hole=0.55,
            )
        ]
    )
    fig.update_layout(
        title=title,
        height=390,
        margin=dict(l=10, r=10, t=52, b=10),
    )
    return fig


def make_gauge(score: float, title: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "/100"},
            title={"text": title},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#6C5CE7"},
                "steps": [
                    {"range": [0, 50], "color": "#FEE2E2"},
                    {"range": [50, 70], "color": "#FEF3C7"},
                    {"range": [70, 85], "color": "#DBEAFE"},
                    {"range": [85, 100], "color": "#D1FAE5"},
                ],
            },
        )
    )
    fig.update_layout(height=310, margin=dict(l=15, r=15, t=45, b=15))
    return fig


def severity_rank(value: Any) -> int:
    order = {
        "critical": 0,
        "high": 1,
        "major": 1,
        "medium": 2,
        "moderate": 2,
        "minor": 3,
        "low": 3,
        "review": 4,
        "none": 5,
    }
    return order.get(str(value).lower(), 6)


def show_verdict(result: dict[str, Any]) -> None:
    audit_score = result.get("audit_score", {})
    readiness = str(audit_score.get("readiness", "")).lower()

    if readiness in {"good_starting_point"}:
        css = "verdict verdict-good"
        text = "Good Starting Point: No major blocking issue detected by current audit checks."
    elif readiness in {"needs_review"}:
        css = "verdict verdict-review"
        text = "Needs Review: Audit passed execution, but flagged items need human review."
    else:
        css = "verdict verdict-risk"
        text = "High Review Needed: Dataset has important quality or leakage-risk signals."

    st.markdown(f'<div class="{css}">{text}</div>', unsafe_allow_html=True)


def show_stage_timeline(result: dict[str, Any] | None = None) -> None:
    """
    Show a simple enterprise-style audit stage timeline.
    """
    stages = [
        "Load",
        "Profile",
        "Problem Type",
        "Quality",
        "Leakage",
        "Imbalance",
        "Metrics",
        "Baselines",
        "Explainability",
        "Report",
    ]

    timings = {}
    if isinstance(result, dict):
        timings = result.get("execution_summary", {}).get("node_timings", {}) or {}

    pills = []
    for stage in stages:
        key = stage.lower().replace(" ", "_")
        seconds = timings.get(key)
        label = f"{stage} · {seconds}s" if seconds is not None else stage
        pills.append(f'<span class="stage-pill">{label}</span>')

    st.markdown("".join(pills), unsafe_allow_html=True)


def show_hero() -> None:
    st.markdown(
        f"""
        <div class="hero">
            <h1>🤖 Agentic ML Audit Copilot</h1>
            <p>
                Human-in-the-loop ML audit system for pre-training dataset review:
                data quality, leakage risk, metric recommendation, imbalance detection,
                baseline models, MLflow tracking, feature importance, real SHAP summaries,
                and grounded AI explanations.
            </p>
            <div class="hero-badges">
                <span class="badge">⚙️ Deterministic-first</span>
                <span class="badge">🧑‍💻 Human-in-the-loop</span>
                <span class="badge">📊 Plotly Dashboard</span>
                <span class="badge">🧪 MLflow</span>
                <span class="badge">🔍 SHAP</span>
                <span class="badge">v{get_project_version()}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def show_sidebar() -> None:
    with st.sidebar:
        st.header("⚙️ Audit Control Center")

        st.markdown("### Workflow")
        st.markdown(
            """
            1. Upload CSV  
            2. Select target  
            3. Run deterministic audit  
            4. Review risks + baselines  
            5. Human approves/rejects flags  
            """
        )

        st.divider()

        st.markdown("### Current Config")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Max Upload", f"{get_config_value('streamlit.max_upload_mb', 25)} MB")
        with c2:
            st.metric("CV", str(get_config_value("modeling.enable_cross_validation", False)))

        st.caption(
            "Python performs ML checks. The LLM only explains completed deterministic results."
        )

        st.divider()

        if st.button("🧹 Clear Current Audit", use_container_width=True):
            reset_audit_state()
            st.rerun()


# ---------------------------------------------------------------------------
# Upload section
# ---------------------------------------------------------------------------
def show_upload_panel() -> None:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📤 Upload Dataset")
    show_stage_timeline(st.session_state.get("audit_result"))

    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"],
        help="Only CSV files are currently supported.",
    )

    if uploaded_file is None:
        st.info("Upload a CSV dataset to begin the audit.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

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

    st.success(f"Loaded preview successfully: {uploaded_file.name} ({file_size_mb:.2f} MB)")

    col1, col2, col3 = st.columns(3)
    col1.metric("Preview Rows", len(df_preview))
    col2.metric("Columns", len(df_preview.columns))
    col3.metric("File Size", f"{file_size_mb:.2f} MB")

    with st.expander("Preview dataset", expanded=True):
        st.dataframe(df_preview.head(20), use_container_width=True)

    target_column = st.selectbox(
        "Select target column",
        options=df_preview.columns.tolist(),
        index=len(df_preview.columns) - 1 if len(df_preview.columns) > 0 else 0,
    )

    if target_column:
        show_target_distribution(
            df_preview,
            target_column,
            compact=True,
            chart_key="upload_target_distribution",
        )

    run_col, clear_col = st.columns([2, 1])

    with run_col:
        run_clicked = st.button(
            "🚀 Run Full Audit",
            type="primary",
            use_container_width=True,
        )

    with clear_col:
        if st.button("Reset", use_container_width=True):
            reset_audit_state()
            cleanup_file(file_path)
            st.rerun()

    if run_clicked:
        run_audit_and_store(file_path, target_column, df_preview)

    st.markdown("</div>", unsafe_allow_html=True)


def run_audit_and_store(
    file_path: Path,
    target_column: str,
    df_preview: pd.DataFrame,
) -> None:
    try:
        progress = st.progress(0)
        status = st.empty()

        steps = [
            "Loading dataset",
            "Profiling dataset",
            "Checking data quality",
            "Detecting leakage risks",
            "Training baseline models",
            "Running explainability",
            "Generating audit report",
        ]

        start_time = time.perf_counter()

        for idx, step in enumerate(steps[:-1], start=1):
            status.info(f"Step {idx}/{len(steps)}: {step}...")
            progress.progress(idx / len(steps))
            time.sleep(0.05)

        with st.spinner("Running deterministic audit + grounded AI report..."):
            result = run_audit_workflow(
                dataset_path=str(file_path),
                target_column=target_column,
            )

        runtime_seconds = round(time.perf_counter() - start_time, 2)

        status.success("Audit completed successfully.")
        progress.progress(1.0)

        result = remove_non_serializable_objects(result)

        st.session_state["audit_result"] = result
        st.session_state["df_preview"] = df_preview
        st.session_state["target_column"] = target_column
        st.session_state["chat_history"] = []
        st.session_state["last_runtime_seconds"] = runtime_seconds

        st.success(f"Audit completed in {runtime_seconds} seconds.")

    except Exception as error:
        st.error(f"Audit failed: {error}")

    finally:
        cleanup_file(file_path)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
def show_target_distribution(
    df: pd.DataFrame,
    target_column: str,
    compact: bool = False,
    chart_key: str = "target_distribution",
) -> None:
    st.subheader("🎯 Target Distribution")

    target_counts = df[target_column].value_counts(dropna=False)

    chart_df = pd.DataFrame(
        {
            "Class": target_counts.index.astype(str),
            "Count": target_counts.values,
        }
    )

    if len(chart_df) <= 12:
        fig = make_donut_chart(
            labels=chart_df["Class"].tolist(),
            values=chart_df["Count"].tolist(),
            title="Target Distribution",
        )
    else:
        fig = make_bar_chart(
            chart_df.head(25),
            x="Class",
            y="Count",
            title="Top Target Values",
        )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key=chart_key,
    )

    if not compact:
        with st.expander("View target counts"):
            st.dataframe(chart_df, use_container_width=True)


def show_top_kpi_row(result: dict[str, Any]) -> None:
    profile = result.get("profile", {})
    shape = profile.get("shape", {})
    audit_score = result.get("audit_score", {})
    leakage_count = result.get("leakage", {}).get("total_possible_leakage_risks", 0)
    best_model = result.get("baseline_results", {}).get("best_model", {})

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Rows", shape.get("rows", "N/A"))
    col2.metric("Columns", shape.get("columns", "N/A"))
    col3.metric("Audit Score", audit_score.get("score", "N/A"))
    col4.metric("Leakage Risks", leakage_count)
    col5.metric("Best Baseline", best_model.get("model_name", "N/A"))


def show_dataset_overview(profile: dict[str, Any]) -> None:
    st.subheader("📊 Dataset Overview")

    shape = profile.get("shape", {})
    column_types = profile.get("column_types", {})

    numeric_columns = profile.get("numeric_columns") or column_types.get("numeric_columns", [])
    categorical_columns = profile.get("categorical_columns") or column_types.get("categorical_columns", [])
    datetime_columns = profile.get("datetime_columns") or column_types.get("datetime_columns", [])
    boolean_columns = profile.get("boolean_columns") or column_types.get("boolean_columns", [])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", shape.get("rows", "N/A"))
    c2.metric("Columns", shape.get("columns", "N/A"))
    c3.metric("Duplicate Rows", profile.get("duplicate_rows", 0))
    c4.metric("Memory MB", profile.get("memory_usage_mb", "N/A"))

    type_df = pd.DataFrame(
        {
            "Type": ["Numeric", "Categorical", "Datetime", "Boolean"],
            "Count": [
                len(numeric_columns),
                len(categorical_columns),
                len(datetime_columns),
                len(boolean_columns),
            ],
        }
    )

    fig = make_bar_chart(type_df, x="Type", y="Count", title="Column Type Distribution")
    st.plotly_chart(fig, use_container_width=True, key="overview_column_types")

    warnings = profile.get("warnings", [])
    if warnings:
        with st.expander("Profiler warnings"):
            for warning in warnings:
                st.warning(warning)


def show_data_quality(data_quality: dict[str, Any]) -> None:
    st.subheader("🧩 Data Quality Audit")

    quality_score = data_quality.get("quality_score", {})
    score = safe_number(quality_score.get("score", 0))
    health = quality_score.get("health_label", "unknown")
    target_quality = data_quality.get("target_quality", {})
    warnings = data_quality.get("warnings", [])

    col1, col2 = st.columns([1, 2])

    with col1:
        st.plotly_chart(
            make_gauge(score, "Dataset Quality Score"),
            use_container_width=True,
            key="data_quality_score_gauge",
        )

    with col2:
        c1, c2, c3 = st.columns(3)
        c1.metric("Health", health)
        c2.metric("Duplicate Rows", data_quality.get("duplicate_rows", 0))
        c3.metric("Target Missing %", target_quality.get("missing_percent", "N/A"))

        if warnings:
            for warning in warnings:
                if warning == "No major basic data quality issues detected.":
                    st.success(warning)
                else:
                    st.warning(warning)

    missing_values = data_quality.get("missing_values", {})

    st.markdown("#### Missing Values")
    if missing_values:
        missing_df = pd.DataFrame(
            [
                {
                    "Column": column,
                    "Missing Count": values.get("missing_count", 0),
                    "Missing %": values.get("missing_percent", 0),
                }
                for column, values in missing_values.items()
            ]
        ).sort_values("Missing %", ascending=False)

        st.dataframe(missing_df, use_container_width=True)
        fig = make_bar_chart(
            missing_df.head(25),
            x="Missing %",
            y="Column",
            title="Top Missing Value Columns",
            orientation="h",
        )
        st.plotly_chart(fig, use_container_width=True, key="missing_values_bar")
    else:
        st.success("No missing values found in feature columns.")

    col_a, col_b = st.columns(2)

    with col_a:
        show_quality_table("High Missing Columns", data_quality.get("high_missing_columns", []))
        show_quality_table("Null-only Columns", [{"column": col} for col in data_quality.get("null_only_columns", [])])
        show_quality_table("Constant Columns", [{"column": col} for col in data_quality.get("constant_columns", [])])
        show_quality_table("Near-constant Columns", data_quality.get("near_constant_columns", []))

    with col_b:
        show_quality_table("High Cardinality Columns", data_quality.get("high_cardinality_columns", []))
        show_quality_table("Possible ID Columns", data_quality.get("possible_id_columns", []))
        show_quality_table("Infinite Values", dict_to_records(data_quality.get("infinite_values", {})))
        show_quality_table("Outlier Columns", data_quality.get("outlier_columns", []))

    actions = data_quality.get("recommended_actions", [])
    if actions:
        st.markdown("#### Recommended Actions")
        for action in actions:
            st.info(action)


def dict_to_records(data: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for key, value in data.items():
        row = {"column": key}
        if isinstance(value, dict):
            row.update(value)
        else:
            row["value"] = value
        records.append(row)
    return records


def show_quality_table(title: str, records: list[dict[str, Any]]) -> None:
    with st.expander(title):
        if records:
            st.dataframe(pd.DataFrame(records), use_container_width=True)
        else:
            st.success("None detected.")


def show_metric_recommendation(metric_result: dict[str, Any]) -> None:
    st.subheader("📌 Metric Recommendation")

    primary_metric = metric_result.get("primary_metric", "N/A")
    scoring_metric = metric_result.get("scoring_metric", "N/A")
    reason = metric_result.get("reason", "N/A")
    recommended_metrics = metric_result.get("recommended_metrics", [])

    c1, c2, c3 = st.columns(3)
    c1.metric("Primary Metric", primary_metric)
    c2.metric("Sklearn Scoring", scoring_metric)
    c3.metric("Higher is Better", str(metric_result.get("higher_is_better", "N/A")))

    st.info(reason)

    if recommended_metrics:
        metric_df = pd.DataFrame({"Recommended Metrics": recommended_metrics})
        st.dataframe(metric_df, use_container_width=True)


def show_class_imbalance(imbalance_result: dict[str, Any]) -> None:
    st.subheader("⚖️ Class Imbalance")

    if not imbalance_result.get("is_applicable", True):
        st.info(imbalance_result.get("message", "Not applicable."))
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Imbalance Ratio", imbalance_result.get("imbalance_ratio", "N/A"))
    c2.metric("Severity", imbalance_result.get("imbalance_severity", "N/A"))
    c3.metric("Minority Class", imbalance_result.get("minority_class", "N/A"))
    c4.metric("Classes", imbalance_result.get("num_classes", "N/A"))

    class_counts = imbalance_result.get("class_counts", {})
    if class_counts:
        class_df = pd.DataFrame(
            {
                "Class": list(class_counts.keys()),
                "Count": list(class_counts.values()),
            }
        )

        fig = make_donut_chart(
            labels=class_df["Class"].tolist(),
            values=class_df["Count"].tolist(),
            title="Class Distribution",
        )
        st.plotly_chart(fig, use_container_width=True, key="imbalance_class_donut")
        st.dataframe(class_df, use_container_width=True)

    warning = imbalance_result.get("warning")
    if warning:
        st.warning(warning)

    actions = imbalance_result.get("recommended_actions", [])
    if actions:
        st.markdown("#### Recommended Actions")
        for action in actions:
            st.info(action)


def show_leakage(leakage_result: dict[str, Any]) -> None:
    st.subheader("🚨 Possible Leakage Risks")

    total_risks = leakage_result.get("total_possible_leakage_risks", 0)
    severity = leakage_result.get("overall_severity", "none")
    risk_summary = leakage_result.get("risk_summary", {})
    all_risks = leakage_result.get("all_risks", [])

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Possible Risks", total_risks)
    c2.metric("Overall Severity", severity)
    c3.metric("Critical Risks", risk_summary.get("critical", 0))

    if risk_summary:
        risk_df = pd.DataFrame(
            [{"Risk Level": key, "Count": value} for key, value in risk_summary.items()]
        )
        risk_df["sort_rank"] = risk_df["Risk Level"].map(severity_rank)
        risk_df = risk_df.sort_values("sort_rank").drop(columns=["sort_rank"])

        fig = make_bar_chart(risk_df, x="Risk Level", y="Count", title="Leakage Risk Summary")
        st.plotly_chart(fig, use_container_width=True, key="leakage_summary_bar")

    if not all_risks:
        st.success("No possible leakage risks detected.")
        st.caption("This does not guarantee absence of leakage. It means no risk was detected by current deterministic checks.")
        return

    risk_df = pd.DataFrame(all_risks)
    if "risk_level" in risk_df.columns:
        risk_df["severity_rank"] = risk_df["risk_level"].map(severity_rank)
        risk_df = risk_df.sort_values("severity_rank").drop(columns=["severity_rank"])

    st.dataframe(risk_df, use_container_width=True)

    for risk in all_risks:
        risk_level = risk.get("risk_level", "unknown").upper()
        column = risk.get("column", "N/A")
        reason = risk.get("reason", "N/A")

        with st.expander(f"{risk_level} risk: {column}"):
            st.write(reason)
            st.json(risk)

    actions = leakage_result.get("recommended_actions", [])
    if actions:
        st.markdown("#### Recommended Actions")
        for action in actions:
            st.info(action)


def show_baseline_results(baseline_result: dict[str, Any]) -> None:
    st.subheader("🤖 Baseline Model Benchmark")

    best_model = baseline_result.get("best_model", {})
    results = baseline_result.get("results", {})
    evaluation_details = baseline_result.get("evaluation_details", {})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Model", best_model.get("model_name", "N/A"))
    c2.metric("Selection Metric", best_model.get("selection_metric", "N/A"))
    c3.metric("Best Score", best_model.get("score", "N/A"))
    c4.metric("CV Enabled", str(evaluation_details.get("cross_validation_enabled", False)))

    cv_warning = evaluation_details.get("cv_warning")
    if cv_warning:
        st.warning(cv_warning)

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
            key="baseline_metric_select",
        )
        chart_df = result_df[["Model", selected_metric]]
        fig = make_bar_chart(
            chart_df,
            x="Model",
            y=selected_metric,
            title=f"Model Comparison by {selected_metric}",
        )
        st.plotly_chart(fig, use_container_width=True, key="baseline_comparison_bar")

    confusion_matrices = evaluation_details.get("confusion_matrices", {})
    if confusion_matrices:
        st.markdown("#### Confusion Matrices")
        for model_name, matrix in confusion_matrices.items():
            with st.expander(model_name):
                matrix_df = pd.DataFrame(matrix)
                fig = px.imshow(
                    matrix_df,
                    text_auto=True,
                    title=f"Confusion Matrix - {model_name}",
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                )
                fig.update_layout(height=420)
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key=f"confusion_matrix_{model_name}",
                )


def show_explainability(explainability: dict[str, Any]) -> None:
    st.subheader("🔍 Explainability")

    if not explainability:
        st.info("Explainability results not available.")
        return

    if not explainability.get("enabled", False):
        st.info(explainability.get("message", "Explainability is disabled."))
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Available", str(explainability.get("available", False)))
    c2.metric("Best Model", explainability.get("best_model_name", "N/A"))
    c3.metric("Model Type", explainability.get("model_type", "N/A"))
    c4.metric("Sample Rows", explainability.get("sample_rows_used", "N/A"))

    summary = explainability.get("summary", {})
    if summary.get("message"):
        st.info(summary["message"])

    builtin = explainability.get("builtin_feature_importance", {})
    shap_result = explainability.get("shap", {})

    source_choice = st.radio(
        "Explainability source",
        options=["Built-in Feature Importance", "Real SHAP"],
        horizontal=True,
        key="explainability_source_choice",
    )

    if source_choice == "Real SHAP":
        show_real_shap(shap_result)
    else:
        show_builtin_importance(builtin)

    with st.expander("Explainability Notes"):
        for note in explainability.get("notes", []):
            st.caption(note)

    with st.expander("Raw Explainability JSON"):
        st.json(explainability)


def show_builtin_importance(builtin: dict[str, Any]) -> None:
    if not builtin.get("available", False):
        st.warning(builtin.get("message", "Built-in feature importance unavailable."))
        return

    top_features = builtin.get("top_features", [])

    if not top_features:
        st.info("No built-in feature importance records available.")
        return

    feature_df = pd.DataFrame(top_features)
    st.markdown("#### Built-in Feature Importance")
    st.dataframe(feature_df, use_container_width=True)

    value_col = "absolute_importance" if "absolute_importance" in feature_df.columns else "importance"
    chart_df = feature_df[["feature", value_col]].copy()
    chart_df = chart_df.sort_values(value_col, ascending=True)

    fig = px.bar(
        chart_df,
        x=value_col,
        y="feature",
        orientation="h",
        title=f"Built-in Importance - {builtin.get('method', 'model')}",
    )
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=52, b=10))
    st.plotly_chart(fig, use_container_width=True, key="builtin_importance_bar")


def show_real_shap(shap_result: dict[str, Any]) -> None:
    if not shap_result.get("available", False):
        st.warning(shap_result.get("message", "SHAP is unavailable."))
        return

    shap_features = shap_result.get("global_importance", [])

    if shap_features:
        shap_df = pd.DataFrame(shap_features)
        st.markdown("#### Global SHAP Importance")
        st.dataframe(shap_df, use_container_width=True)

        if "mean_abs_shap" in shap_df.columns:
            chart_df = shap_df[["feature", "mean_abs_shap"]].copy()
            chart_df = chart_df.sort_values("mean_abs_shap", ascending=True)

            fig = px.bar(
                chart_df,
                x="mean_abs_shap",
                y="feature",
                orientation="h",
                title="Mean Absolute SHAP Value",
            )
            fig.update_layout(height=520, margin=dict(l=10, r=10, t=52, b=10))
            st.plotly_chart(fig, use_container_width=True, key="real_shap_global_bar")
    else:
        st.info("No global SHAP records available.")

    pos = shap_result.get("positive_contributors", [])
    neg = shap_result.get("negative_contributors", [])

    col_pos, col_neg = st.columns(2)

    with col_pos:
        st.markdown("#### Top Positive SHAP Contributors")
        if pos:
            st.dataframe(pd.DataFrame(pos), use_container_width=True)
        else:
            st.info("No positive contributors found.")

    with col_neg:
        st.markdown("#### Top Negative SHAP Contributors")
        if neg:
            st.dataframe(pd.DataFrame(neg), use_container_width=True)
        else:
            st.info("No negative contributors found.")

    plots = shap_result.get("plots", {})
    bar_plot = plots.get("bar_plot_base64")
    beeswarm_plot = plots.get("beeswarm_plot_base64")

    if bar_plot:
        st.markdown("#### SHAP Bar Plot")
        st.image(base64.b64decode(bar_plot), use_container_width=True)

    if beeswarm_plot:
        st.markdown("#### SHAP Beeswarm Plot")
        st.image(base64.b64decode(beeswarm_plot), use_container_width=True)

    local_explanations = shap_result.get("local_explanations", [])
    if local_explanations:
        st.markdown("#### Local SHAP Explanations")
        selected_sample = st.selectbox(
            "Select sampled row",
            options=list(range(len(local_explanations))),
            format_func=lambda i: (
                f"Sample {local_explanations[i].get('sample_position')} | "
                f"Index {local_explanations[i].get('sample_index')}"
            ),
            key="local_shap_sample_select",
        )

        local = local_explanations[selected_sample]
        local_df = pd.DataFrame(local.get("top_contributors", []))

        if not local_df.empty:
            st.dataframe(local_df, use_container_width=True)

            if "shap_value" in local_df.columns:
                local_chart = local_df[["feature", "shap_value"]].copy()
                local_chart = local_chart.sort_values("shap_value")

                fig = px.bar(
                    local_chart,
                    x="shap_value",
                    y="feature",
                    orientation="h",
                    title="Local SHAP Contributions",
                )
                fig.update_layout(height=520, margin=dict(l=10, r=10, t=52, b=10))
                st.plotly_chart(fig, use_container_width=True, key="local_shap_bar")


def show_mlflow_results(mlflow_results: dict[str, Any]) -> None:
    st.subheader("🧪 MLflow Tracking")

    if not mlflow_results:
        st.warning("MLflow results not available.")
        return

    models_logged = mlflow_results.get("models_logged", [])
    model_logged = mlflow_results.get("model_logged", False)

    c1, c2, c3 = st.columns(3)
    c1.metric("Experiment", mlflow_results.get("experiment_name", "N/A"))
    c2.metric("Runs Logged", len(models_logged))
    c3.metric("Best Model Logged", str(model_logged))

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

    if mlflow_results.get("error"):
        st.warning(mlflow_results["error"])


def show_human_review(human_review: dict[str, Any]) -> None:
    st.subheader("🧑‍💻 Human Review")

    if not human_review:
        st.info("Human review summary not available.")
        return

    c1, c2 = st.columns(2)
    c1.metric("Requires Review", str(human_review.get("requires_human_review", False)))
    c2.metric("Review Items", human_review.get("review_items_count", 0))

    st.info(human_review.get("message", "Human review is recommended."))

    review_items = human_review.get("review_items", [])

    if review_items:
        review_df = pd.DataFrame(review_items)
        if "severity" in review_df.columns:
            review_df["severity_rank"] = review_df["severity"].map(severity_rank)
            review_df = review_df.sort_values("severity_rank").drop(columns=["severity_rank"])
        st.dataframe(review_df, use_container_width=True)


def show_ai_report(report_text: str) -> None:
    st.subheader("📄 Generated Audit Report")

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
    st.subheader("⬇️ Downloads")

    report_text = result.get("audit_report", "")
    json_text = to_json_download(result)

    baseline_results = result.get("baseline_results", {})
    baseline_csv = ""
    if baseline_results.get("results"):
        rows = []
        for model_name, metrics in baseline_results["results"].items():
            row = {"Model": model_name}
            row.update(metrics)
            rows.append(row)

        baseline_df = sanitize_dataframe_for_csv(pd.DataFrame(rows))
        baseline_csv = baseline_df.to_csv(index=False)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.download_button(
            label="Download Report (.md)",
            data=report_text,
            file_name="audit_report.md",
            mime="text/markdown",
            disabled=not bool(report_text),
            use_container_width=True,
        )

    with c2:
        st.download_button(
            label="Download Result (.json)",
            data=json_text,
            file_name="audit_result.json",
            mime="application/json",
            use_container_width=True,
        )

    with c3:
        st.download_button(
            label="Download Model Metrics (.csv)",
            data=baseline_csv,
            file_name="baseline_metrics.csv",
            mime="text/csv",
            disabled=not bool(baseline_csv),
            use_container_width=True,
        )


def compact_chat_context(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "target_column": result.get("target_column"),
        "problem_type": result.get("problem_type"),
        "audit_score": result.get("audit_score"),
        "leakage": result.get("leakage"),
        "metric_recommendation": result.get("metric_recommendation"),
        "class_imbalance": result.get("class_imbalance"),
        "baseline_results": result.get("baseline_results"),
        "human_review": result.get("human_review"),
    }


def answer_audit_question(audit_result: dict[str, Any], user_question: str) -> str:
    response = ask_about_audit(
        audit_context=compact_chat_context(audit_result),
        user_question=user_question,
    )

    if response:
        return response

    return (
        "LLM response is not available right now. Please review the audit report "
        "and deterministic sections above."
    )


def show_audit_chat(result: dict[str, Any]) -> None:
    st.subheader("💬 Ask AI About This Audit")

    suggested_questions = [
        "Why was this primary metric recommended?",
        "Which leakage risks should I review first?",
        "Is this dataset ready for final model training?",
        "Which baseline model performed best and why?",
        "What should I improve before tuning models?",
    ]

    cols = st.columns(len(suggested_questions))
    for idx, question in enumerate(suggested_questions):
        with cols[idx]:
            if st.button(question, key=f"suggested_q_{idx}", use_container_width=True):
                st.session_state["pending_question"] = question

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Limit chat history window to last 5 turns.
    st.session_state["chat_history"] = st.session_state["chat_history"][-10:]

    for message in st.session_state["chat_history"]:
        role = message.get("role", "assistant")
        content = message.get("content", "")

        with st.chat_message(role):
            st.markdown(content)

    user_question = st.chat_input(
        "Ask about this audit. Example: Why is F1 Score recommended?"
    )

    pending_question = st.session_state.pop("pending_question", None)
    final_question = pending_question or user_question

    if final_question:
        st.session_state["chat_history"].append(
            {"role": "user", "content": final_question}
        )

        with st.chat_message("user"):
            st.markdown(final_question)

        with st.chat_message("assistant"):
            with st.spinner("Generating grounded answer..."):
                answer = answer_audit_question(
                    audit_result=result,
                    user_question=final_question.strip(),
                )
            st.markdown(answer)

        st.session_state["chat_history"].append(
            {"role": "assistant", "content": answer}
        )
        st.session_state["chat_history"] = st.session_state["chat_history"][-10:]

    if st.button("Clear Chat"):
        st.session_state["chat_history"] = []
        st.rerun()


def show_result_dashboard() -> None:
    if "audit_result" not in st.session_state:
        return

    result = st.session_state["audit_result"]
    df_preview = st.session_state["df_preview"]
    target_column = st.session_state["target_column"]

    st.divider()

    show_verdict(result)
    show_top_kpi_row(result)

    runtime = st.session_state.get("last_runtime_seconds")
    if runtime:
        st.caption(f"Last audit runtime: {runtime} seconds")

    st.divider()

    tabs = st.tabs(
        [
            "🏠 Overview",
            "🧩 Data Quality",
            "📌 Metrics",
            "⚖️ Imbalance",
            "🚨 Leakage",
            "🤖 Models",
            "🔍 Explainability",
            "🧑‍💻 Human Review",
            "🧪 MLflow",
            "📄 Report",
            "⬇️ Downloads",
        ]
    )

    with tabs[0]:
        show_dataset_overview(result.get("profile", {}))
        show_target_distribution(
            df_preview,
            target_column,
            chart_key="overview_target_distribution",
        )

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
        show_explainability(result.get("explainability", {}))

    with tabs[7]:
        show_human_review(result.get("human_review", {}))

    with tabs[8]:
        show_mlflow_results(result.get("mlflow_results", {}))

    with tabs[9]:
        show_ai_report(result.get("audit_report", ""))

    with tabs[10]:
        show_downloads(result)

    st.divider()
    show_audit_chat(result)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
show_hero()
show_sidebar()
show_upload_panel()
show_result_dashboard()

st.markdown(
    f"""
    <div class="footer">
        Agentic ML Audit Copilot v{get_project_version()} · Deterministic-first · Human-in-the-loop · SHAP-ready.
    </div>
    """,
    unsafe_allow_html=True,
)
