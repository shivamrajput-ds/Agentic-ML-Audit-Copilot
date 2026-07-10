from __future__ import annotations

import base64
import binascii
import json
import math
import sys
import time
import uuid
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.audit.llm_report import ask_about_audit
from src.audit.workflow import run_audit_workflow
from src.utils.config import get_config_value
from src.utils.exceptions import AuditCopilotException


def resolve_project_path(path_value: str | Path) -> Path:
    """Resolve project-relative paths safely for Streamlit runtime."""
    path = Path(path_value)

    if path.is_absolute():
        return path

    return ROOT_DIR / path


UPLOAD_DIR = resolve_project_path(
    get_config_value("streamlit.upload_dir", "data/uploads"),
)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

TRUE_VALUES = {"true", "1", "yes", "y", "on"}
FALSE_VALUES = {"false", "0", "no", "n", "off"}
BLOCKED_DOWNLOAD_KEYS = {
    "trained_model_objects",
    "runtime_objects",
    "model_object",
    "sample_features",
    "sample_target",
    "train_features",
    "test_features",
    "label_encoder",
    "df",
}

PLOTLY_COLOR_SEQUENCE = [
    "#7C3AED",
    "#06B6D4",
    "#F97316",
    "#10B981",
    "#EF4444",
    "#F59E0B",
    "#2563EB",
    "#DB2777",
]
PLOTLY_TEMPLATE = "plotly_white"

st.set_page_config(
    page_title="Agentic ML Audit Copilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------


def inject_custom_css() -> None:
    """Inject premium dashboard CSS."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

        :root {
            --violet: #7C3AED;
            --violet-dark: #4C1D95;
            --cyan: #06B6D4;
            --blue: #2563EB;
            --orange: #F97316;
            --green: #10B981;
            --red: #EF4444;
            --amber: #F59E0B;
            --slate: #0F172A;
            --muted: #64748B;
            --card: rgba(255,255,255,0.82);
            --card-solid: #FFFFFF;
            --line: rgba(124,58,237,0.13);
            --soft-line: rgba(15,23,42,0.08);
            --shadow-lg: 0 28px 80px rgba(30,41,59,0.14);
            --shadow-md: 0 16px 44px rgba(30,41,59,0.10);
            --shadow-sm: 0 10px 28px rgba(30,41,59,0.07);
        }

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }

        .stApp {
            color: var(--slate);
            background:
                radial-gradient(circle at 6% 7%, rgba(124,58,237,0.22), transparent 28%),
                radial-gradient(circle at 90% 8%, rgba(6,182,212,0.22), transparent 30%),
                radial-gradient(circle at 50% 98%, rgba(249,115,22,0.15), transparent 36%),
                linear-gradient(135deg, #F8FAFC 0%, #EEF2FF 32%, #F5F3FF 62%, #ECFEFF 100%);
            background-attachment: fixed;
        }

        .stApp:before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            background-image:
                linear-gradient(rgba(15,23,42,0.035) 1px, transparent 1px),
                linear-gradient(90deg, rgba(15,23,42,0.035) 1px, transparent 1px);
            background-size: 42px 42px;
            mask-image: radial-gradient(circle at 50% 20%, black, transparent 76%);
            z-index: 0;
        }

        .block-container {
            padding-top: 1.1rem;
            padding-bottom: 3rem;
            max-width: 1540px;
            position: relative;
            z-index: 1;
        }

        #MainMenu, footer, header {
            visibility: hidden;
        }

        [data-testid="stSidebar"] {
            background:
                radial-gradient(circle at 20% 5%, rgba(124,58,237,0.14), transparent 26%),
                linear-gradient(180deg, rgba(255,255,255,0.94) 0%, rgba(241,245,249,0.91) 100%);
            border-right: 1px solid rgba(124,58,237,0.14);
            box-shadow: 14px 0 42px rgba(30,41,59,0.06);
        }

        [data-testid="stSidebar"] * {
            color: var(--slate);
        }

        .hero {
            position: relative;
            overflow: hidden;
            padding: 2.65rem 2.55rem;
            border-radius: 34px;
            color: white;
            margin-bottom: 1.45rem;
            background:
                radial-gradient(circle at 88% 10%, rgba(255,255,255,0.32), transparent 18%),
                radial-gradient(circle at 12% 16%, rgba(34,211,238,0.23), transparent 21%),
                linear-gradient(135deg, #0B1026 0%, #311063 29%, #6D28D9 55%, #0891B2 100%);
            box-shadow: 0 34px 90px rgba(76,29,149,0.34);
            border: 1px solid rgba(255,255,255,0.25);
        }

        .hero:before {
            content: "";
            position: absolute;
            width: 500px;
            height: 500px;
            right: -185px;
            top: -205px;
            border-radius: 999px;
            background: rgba(255,255,255,0.15);
            filter: blur(1px);
        }

        .hero:after {
            content: "";
            position: absolute;
            inset: 0;
            background-image:
                linear-gradient(rgba(255,255,255,0.075) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,0.075) 1px, transparent 1px);
            background-size: 34px 34px;
            mask-image: linear-gradient(90deg, rgba(0,0,0,0.35), transparent 80%);
            pointer-events: none;
        }

        .hero h1,
        .hero p,
        .hero-badges,
        .hero-kpis {
            position: relative;
            z-index: 1;
        }

        .hero h1 {
            margin: 0;
            font-size: clamp(2.1rem, 4vw, 3.35rem);
            letter-spacing: -0.065em;
            line-height: 1.01;
            font-weight: 950;
        }

        .hero p {
            margin: 0.9rem 0 0 0;
            opacity: 0.94;
            font-size: 1.04rem;
            max-width: 1040px;
            line-height: 1.68;
            font-weight: 520;
        }

        .hero-badges, .hero-kpis {
            display: flex;
            flex-wrap: wrap;
            gap: 0.62rem;
            margin-top: 1.25rem;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            gap: 0.38rem;
            padding: 0.48rem 0.82rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.16);
            border: 1px solid rgba(255,255,255,0.30);
            color: white;
            font-size: 0.82rem;
            font-weight: 850;
            backdrop-filter: blur(12px);
        }

        .hero-stat {
            min-width: 132px;
            padding: 0.86rem 1rem;
            border-radius: 20px;
            background: rgba(255,255,255,0.13);
            border: 1px solid rgba(255,255,255,0.22);
            backdrop-filter: blur(14px);
        }

        .hero-stat strong {
            display: block;
            font-size: 1.15rem;
            font-weight: 950;
            letter-spacing: -0.03em;
        }

        .hero-stat span {
            display: block;
            margin-top: 0.12rem;
            font-size: 0.74rem;
            opacity: 0.86;
            font-weight: 750;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        .section-card {
            border: 1px solid var(--line);
            border-radius: 28px;
            padding: 1.32rem 1.38rem;
            background:
                linear-gradient(180deg, rgba(255,255,255,0.88) 0%, rgba(248,250,252,0.78) 100%);
            backdrop-filter: blur(18px);
            box-shadow: var(--shadow-md);
            margin-bottom: 1.18rem;
        }

        .glass-panel {
            border: 1px solid rgba(255,255,255,0.35);
            border-radius: 26px;
            padding: 1.1rem 1.2rem;
            background:
                linear-gradient(135deg, rgba(255,255,255,0.82), rgba(239,246,255,0.64));
            box-shadow: var(--shadow-sm);
        }

        div[data-testid="stMetric"] {
            background:
                linear-gradient(180deg, rgba(255,255,255,0.98) 0%, rgba(248,250,252,0.91) 100%);
            border: 1px solid var(--soft-line);
            border-radius: 22px;
            padding: 1.08rem 1.02rem 0.88rem 1.02rem;
            box-shadow: var(--shadow-sm);
            transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
        }

        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            border-color: rgba(124,58,237,0.32);
            box-shadow: 0 20px 48px rgba(30,41,59,0.12);
        }

        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] div,
        div[data-testid="stMetricLabel"],
        div[data-testid="stMetricValue"] {
            color: var(--slate) !important;
        }

        div[data-testid="stMetricValue"] {
            font-weight: 950 !important;
            letter-spacing: -0.042em;
        }

        [data-testid="stFileUploader"] {
            border: 2px dashed rgba(124,58,237,0.36);
            border-radius: 26px;
            padding: 1.1rem;
            background:
                radial-gradient(circle at 88% 12%, rgba(6,182,212,0.11), transparent 32%),
                linear-gradient(135deg, rgba(255,255,255,0.96), rgba(238,242,255,0.86));
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.9), var(--shadow-sm);
        }

        [data-testid="stFileUploader"] section {
            border-radius: 22px;
        }

        .stButton > button,
        .stDownloadButton > button {
            border-radius: 16px !important;
            font-weight: 900 !important;
            letter-spacing: -0.01em;
            border: 1px solid rgba(124,58,237,0.20) !important;
            box-shadow: 0 12px 26px rgba(124,58,237,0.14);
            transition: transform 150ms ease, box-shadow 150ms ease, border-color 150ms ease;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            transform: translateY(-1px);
            border-color: rgba(124,58,237,0.38) !important;
            box-shadow: 0 18px 36px rgba(124,58,237,0.19);
        }

        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #7C3AED 0%, #2563EB 47%, #06B6D4 100%) !important;
            border: none !important;
            color: white !important;
        }

        button[data-baseweb="tab"] {
            font-weight: 900;
            border-radius: 999px;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        [data-testid="stTabs"] [role="tablist"] {
            gap: 0.35rem;
            background: rgba(255,255,255,0.66);
            border: 1px solid rgba(124,58,237,0.12);
            border-radius: 999px;
            padding: 0.38rem;
            box-shadow: 0 12px 30px rgba(30,41,59,0.06);
            backdrop-filter: blur(14px);
        }

        .stage-pill {
            display: inline-block;
            margin: 0.18rem 0.23rem 0.18rem 0;
            padding: 0.46rem 0.76rem;
            border-radius: 999px;
            background: rgba(124,58,237,0.10);
            color: #4C1D95;
            font-size: 0.78rem;
            font-weight: 900;
            border: 1px solid rgba(124,58,237,0.14);
        }

        .verdict {
            padding: 1.05rem 1.15rem;
            border-radius: 22px;
            font-weight: 900;
            border: 1px solid rgba(0,0,0,0.06);
            margin: 0.52rem 0 1.12rem 0;
            box-shadow: var(--shadow-sm);
        }

        .verdict-good {
            background: linear-gradient(135deg, rgba(16,185,129,0.16), rgba(209,250,229,0.88));
            color: #065F46;
        }

        .verdict-review {
            background: linear-gradient(135deg, rgba(245,158,11,0.18), rgba(254,243,199,0.90));
            color: #92400E;
        }

        .verdict-risk {
            background: linear-gradient(135deg, rgba(239,68,68,0.17), rgba(254,226,226,0.90));
            color: #991B1B;
        }

        .hitl-command {
            position: relative;
            overflow: hidden;
            border-radius: 30px;
            padding: 1.45rem 1.5rem;
            margin: 0.6rem 0 1.25rem 0;
            color: white;
            background:
                radial-gradient(circle at 86% 12%, rgba(255,255,255,0.28), transparent 18%),
                linear-gradient(135deg, #1E1B4B 0%, #6D28D9 54%, #0891B2 100%);
            border: 1px solid rgba(255,255,255,0.24);
            box-shadow: 0 26px 68px rgba(76,29,149,0.27);
        }

        .hitl-command h3 {
            color: white;
            margin: 0;
            font-size: 1.55rem;
            font-weight: 950;
            letter-spacing: -0.045em;
        }

        .hitl-command p {
            color: rgba(255,255,255,0.90);
            margin: 0.48rem 0 0 0;
            line-height: 1.58;
            font-weight: 560;
        }

        .hitl-progress-track {
            height: 12px;
            border-radius: 999px;
            background: rgba(255,255,255,0.20);
            overflow: hidden;
            margin-top: 1rem;
            border: 1px solid rgba(255,255,255,0.18);
        }

        .hitl-progress-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #10B981, #22D3EE, #FFFFFF);
        }

        .risk-card {
            border-radius: 24px;
            padding: 1rem 1.08rem;
            margin: 0.65rem 0;
            background: rgba(255,255,255,0.84);
            border: 1px solid rgba(15,23,42,0.08);
            box-shadow: var(--shadow-sm);
        }

        .risk-critical { border-left: 7px solid #DC2626; }
        .risk-high { border-left: 7px solid #F97316; }
        .risk-medium, .risk-moderate { border-left: 7px solid #F59E0B; }
        .risk-low { border-left: 7px solid #10B981; }
        .risk-review, .risk-unknown { border-left: 7px solid #7C3AED; }

        .risk-title {
            font-weight: 950;
            letter-spacing: -0.035em;
            color: #0F172A;
            font-size: 1.02rem;
            margin-bottom: 0.18rem;
        }

        .risk-meta {
            color: #64748B;
            font-size: 0.82rem;
            font-weight: 760;
            margin-bottom: 0.65rem;
        }

        .risk-reason {
            color: #334155;
            line-height: 1.58;
            font-size: 0.92rem;
            margin-bottom: 0.2rem;
        }

        .decision-chip {
            display: inline-flex;
            align-items: center;
            gap: .35rem;
            padding: .35rem .62rem;
            border-radius: 999px;
            background: rgba(124,58,237,0.10);
            color: #4C1D95;
            font-size: .78rem;
            font-weight: 900;
            border: 1px solid rgba(124,58,237,0.15);
        }

        .gate-final {
            border-radius: 28px;
            padding: 1.25rem 1.35rem;
            background:
                radial-gradient(circle at 90% 12%, rgba(6,182,212,0.13), transparent 28%),
                linear-gradient(135deg, rgba(255,255,255,0.94), rgba(248,250,252,0.84));
            border: 1px solid rgba(124,58,237,0.18);
            box-shadow: var(--shadow-md);
            margin-top: 1rem;
        }

        div[data-testid="stAlert"] {
            border-radius: 18px;
            border: 1px solid rgba(15,23,42,0.06);
            box-shadow: 0 9px 22px rgba(30,41,59,0.055);
        }

        div[data-testid="stDataFrame"] {
            border-radius: 19px;
            overflow: hidden;
            box-shadow: 0 11px 28px rgba(30,41,59,0.055);
            border: 1px solid rgba(15,23,42,0.06);
        }

        h1, h2, h3, h4 {
            letter-spacing: -0.04em;
        }

        h2, h3 {
            color: var(--slate);
            font-weight: 950;
        }

        .footer {
            margin-top: 2rem;
            padding: 1.15rem;
            border-radius: 22px;
            border: 1px solid rgba(124,58,237,0.12);
            background: rgba(255,255,255,0.70);
            color: var(--muted);
            font-size: 0.87rem;
            text-align: center;
            box-shadow: var(--shadow-sm);
            backdrop-filter: blur(14px);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def as_bool(value: Any, default: bool = False) -> bool:
    """Convert config values safely into boolean."""
    if isinstance(value, bool):
        return value

    if value is None:
        return default

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in TRUE_VALUES:
            return True
        if normalized in FALSE_VALUES:
            return False
        return default

    return bool(value)


def get_int_config(path: str, default: int, minimum: int | None = None) -> int:
    """Read integer config values with safe fallback and optional lower bound."""
    try:
        value = int(get_config_value(path, default))
    except (TypeError, ValueError):
        value = int(default)

    if minimum is not None:
        return max(minimum, value)

    return value


def get_float_config(path: str, default: float, minimum: float | None = None) -> float:
    """Read float config values with safe fallback and optional lower bound."""
    try:
        value = float(get_config_value(path, default))
    except (TypeError, ValueError):
        value = float(default)

    if not math.isfinite(value):
        value = float(default)

    if minimum is not None:
        return max(float(minimum), value)

    return value


def get_project_version() -> str:
    """Return project version from config."""
    return str(get_config_value("project.version", "1.0.0"))


def normalize_extensions(value: Any, default: list[str]) -> list[str]:
    """Normalize extension config into Streamlit file_uploader format."""
    if isinstance(value, str):
        raw_items = [item.strip() for item in value.split(",")]
    elif isinstance(value, list):
        raw_items = [str(item).strip() for item in value]
    else:
        raw_items = default

    cleaned = []
    for item in raw_items:
        if not item:
            continue
        cleaned.append(item.lower().lstrip("."))

    return sorted(set(cleaned or default))


def get_allowed_upload_extensions() -> list[str]:
    """Return allowed upload extensions without dot for Streamlit uploader."""
    return normalize_extensions(
        get_config_value("streamlit.allowed_extensions", ["csv"]),
        default=["csv"],
    )


def get_uploaded_file_size_mb(uploaded_file: Any) -> float:
    """Return uploaded file size without forcing a full file copy when possible."""
    size = getattr(uploaded_file, "size", None)

    if isinstance(size, int | float):
        return round(float(size) / (1024 * 1024), 4)

    position = uploaded_file.tell()
    uploaded_file.seek(0, 2)
    bytes_size = uploaded_file.tell()
    uploaded_file.seek(position)
    return round(bytes_size / (1024 * 1024), 4)


def get_upload_signature(uploaded_file: Any) -> str:
    """Build a stable signature for deciding when to reset audit state."""
    return "|".join(
        [
            str(getattr(uploaded_file, "name", "")),
            str(getattr(uploaded_file, "size", "")),
            str(getattr(uploaded_file, "type", "")),
        ],
    )


def validate_uploaded_file(uploaded_file: Any) -> None:
    """Validate Streamlit upload metadata before saving."""
    safe_name = Path(getattr(uploaded_file, "name", "") or "").name
    if not safe_name:
        st.error("Uploaded file must have a valid filename.")
        st.stop()

    extension = Path(safe_name).suffix.lower().lstrip(".")
    allowed_extensions = get_allowed_upload_extensions()
    if extension not in allowed_extensions:
        st.error(
            "Unsupported file type. Allowed: "
            + ", ".join(f".{item}" for item in allowed_extensions),
        )
        st.stop()

    max_upload_mb = get_float_config("streamlit.max_upload_mb", 25.0, minimum=0.1)
    file_size_mb = get_uploaded_file_size_mb(uploaded_file)

    if file_size_mb <= 0:
        st.error("Uploaded file is empty.")
        st.stop()

    if file_size_mb > max_upload_mb:
        st.error(f"File too large. Maximum allowed size is {max_upload_mb} MB.")
        st.stop()


def save_uploaded_file(uploaded_file: Any) -> Path:
    """Save uploaded file to a unique project-local path."""
    safe_filename = Path(uploaded_file.name).name
    file_path = UPLOAD_DIR / f"{uuid.uuid4()}_{safe_filename}"

    uploaded_file.seek(0)
    try:
        with file_path.open("wb") as output_file:
            output_file.write(uploaded_file.read())
    except OSError as error:
        st.error(f"Could not save uploaded file: {error}")
        st.stop()
    finally:
        uploaded_file.seek(0)

    return file_path


def json_safe_value(value: Any) -> Any:
    """Recursively convert values into JSON/download-safe objects."""
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if str(key) in BLOCKED_DOWNLOAD_KEYS:
                continue
            cleaned[str(key)] = json_safe_value(item)
        return cleaned

    if isinstance(value, list):
        return [json_safe_value(item) for item in value]

    if isinstance(value, tuple | set):
        return [json_safe_value(item) for item in value]

    if isinstance(value, pd.DataFrame):
        return {
            "type": "DataFrame",
            "shape": list(value.shape),
            "columns": [str(column) for column in value.columns],
        }

    if isinstance(value, pd.Series):
        return {
            "type": "Series",
            "shape": list(value.shape),
            "name": str(value.name),
        }

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, str | int | bool) or value is None:
        return value

    return str(value)


def cleanup_file(file_path: Path) -> None:
    """Delete uploaded file if Streamlit cleanup is enabled."""
    cleanup_enabled = as_bool(
        get_config_value("streamlit.cleanup_uploaded_files", False),
    )

    if cleanup_enabled and file_path.exists():
        try:
            file_path.unlink(missing_ok=True)
        except OSError as error:
            st.warning(f"Could not clean uploaded file: {error}")


def reset_audit_state() -> None:
    """Clear Streamlit session state for the current audit."""
    keys_to_clear = [
        "audit_result",
        "df_preview",
        "target_column",
        "uploaded_filename",
        "uploaded_file_signature",
        "chat_history",
        "last_runtime_seconds",
        "pending_question",
        "human_review_decisions",
        "final_human_review_decision",
        "human_review_export",
        "audit_dataset_path",
        "audit_phase",
    ]

    for key in keys_to_clear:
        st.session_state.pop(key, None)


def safe_dataframe_preview(file_path: Path) -> pd.DataFrame:
    """Read a safe CSV preview for target selection and visual checks."""
    preview_rows = get_int_config(
        "dataset.profiling_sample_rows",
        100_000,
        minimum=1,
    )

    try:
        return pd.read_csv(file_path, nrows=preview_rows, low_memory=False)
    except UnicodeDecodeError:
        try:
            return pd.read_csv(
                file_path,
                nrows=preview_rows,
                low_memory=False,
                encoding="latin1",
            )
        except (
            OSError,
            UnicodeDecodeError,
            pd.errors.ParserError,
            pd.errors.EmptyDataError,
        ) as error:
            st.error(f"Failed to read CSV file: {error}")
            st.stop()
    except (
        OSError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
    ) as error:
        st.error(f"Failed to read CSV file: {error}")
        st.stop()


def remove_non_serializable_objects(result: dict[str, Any]) -> dict[str, Any]:
    """Remove runtime/sklearn objects from Streamlit session/download payload."""
    cleaned = json_safe_value(result)
    return cleaned if isinstance(cleaned, dict) else {}


def sanitize_csv_cell(value: Any) -> Any:
    """Prevent CSV formula injection for spreadsheet downloads."""
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@")):
        return "'" + value

    return value


def sanitize_dataframe_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize dataframe cells before CSV download."""
    return df.apply(lambda column: column.map(sanitize_csv_cell))


def to_json_download(data: dict[str, Any]) -> str:
    """Serialize audit result for JSON download."""
    safe_data = json_safe_value(data)
    return json.dumps(safe_data, indent=2, ensure_ascii=False, default=str)


def safe_number(value: Any, default: float = 0.0) -> float:
    """Convert numeric-like values safely."""
    try:
        if value is None or value == "N/A":
            return default
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def make_bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    orientation: str = "v",
) -> go.Figure:
    """Create a consistent Plotly bar chart."""
    fig = px.bar(
        df,
        x=x,
        y=y,
        title=title,
        orientation=orientation,
        text=y if orientation == "v" else x,
        color_discrete_sequence=PLOTLY_COLOR_SEQUENCE,
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(
        height=410,
        margin=dict(l=16, r=16, t=60, b=16),
        title_font_size=19,
        title_font_color="#0F172A",
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.72)",
        font=dict(family="Inter, sans-serif", color="#0F172A"),
    )
    fig.update_xaxes(gridcolor="rgba(15,23,42,0.07)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(15,23,42,0.07)", zeroline=False)
    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
        marker_line_width=0,
        opacity=0.92,
    )
    return fig


def make_donut_chart(labels: list[Any], values: list[Any], title: str) -> go.Figure:
    """Create a consistent Plotly donut chart."""
    fig = go.Figure(
        data=[
            go.Pie(
                labels=[str(label) for label in labels],
                values=values,
                hole=0.58,
                marker=dict(
                    colors=PLOTLY_COLOR_SEQUENCE, line=dict(color="#FFFFFF", width=3)
                ),
                textinfo="percent+label",
            ),
        ],
    )
    fig.update_layout(
        title=title,
        height=410,
        margin=dict(l=16, r=16, t=60, b=16),
        title_font_size=19,
        title_font_color="#0F172A",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#0F172A"),
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5
        ),
    )
    return fig


def make_gauge(score: float, title: str) -> go.Figure:
    """Create an audit/quality score gauge."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "/100"},
            title={"text": title},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#7C3AED", "thickness": 0.24},
                "bgcolor": "rgba(255,255,255,0.72)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": "#FEE2E2"},
                    {"range": [50, 70], "color": "#FEF3C7"},
                    {"range": [70, 85], "color": "#DBEAFE"},
                    {"range": [85, 100], "color": "#D1FAE5"},
                ],
                "threshold": {
                    "line": {"color": "#0F172A", "width": 3},
                    "thickness": 0.72,
                    "value": score,
                },
            },
        ),
    )
    fig.update_layout(
        height=325,
        margin=dict(l=18, r=18, t=52, b=18),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#0F172A"),
    )
    return fig


def severity_rank(value: Any) -> int:
    """Sort severity labels in risk-first order."""
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


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------
def show_verdict(result: dict[str, Any]) -> None:
    """Show audit readiness verdict."""
    audit_score = result.get("audit_score", {})
    workflow_status = str(result.get("workflow_status", "")).lower()
    readiness = (
        str(audit_score.get("readiness", "")).lower()
        if isinstance(audit_score, dict)
        else ""
    )

    if workflow_status == "blocked_for_review":
        css = "verdict verdict-risk"
        text = (
            "Blocked for Human Review: Critical risk signals were detected, so "
            "baseline modeling may have been skipped depending on config."
        )
    elif readiness in {"good_starting_point"}:
        css = "verdict verdict-good"
        text = "Good Starting Point: No major blocking issue detected by current audit checks."
    elif readiness in {"needs_review"}:
        css = "verdict verdict-review"
        text = (
            "Needs Review: Audit passed execution, but flagged items need human review."
        )
    else:
        css = "verdict verdict-risk"
        text = (
            "High Review Needed: Dataset has important quality or leakage-risk signals."
        )

    st.markdown(f'<div class="{css}">{text}</div>', unsafe_allow_html=True)


def show_stage_timeline(result: dict[str, Any] | None = None) -> None:
    """Show simple audit stage timeline."""
    stage_map = [
        ("Load", "load_dataset"),
        ("Profile", "profile"),
        ("Problem Type", "problem_detection"),
        ("Quality", "data_quality"),
        ("Leakage", "leakage"),
        ("Imbalance", "imbalance"),
        ("Risk Router", "decision_router"),
        ("Metrics", "metrics"),
        ("Baselines", "baseline"),
        ("Explainability", "explainability"),
        ("Report", "report"),
    ]

    timings: dict[str, Any] = {}
    stage_status: dict[str, str] = {}

    if isinstance(result, dict):
        execution_summary = result.get("execution_summary", {})
        if isinstance(execution_summary, dict):
            timings = execution_summary.get("node_timings", {}) or {}
            for item in execution_summary.get("stage_status", []) or []:
                if isinstance(item, dict):
                    stage_status[str(item.get("stage"))] = str(item.get("status"))

    pills = []
    for label, key in stage_map:
        seconds = timings.get(key)
        status = stage_status.get(key)
        suffix_parts = []
        if seconds is not None:
            suffix_parts.append(f"{seconds}s")
        if status and status != "completed":
            suffix_parts.append(status)
        suffix = " · " + " · ".join(suffix_parts) if suffix_parts else ""
        pills.append(f'<span class="stage-pill">{label}{suffix}</span>')

    st.markdown("".join(pills), unsafe_allow_html=True)


def show_hero() -> None:
    """Render hero banner."""
    st.markdown(
        f"""
        <div class="hero">
            <h1>Agentic ML Audit Copilot</h1>
            <p>
                Premium pre-training ML audit cockpit with a real human-in-the-loop gate.
                Upload a tabular dataset, review deterministic risks, approve or block modeling,
                then generate baselines, explainability, MLflow tracking, and a grounded report.
            </p>
            <div class="hero-badges">
                <span class="badge">⚙️ Deterministic-first</span>
                <span class="badge">🧑‍💻 Human Gate</span>
                <span class="badge">🚨 Leakage Review</span>
                <span class="badge">📊 Executive Dashboard</span>
                <span class="badge">🔍 SHAP-ready</span>
                <span class="badge">v{get_project_version()}</span>
            </div>
            <div class="hero-kpis">
                <div class="hero-stat"><strong>Audit</strong><span>before training</span></div>
                <div class="hero-stat"><strong>HITL</strong><span>approve risks</span></div>
                <div class="hero-stat"><strong>MLflow</strong><span>track baselines</span></div>
                <div class="hero-stat"><strong>Report</strong><span>grounded output</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_sidebar() -> None:
    """Render sidebar controls."""
    with st.sidebar:
        st.markdown(
            """
            <div style="padding:1.08rem;border-radius:24px;background:linear-gradient(135deg,#111827,#6D28D9 55%,#06B6D4);color:white;box-shadow:0 18px 40px rgba(76,29,149,.25);margin-bottom:1rem;">
                <div style="font-size:1.16rem;font-weight:950;letter-spacing:-.035em;color:white;">⚙️ Audit Control Center</div>
                <div style="font-size:.82rem;opacity:.90;margin-top:.28rem;color:white;">Upload → Review Gate → Baseline → Explain</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Workflow Mode")
        st.markdown(
            """
            <span class="decision-chip">Human Gate enabled</span>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "The app pauses after deterministic risk checks. Modeling continues only after reviewer approval.",
        )

        st.divider()

        st.markdown("### Audit Pipeline")
        st.markdown(
            """
            - CSV upload and profiling  
            - Problem type detection  
            - Quality + leakage + imbalance checks  
            - Risk aggregator + human gate  
            - Baseline models + MLflow + SHAP  
            - Final audit report  
            """,
        )

        st.divider()

        st.markdown("### Current Config")
        c1, c2 = st.columns(2)
        with c1:
            st.metric(
                "Upload",
                f"{get_config_value('streamlit.max_upload_mb', 25)} MB",
            )
        with c2:
            st.metric(
                "CV",
                str(get_config_value("modeling.enable_cross_validation", False)),
            )

        st.caption(
            "LLM is used only for explanations/reporting. Python performs all ML checks.",
        )

        st.divider()

        if st.button("🧹 Clear Current Audit", use_container_width=True):
            reset_audit_state()
            st.rerun()


def show_upload_panel() -> None:
    """Render upload panel and run audit trigger."""
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📤 Dataset Intake")
    st.markdown(
        """
        <div class="glass-panel">
            <strong>Upload a CSV and select the target column.</strong><br>
            <span style="color:#64748B;">
            The first run opens the human gate. Baseline modeling continues only after reviewer approval.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    show_stage_timeline(st.session_state.get("audit_result"))

    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=get_allowed_upload_extensions(),
        help="Only CSV files are currently supported.",
    )

    if uploaded_file is None:
        st.info("Upload a CSV dataset to begin the audit.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    validate_uploaded_file(uploaded_file)

    current_signature = get_upload_signature(uploaded_file)
    if st.session_state.get("uploaded_file_signature") != current_signature:
        reset_audit_state()
        st.session_state["uploaded_file_signature"] = current_signature
        st.session_state["uploaded_filename"] = uploaded_file.name

    file_path = save_uploaded_file(uploaded_file)
    file_size_mb = get_uploaded_file_size_mb(uploaded_file)
    df_preview = safe_dataframe_preview(file_path)

    st.success(
        f"Loaded preview successfully: {uploaded_file.name} ({file_size_mb:.2f} MB)"
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Preview Rows", len(df_preview))
    col2.metric("Columns", len(df_preview.columns))
    col3.metric("File Size", f"{file_size_mb:.2f} MB")
    col4.metric("Gate Mode", "Human")

    with st.expander("Preview dataset", expanded=True):
        st.dataframe(df_preview.head(20), use_container_width=True)

    if df_preview.empty or len(df_preview.columns) == 0:
        cleanup_file(file_path)
        st.error("CSV preview is empty or has no columns.")
        st.stop()

    target_column = st.selectbox(
        "Select target column",
        options=[str(column) for column in df_preview.columns.tolist()],
        index=len(df_preview.columns) - 1,
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
            "🚀 Run Audit + Open Human Gate",
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
    else:
        cleanup_file(file_path)

    st.markdown("</div>", unsafe_allow_html=True)


def run_audit_and_store(
    file_path: Path,
    target_column: str,
    df_preview: pd.DataFrame,
) -> None:
    """Run audit workflow and store result in Streamlit session state."""
    try:
        progress = st.progress(0)
        status = st.empty()

        steps = [
            "Loading dataset",
            "Profiling dataset",
            "Running parallel audit checks",
            "Aggregating risks",
            "Routing HITL decision",
            "Training baseline models when allowed",
            "Generating explainability/report",
        ]

        start_time = time.perf_counter()

        for index, step in enumerate(steps[:-1], start=1):
            status.info(f"Step {index}/{len(steps)}: {step}...")
            progress.progress(index / len(steps))
            time.sleep(0.03)

        with st.spinner("Running deterministic audit + grounded AI report..."):
            result = run_audit_workflow(
                dataset_path=str(file_path),
                target_column=target_column,
                workflow_mode="human_gate",
            )

        runtime_seconds = round(time.perf_counter() - start_time, 2)

        result = remove_non_serializable_objects(result)

        st.session_state["audit_result"] = result
        st.session_state["df_preview"] = df_preview.copy()
        st.session_state["target_column"] = target_column
        st.session_state["audit_dataset_path"] = str(file_path)
        st.session_state["audit_phase"] = (
            "waiting_for_human_approval"
            if str(result.get("workflow_status", "")).lower()
            == "waiting_for_human_approval"
            else "completed"
        )
        st.session_state["chat_history"] = []
        st.session_state["last_runtime_seconds"] = runtime_seconds

        status.success("Audit completed successfully.")
        progress.progress(1.0)
        st.success(f"Audit completed in {runtime_seconds} seconds.")

    except AuditCopilotException as error:
        st.error(error.user_message() if hasattr(error, "user_message") else str(error))
        with st.expander("Technical error detail"):
            st.code(str(error))
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
        OSError,
    ) as error:
        st.error(f"Audit failed: {error}")

    finally:
        if st.session_state.get("audit_phase") != "waiting_for_human_approval":
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
    """Display target distribution chart."""
    st.subheader("🎯 Target Distribution")

    target_counts = df[target_column].value_counts(dropna=False)
    chart_df = pd.DataFrame(
        {
            "Class": target_counts.index.astype(str),
            "Count": target_counts.values,
        },
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

    st.plotly_chart(fig, use_container_width=True, key=chart_key)

    if not compact:
        with st.expander("View target counts"):
            st.dataframe(chart_df, use_container_width=True)


def show_workflow_decision(result: dict[str, Any]) -> None:
    """Show workflow status, router decision, and risk aggregation summary."""
    st.subheader("🧭 Workflow Decision")

    execution_summary = result.get("execution_summary", {})
    risk_summary = result.get("risk_aggregator", {})
    decision_router = result.get("decision_router", {})

    if isinstance(execution_summary, dict):
        decision_router = decision_router or execution_summary.get(
            "decision_router", {}
        )

    if not isinstance(risk_summary, dict):
        risk_summary = {}

    if not isinstance(decision_router, dict):
        decision_router = {}

    c1, c2, c3 = st.columns(3)
    c1.metric("Workflow Status", result.get("workflow_status", "N/A"))
    c2.metric("Router Decision", decision_router.get("decision", "N/A"))
    c3.metric("Risk Items", risk_summary.get("risk_items_count", 0))

    router_message = decision_router.get("message")
    risk_message = risk_summary.get("message")

    if router_message:
        st.info(router_message)

    if risk_message:
        st.caption(risk_message)

    critical_blockers = risk_summary.get("critical_blockers", [])
    if critical_blockers:
        st.warning("Critical blockers: " + ", ".join(map(str, critical_blockers)))

    risk_items = risk_summary.get("risk_items", [])
    if risk_items:
        with st.expander("Risk aggregation details"):
            st.dataframe(pd.DataFrame(risk_items), use_container_width=True)


def show_top_kpi_row(result: dict[str, Any]) -> None:
    """Show main KPI metric row."""
    profile = result.get("profile", {})
    shape = profile.get("shape", {}) if isinstance(profile, dict) else {}
    audit_score = result.get("audit_score", {})
    leakage = result.get("leakage", {})
    baseline = result.get("baseline_results", {})

    leakage_count = (
        leakage.get("total_possible_leakage_risks", 0)
        if isinstance(leakage, dict)
        else 0
    )
    best_model = baseline.get("best_model", {}) if isinstance(baseline, dict) else {}

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Rows", shape.get("rows", "N/A"))
    col2.metric("Columns", shape.get("columns", "N/A"))
    col3.metric("Audit Score", audit_score.get("score", "N/A"))
    col4.metric("Leakage Risks", leakage_count)
    col5.metric("Best Baseline", best_model.get("model_name", "N/A"))


def show_dataset_overview(profile: dict[str, Any]) -> None:
    """Show dataset profile overview."""
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
    boolean_columns = profile.get("boolean_columns") or column_types.get(
        "boolean_columns", []
    )

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
        },
    )

    fig = make_bar_chart(type_df, x="Type", y="Count", title="Column Type Distribution")
    st.plotly_chart(fig, use_container_width=True, key="overview_column_types")

    warnings = profile.get("warnings", [])
    if warnings:
        with st.expander("Profiler warnings"):
            for warning in warnings:
                st.warning(warning)


def dict_to_records(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert nested dict to row records."""
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
    """Show expandable quality table."""
    with st.expander(title):
        if records:
            st.dataframe(pd.DataFrame(records), use_container_width=True)
        else:
            st.success("None detected.")


def show_data_quality(data_quality: dict[str, Any]) -> None:
    """Show data quality audit section."""
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
            ],
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
        show_quality_table(
            "High Missing Columns",
            data_quality.get("high_missing_columns", []),
        )
        show_quality_table(
            "Null-only Columns",
            [{"column": col} for col in data_quality.get("null_only_columns", [])],
        )
        show_quality_table(
            "Constant Columns",
            [{"column": col} for col in data_quality.get("constant_columns", [])],
        )
        show_quality_table(
            "Near-constant Columns",
            data_quality.get("near_constant_columns", []),
        )

    with col_b:
        show_quality_table(
            "High Cardinality Columns",
            data_quality.get("high_cardinality_columns", []),
        )
        show_quality_table(
            "Possible ID Columns",
            data_quality.get("possible_id_columns", []),
        )
        show_quality_table(
            "Infinite Values",
            dict_to_records(data_quality.get("infinite_values", {})),
        )
        show_quality_table(
            "Outlier Columns",
            data_quality.get("outlier_columns", []),
        )

    actions = data_quality.get("recommended_actions", [])
    if actions:
        st.markdown("#### Recommended Actions")
        for action in actions:
            st.info(action)


def show_metric_recommendation(metric_result: dict[str, Any]) -> None:
    """Show metric recommendation section."""
    st.subheader("📌 Metric Recommendation")

    if not metric_result:
        st.info("Metric recommendation is not available.")
        return

    if metric_result.get("skipped"):
        st.warning(metric_result.get("message", "Metric recommendation was skipped."))
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Primary Metric", metric_result.get("primary_metric", "N/A"))
    c2.metric(
        "Sklearn Scoring",
        metric_result.get(
            "sklearn_scoring_metric",
            metric_result.get("scoring_metric", "N/A"),
        ),
    )
    c3.metric("Higher is Better", str(metric_result.get("higher_is_better", "N/A")))

    st.info(metric_result.get("reason", "N/A"))

    warnings = metric_result.get("warnings", [])
    for warning in warnings:
        st.warning(warning)

    recommended_metrics = metric_result.get("recommended_metrics", [])
    if recommended_metrics:
        st.dataframe(
            pd.DataFrame({"Recommended Metrics": recommended_metrics}),
            use_container_width=True,
        )


def show_class_imbalance(imbalance_result: dict[str, Any]) -> None:
    """Show class imbalance section."""
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
            },
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
    """Show leakage risk section."""
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
            [
                {"Risk Level": key, "Count": value}
                for key, value in risk_summary.items()
            ],
        )
        risk_df["sort_rank"] = risk_df["Risk Level"].map(severity_rank)
        risk_df = risk_df.sort_values("sort_rank").drop(columns=["sort_rank"])

        fig = make_bar_chart(
            risk_df,
            x="Risk Level",
            y="Count",
            title="Leakage Risk Summary",
        )
        st.plotly_chart(fig, use_container_width=True, key="leakage_summary_bar")

    if not all_risks:
        st.success("No possible leakage risks detected.")
        st.caption(
            "This does not guarantee absence of leakage. It means no risk was detected by current deterministic checks.",
        )
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
    """Show baseline model benchmark section."""
    st.subheader("🤖 Baseline Model Benchmark")

    if not baseline_result:
        st.warning("No baseline results available.")
        return

    if baseline_result.get("skipped"):
        st.warning(baseline_result.get("message", "Baseline modeling was skipped."))
        return

    best_model = baseline_result.get("best_model", {})
    results = baseline_result.get("results", {})
    evaluation_details = baseline_result.get("evaluation_details", {})

    if not isinstance(best_model, dict):
        best_model = {}
    if not isinstance(results, dict):
        results = {}
    if not isinstance(evaluation_details, dict):
        evaluation_details = {}

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Model", best_model.get("model_name", "N/A"))
    c2.metric("Selection Metric", best_model.get("selection_metric", "N/A"))
    c3.metric("Best Score", best_model.get("score", "N/A"))
    c4.metric(
        "CV Enabled",
        str(evaluation_details.get("cross_validation_enabled", False)),
    )

    cv_warning = evaluation_details.get("cv_warning")
    if cv_warning:
        st.warning(cv_warning)

    note = baseline_result.get("note")
    if note:
        st.info(note)

    warnings = baseline_result.get("warnings", [])
    for warning in warnings:
        st.warning(warning)

    if not results:
        st.warning("No baseline results available.")
        return

    rows = []
    for model_name, metrics in results.items():
        row = {"Model": model_name}
        if isinstance(metrics, dict):
            row.update(metrics)
        rows.append(row)

    result_df = pd.DataFrame(rows)
    st.dataframe(result_df, use_container_width=True)

    metric_columns = [
        column
        for column in result_df.columns
        if column != "Model" and pd.api.types.is_numeric_dtype(result_df[column])
    ]

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
            with st.expander(str(model_name)):
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


def show_builtin_importance(builtin: dict[str, Any]) -> None:
    """Show built-in model feature importance."""
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

    value_col = (
        "absolute_importance"
        if "absolute_importance" in feature_df.columns
        else "importance"
    )
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
    """Show real SHAP global/local explanations."""
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

    try:
        if bar_plot:
            st.markdown("#### SHAP Bar Plot")
            st.image(base64.b64decode(bar_plot), use_container_width=True)

        if beeswarm_plot:
            st.markdown("#### SHAP Beeswarm Plot")
            st.image(base64.b64decode(beeswarm_plot), use_container_width=True)
    except (ValueError, TypeError, binascii.Error) as error:
        st.warning(f"Could not render SHAP image: {error}")

    local_explanations = shap_result.get("local_explanations", [])
    if local_explanations:
        st.markdown("#### Local SHAP Explanations")
        selected_sample = st.selectbox(
            "Select sampled row",
            options=list(range(len(local_explanations))),
            format_func=lambda index: (
                f"Sample {local_explanations[index].get('sample_position')} | "
                f"Index {local_explanations[index].get('sample_index')}"
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


def show_explainability(explainability: dict[str, Any]) -> None:
    """Show explainability section."""
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
    if isinstance(summary, dict) and summary.get("message"):
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
        show_real_shap(shap_result if isinstance(shap_result, dict) else {})
    else:
        show_builtin_importance(builtin if isinstance(builtin, dict) else {})

    with st.expander("Explainability Notes"):
        for note in explainability.get("notes", []):
            st.caption(note)

    with st.expander("Raw Explainability JSON"):
        st.json(explainability)


def show_mlflow_results(mlflow_results: dict[str, Any]) -> None:
    """Show MLflow results section with human-gate aware messaging."""
    st.subheader("🧪 MLflow Tracking")

    result_context = st.session_state.get("audit_result", {})
    workflow_status = ""
    if isinstance(result_context, dict):
        workflow_status = str(result_context.get("workflow_status", "")).lower()

    if not isinstance(mlflow_results, dict):
        mlflow_results = {}

    if not mlflow_results:
        if workflow_status == "waiting_for_human_approval":
            st.markdown(
                """
                <div class="gate-final">
                    <h3 style="margin-top:0;">MLflow Tracking Paused</h3>
                    <p style="color:#475569;line-height:1.58;margin-bottom:.25rem;">
                        The audit is currently stopped at the Human Review Gate. This is expected.
                        Baseline model training, MLflow experiment logging, SHAP explainability,
                        and final report generation will start only after reviewer approval.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            c1, c2, c3 = st.columns(3)
            c1.metric("Workflow", "Human Gate")
            c2.metric("MLflow Status", "Paused")
            c3.metric("Next Step", "Approve Review")
            st.info(
                "Go to the Human Gate tab, review every risk item, choose an approving final decision, then click Continue Modeling After Human Approval.",
            )
            return

        st.warning("MLflow results are not available for this run.")
        st.caption(
            "Possible reasons: MLflow is disabled in config, baseline modeling was skipped, or the workflow did not reach the tracking stage.",
        )
        return

    if mlflow_results.get("skipped"):
        st.warning(mlflow_results.get("message", "MLflow tracking was skipped."))
        return

    if mlflow_results.get("enabled") is False:
        st.info(mlflow_results.get("message", "MLflow tracking is disabled."))
        return

    models_logged = (
        mlflow_results.get("models_logged") or mlflow_results.get("logged_models") or []
    )
    if not isinstance(models_logged, list):
        models_logged = [str(models_logged)]

    model_logged = mlflow_results.get(
        "model_logged",
        bool(models_logged or mlflow_results.get("logged_model_uri")),
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Experiment", mlflow_results.get("experiment_name", "N/A"))
    c2.metric("Run ID", mlflow_results.get("run_id", "N/A"))
    c3.metric("Runs/Models Logged", len(models_logged))
    c4.metric("Best Model Logged", str(model_logged))

    run_name = mlflow_results.get("run_name")
    if run_name:
        st.caption(f"Run name: {run_name}")

    if models_logged:
        st.markdown("#### Models Logged")
        st.dataframe(
            pd.DataFrame({"Logged Models": [str(model) for model in models_logged]}),
            use_container_width=True,
        )

    logged_uri = mlflow_results.get("logged_model_uri") or mlflow_results.get(
        "model_uri",
    )
    if logged_uri:
        st.markdown("#### Logged Best Model URI")
        st.code(str(logged_uri))

    tracking_uri = mlflow_results.get("tracking_uri")
    if tracking_uri:
        st.caption(f"Tracking URI: {tracking_uri}")

    message = mlflow_results.get("message")
    if message:
        st.info(str(message))

    if mlflow_results.get("error"):
        st.warning(str(mlflow_results["error"]))

    with st.expander("Raw MLflow Tracking JSON"):
        st.json(mlflow_results)


def get_review_item_key(item: dict[str, Any], index: int) -> str:
    """Return stable key for a human review item."""
    category = str(item.get("category", "unknown"))
    severity = str(item.get("severity", "unknown"))
    column = str(item.get("column", "none"))
    reason = str(item.get("reason", ""))[:80]
    return f"{index}_{category}_{severity}_{column}_{abs(hash(reason))}"


def normalize_review_items(review_items: Any) -> list[dict[str, Any]]:
    """Normalize human-review items into dictionaries."""
    if not isinstance(review_items, list):
        return []

    normalized = []
    for index, item in enumerate(review_items):
        if isinstance(item, dict):
            row: dict[str, Any] = dict(item)
        else:
            row = {"reason": str(item)}

        row.setdefault("category", "review")
        row.setdefault("severity", "review")
        row.setdefault("column", None)
        row.setdefault("suggested_decision", "review_before_modeling")
        row.setdefault("status", "pending_human_review")
        row["_review_key"] = get_review_item_key(row, index)
        normalized.append(row)

    return normalized


def get_review_decision_store(
    review_items: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Initialize or return Streamlit session store for human review decisions."""
    if "human_review_decisions" not in st.session_state:
        st.session_state["human_review_decisions"] = {}

    store = st.session_state["human_review_decisions"]

    for item in review_items:
        review_key = item["_review_key"]
        if review_key not in store:
            store[review_key] = {
                "decision": "pending_human_review",
                "reviewer_note": "",
            }

    valid_keys = {item["_review_key"] for item in review_items}
    for old_key in list(store.keys()):
        if old_key not in valid_keys:
            store.pop(old_key, None)

    return store


def get_decision_label(value: str) -> str:
    """Return human-friendly label for review decisions."""
    labels = {
        "pending_human_review": "⏳ Pending review",
        "accept_flag_fix_later": "✅ Accept flag, fix later",
        "accept_risk_continue": "🟢 Accept risk and continue",
        "needs_data_fix": "🛠️ Needs data fix before modeling",
        "reject_modeling": "🛑 Reject modeling for now",
        "false_positive": "⚪ Mark as false positive",
    }
    return labels.get(value, value)


def build_human_review_export(
    human_review: dict[str, Any],
    review_items: list[dict[str, Any]],
    decisions: dict[str, dict[str, Any]],
    final_decision: str,
) -> dict[str, Any]:
    """Build downloadable human-review decision payload."""
    reviewed_items = []

    for item in review_items:
        review_key = item["_review_key"]
        decision_data = decisions.get(review_key, {})
        clean_item = {key: value for key, value in item.items() if key != "_review_key"}
        clean_item.update(
            {
                "human_decision": decision_data.get(
                    "decision",
                    "pending_human_review",
                ),
                "reviewer_note": decision_data.get("reviewer_note", ""),
            },
        )
        reviewed_items.append(clean_item)

    pending_count = sum(
        1
        for item in reviewed_items
        if item.get("human_decision") == "pending_human_review"
    )

    return {
        "requires_human_review": human_review.get("requires_human_review", False),
        "review_items_count": len(reviewed_items),
        "pending_items_count": pending_count,
        "final_human_decision": final_decision,
        "reviewed_items": reviewed_items,
    }


def is_human_review_pending(
    review_items: list[dict[str, Any]], decisions: dict[str, dict[str, Any]]
) -> bool:
    """Return whether at least one review decision is pending."""
    return any(
        decisions.get(item["_review_key"], {}).get("decision") == "pending_human_review"
        for item in review_items
    )


def is_final_decision_approved(final_decision: str) -> bool:
    """Return whether final human decision allows modeling."""
    return final_decision in {
        "approved_for_baseline_experiment_only",
        "approved_with_known_risks",
    }


def continue_modeling_after_human_review(export_payload: dict[str, Any]) -> None:
    """Run full modeling workflow after explicit human approval."""
    dataset_path = st.session_state.get("audit_dataset_path")
    target_column = st.session_state.get("target_column")

    if not dataset_path or not Path(str(dataset_path)).exists():
        st.error(
            "Saved uploaded dataset was not found. Please upload the CSV again and rerun the review gate.",
        )
        return

    if not target_column:
        st.error("Target column missing from session. Please rerun the audit.")
        return

    with st.spinner(
        "Human approval received. Continuing deterministic modeling, MLflow, SHAP, and report generation..."
    ):
        start_time = time.perf_counter()
        result = run_audit_workflow(
            dataset_path=str(dataset_path),
            target_column=str(target_column),
            workflow_mode="human_approved",
            human_review_decision=export_payload,
        )
        runtime_seconds = round(time.perf_counter() - start_time, 2)

    result = remove_non_serializable_objects(result)
    result["human_review_decisions"] = export_payload

    st.session_state["audit_result"] = result
    st.session_state["last_runtime_seconds"] = runtime_seconds
    st.session_state["audit_phase"] = "completed_after_human_approval"

    st.success(
        f"Modeling continued after human approval. Completed in {runtime_seconds} seconds."
    )
    st.rerun()


def show_human_gate_banner(result: dict[str, Any]) -> None:
    """Show prominent human gate status above dashboard tabs."""
    status = str(result.get("workflow_status", "")).lower()
    decision_router = result.get("decision_router", {})
    human_review = result.get("human_review", {})

    if not isinstance(decision_router, dict):
        decision_router = {}
    if not isinstance(human_review, dict):
        human_review = {}

    if status == "waiting_for_human_approval":
        review_items = normalize_review_items(human_review.get("review_items", []))
        total_items = len(review_items)
        st.markdown(
            f"""
            <div class="hitl-command">
                <h3>🧑‍💻 Human Gate Active</h3>
                <p>
                    Deterministic risk checks are complete. The workflow is paused before baseline modeling,
                    MLflow logging, SHAP, and final reporting. Review all risk items, document decisions,
                    then approve continuation.
                </p>
                <div class="hero-kpis">
                    <div class="hero-stat"><strong>{total_items}</strong><span>review items</span></div>
                    <div class="hero-stat"><strong>Paused</strong><span>modeling gate</span></div>
                    <div class="hero-stat"><strong>Required</strong><span>human approval</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        router_message = decision_router.get("message")
        if router_message:
            st.caption(router_message)
    elif result.get("human_review_decisions"):
        st.markdown(
            """
            <div class="verdict verdict-good">
                ✅ Human Approval Recorded: reviewer decisions were captured and the modeling workflow continued.
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif human_review.get("requires_human_review"):
        st.markdown(
            """
            <div class="verdict verdict-review">
                🟡 Review Recommended: audit completed, but reviewer decisions should be documented before production use.
            </div>
            """,
            unsafe_allow_html=True,
        )


def show_human_review(human_review: dict[str, Any]) -> None:
    """Show premium HITL review panel with reviewer decisions and continuation gate."""
    st.subheader("🧑‍💻 Human Review Gate")

    if not human_review:
        st.info("Human review summary not available.")
        return

    review_items = normalize_review_items(human_review.get("review_items", []))
    decisions = get_review_decision_store(review_items)
    requires_review = bool(human_review.get("requires_human_review", False))
    pending_count = sum(
        1
        for item in review_items
        if decisions.get(item["_review_key"], {}).get("decision")
        == "pending_human_review"
    )
    reviewed_count = max(0, len(review_items) - pending_count)
    progress_percent = (
        100
        if not review_items
        else round(
            (reviewed_count / len(review_items)) * 100,
            1,
        )
    )

    severity_counts: dict[str, int] = {}
    for item in review_items:
        severity = str(item.get("severity", "review")).lower()
        severity_counts[severity] = severity_counts.get(severity, 0) + 1

    st.markdown(
        f"""
        <div class="hitl-command">
            <h3>Human Risk Review Cockpit</h3>
            <p>
                The system flags possible issues only. A reviewer must decide whether each item is valid,
                acceptable for baseline experimentation, a false positive, or a blocker that needs data fixes.
            </p>
            <div class="hero-kpis">
                <div class="hero-stat"><strong>{requires_review}</strong><span>gate required</span></div>
                <div class="hero-stat"><strong>{len(review_items)}</strong><span>review items</span></div>
                <div class="hero-stat"><strong>{pending_count}</strong><span>pending</span></div>
                <div class="hero-stat"><strong>{progress_percent}%</strong><span>completed</span></div>
            </div>
            <div class="hitl-progress-track">
                <div class="hitl-progress-fill" style="width:{progress_percent}%;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    message = human_review.get("message")
    if message:
        st.info(str(message))

    if not review_items:
        st.success("No review items were generated by the deterministic audit.")
        return

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Critical", severity_counts.get("critical", 0))
    s2.metric("High", severity_counts.get("high", 0))
    s3.metric(
        "Medium",
        severity_counts.get("medium", 0) + severity_counts.get("moderate", 0),
    )
    s4.metric(
        "Low/Review", severity_counts.get("low", 0) + severity_counts.get("review", 0)
    )

    st.markdown("### Reviewer Action Panel")

    bulk_col1, bulk_col2, bulk_col3, bulk_col4 = st.columns(4)

    with bulk_col1:
        if st.button("🟢 Accept all risks", use_container_width=True):
            for item in review_items:
                decisions[item["_review_key"]]["decision"] = "accept_risk_continue"
            st.rerun()

    with bulk_col2:
        if st.button("✅ Fix later", use_container_width=True):
            for item in review_items:
                decisions[item["_review_key"]]["decision"] = "accept_flag_fix_later"
            st.rerun()

    with bulk_col3:
        if st.button("🛠️ Data fix needed", use_container_width=True):
            for item in review_items:
                decisions[item["_review_key"]]["decision"] = "needs_data_fix"
            st.rerun()

    with bulk_col4:
        if st.button("↩️ Reset", use_container_width=True):
            for item in review_items:
                decisions[item["_review_key"]] = {
                    "decision": "pending_human_review",
                    "reviewer_note": "",
                }
            st.rerun()

    filter_col1, filter_col2 = st.columns([1, 2])
    with filter_col1:
        severity_filter = st.selectbox(
            "Filter by severity",
            options=["all", "critical", "high", "medium", "moderate", "low", "review"],
            key="human_review_severity_filter",
        )
    with filter_col2:
        search_text = (
            st.text_input(
                "Search column/category/reason",
                placeholder="Example: leakage, missing, target_copy",
                key="human_review_search",
            )
            .strip()
            .lower()
        )

    decision_options = [
        "pending_human_review",
        "accept_flag_fix_later",
        "accept_risk_continue",
        "needs_data_fix",
        "reject_modeling",
        "false_positive",
    ]

    severity_emoji = {
        "critical": "🔴",
        "high": "🟠",
        "medium": "🟡",
        "moderate": "🟡",
        "review": "🔵",
        "low": "🟢",
    }

    visible_items = []
    for item in review_items:
        severity = str(item.get("severity", "review")).lower()
        searchable = " ".join(
            [
                str(item.get("category", "")),
                str(item.get("severity", "")),
                str(item.get("column", "")),
                str(item.get("reason", "")),
                str(item.get("suggested_decision", "")),
            ],
        ).lower()

        if severity_filter != "all" and severity != severity_filter:
            continue

        if search_text and search_text not in searchable:
            continue

        visible_items.append(item)

    if not visible_items:
        st.warning("No review items match the current filter.")
    else:
        st.caption(
            f"Showing {len(visible_items)} of {len(review_items)} review item(s)."
        )

    for index, item in enumerate(visible_items, start=1):
        review_key = item["_review_key"]
        severity = str(item.get("severity", "review")).lower()
        category = str(item.get("category", "review"))
        column = item.get("column")
        reason = str(item.get("reason", "Review required."))
        suggested = str(item.get("suggested_decision", "review_before_modeling"))
        current_decision = decisions.get(review_key, {}).get(
            "decision",
            "pending_human_review",
        )
        if current_decision not in decision_options:
            current_decision = "pending_human_review"

        css_severity = (
            severity
            if severity
            in {
                "critical",
                "high",
                "medium",
                "moderate",
                "low",
                "review",
            }
            else "unknown"
        )
        column_text = (
            "No specific column" if column in {None, "", "None"} else str(column)
        )

        st.markdown(
            f"""
            <div class="risk-card risk-{css_severity}">
                <div class="risk-title">{severity_emoji.get(severity, "🔵")} {category} · {severity.upper()}</div>
                <div class="risk-meta">Column: {column_text} · Suggested: {suggested}</div>
                <div class="risk-reason">{reason}</div>
                <span class="decision-chip">{get_decision_label(current_decision)}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander(
            f"Review controls for {category} · {column_text}", expanded=index <= 2
        ):
            control_left, control_right = st.columns([1.35, 1])

            with control_left:
                selected_decision = st.selectbox(
                    "Reviewer decision",
                    options=decision_options,
                    index=decision_options.index(current_decision),
                    format_func=get_decision_label,
                    key=f"review_decision_{review_key}",
                )

                reviewer_note = st.text_area(
                    "Reviewer note / domain justification",
                    value=decisions.get(review_key, {}).get("reviewer_note", ""),
                    placeholder=(
                        "Example: This column is available before prediction time, "
                        "so it is acceptable for baseline experimentation only."
                    ),
                    key=f"review_note_{review_key}",
                    height=105,
                )

                decisions[review_key] = {
                    "decision": selected_decision,
                    "reviewer_note": reviewer_note.strip(),
                }

            with control_right:
                st.metric("Severity", severity.upper())
                st.metric("Category", category)
                st.caption(f"Suggested decision: `{suggested}`")

            show_raw_item = st.checkbox(
                "Show raw review item",
                key=f"show_raw_review_item_{review_key}",
            )

            if show_raw_item:
                st.json(
                    {key: value for key, value in item.items() if key != "_review_key"}
                )

    pending_review = is_human_review_pending(review_items, decisions)
    pending_count = sum(
        1
        for item in review_items
        if decisions.get(item["_review_key"], {}).get("decision")
        == "pending_human_review"
    )

    st.markdown(
        """
        <div class="gate-final">
            <h3 style="margin-top:0;">Final Human Gate</h3>
            <p style="color:#475569;line-height:1.58;margin-bottom:.25rem;">
                Final approval controls whether the workflow can continue to baseline modeling,
                MLflow tracking, SHAP/explainability, and final reporting.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if pending_review:
        st.error(
            f"{pending_count} review item(s) are still pending. Decide every item before continuing modeling.",
        )
        final_options = ["not_ready_pending_review"]
    else:
        st.success("All review items have a human decision.")
        final_options = [
            "approved_for_baseline_experiment_only",
            "approved_with_known_risks",
            "pause_and_fix_data_first",
            "reject_modeling_until_fixed",
        ]

    current_final = st.session_state.get("final_human_review_decision")
    if current_final not in final_options:
        current_final = final_options[0]

    final_decision = st.radio(
        "Final human decision",
        options=final_options,
        index=final_options.index(current_final),
        format_func=lambda value: value.replace("_", " ").title(),
        horizontal=False,
        key="final_human_review_decision",
    )

    export_payload = build_human_review_export(
        human_review=human_review,
        review_items=review_items,
        decisions=decisions,
        final_decision=final_decision,
    )
    export_payload["approved_for_modeling"] = is_final_decision_approved(final_decision)

    st.session_state["human_review_export"] = export_payload

    if "audit_result" in st.session_state:
        st.session_state["audit_result"]["human_review_decisions"] = export_payload

    approve_col, download_col = st.columns([2, 1])

    with approve_col:
        can_continue = (
            not pending_review
            and is_final_decision_approved(final_decision)
            and str(st.session_state.get("audit_phase", "")).lower()
            == "waiting_for_human_approval"
        )
        if st.button(
            "🚀 Continue Modeling After Human Approval",
            type="primary",
            use_container_width=True,
            disabled=not can_continue,
        ):
            continue_modeling_after_human_review(export_payload)

        if not can_continue:
            st.caption(
                "Unlocks only when all items are reviewed, final decision approves continuation, and workflow is paused at the human gate.",
            )

    with download_col:
        st.download_button(
            label="Download Review JSON",
            data=json.dumps(export_payload, indent=2, ensure_ascii=False),
            file_name="human_review_decisions.json",
            mime="application/json",
            use_container_width=True,
        )

    with st.expander("Reviewed Items Summary", expanded=not pending_review):
        reviewed_df = pd.DataFrame(export_payload["reviewed_items"])
        st.dataframe(reviewed_df, use_container_width=True)


def show_ai_report(report_text: str) -> None:
    """Show generated audit report."""
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
    """Show report/result download buttons."""
    st.subheader("⬇️ Downloads")

    report_text = result.get("audit_report", "")
    json_text = to_json_download(result)

    baseline_results = result.get("baseline_results", {})
    baseline_csv = ""

    if isinstance(baseline_results, dict) and baseline_results.get("results"):
        rows = []
        for model_name, metrics in baseline_results["results"].items():
            row = {"Model": model_name}
            if isinstance(metrics, dict):
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
    """Build compact audit chat context."""
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
    """Answer a grounded question about the audit."""
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
    """Show audit Q&A chat."""
    st.subheader("💬 Ask AI About This Audit")

    suggested_questions = [
        "Why was this primary metric recommended?",
        "Which leakage risks should I review first?",
        "Is this dataset ready for final model training?",
        "Which baseline model performed best and why?",
        "What should I improve before tuning models?",
    ]

    cols = st.columns(len(suggested_questions))
    for index, question in enumerate(suggested_questions):
        with cols[index]:
            if st.button(
                question, key=f"suggested_q_{index}", use_container_width=True
            ):
                st.session_state["pending_question"] = question

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    st.session_state["chat_history"] = st.session_state["chat_history"][-10:]

    for message in st.session_state["chat_history"]:
        role = message.get("role", "assistant")
        content = message.get("content", "")

        with st.chat_message(role):
            st.markdown(content)

    user_question = st.chat_input(
        "Ask about this audit. Example: Why is F1 Score recommended?",
    )

    pending_question = st.session_state.pop("pending_question", None)
    final_question = pending_question or user_question

    if isinstance(final_question, str) and final_question.strip():
        clean_question = final_question.strip()
        st.session_state["chat_history"].append(
            {"role": "user", "content": clean_question},
        )

        with st.chat_message("user"):
            st.markdown(clean_question)

        with st.chat_message("assistant"):
            with st.spinner("Generating grounded answer..."):
                answer = answer_audit_question(
                    audit_result=result,
                    user_question=clean_question,
                )
            st.markdown(answer)

        st.session_state["chat_history"].append(
            {"role": "assistant", "content": answer},
        )
        st.session_state["chat_history"] = st.session_state["chat_history"][-10:]

    if st.button("Clear Chat"):
        st.session_state["chat_history"] = []
        st.rerun()


def show_report(result: dict[str, Any]) -> None:
    """Show final deterministic/LLM audit report safely."""
    st.subheader("📄 Final Audit Report")

    report_candidates = [
        result.get("llm_report"),
        result.get("report"),
        result.get("final_report"),
        result.get("audit_report"),
    ]

    report_data = None
    for candidate in report_candidates:
        if candidate is None:
            continue
        if isinstance(candidate, str) and not candidate.strip():
            continue
        if isinstance(candidate, dict) and not candidate:
            continue
        if isinstance(candidate, list) and not candidate:
            continue
        report_data = candidate
        break

    if report_data is None:
        st.info("Final report is not available yet.")
        if (
            str(result.get("workflow_status", "")).lower()
            == "waiting_for_human_approval"
        ):
            st.warning(
                "Workflow is paused at the human gate. Approve continuation to generate the final report.",
            )
        return

    if isinstance(report_data, dict):
        report_text = (
            report_data.get("report_markdown")
            or report_data.get("markdown")
            or report_data.get("report")
            or report_data.get("content")
            or report_data.get("text")
        )

        if report_text:
            st.markdown(str(report_text))
        else:
            st.json(report_data)

        saved_path = report_data.get("saved_path") or report_data.get("output_path")
        if saved_path:
            st.caption(f"Saved report path: {saved_path}")

        return

    if isinstance(report_data, list):
        for item in report_data:
            if isinstance(item, dict):
                st.json(item)
            else:
                st.markdown(str(item))
        return

    st.markdown(str(report_data))


def show_result_dashboard() -> None:
    """Render full result dashboard after audit completes."""
    if "audit_result" not in st.session_state:
        return

    result = st.session_state["audit_result"]
    df_preview = st.session_state["df_preview"]
    target_column = st.session_state["target_column"]

    st.divider()

    show_verdict(result)
    show_human_gate_banner(result)
    show_top_kpi_row(result)

    runtime = st.session_state.get("last_runtime_seconds")
    if runtime:
        st.caption(f"Last audit runtime: {runtime} seconds")

    st.divider()

    tabs = st.tabs(
        [
            "🏠 Executive",
            "🧑‍💻 Human Gate",
            "🧩 Data Quality",
            "🚨 Leakage",
            "⚖️ Imbalance",
            "📌 Metrics",
            "🤖 Models",
            "🔍 Explainability",
            "🧪 MLflow",
            "💬 Audit Q&A",
            "📄 Report",
            "⬇️ Downloads",
        ],
    )

    with tabs[0]:
        show_workflow_decision(result)
        show_dataset_overview(result.get("profile", {}))
        if isinstance(df_preview, pd.DataFrame) and target_column in df_preview.columns:
            show_target_distribution(
                df_preview,
                target_column,
                chart_key="dashboard_target_distribution",
            )

    with tabs[1]:
        show_human_review(result.get("human_review", {}))

    with tabs[2]:
        show_data_quality(result.get("data_quality", {}))

    with tabs[3]:
        show_leakage(result.get("leakage", {}))

    with tabs[4]:
        show_class_imbalance(result.get("class_imbalance", {}))

    with tabs[5]:
        show_metric_recommendation(result.get("metric_recommendation", {}))

    with tabs[6]:
        show_baseline_results(result.get("baseline_results", {}))

    with tabs[7]:
        show_explainability(result.get("explainability", {}))

    with tabs[8]:
        show_mlflow_results(
            result.get("mlflow_tracking")
            or result.get("mlflow_results")
            or result.get("mlflow")
            or {},
        )

    with tabs[9]:
        show_audit_chat(result)

    with tabs[10]:
        show_report(result)

    with tabs[11]:
        show_downloads(result)


def main() -> None:
    """Run Streamlit app."""
    inject_custom_css()
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


if __name__ == "__main__":
    main()
