from __future__ import annotations

import json
import math
import time
import uuid
from json import JSONDecodeError
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

from src.audit.workflow import (
    WORKFLOW_MODE_AUTO,
    WORKFLOW_MODE_HUMAN_APPROVED,
    WORKFLOW_MODE_HUMAN_GATE,
    run_audit_workflow,
)
from src.utils.config import get_config_value
from src.utils.exceptions import AuditCopilotException
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_UPLOAD_DIR = Path("data/uploads")
DEFAULT_MAX_UPLOAD_MB = 25.0
DEFAULT_UPLOAD_CHUNK_SIZE_BYTES = 1024 * 1024

TRUE_VALUES = {"true", "1", "yes", "y", "on"}
FALSE_VALUES = {"false", "0", "no", "n", "off"}

DEFAULT_ALLOWED_CONTENT_TYPES = {
    "text/csv",
    "application/csv",
    "application/vnd.ms-excel",
    "text/plain",
    "application/octet-stream",
}

ALLOWED_WORKFLOW_MODES = {
    WORKFLOW_MODE_AUTO,
    WORKFLOW_MODE_HUMAN_GATE,
    WORKFLOW_MODE_HUMAN_APPROVED,
}

RESPONSE_BLOCKED_KEYS = {
    "df",
    "trained_model_objects",
    "runtime_objects",
    "model_object",
    "sample_features",
    "sample_target",
    "train_features",
    "test_features",
    "label_encoder",
    "preprocessor",
}

BAD_REQUEST_KEYWORDS = {
    "target column",
    "dataset is empty",
    "must contain at least",
    "not found",
    "invalid",
    "unsupported",
    "empty",
    "parsing failed",
    "malformed",
}


OPENAPI_TAGS = [
    {
        "name": "System",
        "description": "Service health, API metadata, and workflow guidance.",
    },
    {
        "name": "Audit",
        "description": (
            "Run full or summary dataset audits. Use these endpoints when you want "
            "a direct API response from an uploaded CSV file."
        ),
    },
    {
        "name": "Human Review",
        "description": (
            "Human-in-the-loop endpoints. First run the review gate, review every "
            "risk item, then send a decision payload to continue modeling."
        ),
    },
]


app = FastAPI(
    title="Agentic ML Audit Copilot API",
    description=(
        "Production-style Human-in-the-loop API for auditing tabular ML datasets "
        "before model training. The API supports deterministic pre-training checks, "
        "risk aggregation, human approval gates, baseline modeling, explainability, "
        "MLflow tracking, and audit reports.\n\n"
        "Typical HITL flow:\n"
        "1. POST /audit/review-gate with CSV + target_column.\n"
        "2. Read human_review.review_items from the response.\n"
        "3. Build a decision payload using /human-review/decision-template.\n"
        "4. POST /audit/after-human-approval with the same CSV, target_column, and "
        "human_review_decision_json.\n\n"
        "For demo mode, /audit with workflow_mode=human_gate gives the same first-step gate behavior."
    ),
    version=str(get_config_value("project.version", "1.0.0")),
    openapi_tags=OPENAPI_TAGS,
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


def get_float_config(
    path: str,
    default: float,
    minimum: float | None = None,
) -> float:
    """Read float config values with safe fallback and optional lower bound."""
    try:
        value = float(get_config_value(path, default))
    except (TypeError, ValueError):
        value = float(default)

    if not math.isfinite(value):
        value = float(default)

    if minimum is not None:
        return max(minimum, value)

    return value


def normalize_list(value: Any, default: list[str]) -> list[str]:
    """Normalize config values that should be string lists."""
    if isinstance(value, list):
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        return cleaned or default

    if isinstance(value, str) and value.strip():
        cleaned = [item.strip() for item in value.split(",") if item.strip()]
        return cleaned or default

    return default


def resolve_upload_dir() -> Path:
    """Resolve and create upload directory."""
    raw_upload_dir = get_config_value("api.upload_dir", str(DEFAULT_UPLOAD_DIR))
    upload_dir = Path(str(raw_upload_dir)).expanduser()

    if not upload_dir.is_absolute():
        upload_dir = Path.cwd() / upload_dir

    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


def get_cors_origins() -> list[str]:
    """
    Read CORS origins from config.

    For portfolio/demo mode, wildcard origins are acceptable. In production,
    configure exact frontend domains and keep credentials disabled when wildcard
    origins are used.
    """
    origins = normalize_list(
        get_config_value("api.cors_allow_origins", ["*"]),
        default=["*"],
    )
    return origins or ["*"]


_cors_origins = get_cors_origins()
_cors_allows_wildcard = "*" in _cors_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=not _cors_allows_wildcard,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_metadata(request: Request, call_next: Any) -> Any:
    """Add request id and process time to every API response."""
    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()
    request.state.request_id = request_id

    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Unhandled request failure. request_id=%s", request_id)
        raise

    process_time = round(time.perf_counter() - start_time, 4)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.exception_handler(AuditCopilotException)
async def audit_exception_handler(
    request: Request,
    error: AuditCopilotException,
) -> JSONResponse:
    """Return consistent JSON for project-level exceptions."""
    request_id = getattr(request.state, "request_id", None)
    http_error = map_audit_exception_to_http(error)

    content: dict[str, Any] = {
        "error": True,
        "request_id": request_id,
        "error_type": error.__class__.__name__,
        "detail": http_error.detail,
    }

    expose_detail = as_bool(get_config_value("api.expose_error_detail", False))
    error_detail = getattr(error, "error_detail", None)
    if expose_detail and error_detail:
        content["error_detail"] = error_detail

    return JSONResponse(status_code=http_error.status_code, content=content)


@app.exception_handler(HTTPException)
async def http_exception_handler(
    request: Request,
    error: HTTPException,
) -> JSONResponse:
    """Return consistent JSON for FastAPI HTTP exceptions."""
    request_id = getattr(request.state, "request_id", None)
    return JSONResponse(
        status_code=error.status_code,
        content={
            "error": True,
            "request_id": request_id,
            "error_type": "HTTPException",
            "detail": error.detail,
        },
        headers=getattr(error, "headers", None),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request,
    error: Exception,
) -> JSONResponse:
    """Return a safe response for unexpected server errors."""
    request_id = getattr(request.state, "request_id", None)
    logger.exception("Unhandled API error. request_id=%s error=%s", request_id, error)
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "request_id": request_id,
            "error_type": "InternalServerError",
            "detail": (
                "Unexpected server error. Check server logs with the X-Request-ID "
                "header for details."
            ),
        },
    )


def get_allowed_extensions() -> set[str]:
    """Get allowed upload extensions from config."""
    raw_extensions = get_config_value("api.allowed_extensions", [".csv"])
    extensions = normalize_list(raw_extensions, default=[".csv"])

    allowed = {
        extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        for extension in extensions
    }
    return allowed or {".csv"}


def get_allowed_content_types() -> set[str]:
    """
    Return allowed MIME types for CSV uploads.

    Browser MIME values vary, so the default is intentionally permissive. Enable
    api.strict_content_type_check=true to reject unexpected values.
    """
    raw_types = get_config_value(
        "api.allowed_content_types",
        sorted(DEFAULT_ALLOWED_CONTENT_TYPES),
    )
    allowed_types = {
        item.lower().strip()
        for item in normalize_list(
            raw_types,
            default=sorted(DEFAULT_ALLOWED_CONTENT_TYPES),
        )
    }
    return allowed_types or set(DEFAULT_ALLOWED_CONTENT_TYPES)


def sanitize_filename(filename: str | None) -> str:
    """Return safe basename for uploaded file."""
    safe_filename = Path(filename or "uploaded.csv").name.strip()
    return safe_filename or "uploaded.csv"


def get_file_size_mb(file_path: Path) -> float:
    """Return file size in megabytes."""
    return round(file_path.stat().st_size / (1024 * 1024), 4)


def get_max_upload_bytes() -> int:
    """Return max upload size in bytes."""
    max_upload_mb = get_float_config(
        "api.max_upload_mb",
        DEFAULT_MAX_UPLOAD_MB,
        minimum=0.001,
    )
    return int(max_upload_mb * 1024 * 1024)


def normalize_workflow_mode(workflow_mode: str | None) -> str:
    """Validate and normalize workflow mode."""
    clean_mode = str(workflow_mode or WORKFLOW_MODE_AUTO).strip().lower()
    if clean_mode not in ALLOWED_WORKFLOW_MODES:
        allowed = ", ".join(sorted(ALLOWED_WORKFLOW_MODES))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported workflow mode '{workflow_mode}'. Allowed: {allowed}",
        )

    return clean_mode


def parse_human_review_decision(raw_decision: str | None) -> dict[str, Any] | None:
    """Parse optional human review decision JSON from a form field."""
    if raw_decision is None or not raw_decision.strip():
        return None

    try:
        parsed = json.loads(raw_decision)
    except JSONDecodeError as error:
        raise HTTPException(
            status_code=400,
            detail="human_review_decision_json must be valid JSON.",
        ) from error

    if not isinstance(parsed, dict):
        raise HTTPException(
            status_code=400,
            detail="human_review_decision_json must decode to a JSON object.",
        )

    validate_human_review_decision_payload(parsed)
    return parsed


def validate_upload_metadata(file: UploadFile, target_column: str) -> str:
    """Validate request metadata before saving file."""
    safe_filename = sanitize_filename(file.filename)
    extension = Path(safe_filename).suffix.lower()

    if not extension:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must have an extension.",
        )

    allowed_extensions = get_allowed_extensions()
    if extension not in allowed_extensions:
        allowed = ", ".join(sorted(allowed_extensions))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{extension}'. Allowed: {allowed}",
        )

    content_type = (file.content_type or "").lower().strip()
    allowed_types = get_allowed_content_types()
    strict_content_type = as_bool(
        get_config_value("api.strict_content_type_check", False),
        default=False,
    )

    if content_type and content_type not in allowed_types:
        message = (
            f"Unexpected upload content type '{content_type}' for file "
            f"'{safe_filename}'."
        )
        if strict_content_type:
            raise HTTPException(status_code=400, detail=message)
        logger.warning(message)

    clean_target_column = str(target_column or "").strip()
    if not clean_target_column:
        raise HTTPException(status_code=400, detail="Target column is required.")

    max_target_column_chars = get_int_config(
        "api.max_target_column_chars",
        200,
        minimum=1,
    )
    if len(clean_target_column) > max_target_column_chars:
        raise HTTPException(status_code=400, detail="Target column name is too long.")

    return clean_target_column


def save_upload_to_disk(file: UploadFile) -> Path:
    """
    Save uploaded file to a unique path with streaming size enforcement.

    The size check happens while copying, so a very large upload is rejected before
    the full file is persisted on disk.
    """
    upload_dir = resolve_upload_dir()
    safe_filename = sanitize_filename(file.filename)
    unique_filename = f"{uuid.uuid4().hex}_{safe_filename}"
    file_path = upload_dir / unique_filename

    chunk_size = get_int_config(
        "api.upload_chunk_size_bytes",
        DEFAULT_UPLOAD_CHUNK_SIZE_BYTES,
        minimum=1024,
    )
    max_upload_bytes = get_max_upload_bytes()
    bytes_written = 0

    try:
        try:
            file.file.seek(0)
        except (OSError, AttributeError):
            logger.debug(
                "Could not seek uploaded file before saving: %s",
                safe_filename,
            )

        with file_path.open("wb") as buffer:
            while True:
                chunk = file.file.read(chunk_size)
                if not chunk:
                    break

                bytes_written += len(chunk)
                if bytes_written > max_upload_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "File too large. Maximum allowed size is "
                            f"{round(max_upload_bytes / (1024 * 1024), 2)} MB."
                        ),
                    )

                buffer.write(chunk)

        return file_path

    except HTTPException:
        file_path.unlink(missing_ok=True)
        raise
    except OSError as error:
        file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail="Uploaded file could not be saved.",
        ) from error
    finally:
        try:
            file.file.close()
        except OSError:
            logger.warning("Failed to close uploaded file handle: %s", safe_filename)


def validate_saved_file(file_path: Path) -> None:
    """Validate saved upload size and non-empty content."""
    if not file_path.exists():
        raise HTTPException(status_code=500, detail="Uploaded file could not be saved.")

    if not file_path.is_file():
        raise HTTPException(
            status_code=500,
            detail="Uploaded path is not a valid file.",
        )

    if file_path.stat().st_size == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    max_upload_mb = get_float_config(
        "api.max_upload_mb",
        DEFAULT_MAX_UPLOAD_MB,
        minimum=0.001,
    )
    file_size_mb = get_file_size_mb(file_path)

    if file_size_mb > max_upload_mb:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {max_upload_mb} MB.",
        )


def to_json_safe(value: Any) -> Any:
    """Recursively convert response values into JSON-safe lightweight objects."""
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if str(key) in RESPONSE_BLOCKED_KEYS:
                continue
            cleaned[str(key)] = to_json_safe(item)
        return cleaned

    if isinstance(value, list):
        return [to_json_safe(item) for item in value]

    if isinstance(value, tuple | set):
        return [to_json_safe(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, str | int | bool) or value is None:
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if hasattr(value, "shape") and hasattr(value, "columns"):
        shape = getattr(value, "shape", [])
        columns = getattr(value, "columns", [])
        return {
            "type": value.__class__.__name__,
            "shape": list(shape),
            "columns": [str(column) for column in columns],
        }

    return str(value)


def strip_non_serializable_objects(audit_result: dict[str, Any]) -> dict[str, Any]:
    """Remove sklearn pipelines, DataFrames, and runtime objects from API response."""
    safe_result = to_json_safe(audit_result)
    return safe_result if isinstance(safe_result, dict) else {}


def cleanup_uploaded_file(file_path: Path | None) -> None:
    """Delete uploaded file if cleanup is enabled in config."""
    should_cleanup = as_bool(
        get_config_value("api.cleanup_uploaded_files", True),
        default=True,
    )

    if should_cleanup and file_path and file_path.exists():
        try:
            file_path.unlink(missing_ok=True)
            logger.info("Cleaned uploaded file: %s", file_path)
        except OSError as error:
            logger.warning("Failed to clean uploaded file %s: %s", file_path, error)


def map_audit_exception_to_http(error: AuditCopilotException) -> HTTPException:
    """Convert project exceptions to user-friendly HTTP errors."""
    status_code = int(getattr(error, "status_code", 500) or 500)

    if status_code < 400 or status_code > 599:
        status_code = 500

    message = error.user_message() if hasattr(error, "user_message") else str(error)

    if status_code == 500:
        lowered = str(error).lower()
        if any(keyword in lowered for keyword in BAD_REQUEST_KEYWORDS):
            status_code = 400

    return HTTPException(status_code=status_code, detail=message)


def make_summary_response(full_result: dict[str, Any]) -> dict[str, Any]:
    """Build lightweight response from full audit result."""
    leakage = full_result.get("leakage", {}) or {}
    baseline_results = full_result.get("baseline_results", {}) or {}
    risk_aggregator = full_result.get("risk_aggregator", {}) or {}

    if not isinstance(leakage, dict):
        leakage = {}

    if not isinstance(baseline_results, dict):
        baseline_results = {}

    if not isinstance(risk_aggregator, dict):
        risk_aggregator = {}

    return {
        "message": full_result.get("message"),
        "target_column": full_result.get("target_column"),
        "problem_type": full_result.get("problem_type"),
        "workflow_status": full_result.get("workflow_status"),
        "audit_score": full_result.get("audit_score"),
        "human_review": full_result.get("human_review"),
        "execution_summary": full_result.get("execution_summary"),
        "risk_summary": {
            "has_blockers": risk_aggregator.get("has_blockers", False),
            "requires_human_review": risk_aggregator.get(
                "requires_human_review",
                False,
            ),
            "risk_items_count": risk_aggregator.get("risk_items_count", 0),
        },
        "leakage_summary": {
            "total_possible_leakage_risks": leakage.get(
                "total_possible_leakage_risks",
                0,
            ),
            "overall_severity": leakage.get("overall_severity", "none"),
        },
        "best_model": baseline_results.get("best_model"),
    }


async def execute_audit_request(
    file: UploadFile,
    target_column: str,
    workflow_mode: str = WORKFLOW_MODE_AUTO,
    human_review_decision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Shared implementation for full and summary audit endpoints."""
    file_path: Path | None = None

    try:
        clean_mode = normalize_workflow_mode(workflow_mode)
        clean_target_column = validate_upload_metadata(file, target_column)

        logger.info(
            "Received audit request file=%s target=%s workflow_mode=%s",
            file.filename,
            clean_target_column,
            clean_mode,
        )

        file_path = save_upload_to_disk(file)
        validate_saved_file(file_path)

        logger.info(
            "Uploaded file saved at: %s size_mb=%s",
            file_path,
            get_file_size_mb(file_path),
        )

        audit_result = await run_in_threadpool(
            run_audit_workflow,
            dataset_path=str(file_path),
            target_column=clean_target_column,
            workflow_mode=clean_mode,
            human_review_decision=human_review_decision,
        )

        safe_result = strip_non_serializable_objects(audit_result)
        report_save_result = safe_result.get("report_save_result", {})
        if not isinstance(report_save_result, dict):
            report_save_result = {}

        return {
            "message": "Audit completed successfully.",
            "target_column": clean_target_column,
            "problem_type": safe_result.get("problem_type"),
            "workflow_status": safe_result.get("workflow_status"),
            "workflow_mode": clean_mode,
            "audit_score": safe_result.get("audit_score"),
            "human_review": safe_result.get("human_review"),
            "execution_summary": safe_result.get("execution_summary"),
            "profile": safe_result.get("profile"),
            "problem_detection": safe_result.get("problem_detection"),
            "data_quality": safe_result.get("data_quality"),
            "parallel_audit": safe_result.get("parallel_audit"),
            "risk_aggregator": safe_result.get("risk_aggregator"),
            "decision_router": safe_result.get("decision_router"),
            "metric_recommendation": safe_result.get("metric_recommendation"),
            "class_imbalance": safe_result.get("class_imbalance"),
            "leakage": safe_result.get("leakage"),
            "baseline_results": safe_result.get("baseline_results"),
            "explainability": safe_result.get("explainability"),
            "mlflow_results": safe_result.get("mlflow_results"),
            "audit_report": safe_result.get("audit_report"),
            "report_path": report_save_result.get("report_path"),
        }

    except HTTPException:
        raise
    except AuditCopilotException as error:
        logger.error("Audit failed: %s", error)
        raise map_audit_exception_to_http(error) from error
    finally:
        cleanup_uploaded_file(file_path)


def get_api_capabilities() -> dict[str, Any]:
    """Return API capabilities for docs/root responses."""
    return {
        "supported_file_types": sorted(get_allowed_extensions()),
        "max_upload_mb": round(get_max_upload_bytes() / (1024 * 1024), 2),
        "workflow_modes": {
            WORKFLOW_MODE_AUTO: (
                "Config-driven routing. Good for backward-compatible demo usage."
            ),
            WORKFLOW_MODE_HUMAN_GATE: (
                "Run deterministic checks and pause before modeling when human review is required."
            ),
            WORKFLOW_MODE_HUMAN_APPROVED: (
                "Continue modeling after a reviewer sends a valid human review decision payload."
            ),
        },
        "hitl_steps": [
            "POST /audit/review-gate with file and target_column.",
            "Inspect human_review.review_items in the response.",
            "Create reviewer decisions for every item.",
            "Set final_human_decision to an approval value.",
            "POST /audit/after-human-approval with the same CSV and human_review_decision_json.",
        ],
        "approval_final_decisions": [
            "approved_for_baseline_experiment_only",
            "approved_with_known_risks",
        ],
        "blocking_final_decisions": [
            "pause_and_fix_data_first",
            "reject_modeling_until_fixed",
            "not_ready_pending_review",
        ],
    }


def get_human_review_decision_template() -> dict[str, Any]:
    """Return a reusable decision payload template for HITL clients."""
    return {
        "requires_human_review": True,
        "review_items_count": 1,
        "pending_items_count": 0,
        "final_human_decision": "approved_for_baseline_experiment_only",
        "approved_for_modeling": True,
        "reviewed_items": [
            {
                "category": "possible_leakage",
                "severity": "high",
                "column": "target_copy",
                "reason": "Column appears to duplicate or strongly proxy the target.",
                "suggested_decision": "review_before_modeling",
                "status": "pending_human_review",
                "human_decision": "accept_risk_continue",
                "reviewer_note": (
                    "Reviewer accepts this risk for baseline experimentation only. "
                    "This column should be removed before production modeling."
                ),
            },
        ],
    }


def validate_human_review_decision_payload(payload: dict[str, Any]) -> None:
    """Validate minimum HITL decision payload structure."""
    final_decision = str(payload.get("final_human_decision", "")).strip()
    reviewed_items = payload.get("reviewed_items", [])
    approved_for_modeling = bool(payload.get("approved_for_modeling", False))

    valid_final_decisions = {
        "approved_for_baseline_experiment_only",
        "approved_with_known_risks",
        "pause_and_fix_data_first",
        "reject_modeling_until_fixed",
        "not_ready_pending_review",
    }
    approval_final_decisions = {
        "approved_for_baseline_experiment_only",
        "approved_with_known_risks",
    }

    if final_decision not in valid_final_decisions:
        raise HTTPException(
            status_code=400,
            detail=(
                "human_review_decision_json.final_human_decision is invalid. "
                "Use /human-review/decision-template for the expected structure."
            ),
        )

    if not isinstance(reviewed_items, list):
        raise HTTPException(
            status_code=400,
            detail="human_review_decision_json.reviewed_items must be a list.",
        )

    try:
        pending_items_count = int(payload.get("pending_items_count", 0))
    except (TypeError, ValueError) as error:
        raise HTTPException(
            status_code=400,
            detail="human_review_decision_json.pending_items_count must be an integer.",
        ) from error

    if pending_items_count > 0 and approved_for_modeling:
        raise HTTPException(
            status_code=400,
            detail="Cannot approve modeling while pending_items_count is non-zero.",
        )

    if final_decision in approval_final_decisions and not approved_for_modeling:
        raise HTTPException(
            status_code=400,
            detail=(
                "approved_for_modeling must be true when final_human_decision approves continuation."
            ),
        )


@app.get(
    "/",
    tags=["System"],
    summary="API welcome and endpoint map",
)
def root() -> dict[str, Any]:
    """Return API landing information and key endpoint links."""
    return {
        "message": "Agentic ML Audit Copilot API is running.",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
        "metadata": "/metadata",
        "workflow_guide": "/workflow-guide",
        "human_review_template": "/human-review/decision-template",
        "audit_endpoint": "/audit",
        "summary_endpoint": "/audit/summary",
        "review_gate_endpoint": "/audit/review-gate",
        "approval_endpoint": "/audit/after-human-approval",
        "human_in_the_loop": True,
        "capabilities": get_api_capabilities(),
    }


@app.get(
    "/health",
    tags=["System"],
    summary="Health check",
)
def health_check() -> dict[str, Any]:
    """Return service health and version."""
    return {
        "status": "healthy",
        "service": "agentic-ml-audit-copilot",
        "version": str(get_config_value("project.version", "1.0.0")),
    }


@app.get(
    "/metadata",
    tags=["System"],
    summary="API metadata and limits",
)
def metadata() -> dict[str, Any]:
    """Return upload limits, workflow modes, and API capabilities."""
    return {
        "service": "agentic-ml-audit-copilot",
        "version": str(get_config_value("project.version", "1.0.0")),
        "capabilities": get_api_capabilities(),
    }


@app.get(
    "/workflow-guide",
    tags=["Human Review"],
    summary="Human-in-the-loop workflow guide",
)
def workflow_guide() -> dict[str, Any]:
    """Explain how to use the Human Review Gate workflow."""
    return {
        "title": "Human-in-the-loop audit workflow",
        "why_human_review_exists": (
            "The API flags possible risks only. A reviewer decides whether each item is "
            "valid, acceptable for baseline experimentation, a false positive, or a blocker."
        ),
        "step_1_review_gate": {
            "endpoint": "POST /audit/review-gate",
            "form_fields": {
                "file": "CSV dataset upload.",
                "target_column": "Target column name to audit.",
            },
            "expected_response": (
                "workflow_status='waiting_for_human_approval' when review is required. "
                "Read human_review.review_items."
            ),
        },
        "step_2_decision_payload": {
            "endpoint": "GET /human-review/decision-template",
            "instruction": (
                "Create reviewed_items with human_decision and reviewer_note for every review item."
            ),
            "approval_values": [
                "approved_for_baseline_experiment_only",
                "approved_with_known_risks",
            ],
        },
        "step_3_continue_after_approval": {
            "endpoint": "POST /audit/after-human-approval",
            "form_fields": {
                "file": "Send the same CSV again because the API is stateless.",
                "target_column": "Same target column.",
                "human_review_decision_json": (
                    "JSON string containing final_human_decision, approved_for_modeling, "
                    "pending_items_count, and reviewed_items."
                ),
            },
            "result": (
                "Runs metric recommendation, baseline models, MLflow tracking, explainability, "
                "and final report when approved."
            ),
        },
    }


@app.get(
    "/human-review/decision-template",
    tags=["Human Review"],
    summary="Human review decision JSON template",
)
def human_review_decision_template() -> dict[str, Any]:
    """Return a copy-paste-ready JSON payload for after-human-approval requests."""
    return {
        "message": (
            "Use this JSON as the human_review_decision_json form field after reviewing "
            "the review_items returned by /audit/review-gate."
        ),
        "template": get_human_review_decision_template(),
    }


@app.get(
    "/audit/modes",
    tags=["Audit"],
    summary="Available audit workflow modes",
)
def audit_modes() -> dict[str, Any]:
    """Return supported workflow modes and when to use each one."""
    return get_api_capabilities()["workflow_modes"]


@app.post(
    "/audit",
    tags=["Audit"],
    summary="Run full audit workflow",
    description=(
        "Upload a CSV and run the audit. Use workflow_mode='human_gate' to pause "
        "before modeling, or workflow_mode='human_approved' with a decision payload "
        "to continue after human approval."
    ),
)
async def run_audit(
    file: Annotated[
        UploadFile,
        File(description="CSV dataset file."),
    ],
    target_column: Annotated[
        str,
        Form(description="Name of the target column in the CSV."),
    ],
    workflow_mode: Annotated[
        str,
        Form(description="Workflow mode: auto, human_gate, or human_approved."),
    ] = WORKFLOW_MODE_AUTO,
    human_review_decision_json: Annotated[
        str | None,
        Form(
            description=(
                "Optional JSON string. Required when workflow_mode='human_approved'. "
                "Use GET /human-review/decision-template for the expected structure."
            ),
        ),
    ] = None,
) -> dict[str, Any]:
    """
    Upload a CSV dataset and run the ML audit workflow.

    workflow_mode options:
    - auto: backward-compatible config-driven routing.
    - human_gate: pause before modeling when review items exist.
    - human_approved: continue only with a human review decision payload.
    """
    normalized_mode = normalize_workflow_mode(workflow_mode)
    human_review_decision = parse_human_review_decision(human_review_decision_json)
    if (
        normalized_mode == WORKFLOW_MODE_HUMAN_APPROVED
        and human_review_decision is None
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "human_review_decision_json is required when workflow_mode='human_approved'. "
                "Use /human-review/decision-template for the expected structure."
            ),
        )

    return await execute_audit_request(
        file=file,
        target_column=target_column,
        workflow_mode=normalized_mode,
        human_review_decision=human_review_decision,
    )


@app.post(
    "/audit/review-gate",
    tags=["Human Review"],
    summary="Run audit and pause at human review gate",
    description=(
        "First HITL step. Upload a CSV and target column. The API runs deterministic "
        "quality/leakage/imbalance checks and returns human_review.review_items. "
        "Modeling is paused until approval."
    ),
)
async def run_audit_review_gate_endpoint(
    file: Annotated[
        UploadFile,
        File(description="CSV dataset file."),
    ],
    target_column: Annotated[
        str,
        Form(description="Target column to audit."),
    ],
) -> dict[str, Any]:
    """Run deterministic checks and pause at the human gate when review is needed."""
    return await execute_audit_request(
        file=file,
        target_column=target_column,
        workflow_mode=WORKFLOW_MODE_HUMAN_GATE,
    )


@app.post(
    "/audit/after-human-approval",
    tags=["Human Review"],
    summary="Continue workflow after human approval",
    description=(
        "Second HITL step. Send the same CSV again plus the human review decision JSON. "
        "When approved, the API continues to metrics, baseline models, MLflow, "
        "explainability, and report generation."
    ),
)
async def run_audit_after_human_approval_endpoint(
    file: Annotated[
        UploadFile,
        File(description="Same CSV dataset used in /audit/review-gate."),
    ],
    target_column: Annotated[
        str,
        Form(description="Same target column used in /audit/review-gate."),
    ],
    human_review_decision_json: Annotated[
        str,
        Form(
            description=(
                "JSON string with final_human_decision, approved_for_modeling, "
                "pending_items_count, and reviewed_items. See /human-review/decision-template."
            ),
        ),
    ],
) -> dict[str, Any]:
    """
    Continue the workflow after explicit human approval.

    Because this API is stateless, the client should send the dataset again along
    with the human review decision JSON exported from the review gate.
    """
    human_review_decision = parse_human_review_decision(human_review_decision_json)
    if human_review_decision is None:
        raise HTTPException(
            status_code=400,
            detail="human_review_decision_json is required for approval workflow.",
        )

    return await execute_audit_request(
        file=file,
        target_column=target_column,
        workflow_mode=WORKFLOW_MODE_HUMAN_APPROVED,
        human_review_decision=human_review_decision,
    )


@app.post(
    "/audit/summary",
    tags=["Audit"],
    summary="Run audit and return lightweight summary",
    description=(
        "Upload a CSV and receive only top-level audit status, score, human review, "
        "risk summary, leakage summary, and best model metadata."
    ),
)
async def run_audit_summary(
    file: Annotated[
        UploadFile,
        File(description="CSV dataset file."),
    ],
    target_column: Annotated[
        str,
        Form(description="Name of the target column in the CSV."),
    ],
    workflow_mode: Annotated[
        str,
        Form(description="Workflow mode: auto, human_gate, or human_approved."),
    ] = WORKFLOW_MODE_AUTO,
) -> dict[str, Any]:
    """Upload a CSV dataset and return only the top-level audit summary."""
    full_result = await execute_audit_request(
        file=file,
        target_column=target_column,
        workflow_mode=workflow_mode,
    )
    return make_summary_response(full_result)
