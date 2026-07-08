from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from src.audit.workflow import run_audit_workflow
from src.utils.config import get_config_value
from src.utils.exceptions import AuditCopilotException
from src.utils.logger import get_logger


logger = get_logger(__name__)

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


app = FastAPI(
    title="Agentic ML Audit Copilot API",
    description=(
        "Human-in-the-loop API for auditing tabular ML datasets before model training."
    ),
    version=str(get_config_value("project.version", "1.0.0")),
)


# Portfolio/demo default. Restrict this in real production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_config_value("api.cors_allow_origins", ["*"]),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def as_bool(value: Any) -> bool:
    """
    Convert config values safely into boolean.
    """
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.lower().strip() in {"true", "1", "yes", "y"}

    return bool(value)


def get_allowed_extensions() -> set[str]:
    """
    Get allowed upload extensions from config.
    """
    raw_extensions = get_config_value("api.allowed_extensions", [".csv"])

    if not isinstance(raw_extensions, list):
        return {".csv"}

    return {str(ext).lower().strip() for ext in raw_extensions}


def get_file_size_mb(file_path: Path) -> float:
    """
    Return file size in megabytes.
    """
    return file_path.stat().st_size / (1024 * 1024)


def validate_upload_metadata(file: UploadFile, target_column: str) -> str:
    """
    Validate request metadata before saving file.
    """
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must have a valid filename.",
        )

    safe_filename = Path(file.filename).name
    extension = Path(safe_filename).suffix.lower()

    if extension not in get_allowed_extensions():
        allowed = ", ".join(sorted(get_allowed_extensions()))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{extension}'. Allowed: {allowed}",
        )

    clean_target_column = target_column.strip()

    if not clean_target_column:
        raise HTTPException(
            status_code=400,
            detail="Target column is required.",
        )

    return clean_target_column


def save_upload_to_disk(file: UploadFile) -> Path:
    """
    Save uploaded file to a unique path.
    """
    safe_filename = Path(file.filename or "uploaded.csv").name
    unique_filename = f"{uuid.uuid4()}_{safe_filename}"
    file_path = UPLOAD_DIR / unique_filename

    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        try:
            file.file.close()
        except Exception:
            pass

    return file_path


def validate_saved_file(file_path: Path) -> None:
    """
    Validate saved upload size and non-empty content.
    """
    if not file_path.exists():
        raise HTTPException(
            status_code=500,
            detail="Uploaded file could not be saved.",
        )

    if file_path.stat().st_size == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    max_upload_mb = float(get_config_value("api.max_upload_mb", 25))
    file_size_mb = get_file_size_mb(file_path)

    if file_size_mb > max_upload_mb:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {max_upload_mb} MB.",
        )


def strip_non_serializable_objects(audit_result: dict[str, Any]) -> dict[str, Any]:
    """
    Remove sklearn pipelines, DataFrames, and other runtime objects from API response.
    """
    cleaned = dict(audit_result)

    baseline_results = dict(cleaned.get("baseline_results", {}))
    baseline_results.pop("trained_model_objects", None)
    baseline_results.pop("runtime_objects", None)
    cleaned["baseline_results"] = baseline_results

    cleaned.pop("df", None)

    return cleaned


def cleanup_uploaded_file(file_path: Path | None) -> None:
    """
    Delete uploaded file if cleanup is enabled in config.
    """
    should_cleanup = as_bool(get_config_value("api.cleanup_uploaded_files", True))

    if should_cleanup and file_path and file_path.exists():
        try:
            file_path.unlink(missing_ok=True)
            logger.info("Cleaned uploaded file: %s", file_path)
        except Exception as error:
            logger.warning("Failed to clean uploaded file %s: %s", file_path, error)


def map_audit_exception_to_http(error: AuditCopilotException) -> HTTPException:
    """
    Convert project exceptions to user-friendly HTTP errors.
    """
    message = str(error)
    lowered = message.lower()

    if (
        "target column" in lowered
        or "dataset is empty" in lowered
        or "must contain at least" in lowered
        or "not found" in lowered
        or "invalid" in lowered
    ):
        return HTTPException(status_code=400, detail=message)

    return HTTPException(status_code=500, detail=message)


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "message": "Agentic ML Audit Copilot API is running.",
        "docs": "/docs",
        "health": "/health",
        "human_in_the_loop": True,
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    return {
        "status": "healthy",
        "service": "agentic-ml-audit-copilot",
        "version": str(get_config_value("project.version", "1.0.0")),
    }


@app.post("/audit")
async def run_audit(
    file: UploadFile = File(...),
    target_column: str = Form(...),
) -> dict[str, Any]:
    """
    Upload a CSV dataset and run the full ML audit workflow.

    The workflow is CPU-heavy, so it runs in a threadpool to avoid blocking
    FastAPI's event loop.
    """
    file_path: Path | None = None

    try:
        logger.info("Received audit request for file=%s", file.filename)

        clean_target_column = validate_upload_metadata(file, target_column)

        file_path = save_upload_to_disk(file)
        validate_saved_file(file_path)

        logger.info("Uploaded file saved at: %s", file_path)

        audit_result = await run_in_threadpool(
            run_audit_workflow,
            dataset_path=str(file_path),
            target_column=clean_target_column,
        )

        safe_result = strip_non_serializable_objects(audit_result)

        return {
            "message": "Audit completed successfully.",
            "target_column": clean_target_column,
            "problem_type": safe_result.get("problem_type"),
            "audit_score": safe_result.get("audit_score"),
            "human_review": safe_result.get("human_review"),
            "execution_summary": safe_result.get("execution_summary"),
            "profile": safe_result.get("profile"),
            "problem_detection": safe_result.get("problem_detection"),
            "data_quality": safe_result.get("data_quality"),
            "metric_recommendation": safe_result.get("metric_recommendation"),
            "class_imbalance": safe_result.get("class_imbalance"),
            "leakage": safe_result.get("leakage"),
            "baseline_results": safe_result.get("baseline_results"),
            "explainability": safe_result.get("explainability"),
            "mlflow_results": safe_result.get("mlflow_results"),
            "audit_report": safe_result.get("audit_report"),
            "report_path": safe_result.get("report_save_result", {}).get(
                "report_path"
            ),
        }

    except HTTPException:
        raise

    except AuditCopilotException as error:
        logger.error("Audit failed: %s", error)
        raise map_audit_exception_to_http(error) from error

    except Exception as error:
        logger.exception("Unexpected API error during audit execution.")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected server error during audit execution: {error}",
        ) from error

    finally:
        cleanup_uploaded_file(file_path)


@app.post("/audit/summary")
async def run_audit_summary(
    file: UploadFile = File(...),
    target_column: str = Form(...),
) -> dict[str, Any]:
    """
    Lightweight endpoint that returns only the top-level audit summary.

    Useful for integrations that do not need full report payloads.
    """
    full_result = await run_audit(file=file, target_column=target_column)

    return {
        "message": full_result.get("message"),
        "target_column": full_result.get("target_column"),
        "problem_type": full_result.get("problem_type"),
        "audit_score": full_result.get("audit_score"),
        "human_review": full_result.get("human_review"),
        "execution_summary": full_result.get("execution_summary"),
        "leakage_summary": {
            "total_possible_leakage_risks": full_result.get("leakage", {}).get(
                "total_possible_leakage_risks", 0
            ),
            "overall_severity": full_result.get("leakage", {}).get(
                "overall_severity", "none"
            ),
        },
        "best_model": full_result.get("baseline_results", {}).get("best_model"),
    }
