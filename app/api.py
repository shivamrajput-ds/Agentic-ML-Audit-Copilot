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

app = FastAPI(
    title="Agentic ML Audit Copilot API",
    description="API for auditing tabular ML datasets before model training.",
    version=str(get_config_value("project.version", "1.0.0")),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.lower().strip() in {"true", "1", "yes", "y"}

    return bool(value)


def get_file_size_mb(file_path: Path) -> float:
    return file_path.stat().st_size / (1024 * 1024)


def get_allowed_extensions() -> list[str]:
    extensions = get_config_value("api.allowed_extensions", [".csv"])

    if not isinstance(extensions, list):
        return [".csv"]

    return [str(ext).lower().strip() for ext in extensions]


def remove_non_serializable_objects(audit_result: dict[str, Any]) -> dict[str, Any]:
    """
    Remove heavy/non-serializable runtime objects from API response.
    """
    cleaned = dict(audit_result)

    baseline_results = dict(cleaned.get("baseline_results", {}))
    baseline_results.pop("trained_model_objects", None)

    cleaned["baseline_results"] = baseline_results

    return cleaned


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Agentic ML Audit Copilot API is running.",
        "docs": "/docs",
        "health": "/health",
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
    Upload a tabular dataset and run the full ML audit workflow.
    """
    file_path: Path | None = None

    try:
        logger.info("Received audit request")

        max_upload_mb = float(get_config_value("api.max_upload_mb", 25))
        allowed_extensions = get_allowed_extensions()

        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file must have a valid filename.",
            )

        safe_filename = Path(file.filename).name
        file_extension = Path(safe_filename).suffix.lower()

        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed extensions: {allowed_extensions}",
            )

        clean_target_column = target_column.strip()

        if not clean_target_column:
            raise HTTPException(
                status_code=400,
                detail="Target column is required.",
            )

        unique_filename = f"{uuid.uuid4()}_{safe_filename}"
        file_path = UPLOAD_DIR / unique_filename

        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_size_mb = get_file_size_mb(file_path)

        if file_size_mb > max_upload_mb:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum allowed size is {max_upload_mb} MB.",
            )

        logger.info("Uploaded file saved at: %s", file_path)

        audit_result = await run_in_threadpool(
            run_audit_workflow,
            dataset_path=str(file_path),
            target_column=clean_target_column,
        )

        cleaned_result = remove_non_serializable_objects(audit_result)

        response = {
            "message": "Audit completed successfully.",
            "target_column": clean_target_column,
            "problem_type": cleaned_result.get("problem_type"),
            "profile": cleaned_result.get("profile"),
            "data_quality": cleaned_result.get("data_quality"),
            "metric_recommendation": cleaned_result.get("metric_recommendation"),
            "class_imbalance": cleaned_result.get("class_imbalance"),
            "leakage": cleaned_result.get("leakage"),
            "baseline_results": cleaned_result.get("baseline_results"),
            "mlflow_results": cleaned_result.get("mlflow_results"),
            "audit_report": cleaned_result.get("audit_report"),
            "report_path": cleaned_result.get("report_save_result", {}).get(
                "report_path"
            ),
        }

        logger.info("Audit request completed successfully")
        return response

    except HTTPException:
        raise

    except AuditCopilotException as error:
        logger.exception("Audit failed.")
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error

    except Exception as error:
        logger.exception("Unexpected API error.")
        raise HTTPException(
            status_code=500,
            detail="Unexpected server error during audit execution.",
        ) from error

    finally:
        should_cleanup = as_bool(get_config_value("api.cleanup_uploaded_files", True))

        if should_cleanup and file_path and file_path.exists():
            file_path.unlink(missing_ok=True)
            logger.info("Cleaned uploaded file: %s", file_path)