from pathlib import Path
import shutil
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool
from src.audit.workflow import run_audit_workflow
from src.utils.config import get_config_value
from src.utils.exceptions import AuditCopilotException
from src.utils.logger import get_logger


logger = get_logger(__name__)


app = FastAPI(
    title="Agentic ML Audit Copilot API",
    description="API for auditing tabular ML datasets before model training.",
    version="1.0.0",
)


UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/")
def root() -> dict[str, str]:
    """
    Root endpoint.
    """
    return {
        "message": "Agentic ML Audit Copilot API is running.",
        "docs": "/docs",
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    """
    Health check endpoint.
    """
    return {
        "status": "healthy",
        "service": "agentic-ml-audit-copilot",
    }


@app.post("/audit")
async def run_audit(
    file: UploadFile = File(...),
    target_column: str = Form(...),
) -> dict:
    """
    Upload a CSV dataset and run the full ML audit workflow.
    """
    file_path: Path | None = None

    try:
        logger.info("Received audit request")

        max_upload_mb = float(get_config_value("api.max_upload_mb", 25))
        cleanup_uploaded_files = bool(
            get_config_value("api.cleanup_uploaded_files", True)
        )

        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file must have a valid filename.",
            )

        if not file.filename.lower().endswith(".csv"):
            raise HTTPException(
                status_code=400,
                detail="Only CSV files are supported currently.",
            )

        if not target_column or not target_column.strip():
            raise HTTPException(
                status_code=400,
                detail="Target column is required.",
            )

        safe_filename = Path(file.filename).name
        unique_filename = f"{uuid.uuid4()}_{safe_filename}"
        file_path = UPLOAD_DIR / unique_filename

        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_size_mb = file_path.stat().st_size / (1024 * 1024)

        if file_size_mb > max_upload_mb:
            file_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum allowed size is {max_upload_mb} MB.",
            )

        logger.info(f"Uploaded file saved at: {file_path}")

        audit_result = await run_in_threadpool(
    run_audit_workflow,
    dataset_path=str(file_path),
    target_column=target_column.strip(),
)

        baseline_results = audit_result.get("baseline_results", {}).copy()
        baseline_results.pop("trained_model_objects", None)

        response = {
            "message": "Audit completed successfully.",
            "target_column": target_column.strip(),
            "problem_type": audit_result.get("problem_type"),
            "profile": audit_result.get("profile"),
            "data_quality": audit_result.get("data_quality"),
            "metric_recommendation": audit_result.get("metric_recommendation"),
            "class_imbalance": audit_result.get("class_imbalance"),
            "leakage": audit_result.get("leakage"),
            "baseline_results": baseline_results,
            "mlflow_results": audit_result.get("mlflow_results"),
            "audit_report": audit_result.get("audit_report"),
            "report_path": audit_result.get("report_save_result", {}).get(
                "report_path"
            ),
        }

        logger.info("Audit request completed successfully")
        return response

    except HTTPException:
        raise

    except AuditCopilotException as error:
        logger.error(f"Audit failed: {error}")
        raise HTTPException(status_code=500, detail=str(error)) from error

    except Exception as error:
        logger.error(f"Unexpected API error: {error}")
        raise HTTPException(
            status_code=500,
            detail="Unexpected server error during audit execution.",
        ) from error

    finally:
        cleanup_uploaded_files = bool(
            get_config_value("api.cleanup_uploaded_files", True)
        )

        if cleanup_uploaded_files and file_path and file_path.exists():
            file_path.unlink(missing_ok=True)