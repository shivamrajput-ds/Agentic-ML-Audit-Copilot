from __future__ import annotations

from pathlib import Path
from typing import Any

from src.audit.baseline_models import (
    get_sample_features_for_explainability,
    strip_runtime_objects,
    train_baseline_models,
)
from src.audit.class_imbalance import detect_class_imbalance
from src.audit.data_quality import run_data_quality_audit
from src.audit.explainability import run_model_explainability
from src.audit.leakage import run_leakage_check
from src.audit.llm_report import build_audit_report, save_audit_report
from src.audit.metric_recommender import recommend_metrics
from src.audit.mlflow_tracker import track_baseline_experiment
from src.audit.problem_detector import detect_problem_type
from src.audit.profiler import load_dataset, profile_dataset
from src.utils.config import get_config_value
from src.utils.exceptions import AgentWorkflowError
from src.utils.logger import get_logger


logger = get_logger(__name__)


def as_bool(value: Any) -> bool:
    """
    Convert config values safely into boolean.
    """
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.lower().strip() in {"true", "1", "yes", "y"}

    return bool(value)


def run_audit_workflow(
    dataset_path: str | Path,
    target_column: str,
) -> dict[str, Any]:
    """
    Run full deterministic ML audit workflow.

    Flow:
    1. Load dataset
    2. Profile dataset
    3. Detect problem type
    4. Run data quality audit
    5. Detect leakage risks
    6. Detect class imbalance
    7. Recommend metrics
    8. Train baseline models
    9. Track MLflow experiment
    10. Run explainability
    11. Generate final audit report
    12. Save report

    Notes:
    - Python performs all ML computations.
    - LLM is only used for explanation/report writing.
    - Baseline models are sanity-check models, not final optimized models.
    """
    try:
        logger.info("Starting full audit workflow")

        if dataset_path is None or str(dataset_path).strip() == "":
            raise AgentWorkflowError("Dataset path is required.")

        if target_column is None or str(target_column).strip() == "":
            raise AgentWorkflowError("Target column is required.")

        clean_target_column = str(target_column).strip()

        df = load_dataset(dataset_path)

        profile = profile_dataset(
            df=df,
            target_column=clean_target_column,
        )

        problem_info = detect_problem_type(
            df=df,
            target_column=clean_target_column,
        )
        problem_type = problem_info["problem_type"]

        data_quality = run_data_quality_audit(
            df=df,
            target_column=clean_target_column,
        )

        leakage = run_leakage_check(
            df=df,
            target_column=clean_target_column,
        )

        class_imbalance = detect_class_imbalance(
            df=df,
            target_column=clean_target_column,
            problem_type=problem_type,
        )

        imbalance_severity = (
            class_imbalance.get("imbalance_severity")
            if class_imbalance.get("is_applicable", False)
            else None
        )

        metric_recommendation = recommend_metrics(
            problem_type=problem_type,
            imbalance_severity=imbalance_severity,
        )

        baseline_results = train_baseline_models(
            df=df,
            target_column=clean_target_column,
            problem_type=problem_type,
        )

        sample_features = get_sample_features_for_explainability(baseline_results)

        mlflow_results = run_mlflow_tracking_safely(
            baseline_results=baseline_results,
            sample_features=sample_features,
        )

        explainability = run_explainability_safely(
            baseline_results=baseline_results,
            sample_features=sample_features,
        )

        audit_results_with_runtime: dict[str, Any] = {
            "target_column": clean_target_column,
            "problem_type": problem_type,
            "dataset_path": str(dataset_path),
            "profile": profile,
            "problem_detection": problem_info,
            "data_quality": data_quality,
            "leakage": leakage,
            "class_imbalance": class_imbalance,
            "metric_recommendation": metric_recommendation,
            "baseline_results": baseline_results,
            "mlflow_results": mlflow_results,
            "explainability": explainability,
        }

        report_input = build_report_safe_results(audit_results_with_runtime)

        audit_report = build_audit_report(report_input)

        report_save_result = save_report_safely(audit_report)

        final_results = build_report_safe_results(audit_results_with_runtime)
        final_results["audit_report"] = audit_report
        final_results["report_save_result"] = report_save_result
        final_results["message"] = "Full audit workflow completed successfully."

        logger.info("Full audit workflow completed successfully")
        return final_results

    except AgentWorkflowError:
        raise

    except Exception as error:
        logger.exception("Audit workflow failed.")
        raise AgentWorkflowError(
            "Audit workflow failed.",
            error_detail=str(error),
        ) from error


def run_mlflow_tracking_safely(
    baseline_results: dict[str, Any],
    sample_features: Any,
) -> dict[str, Any]:
    """
    Run MLflow tracking if enabled.

    MLflow failures should not break the whole audit because the core audit
    findings are still useful.
    """
    try:
        mlflow_enabled = as_bool(get_config_value("mlflow.enabled", True))

        if not mlflow_enabled:
            return {
                "enabled": False,
                "message": "MLflow tracking skipped because mlflow.enabled=false.",
            }

        return track_baseline_experiment(
            baseline_results=baseline_results,
            sample_input=sample_features,
        )

    except Exception as error:
        logger.warning("MLflow tracking failed but workflow will continue: %s", error)
        return {
            "enabled": True,
            "error": str(error),
            "message": "MLflow tracking failed, but audit workflow continued.",
        }


def run_explainability_safely(
    baseline_results: dict[str, Any],
    sample_features: Any,
) -> dict[str, Any]:
    """
    Run explainability if enabled.

    Explainability failures should not break the whole audit.
    """
    try:
        explainability_enabled = as_bool(
            get_config_value("explainability.enabled", False)
        )

        if not explainability_enabled:
            return {
                "enabled": False,
                "available": False,
                "message": "Explainability skipped because explainability.enabled=false.",
            }

        return run_model_explainability(
            baseline_results=baseline_results,
            sample_features=sample_features,
        )

    except Exception as error:
        logger.warning("Explainability failed but workflow will continue: %s", error)
        return {
            "enabled": True,
            "available": False,
            "error": str(error),
            "message": "Explainability failed, but audit workflow continued.",
        }


def save_report_safely(report: str) -> dict[str, Any]:
    """
    Save audit report using config-driven path.

    Saving failure should not destroy completed audit results.
    """
    try:
        default_report_path = str(
            get_config_value("reports.default_report_path", "reports/audit_report.md")
        )

        return save_audit_report(
            report=report,
            output_path=default_report_path,
        )

    except Exception as error:
        logger.warning("Audit report saving failed: %s", error)
        return {
            "report_path": None,
            "error": str(error),
            "message": "Audit report generation completed, but saving failed.",
        }


def build_report_safe_results(audit_results: dict[str, Any]) -> dict[str, Any]:
    """
    Remove runtime-only objects from workflow output.

    This keeps API/Streamlit responses JSON-safe and avoids exposing sklearn objects.
    """
    cleaned = dict(audit_results)

    baseline_results = cleaned.get("baseline_results")

    if isinstance(baseline_results, dict):
        cleaned["baseline_results"] = strip_runtime_objects(baseline_results)

    cleaned.pop("df", None)

    return cleaned


def run_audit_workflow_without_report(
    dataset_path: str | Path,
    target_column: str,
) -> dict[str, Any]:
    """
    Lightweight workflow variant useful for tests.

    Runs deterministic checks and baseline modeling, but skips LLM report generation
    and saving. This is useful when tests should be fast or offline.
    """
    try:
        logger.info("Starting audit workflow without report")

        clean_target_column = str(target_column).strip()

        df = load_dataset(dataset_path)

        profile = profile_dataset(df, clean_target_column)
        problem_info = detect_problem_type(df, clean_target_column)
        problem_type = problem_info["problem_type"]

        data_quality = run_data_quality_audit(df, clean_target_column)
        leakage = run_leakage_check(df, clean_target_column)

        class_imbalance = detect_class_imbalance(
            df=df,
            target_column=clean_target_column,
            problem_type=problem_type,
        )

        imbalance_severity = (
            class_imbalance.get("imbalance_severity")
            if class_imbalance.get("is_applicable", False)
            else None
        )

        metric_recommendation = recommend_metrics(
            problem_type=problem_type,
            imbalance_severity=imbalance_severity,
        )

        baseline_results = train_baseline_models(
            df=df,
            target_column=clean_target_column,
            problem_type=problem_type,
        )

        sample_features = get_sample_features_for_explainability(baseline_results)

        explainability = run_explainability_safely(
            baseline_results=baseline_results,
            sample_features=sample_features,
        )

        result = {
            "target_column": clean_target_column,
            "problem_type": problem_type,
            "dataset_path": str(dataset_path),
            "profile": profile,
            "problem_detection": problem_info,
            "data_quality": data_quality,
            "leakage": leakage,
            "class_imbalance": class_imbalance,
            "metric_recommendation": metric_recommendation,
            "baseline_results": baseline_results,
            "explainability": explainability,
            "message": "Audit workflow without report completed successfully.",
        }

        return build_report_safe_results(result)

    except Exception as error:
        logger.exception("Audit workflow without report failed.")
        raise AgentWorkflowError(
            "Audit workflow without report failed.",
            error_detail=str(error),
        ) from error


if __name__ == "__main__":
    dataset_path = "data/sample/student_mark.csv"
    target_column = "Grade"

    output = run_audit_workflow(
        dataset_path=dataset_path,
        target_column=target_column,
    )

    printable_output = {
        key: value
        for key, value in output.items()
        if key != "audit_report"
    }

    print(printable_output)
