from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, TypedDict

import pandas as pd
from langgraph.graph import END, StateGraph

from src.audit.baseline_models import (
    get_sample_features_for_explainability,
    strip_runtime_objects,
    train_baseline_models,
)
from src.audit.class_imbalance import detect_class_imbalance
from src.audit.data_quality import run_data_quality_audit
from src.audit.explainability import run_model_explainability
from src.audit.leakage import run_leakage_check
from src.audit.metric_recommender import recommend_metrics
from src.audit.mlflow_tracker import track_baseline_experiment
from src.audit.problem_detector import detect_problem_type
from src.audit.profiler import load_dataset, profile_dataset
from src.utils.config import get_config_value
from src.utils.exceptions import AgentWorkflowError
from src.utils.logger import get_logger


logger = get_logger(__name__)


class AuditState(TypedDict, total=False):
    """
    LangGraph state for the audit workflow.

    Important:
    - df is kept only inside workflow runtime.
    - final returned result is JSON-safe and strips runtime objects.
    """

    dataset_path: str
    target_column: str
    df: pd.DataFrame

    started_at: float
    completed_at: float
    runtime_seconds: float

    profile: dict[str, Any]
    problem_detection: dict[str, Any]
    problem_type: str
    data_quality: dict[str, Any]
    leakage: dict[str, Any]
    class_imbalance: dict[str, Any]
    metric_recommendation: dict[str, Any]
    baseline_results: dict[str, Any]
    mlflow_results: dict[str, Any]
    explainability: dict[str, Any]
    audit_report: str
    report_save_result: dict[str, Any]

    node_timings: dict[str, float]
    errors: list[dict[str, Any]]
    warnings: list[str]
    optional_failures: list[dict[str, Any]]

    audit_score: dict[str, Any]
    human_review: dict[str, Any]
    execution_summary: dict[str, Any]
    message: str


NodeFn = Callable[[AuditState], AuditState]


def as_bool(value: Any) -> bool:
    """
    Convert config values safely into boolean.
    """
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.lower().strip() in {"true", "1", "yes", "y"}

    return bool(value)


def now_seconds() -> float:
    return time.perf_counter()


def get_optional_config(path: str, default: bool) -> bool:
    return as_bool(get_config_value(path, default))


def append_warning(state: AuditState, warning: str) -> AuditState:
    warnings = list(state.get("warnings", []))
    warnings.append(warning)
    state["warnings"] = warnings
    return state


def append_error(
    state: AuditState,
    stage: str,
    error: Exception,
    fatal: bool = True,
) -> AuditState:
    errors = list(state.get("errors", []))
    errors.append(
        {
            "stage": stage,
            "fatal": fatal,
            "error_type": error.__class__.__name__,
            "message": str(error),
        }
    )
    state["errors"] = errors
    return state


def append_optional_failure(
    state: AuditState,
    stage: str,
    error: Exception,
) -> AuditState:
    failures = list(state.get("optional_failures", []))
    failures.append(
        {
            "stage": stage,
            "error_type": error.__class__.__name__,
            "message": str(error),
        }
    )
    state["optional_failures"] = failures
    return state


def timed_node(name: str, func: NodeFn) -> NodeFn:
    """
    Wrap a LangGraph node with timing and logging.
    """
    def wrapper(state: AuditState) -> AuditState:
        start = now_seconds()
        logger.info("Workflow node started: %s", name)

        try:
            updated_state = func(state)
        except Exception as error:
            append_error(state, name, error, fatal=True)
            logger.exception("Workflow node failed: %s", name)
            raise

        elapsed = round(now_seconds() - start, 4)
        timings = dict(updated_state.get("node_timings", {}))
        timings[name] = elapsed
        updated_state["node_timings"] = timings

        logger.info("Workflow node completed: %s in %.4fs", name, elapsed)
        return updated_state

    return wrapper


def initialize_state(dataset_path: str | Path, target_column: str) -> AuditState:
    """
    Create initial workflow state.
    """
    if dataset_path is None or str(dataset_path).strip() == "":
        raise AgentWorkflowError("Dataset path is required.")

    if target_column is None or str(target_column).strip() == "":
        raise AgentWorkflowError("Target column is required.")

    return {
        "dataset_path": str(dataset_path),
        "target_column": str(target_column).strip(),
        "started_at": now_seconds(),
        "node_timings": {},
        "errors": [],
        "warnings": [],
        "optional_failures": [],
    }


def load_dataset_node(state: AuditState) -> AuditState:
    dataset_path = state["dataset_path"]
    state["df"] = load_dataset(dataset_path)
    return state


def profile_node(state: AuditState) -> AuditState:
    df = state["df"]
    target_column = state["target_column"]

    state["profile"] = profile_dataset(
        df=df,
        target_column=target_column,
    )
    return state


def problem_detection_node(state: AuditState) -> AuditState:
    df = state["df"]
    target_column = state["target_column"]

    problem_info = detect_problem_type(
        df=df,
        target_column=target_column,
    )

    state["problem_detection"] = problem_info
    state["problem_type"] = str(problem_info["problem_type"])
    return state


def data_quality_node(state: AuditState) -> AuditState:
    df = state["df"]
    target_column = state["target_column"]

    state["data_quality"] = run_data_quality_audit(
        df=df,
        target_column=target_column,
    )
    return state


def leakage_node(state: AuditState) -> AuditState:
    df = state["df"]
    target_column = state["target_column"]

    state["leakage"] = run_leakage_check(
        df=df,
        target_column=target_column,
    )
    return state


def imbalance_node(state: AuditState) -> AuditState:
    df = state["df"]
    target_column = state["target_column"]
    problem_type = state["problem_type"]

    state["class_imbalance"] = detect_class_imbalance(
        df=df,
        target_column=target_column,
        problem_type=problem_type,
    )
    return state


def metric_node(state: AuditState) -> AuditState:
    problem_type = state["problem_type"]
    class_imbalance = state.get("class_imbalance", {})

    imbalance_severity = (
        class_imbalance.get("imbalance_severity")
        if class_imbalance.get("is_applicable", False)
        else None
    )

    state["metric_recommendation"] = recommend_metrics(
        problem_type=problem_type,
        imbalance_severity=imbalance_severity,
    )
    return state


def baseline_node(state: AuditState) -> AuditState:
    df = state["df"]
    target_column = state["target_column"]
    problem_type = state["problem_type"]

    state["baseline_results"] = train_baseline_models(
        df=df,
        target_column=target_column,
        problem_type=problem_type,
    )
    return state


def mlflow_node(state: AuditState) -> AuditState:
    """
    Optional MLflow tracking node.

    Failure should not break the audit.
    """
    if not get_optional_config("mlflow.enabled", True):
        state["mlflow_results"] = {
            "enabled": False,
            "message": "MLflow tracking skipped because mlflow.enabled=false.",
        }
        return state

    try:
        baseline_results = state["baseline_results"]
        sample_features = get_sample_features_for_explainability(baseline_results)

        state["mlflow_results"] = track_baseline_experiment(
            baseline_results=baseline_results,
            sample_input=sample_features,
        )

    except Exception as error:
        logger.warning("MLflow tracking failed but workflow continued: %s", error)
        append_optional_failure(state, "mlflow", error)
        state["mlflow_results"] = {
            "enabled": True,
            "error": str(error),
            "message": "MLflow tracking failed, but audit workflow continued.",
        }

    return state


def explainability_node(state: AuditState) -> AuditState:
    """
    Optional explainability node.

    Failure should not break the audit.
    """
    if not get_optional_config("explainability.enabled", False):
        state["explainability"] = {
            "enabled": False,
            "available": False,
            "message": "Explainability skipped because explainability.enabled=false.",
        }
        return state

    try:
        baseline_results = state["baseline_results"]
        sample_features = get_sample_features_for_explainability(baseline_results)

        state["explainability"] = run_model_explainability(
            baseline_results=baseline_results,
            sample_features=sample_features,
        )

    except Exception as error:
        logger.warning("Explainability failed but workflow continued: %s", error)
        append_optional_failure(state, "explainability", error)
        state["explainability"] = {
            "enabled": True,
            "available": False,
            "error": str(error),
            "message": "Explainability failed, but audit workflow continued.",
        }

    return state


def report_node(state: AuditState) -> AuditState:
    """
    Optional LLM report node with deterministic fallback.

    This imports report functions lazily to avoid hard failure when names differ
    during active refactors.
    """
    if not get_optional_config("llm.enabled", True):
        state["audit_report"] = build_deterministic_fallback_report(state)
        state["report_save_result"] = save_report_safely(state["audit_report"])
        return state

    try:
        from src.audit.llm_report import build_audit_report

        report_input = build_report_safe_results(state)
        state["audit_report"] = build_audit_report(report_input)

    except Exception as error:
        logger.warning("LLM report failed. Using deterministic fallback: %s", error)
        append_optional_failure(state, "llm_report", error)
        state["audit_report"] = build_deterministic_fallback_report(state)

    state["report_save_result"] = save_report_safely(state["audit_report"])
    return state


def save_report_safely(report: str) -> dict[str, Any]:
    """
    Save audit report using available report saver.

    Saving failure should not destroy completed audit results.
    """
    try:
        default_report_path = str(
            get_config_value("reports.default_report_path", "reports/audit_report.md")
        )

        try:
            from src.audit.llm_report import save_audit_report

            return save_audit_report(
                report=report,
                output_path=default_report_path,
            )

        except ImportError:
            report_path = Path(default_report_path)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(report, encoding="utf-8")
            return {
                "report_path": str(report_path),
                "message": "Audit report saved successfully.",
            }

    except Exception as error:
        logger.warning("Audit report saving failed: %s", error)
        return {
            "report_path": None,
            "error": str(error),
            "message": "Audit report generation completed, but saving failed.",
        }


def finalization_node(state: AuditState) -> AuditState:
    """
    Add audit score, HITL summary, execution metadata, and cleanup runtime state.
    """
    state["completed_at"] = now_seconds()
    started_at = float(state.get("started_at", state["completed_at"]))
    state["runtime_seconds"] = round(state["completed_at"] - started_at, 4)

    state["audit_score"] = calculate_audit_score(state)
    state["human_review"] = build_human_review_summary(state)
    state["execution_summary"] = build_execution_summary(state)
    state["message"] = "Full audit workflow completed successfully."

    return state


def calculate_audit_score(state: AuditState) -> dict[str, Any]:
    """
    Calculate portfolio-friendly audit readiness score.

    This is not a scientific score. It is a practical triage score:
    - data quality score from data_quality module
    - leakage severity penalty
    - imbalance penalty
    - optional failure penalty
    """
    score = 100.0
    penalties: list[dict[str, Any]] = []

    def penalty(name: str, value: float, reason: str) -> None:
        nonlocal score
        score -= value
        penalties.append(
            {
                "name": name,
                "penalty": round(float(value), 2),
                "reason": reason,
            }
        )

    data_quality = state.get("data_quality", {})
    quality_score = data_quality.get("quality_score", {})

    dq_score = quality_score.get("score")
    if isinstance(dq_score, (int, float)):
        data_quality_penalty = max(0.0, 100.0 - float(dq_score)) * 0.35
        if data_quality_penalty > 0:
            penalty(
                "data_quality",
                data_quality_penalty,
                "Dataset quality checks found issues.",
            )

    leakage = state.get("leakage", {})
    leakage_severity = str(leakage.get("overall_severity", "none")).lower()
    leakage_count = int(leakage.get("total_possible_leakage_risks", 0) or 0)

    if leakage_severity == "critical":
        penalty("leakage", 30, "Critical possible leakage risk detected.")
    elif leakage_severity == "high":
        penalty("leakage", 20, "High possible leakage risk detected.")
    elif leakage_severity in {"medium", "moderate"}:
        penalty("leakage", 10, "Possible leakage risks require review.")
    elif leakage_count > 0:
        penalty("leakage", 5, "Low leakage-risk signals found.")

    imbalance = state.get("class_imbalance", {})
    if imbalance.get("is_applicable", False):
        severity = str(imbalance.get("imbalance_severity", "low")).lower()
        if severity == "severe":
            penalty("class_imbalance", 12, "Severe class imbalance detected.")
        elif severity == "high":
            penalty("class_imbalance", 8, "High class imbalance detected.")
        elif severity == "moderate":
            penalty("class_imbalance", 4, "Moderate class imbalance detected.")

    optional_failures = state.get("optional_failures", [])
    if optional_failures:
        penalty(
            "optional_failures",
            min(10, len(optional_failures) * 3),
            "Some optional workflow modules failed.",
        )

    final_score = max(0.0, min(100.0, score))

    if final_score >= 85:
        readiness = "good_starting_point"
    elif final_score >= 70:
        readiness = "needs_review"
    elif final_score >= 50:
        readiness = "high_review_needed"
    else:
        readiness = "not_ready"

    return {
        "score": round(final_score, 2),
        "readiness": readiness,
        "penalties": penalties,
        "note": (
            "This score is a triage score, not a guarantee of model readiness. "
            "Human review is required."
        ),
    }


def build_human_review_summary(state: AuditState) -> dict[str, Any]:
    """
    Build Human-in-the-loop review checklist.

    The tool flags possible risks; humans confirm whether they are valid issues.
    """
    items: list[dict[str, Any]] = []

    leakage = state.get("leakage", {})
    for risk in leakage.get("all_risks", []) or []:
        items.append(
            {
                "category": "possible_leakage",
                "severity": risk.get("risk_level", "review"),
                "column": risk.get("column"),
                "reason": risk.get("reason"),
                "suggested_decision": "review_prediction_time_availability",
                "status": "pending_human_review",
            }
        )

    data_quality = state.get("data_quality", {})
    for action in data_quality.get("recommended_actions", []) or []:
        items.append(
            {
                "category": "data_quality",
                "severity": "review",
                "column": None,
                "reason": action,
                "suggested_decision": "accept_or_plan_fix",
                "status": "pending_human_review",
            }
        )

    problem_detection = state.get("problem_detection", {})
    if problem_detection.get("needs_human_review") or problem_detection.get(
        "requires_human_review"
    ):
        items.append(
            {
                "category": "problem_type",
                "severity": "review",
                "column": state.get("target_column"),
                "reason": problem_detection.get(
                    "reason",
                    "Problem type detection needs human review.",
                ),
                "suggested_decision": "confirm_problem_type",
                "status": "pending_human_review",
            }
        )

    audit_score = state.get("audit_score", {})
    requires_review = bool(items) or audit_score.get("readiness") != "good_starting_point"

    return {
        "human_in_the_loop": True,
        "requires_human_review": requires_review,
        "review_items_count": len(items),
        "review_items": items,
        "decision_options": [
            "approved",
            "approved_with_notes",
            "needs_fix",
            "blocked",
        ],
        "message": (
            "This audit is intentionally human-in-the-loop. "
            "The system flags possible risks but does not make final modeling decisions."
        ),
    }


def build_execution_summary(state: AuditState) -> dict[str, Any]:
    """
    Build compact execution summary for UI/API.
    """
    optional_failures = state.get("optional_failures", [])
    errors = state.get("errors", [])

    return {
        "dataset_path": state.get("dataset_path"),
        "target_column": state.get("target_column"),
        "problem_type": state.get("problem_type"),
        "runtime_seconds": state.get("runtime_seconds"),
        "node_timings": state.get("node_timings", {}),
        "fatal_errors_count": len([e for e in errors if e.get("fatal")]),
        "optional_failures_count": len(optional_failures),
        "optional_failures": optional_failures,
        "warnings": state.get("warnings", []),
    }


def build_deterministic_fallback_report(state: AuditState) -> str:
    """
    Generate a deterministic fallback report when LLM report fails/disabled.
    """
    audit_score = calculate_audit_score(state)
    best_model = state.get("baseline_results", {}).get("best_model", {})
    leakage = state.get("leakage", {})
    data_quality = state.get("data_quality", {})
    metric = state.get("metric_recommendation", {})

    lines = [
        "# Agentic ML Audit Report",
        "",
        "## Executive Summary",
        f"- Target column: `{state.get('target_column', 'N/A')}`",
        f"- Problem type: `{state.get('problem_type', 'N/A')}`",
        f"- Audit readiness: `{audit_score.get('readiness')}`",
        f"- Audit score: `{audit_score.get('score')}`",
        "",
        "## Data Quality",
        f"- Quality score: `{data_quality.get('quality_score', {}).get('score', 'N/A')}`",
        f"- Duplicate rows: `{data_quality.get('duplicate_rows', 'N/A')}`",
        "",
        "## Leakage Review",
        f"- Total possible leakage risks: `{leakage.get('total_possible_leakage_risks', 0)}`",
        f"- Overall severity: `{leakage.get('overall_severity', 'none')}`",
        "",
        "## Metric Recommendation",
        f"- Primary metric: `{metric.get('primary_metric', 'N/A')}`",
        "",
        "## Baseline Model",
        f"- Best model: `{best_model.get('model_name', 'N/A')}`",
        f"- Selection metric: `{best_model.get('selection_metric', 'N/A')}`",
        f"- Score: `{best_model.get('score', 'N/A')}`",
        "",
        "## Human Review",
        "This system flags possible risks. A human ML reviewer should confirm whether flagged columns are valid at prediction time.",
    ]

    return "\n".join(lines)


def build_report_safe_results(state: AuditState) -> dict[str, Any]:
    """
    Remove runtime-only objects from workflow output.

    This keeps API/Streamlit responses JSON-safe and avoids exposing sklearn objects.
    """
    cleaned: dict[str, Any] = dict(state)

    baseline_results = cleaned.get("baseline_results")

    if isinstance(baseline_results, dict):
        cleaned["baseline_results"] = strip_runtime_objects(baseline_results)

    cleaned.pop("df", None)
    cleaned.pop("started_at", None)
    cleaned.pop("completed_at", None)

    return cleaned


def build_audit_graph(include_report: bool = True):
    """
    Build compiled LangGraph workflow.

    The graph is deterministic-first. LLM report generation is optional and occurs
    only after Python audit results are complete.
    """
    graph = StateGraph(AuditState)

    graph.add_node("load_dataset", timed_node("load_dataset", load_dataset_node))
    graph.add_node("profile", timed_node("profile", profile_node))
    graph.add_node(
        "problem_detection",
        timed_node("problem_detection", problem_detection_node),
    )
    graph.add_node("data_quality", timed_node("data_quality", data_quality_node))
    graph.add_node("leakage", timed_node("leakage", leakage_node))
    graph.add_node("imbalance", timed_node("imbalance", imbalance_node))
    graph.add_node("metrics", timed_node("metrics", metric_node))
    graph.add_node("baseline", timed_node("baseline", baseline_node))
    graph.add_node("mlflow", timed_node("mlflow", mlflow_node))
    graph.add_node("explainability", timed_node("explainability", explainability_node))

    if include_report:
        graph.add_node("report", timed_node("report", report_node))

    graph.add_node("finalize", timed_node("finalize", finalization_node))

    graph.set_entry_point("load_dataset")
    graph.add_edge("load_dataset", "profile")
    graph.add_edge("profile", "problem_detection")
    graph.add_edge("problem_detection", "data_quality")
    graph.add_edge("data_quality", "leakage")
    graph.add_edge("leakage", "imbalance")
    graph.add_edge("imbalance", "metrics")
    graph.add_edge("metrics", "baseline")
    graph.add_edge("baseline", "mlflow")
    graph.add_edge("mlflow", "explainability")

    if include_report:
        graph.add_edge("explainability", "report")
        graph.add_edge("report", "finalize")
    else:
        graph.add_edge("explainability", "finalize")

    graph.add_edge("finalize", END)

    return graph.compile()


def run_audit_workflow(
    dataset_path: str | Path,
    target_column: str,
) -> dict[str, Any]:
    """
    Run full production-inspired ML audit workflow.

    Includes:
    - LangGraph orchestration
    - deterministic ML checks
    - baseline models
    - optional MLflow
    - optional explainability
    - optional LLM report with deterministic fallback
    - audit score
    - Human-in-the-loop review summary
    """
    try:
        logger.info("Starting full audit workflow")

        initial_state = initialize_state(dataset_path, target_column)
        graph = build_audit_graph(include_report=True)
        final_state = graph.invoke(initial_state)

        result = build_report_safe_results(final_state)

        logger.info("Full audit workflow completed successfully")
        return result

    except AgentWorkflowError:
        raise

    except Exception as error:
        logger.exception("Audit workflow failed.")
        raise AgentWorkflowError(
            "Audit workflow failed.",
            error_detail=str(error),
        ) from error


def run_audit_workflow_without_report(
    dataset_path: str | Path,
    target_column: str,
) -> dict[str, Any]:
    """
    Lightweight workflow variant useful for tests and offline smoke checks.

    Runs deterministic checks, baseline modeling, MLflow/explainability optional
    nodes, audit scoring, and HITL summary, but skips LLM report generation.
    """
    try:
        logger.info("Starting audit workflow without report")

        initial_state = initialize_state(dataset_path, target_column)
        graph = build_audit_graph(include_report=False)
        final_state = graph.invoke(initial_state)

        result = build_report_safe_results(final_state)
        result["audit_report"] = build_deterministic_fallback_report(final_state)
        result["report_save_result"] = {
            "report_path": None,
            "message": "Report generation skipped in without-report workflow.",
        }

        logger.info("Audit workflow without report completed successfully")
        return result

    except Exception as error:
        logger.exception("Audit workflow without report failed.")
        raise AgentWorkflowError(
            "Audit workflow without report failed.",
            error_detail=str(error),
        ) from error


if __name__ == "__main__":
    dataset_path = "data/sample/student_mark.csv"
    target_column = "Grade"

    output = run_audit_workflow_without_report(
        dataset_path=dataset_path,
        target_column=target_column,
    )

    printable_output = {
        key: value
        for key, value in output.items()
        if key != "audit_report"
    }

    print(printable_output)
