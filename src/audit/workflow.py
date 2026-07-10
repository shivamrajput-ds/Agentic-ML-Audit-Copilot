from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, TypeAlias, TypedDict, cast

import pandas as pd
from langgraph.graph import END, StateGraph
from pandas.api.types import is_scalar

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

TRUE_VALUES = {"true", "1", "yes", "y", "on"}
SEVERE_LEVELS = {"critical", "severe", "high"}
REVIEW_LEVELS = {"medium", "moderate", "review"}


class AuditState(TypedDict, total=False):
    """
    LangGraph state for Agentic ML Audit Copilot.

    Important:
    - df is kept only during workflow runtime.
    - final API/UI response is JSON-safe and strips runtime objects.
    - v2 adds parallel audit, risk aggregation, and decision routing.
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
    parallel_audit: dict[str, Any]

    risk_aggregator: dict[str, Any]
    decision_router: dict[str, Any]
    workflow_status: str
    workflow_mode: str
    human_review_decision: dict[str, Any]

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


NodeFn: TypeAlias = Callable[[AuditState], AuditState]
JsonDict: TypeAlias = dict[str, Any]

WORKFLOW_STATUS_RUNNING = "running"
WORKFLOW_STATUS_BLOCKED = "blocked_for_review"
WORKFLOW_STATUS_WAITING_FOR_APPROVAL = "waiting_for_human_approval"
WORKFLOW_STATUS_MODELING = "continue_to_modeling"

WORKFLOW_MODE_AUTO = "auto"
WORKFLOW_MODE_HUMAN_GATE = "human_gate"
WORKFLOW_MODE_HUMAN_APPROVED = "human_approved"
APPROVED_HUMAN_DECISIONS = {
    "approved_for_baseline_experiment_only",
    "approved_with_known_risks",
    "accept_risk_continue",
    "approved",
    "approved_with_notes",
}
BLOCKING_HUMAN_DECISIONS = {
    "pause_and_fix_data_first",
    "reject_modeling_until_fixed",
    "needs_fix",
    "blocked",
    "not_ready_pending_review",
}


def as_bool(value: Any) -> bool:
    """Convert config/env-like values safely into boolean."""
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() in TRUE_VALUES

    return bool(value)


def now_seconds() -> float:
    """Return monotonic timestamp for runtime measurement."""
    return time.perf_counter()


def get_optional_config(path: str, default: bool) -> bool:
    """Read optional boolean config safely."""
    try:
        return as_bool(get_config_value(path, default))
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        logger.warning(
            "Invalid boolean config for %s. Falling back to %s. Error: %s",
            path,
            default,
            error,
        )
        return default


def get_int_config(path: str, default: int) -> int:
    """Read integer config with safe fallback."""
    try:
        value = int(get_config_value(path, default))
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        logger.warning(
            "Invalid integer config for %s. Falling back to %s. Error: %s",
            path,
            default,
            error,
        )
        return default

    return value


def get_float_config(path: str, default: float) -> float:
    """Read float config with safe fallback."""
    try:
        value = float(get_config_value(path, default))
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        logger.warning(
            "Invalid float config for %s. Falling back to %s. Error: %s",
            path,
            default,
            error,
        )
        return default

    return value


def append_warning(state: AuditState, warning: str) -> AuditState:
    """Append workflow warning to state."""
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
    """Append workflow error to state."""
    errors = list(state.get("errors", []))
    errors.append(
        {
            "stage": stage,
            "fatal": fatal,
            "error_type": error.__class__.__name__,
            "message": str(error),
        },
    )
    state["errors"] = errors
    return state


def append_optional_failure(
    state: AuditState,
    stage: str,
    error: Exception,
) -> AuditState:
    """Append optional module failure to state."""
    failures = list(state.get("optional_failures", []))
    failures.append(
        {
            "stage": stage,
            "error_type": error.__class__.__name__,
            "message": str(error),
        },
    )
    state["optional_failures"] = failures
    return state


def require_state_value(state: AuditState, key: str) -> Any:
    """Return a required workflow state value with a clear runtime error."""
    state_mapping = cast(Mapping[str, Any], state)
    if key not in state_mapping:
        raise AgentWorkflowError(f"Workflow state missing required key: {key}")

    return state_mapping[key]


def require_state_str(state: AuditState, key: str) -> str:
    """Return a required workflow state value as a non-empty string."""
    value = require_state_value(state, key)
    if value is None or not str(value).strip():
        raise AgentWorkflowError(
            f"Workflow state key '{key}' must be a non-empty string."
        )

    return str(value)


def require_state_dict(state: AuditState, key: str) -> dict[str, Any]:
    """Return a required workflow state value as a dictionary."""
    value = require_state_value(state, key)
    if not isinstance(value, dict):
        raise AgentWorkflowError(f"Workflow state key '{key}' must be a dictionary.")

    return cast(dict[str, Any], value)


def require_state_dataframe(state: AuditState, key: str = "df") -> pd.DataFrame:
    """Return the runtime dataframe from workflow state."""
    value = require_state_value(state, key)
    if not isinstance(value, pd.DataFrame):
        raise AgentWorkflowError(
            f"Workflow state key '{key}' must be a pandas DataFrame."
        )

    return value


def timed_node(name: str, func: NodeFn) -> NodeFn:
    """
    Wrap a LangGraph node with timing, retry, and logging.

    Optional nodes should handle their own fallback. Required nodes use this retry
    guard before the workflow is marked failed.
    """

    def wrapper(state: AuditState) -> AuditState:
        start = now_seconds()
        max_retries = max(0, get_int_config("workflow.max_retries", 0))
        retry_sleep = max(
            0.0,
            get_float_config("workflow.retry_sleep_seconds", 0.5),
        )
        attempts = max_retries + 1
        last_error: Exception | None = None

        logger.info("Workflow node started: %s", name)

        for attempt in range(1, attempts + 1):
            try:
                updated_state = func(state)
                elapsed = round(now_seconds() - start, 4)
                timings = dict(updated_state.get("node_timings", {}))
                timings[name] = elapsed
                updated_state["node_timings"] = timings

                if attempt > 1:
                    append_warning(
                        updated_state,
                        f"Workflow node '{name}' succeeded after {attempt} attempts.",
                    )

                logger.info("Workflow node completed: %s in %.4fs", name, elapsed)
                return updated_state

            except (
                AgentWorkflowError,
                KeyError,
                TypeError,
                ValueError,
                RuntimeError,
                OSError,
            ) as error:
                last_error = error
                logger.warning(
                    "Workflow node attempt failed: %s attempt=%s/%s error=%s",
                    name,
                    attempt,
                    attempts,
                    error,
                )

                if attempt < attempts:
                    time.sleep(retry_sleep * attempt)

        if last_error is None:
            last_error = AgentWorkflowError(f"Workflow node failed: {name}")

        append_error(state, name, last_error, fatal=True)
        logger.error(
            "Workflow node failed permanently: %s",
            name,
            exc_info=(last_error.__class__, last_error, last_error.__traceback__),
        )
        raise last_error

    return wrapper


def initialize_state(
    dataset_path: str | Path,
    target_column: str,
    workflow_mode: str = WORKFLOW_MODE_AUTO,
    human_review_decision: dict[str, Any] | None = None,
) -> AuditState:
    """Create initial workflow state."""
    if dataset_path is None or not str(dataset_path).strip():
        raise AgentWorkflowError("Dataset path is required.")

    if target_column is None or not str(target_column).strip():
        raise AgentWorkflowError("Target column is required.")

    normalized_mode = str(workflow_mode or WORKFLOW_MODE_AUTO).strip().lower()
    if normalized_mode not in {
        WORKFLOW_MODE_AUTO,
        WORKFLOW_MODE_HUMAN_GATE,
        WORKFLOW_MODE_HUMAN_APPROVED,
    }:
        raise AgentWorkflowError(f"Unsupported workflow mode: {workflow_mode}.")

    return {
        "dataset_path": str(dataset_path),
        "target_column": str(target_column).strip(),
        "started_at": now_seconds(),
        "node_timings": {},
        "errors": [],
        "warnings": [],
        "optional_failures": [],
        "workflow_status": WORKFLOW_STATUS_RUNNING,
        "workflow_mode": normalized_mode,
        "human_review_decision": human_review_decision or {},
    }


def load_dataset_node(state: AuditState) -> AuditState:
    """Load dataset into workflow state."""
    dataset_path = require_state_str(state, "dataset_path")
    state["df"] = load_dataset(dataset_path)
    return state


def profile_node(state: AuditState) -> AuditState:
    """Run dataset profiling."""
    state["profile"] = profile_dataset(
        df=require_state_dataframe(state),
        target_column=require_state_str(state, "target_column"),
    )
    return state


def problem_detection_node(state: AuditState) -> AuditState:
    """Detect ML problem type."""
    problem_info = detect_problem_type(
        df=require_state_dataframe(state),
        target_column=require_state_str(state, "target_column"),
    )

    state["problem_detection"] = problem_info
    state["problem_type"] = str(problem_info["problem_type"])
    return state


def data_quality_node(state: AuditState) -> AuditState:
    """Run data quality audit."""
    state["data_quality"] = run_data_quality_audit(
        df=require_state_dataframe(state),
        target_column=require_state_str(state, "target_column"),
    )
    return state


def leakage_node(state: AuditState) -> AuditState:
    """Run possible leakage-risk checks."""
    state["leakage"] = run_leakage_check(
        df=require_state_dataframe(state),
        target_column=require_state_str(state, "target_column"),
    )
    return state


def imbalance_node(state: AuditState) -> AuditState:
    """Run class imbalance detection for classification problems."""
    problem_type = require_state_str(state, "problem_type")
    target_column = require_state_str(state, "target_column")

    if problem_type == "regression":
        state["class_imbalance"] = {
            "problem_type": problem_type,
            "target_column": target_column,
            "is_applicable": False,
            "message": "Class imbalance detection is not applicable for regression problems.",
        }
        return state

    state["class_imbalance"] = detect_class_imbalance(
        df=require_state_dataframe(state),
        target_column=target_column,
        problem_type=problem_type,
    )
    return state


def run_parallel_task(
    name: str, func: NodeFn, state: AuditState
) -> tuple[str, AuditState]:
    """
    Run one audit task for the parallel audit node.

    Each task receives a shallow state copy and returns only its own updated copy.
    The parent node merges selected output keys afterward to avoid LangGraph
    reducer conflicts.
    """
    task_state = cast(AuditState, dict(state))
    start = now_seconds()
    updated_state = func(task_state)
    elapsed = round(now_seconds() - start, 4)
    timings = dict(updated_state.get("node_timings", {}))
    timings[name] = elapsed
    updated_state["node_timings"] = timings
    return name, updated_state


def parallel_audit_node(state: AuditState) -> AuditState:
    """
    Run independent audit modules concurrently.

    This is implemented as internal parallelism inside one LangGraph node to avoid
    state merge conflicts from multiple graph branches updating the same state.
    """
    if not get_optional_config("workflow.parallel_audit_enabled", True):
        state = data_quality_node(state)
        state = leakage_node(state)
        state = imbalance_node(state)
        state["parallel_audit"] = {
            "enabled": False,
            "message": "Parallel audit disabled; checks ran sequentially.",
        }
        return state

    tasks: dict[str, NodeFn] = {
        "data_quality": data_quality_node,
        "leakage": leakage_node,
        "imbalance": imbalance_node,
    }

    max_workers = max(
        1, min(len(tasks), get_int_config("workflow.parallel_workers", 3))
    )
    completed: list[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_parallel_task, name, func, state): name
            for name, func in tasks.items()
        }

        for future in as_completed(futures):
            name = futures[future]
            try:
                task_name, task_state = future.result()
                completed.append(task_name)

                if task_name == "data_quality":
                    state["data_quality"] = require_state_dict(
                        task_state, "data_quality"
                    )
                elif task_name == "leakage":
                    state["leakage"] = require_state_dict(task_state, "leakage")
                elif task_name == "imbalance":
                    state["class_imbalance"] = require_state_dict(
                        task_state, "class_imbalance"
                    )

                timings = dict(state.get("node_timings", {}))
                timings.update(task_state.get("node_timings", {}))
                state["node_timings"] = timings

            except (
                AgentWorkflowError,
                KeyError,
                TypeError,
                ValueError,
                RuntimeError,
                OSError,
            ) as error:
                logger.exception("Parallel audit task failed: %s", name)
                append_error(state, name, error, fatal=True)
                raise error

    state["parallel_audit"] = {
        "enabled": True,
        "completed_tasks": sorted(completed),
        "max_workers": max_workers,
        "message": "Parallel audit checks completed successfully.",
    }
    return state


def metric_node(state: AuditState) -> AuditState:
    """Recommend metrics based on problem type and imbalance."""
    class_imbalance = state.get("class_imbalance", {})
    if not isinstance(class_imbalance, dict):
        class_imbalance = {}

    imbalance_severity = (
        class_imbalance.get("imbalance_severity")
        if class_imbalance.get("is_applicable", False)
        else None
    )

    state["metric_recommendation"] = recommend_metrics(
        problem_type=require_state_str(state, "problem_type"),
        imbalance_severity=imbalance_severity,
    )
    return state


def baseline_node(state: AuditState) -> AuditState:
    """Train and evaluate baseline models."""
    state["baseline_results"] = train_baseline_models(
        df=require_state_dataframe(state),
        target_column=require_state_str(state, "target_column"),
        problem_type=require_state_str(state, "problem_type"),
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

    if state.get("workflow_status") in {
        WORKFLOW_STATUS_BLOCKED,
        WORKFLOW_STATUS_WAITING_FOR_APPROVAL,
    }:
        state["mlflow_results"] = {
            "enabled": True,
            "skipped": True,
            "message": "MLflow skipped because workflow was blocked for human review.",
        }
        return state

    try:
        baseline_results = require_state_dict(state, "baseline_results")
        sample_features = get_sample_features_for_explainability(baseline_results)

        state["mlflow_results"] = track_baseline_experiment(
            baseline_results=baseline_results,
            sample_input=sample_features,
        )

    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
        OSError,
    ) as error:
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

    if state.get("workflow_status") in {
        WORKFLOW_STATUS_BLOCKED,
        WORKFLOW_STATUS_WAITING_FOR_APPROVAL,
    }:
        state["explainability"] = {
            "enabled": True,
            "available": False,
            "skipped": True,
            "message": "Explainability skipped because workflow was blocked for human review.",
        }
        return state

    try:
        baseline_results = require_state_dict(state, "baseline_results")
        sample_features = get_sample_features_for_explainability(baseline_results)

        state["explainability"] = run_model_explainability(
            baseline_results=baseline_results,
            sample_features=sample_features,
        )

    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
        OSError,
    ) as error:
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

    Report generation failure should not break deterministic audit results.
    """
    if not get_optional_config("llm.enabled", True):
        report = build_deterministic_fallback_report(state)
        state["audit_report"] = report
        state["report_save_result"] = save_report_safely(report)
        return state

    try:
        from src.audit.llm_report import build_audit_report

        report_input = build_report_safe_results(state)
        report = build_audit_report(report_input)

    except (
        ImportError,
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
        OSError,
    ) as error:
        logger.warning("LLM report failed. Using deterministic fallback: %s", error)
        append_optional_failure(state, "llm_report", error)
        report = build_deterministic_fallback_report(state)

    state["audit_report"] = report
    state["report_save_result"] = save_report_safely(report)
    return state


def get_quality_score(state: AuditState) -> float | None:
    """Extract data quality score if available."""
    data_quality = state.get("data_quality", {})
    if not isinstance(data_quality, dict):
        return None

    quality_score = data_quality.get("quality_score", {})
    if not isinstance(quality_score, dict):
        return None

    score = quality_score.get("score")
    if isinstance(score, int | float):
        return float(score)

    return None


def get_leakage_severity(state: AuditState) -> str:
    """Extract leakage severity."""
    leakage = state.get("leakage", {})
    if not isinstance(leakage, dict):
        return "none"

    return str(leakage.get("overall_severity", "none")).lower()


def get_leakage_count(state: AuditState) -> int:
    """Extract total possible leakage-risk count."""
    leakage = state.get("leakage", {})
    if not isinstance(leakage, dict):
        return 0

    try:
        return int(leakage.get("total_possible_leakage_risks", 0) or 0)
    except (TypeError, ValueError):
        return 0


def get_imbalance_severity(state: AuditState) -> str:
    """Extract class imbalance severity."""
    imbalance = state.get("class_imbalance", {})
    if not isinstance(imbalance, dict):
        return "none"

    if not imbalance.get("is_applicable", False):
        return "not_applicable"

    return str(imbalance.get("imbalance_severity", "none")).lower()


def get_final_human_decision(state: AuditState) -> str:
    """Return normalized final human decision from state."""
    decision = state.get("human_review_decision", {})
    if not isinstance(decision, dict):
        return ""

    return str(decision.get("final_human_decision", "")).strip().lower()


def is_human_approved(state: AuditState) -> bool:
    """Return whether a human has approved continuing to modeling."""
    final_decision = get_final_human_decision(state)
    if final_decision in APPROVED_HUMAN_DECISIONS:
        return True

    decision = state.get("human_review_decision", {})
    if isinstance(decision, dict):
        approved = decision.get("approved_for_modeling")
        if isinstance(approved, bool):
            return approved

    return False


def is_human_blocked(state: AuditState) -> bool:
    """Return whether a human decision explicitly blocks modeling."""
    return get_final_human_decision(state) in BLOCKING_HUMAN_DECISIONS


def requires_review_before_modeling(state: AuditState) -> bool:
    """Return whether current risk summary requires a human gate."""
    risk_summary = state.get("risk_aggregator", {})
    if not isinstance(risk_summary, dict):
        return False

    return bool(
        risk_summary.get("requires_human_review")
        or risk_summary.get("has_blockers")
        or risk_summary.get("risk_items_count", 0)
    )


def risk_aggregator_node(state: AuditState) -> AuditState:
    """
    Aggregate audit risks into one decision-ready summary.

    This node does not make the final decision. It converts module-level findings
    into a compact risk object that the decision router can use.
    """
    risk_items: list[dict[str, Any]] = []
    critical_blockers: list[str] = []
    warnings: list[str] = []

    quality_score = get_quality_score(state)
    if quality_score is not None:
        if quality_score < get_float_config(
            "workflow.critical_quality_threshold", 40.0
        ):
            critical_blockers.append("data_quality_below_critical_threshold")
            risk_items.append(
                {
                    "category": "data_quality",
                    "severity": "critical",
                    "reason": f"Data quality score is very low: {quality_score}.",
                },
            )
        elif quality_score < get_float_config(
            "workflow.review_quality_threshold", 70.0
        ):
            warnings.append("data_quality_needs_review")
            risk_items.append(
                {
                    "category": "data_quality",
                    "severity": "review",
                    "reason": f"Data quality score needs review: {quality_score}.",
                },
            )

    leakage_severity = get_leakage_severity(state)
    leakage_count = get_leakage_count(state)

    if leakage_severity in SEVERE_LEVELS:
        critical_blockers.append("severe_possible_leakage")
        risk_items.append(
            {
                "category": "possible_leakage",
                "severity": leakage_severity,
                "reason": "High-severity possible leakage risk detected.",
            },
        )
    elif leakage_count > 0 or leakage_severity in REVIEW_LEVELS:
        warnings.append("possible_leakage_needs_review")
        risk_items.append(
            {
                "category": "possible_leakage",
                "severity": leakage_severity,
                "reason": "Possible leakage risks require human review.",
            },
        )

    imbalance_severity = get_imbalance_severity(state)
    if imbalance_severity in {"severe", "high"}:
        warnings.append("class_imbalance_needs_review")
        risk_items.append(
            {
                "category": "class_imbalance",
                "severity": imbalance_severity,
                "reason": "Class imbalance may affect model evaluation.",
            },
        )

    problem_detection = state.get("problem_detection", {})
    if isinstance(problem_detection, dict) and (
        problem_detection.get("needs_human_review")
        or problem_detection.get("requires_human_review")
    ):
        warnings.append("problem_type_needs_review")
        risk_items.append(
            {
                "category": "problem_type",
                "severity": "review",
                "reason": problem_detection.get(
                    "reason",
                    "Problem type detection requires human review.",
                ),
            },
        )

    has_blockers = bool(critical_blockers)
    has_review_items = bool(risk_items)

    state["risk_aggregator"] = {
        "has_blockers": has_blockers,
        "requires_human_review": has_blockers or has_review_items,
        "critical_blockers": critical_blockers,
        "warnings": warnings,
        "risk_items": risk_items,
        "risk_items_count": len(risk_items),
        "message": (
            "Risk aggregation completed. Human review is required."
            if has_blockers or has_review_items
            else "Risk aggregation completed. No major review blockers found."
        ),
    }
    return state


def should_continue_after_risk_review(state: AuditState) -> bool:
    """
    Decide whether workflow should continue after risk aggregation.

    Modes:
    - auto: existing config-driven behavior for API/backward compatibility.
    - human_gate: stop at HITL whenever review items exist.
    - human_approved: continue only when a positive human decision is supplied.
    """
    risk_summary = state.get("risk_aggregator", {})
    if not isinstance(risk_summary, dict):
        return True

    if is_human_blocked(state):
        return False

    workflow_mode = str(state.get("workflow_mode", WORKFLOW_MODE_AUTO)).lower()

    if workflow_mode == WORKFLOW_MODE_HUMAN_APPROVED:
        return is_human_approved(state)

    if workflow_mode == WORKFLOW_MODE_HUMAN_GATE:
        return not requires_review_before_modeling(state)

    if not risk_summary.get("has_blockers", False):
        return True

    return not get_optional_config("workflow.stop_on_critical_risk", True)


def decision_router_node(state: AuditState) -> AuditState:
    """
    Create explicit HITL/router decision.

    The router can pause before modeling, continue automatically, or continue only
    after an explicit human approval payload.
    """
    risk_summary = state.get("risk_aggregator", {})
    if not isinstance(risk_summary, dict):
        risk_summary = {}

    has_blockers = bool(risk_summary.get("has_blockers", False))
    requires_review = requires_review_before_modeling(state)
    workflow_mode = str(state.get("workflow_mode", WORKFLOW_MODE_AUTO)).lower()
    human_approved = is_human_approved(state)
    human_blocked = is_human_blocked(state)
    continue_after_review = should_continue_after_risk_review(state)

    if human_blocked:
        decision = "human_rejected_modeling"
        state["workflow_status"] = WORKFLOW_STATUS_WAITING_FOR_APPROVAL
        message = (
            "Human reviewer blocked modeling. Fix data/risk items before continuing."
        )
    elif requires_review and not continue_after_review:
        decision = "wait_for_human_approval"
        state["workflow_status"] = WORKFLOW_STATUS_WAITING_FOR_APPROVAL
        message = "Human review required. Workflow paused before metric recommendation and modeling."
    else:
        decision = "continue_to_modeling"
        state["workflow_status"] = WORKFLOW_STATUS_MODELING
        if requires_review and human_approved:
            message = "Human approval received. Workflow continues to metric recommendation and baseline modeling."
        else:
            message = (
                "Workflow continues to metric recommendation and baseline modeling."
            )

    state["decision_router"] = {
        "decision": decision,
        "workflow_mode": workflow_mode,
        "has_blockers": has_blockers,
        "requires_human_review": requires_review,
        "human_approved": human_approved,
        "human_blocked": human_blocked,
        "continue_after_review": continue_after_review,
        "message": message,
    }

    return state


def route_after_decision(state: AuditState) -> str:
    """LangGraph conditional route after decision router."""
    decision_router = state.get("decision_router", {})
    if isinstance(decision_router, dict):
        decision = str(decision_router.get("decision", "continue_to_modeling"))
        if decision in {
            "stop_for_human_review",
            "wait_for_human_approval",
            "human_rejected_modeling",
        }:
            return "human_review"

    return "metrics"


def hitl_review_node(state: AuditState) -> AuditState:
    """
    Build human review output when workflow stops early.

    This is a soft HITL gate for v2 that keeps existing API/Streamlit calls simple.
    True pause/resume can be added later with LangGraph checkpointers and
    separate FastAPI resume endpoints.
    """
    state["audit_score"] = calculate_audit_score(state)
    state["human_review"] = build_human_review_summary(state)

    state["metric_recommendation"] = {
        "skipped": True,
        "message": "Metric recommendation skipped until human review is completed.",
    }
    state["baseline_results"] = {
        "skipped": True,
        "best_model": {},
        "models": [],
        "message": "Baseline modeling skipped until human review is completed.",
    }
    state["mlflow_results"] = {
        "skipped": True,
        "message": "MLflow tracking skipped because baseline modeling did not run.",
    }
    state["explainability"] = {
        "enabled": get_optional_config("explainability.enabled", False),
        "available": False,
        "skipped": True,
        "message": "Explainability skipped because baseline modeling did not run.",
    }

    append_warning(
        state,
        "Workflow stopped before modeling because critical review risks were detected.",
    )
    return state


def save_report_safely(report: str) -> dict[str, Any]:
    """
    Save audit report using available report saver.

    Saving failure should not destroy completed audit results.
    """
    try:
        default_report_path = str(
            get_config_value("reports.default_report_path", "reports/audit_report.md"),
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

    except OSError as error:
        logger.warning("Audit report saving failed: %s", error)
        return {
            "report_path": None,
            "error": str(error),
            "message": "Audit report generation completed, but saving failed.",
        }


def validate_required_state(state: AuditState) -> None:
    """Validate that required workflow outputs exist before final summary."""
    required_keys = [
        "profile",
        "problem_detection",
        "problem_type",
        "data_quality",
        "leakage",
        "class_imbalance",
        "risk_aggregator",
        "decision_router",
    ]

    if state.get("workflow_status") not in {
        WORKFLOW_STATUS_BLOCKED,
        WORKFLOW_STATUS_WAITING_FOR_APPROVAL,
    }:
        required_keys.extend(
            [
                "metric_recommendation",
                "baseline_results",
            ],
        )

    missing = [key for key in required_keys if key not in state]

    if missing:
        raise AgentWorkflowError(
            f"Workflow completed with missing required state keys: {missing}",
        )


def collect_stage_status(state: AuditState) -> list[dict[str, Any]]:
    """Build UI-friendly stage status records."""
    stage_keys = {
        "load_dataset": "df",
        "profile": "profile",
        "problem_detection": "problem_detection",
        "parallel_audit": "parallel_audit",
        "data_quality": "data_quality",
        "leakage": "leakage",
        "imbalance": "class_imbalance",
        "risk_aggregator": "risk_aggregator",
        "decision_router": "decision_router",
        "hitl_review": "human_review",
        "metrics": "metric_recommendation",
        "baseline": "baseline_results",
        "mlflow": "mlflow_results",
        "explainability": "explainability",
        "report": "audit_report",
        "finalize": "execution_summary",
    }

    timings = state.get("node_timings", {})
    optional_failures = {
        str(item.get("stage"))
        for item in state.get("optional_failures", [])
        if isinstance(item, dict)
    }

    stages: list[dict[str, Any]] = []

    for stage, key in stage_keys.items():
        if stage in optional_failures:
            status = "warning"
        elif key in state:
            status = "completed"
        else:
            status = "skipped"

        stages.append(
            {
                "stage": stage,
                "status": status,
                "runtime_seconds": timings.get(stage),
            },
        )

    return stages


def finalization_node(state: AuditState) -> AuditState:
    """Add score, HITL summary, execution metadata, and cleanup runtime state."""
    validate_required_state(state)

    completed_at = now_seconds()
    state["completed_at"] = completed_at
    started_at = float(state.get("started_at", completed_at))
    state["runtime_seconds"] = round(completed_at - started_at, 4)

    if "audit_score" not in state:
        state["audit_score"] = calculate_audit_score(state)

    if "human_review" not in state:
        state["human_review"] = build_human_review_summary(state)

    state["execution_summary"] = build_execution_summary(state)

    if state.get("workflow_status") in {
        WORKFLOW_STATUS_BLOCKED,
        WORKFLOW_STATUS_WAITING_FOR_APPROVAL,
    }:
        state["message"] = (
            "Audit paused at the human review gate. "
            "Modeling will run only after explicit human approval."
        )
    else:
        state["message"] = "Full audit workflow completed successfully."

    return state


def calculate_audit_score(state: AuditState) -> dict[str, Any]:
    """
    Calculate portfolio-friendly audit readiness score.

    This is not a scientific score. It is a practical triage score.
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
            },
        )

    dq_score = get_quality_score(state)
    if dq_score is not None:
        data_quality_penalty = max(0.0, 100.0 - dq_score) * 0.35
        if data_quality_penalty > 0:
            penalty(
                "data_quality",
                data_quality_penalty,
                "Dataset quality checks found issues.",
            )

    leakage_severity = get_leakage_severity(state)
    leakage_count = get_leakage_count(state)

    if leakage_severity == "critical":
        penalty("leakage", 30, "Critical possible leakage risk detected.")
    elif leakage_severity in {"severe", "high"}:
        penalty("leakage", 20, "High possible leakage risk detected.")
    elif leakage_severity in {"medium", "moderate"}:
        penalty("leakage", 10, "Possible leakage risks require review.")
    elif leakage_count > 0:
        penalty("leakage", 5, "Low leakage-risk signals found.")

    imbalance_severity = get_imbalance_severity(state)
    if imbalance_severity == "severe":
        penalty("class_imbalance", 12, "Severe class imbalance detected.")
    elif imbalance_severity == "high":
        penalty("class_imbalance", 8, "High class imbalance detected.")
    elif imbalance_severity == "moderate":
        penalty("class_imbalance", 4, "Moderate class imbalance detected.")

    optional_failures = state.get("optional_failures", [])
    if optional_failures:
        penalty(
            "optional_failures",
            min(10, len(optional_failures) * 3),
            "Some optional workflow modules failed.",
        )

    risk_summary = state.get("risk_aggregator", {})
    if isinstance(risk_summary, dict) and risk_summary.get("has_blockers", False):
        penalty("critical_review", 15, "Critical risks require human review.")

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

    risk_summary = state.get("risk_aggregator", {})
    if isinstance(risk_summary, dict):
        for risk in risk_summary.get("risk_items", []) or []:
            if isinstance(risk, dict):
                items.append(
                    {
                        "category": risk.get("category", "risk_review"),
                        "severity": risk.get("severity", "review"),
                        "column": risk.get("column"),
                        "reason": risk.get("reason"),
                        "suggested_decision": "review_before_modeling",
                        "status": "pending_human_review",
                    },
                )

    leakage = state.get("leakage", {})
    if isinstance(leakage, dict):
        for risk in leakage.get("all_risks", []) or []:
            if not isinstance(risk, dict):
                continue

            items.append(
                {
                    "category": "possible_leakage",
                    "severity": risk.get("risk_level", "review"),
                    "column": risk.get("column"),
                    "reason": risk.get("reason"),
                    "suggested_decision": "review_prediction_time_availability",
                    "status": "pending_human_review",
                },
            )

    data_quality = state.get("data_quality", {})
    if isinstance(data_quality, dict):
        for action in data_quality.get("recommended_actions", []) or []:
            items.append(
                {
                    "category": "data_quality",
                    "severity": "review",
                    "column": None,
                    "reason": action,
                    "suggested_decision": "accept_or_plan_fix",
                    "status": "pending_human_review",
                },
            )

    problem_detection = state.get("problem_detection", {})
    if isinstance(problem_detection, dict) and (
        problem_detection.get("needs_human_review")
        or problem_detection.get("requires_human_review")
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
            },
        )

    audit_score = state.get("audit_score", {})
    readiness = (
        audit_score.get("readiness")
        if isinstance(audit_score, dict)
        else "needs_review"
    )
    requires_review = bool(items) or readiness != "good_starting_point"

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
        "gate_status": state.get("workflow_status"),
        "workflow_mode": state.get("workflow_mode", WORKFLOW_MODE_AUTO),
        "human_approved": is_human_approved(state),
        "final_human_decision": get_final_human_decision(state),
        "next_action": (
            "review_required_before_modeling" if requires_review else "modeling_allowed"
        ),
        "message": (
            "This audit is intentionally human-in-the-loop. "
            "The system flags possible risks but does not make final modeling decisions."
        ),
    }


def build_execution_summary(state: AuditState) -> dict[str, Any]:
    """Build compact execution summary for UI/API."""
    optional_failures = state.get("optional_failures", [])
    errors = state.get("errors", [])
    stage_status = collect_stage_status(state)

    return {
        "dataset_path": state.get("dataset_path"),
        "target_column": state.get("target_column"),
        "problem_type": state.get("problem_type"),
        "workflow_status": state.get("workflow_status"),
        "decision_router": state.get("decision_router", {}),
        "runtime_seconds": state.get("runtime_seconds"),
        "node_timings": state.get("node_timings", {}),
        "fatal_errors_count": len(
            [error for error in errors if error.get("fatal")],
        ),
        "optional_failures_count": len(optional_failures),
        "optional_failures": optional_failures,
        "warnings": state.get("warnings", []),
        "stage_status": stage_status,
        "completed_stages_count": len(
            [stage for stage in stage_status if stage.get("status") == "completed"],
        ),
    }


def build_deterministic_fallback_report(state: AuditState) -> str:
    """Generate a deterministic fallback report when LLM report fails/disabled."""
    audit_score = calculate_audit_score(state)

    baseline_results = state.get("baseline_results", {})
    best_model = (
        baseline_results.get("best_model", {})
        if isinstance(baseline_results, dict)
        else {}
    )

    leakage = state.get("leakage", {})
    data_quality = state.get("data_quality", {})
    metric = state.get("metric_recommendation", {})
    risk_summary = state.get("risk_aggregator", {})
    decision_router = state.get("decision_router", {})

    if not isinstance(best_model, dict):
        best_model = {}

    if not isinstance(leakage, dict):
        leakage = {}

    if not isinstance(data_quality, dict):
        data_quality = {}

    if not isinstance(metric, dict):
        metric = {}

    if not isinstance(risk_summary, dict):
        risk_summary = {}

    if not isinstance(decision_router, dict):
        decision_router = {}

    quality_score = data_quality.get("quality_score", {})
    if not isinstance(quality_score, dict):
        quality_score = {}

    lines = [
        "# Agentic ML Audit Report",
        "",
        "## Executive Summary",
        f"- Target column: `{state.get('target_column', 'N/A')}`",
        f"- Problem type: `{state.get('problem_type', 'N/A')}`",
        f"- Workflow status: `{state.get('workflow_status', 'N/A')}`",
        f"- Router decision: `{decision_router.get('decision', 'N/A')}`",
        f"- Audit readiness: `{audit_score.get('readiness')}`",
        f"- Audit score: `{audit_score.get('score')}`",
        "",
        "## Risk Aggregation",
        f"- Requires human review: `{risk_summary.get('requires_human_review', 'N/A')}`",
        f"- Critical blockers: `{risk_summary.get('critical_blockers', [])}`",
        f"- Risk items: `{risk_summary.get('risk_items_count', 0)}`",
        "",
        "## Data Quality",
        f"- Quality score: `{quality_score.get('score', 'N/A')}`",
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
        (
            "This system flags possible risks. A human ML reviewer should confirm "
            "whether flagged columns are valid at prediction time."
        ),
    ]

    return "\n".join(lines)


def make_json_safe(value: Any) -> Any:
    """Recursively convert common runtime values into JSON-safe objects."""
    if value is None or isinstance(value, str | int | float | bool):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, pd.Timedelta):
        return str(value)

    if is_scalar(value):
        try:
            return value.item()
        except AttributeError:
            return value

    if isinstance(value, Mapping):
        return {str(key): make_json_safe(item) for key, item in value.items()}

    if isinstance(value, list):
        return [make_json_safe(item) for item in value]

    if isinstance(value, tuple):
        return [make_json_safe(item) for item in value]

    if isinstance(value, set):
        return sorted(make_json_safe(item) for item in value)

    return str(value)


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

    return make_json_safe(cleaned)


def build_audit_graph(include_report: bool = True) -> Any:
    """
    Build compiled LangGraph v2 workflow.

    v2 workflow:
    Load -> Profile -> Problem Detection -> Parallel Audit -> Risk Aggregator
    -> Decision Router -> either HITL stop path or modeling path.
    """
    graph = StateGraph(AuditState)

    graph.add_node("node_load_dataset", timed_node("load_dataset", load_dataset_node))
    graph.add_node("node_profile", timed_node("profile", profile_node))
    graph.add_node(
        "node_problem_detection",
        timed_node("problem_detection", problem_detection_node),
    )
    graph.add_node(
        "node_parallel_audit",
        timed_node("parallel_audit", parallel_audit_node),
    )
    graph.add_node(
        "node_risk_aggregator",
        timed_node("risk_aggregator", risk_aggregator_node),
    )
    graph.add_node(
        "node_decision_router",
        timed_node("decision_router", decision_router_node),
    )
    graph.add_node("node_hitl_review", timed_node("hitl_review", hitl_review_node))
    graph.add_node("node_metrics", timed_node("metrics", metric_node))
    graph.add_node("node_baseline", timed_node("baseline", baseline_node))
    graph.add_node("node_mlflow", timed_node("mlflow", mlflow_node))
    graph.add_node(
        "node_explainability",
        timed_node("explainability", explainability_node),
    )

    if include_report:
        graph.add_node("node_report", timed_node("report", report_node))

    graph.add_node("node_finalize", timed_node("finalize", finalization_node))

    graph.set_entry_point("node_load_dataset")
    graph.add_edge("node_load_dataset", "node_profile")
    graph.add_edge("node_profile", "node_problem_detection")
    graph.add_edge("node_problem_detection", "node_parallel_audit")
    graph.add_edge("node_parallel_audit", "node_risk_aggregator")
    graph.add_edge("node_risk_aggregator", "node_decision_router")

    graph.add_conditional_edges(
        "node_decision_router",
        route_after_decision,
        {
            "human_review": "node_hitl_review",
            "metrics": "node_metrics",
        },
    )

    graph.add_edge(
        "node_hitl_review", "node_report" if include_report else "node_finalize"
    )
    graph.add_edge("node_metrics", "node_baseline")
    graph.add_edge("node_baseline", "node_mlflow")
    graph.add_edge("node_mlflow", "node_explainability")

    if include_report:
        graph.add_edge("node_explainability", "node_report")
        graph.add_edge("node_report", "node_finalize")
    else:
        graph.add_edge("node_explainability", "node_finalize")

    graph.add_edge("node_finalize", END)

    return graph.compile()


def run_audit_workflow(
    dataset_path: str | Path,
    target_column: str,
    workflow_mode: str = WORKFLOW_MODE_AUTO,
    human_review_decision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run v2 ML audit workflow.

    Includes deterministic ML checks, internal parallel audit, risk aggregation,
    decision routing, optional MLflow, optional explainability, optional LLM
    report, audit score, and HITL summary.
    """
    try:
        logger.info("Starting v2 audit workflow")

        initial_state = initialize_state(
            dataset_path=dataset_path,
            target_column=target_column,
            workflow_mode=workflow_mode,
            human_review_decision=human_review_decision,
        )
        graph = build_audit_graph(include_report=True)
        final_state = graph.invoke(initial_state)

        result = build_report_safe_results(final_state)

        logger.info("v2 audit workflow completed successfully")
        return result

    except AgentWorkflowError:
        raise
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
        OSError,
    ) as error:
        logger.exception("Audit workflow failed.")
        raise AgentWorkflowError(
            "Audit workflow failed.",
            error_detail=str(error),
        ) from error


def run_audit_workflow_without_report(
    dataset_path: str | Path,
    target_column: str,
    workflow_mode: str = WORKFLOW_MODE_AUTO,
    human_review_decision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Lightweight workflow variant useful for tests and offline smoke checks.

    Runs deterministic checks, risk aggregation, decision routing, baseline
    modeling when safe, optional nodes, audit scoring, and HITL summary, but
    skips LLM report generation.
    """
    try:
        logger.info("Starting v2 audit workflow without report")

        initial_state = initialize_state(
            dataset_path=dataset_path,
            target_column=target_column,
            workflow_mode=workflow_mode,
            human_review_decision=human_review_decision,
        )
        graph = build_audit_graph(include_report=False)
        final_state = graph.invoke(initial_state)

        result = build_report_safe_results(final_state)
        result["audit_report"] = build_deterministic_fallback_report(final_state)
        result["report_save_result"] = {
            "report_path": None,
            "message": "Report generation skipped in without-report workflow.",
        }

        logger.info("v2 audit workflow without report completed successfully")
        return result

    except AgentWorkflowError:
        raise
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
        OSError,
    ) as error:
        logger.exception("Audit workflow without report failed.")
        raise AgentWorkflowError(
            "Audit workflow without report failed.",
            error_detail=str(error),
        ) from error


def run_audit_review_gate(
    dataset_path: str | Path,
    target_column: str,
) -> dict[str, Any]:
    """Run audit only until human review gate when review is required."""
    return run_audit_workflow(
        dataset_path=dataset_path,
        target_column=target_column,
        workflow_mode=WORKFLOW_MODE_HUMAN_GATE,
    )


def run_audit_after_human_approval(
    dataset_path: str | Path,
    target_column: str,
    human_review_decision: dict[str, Any],
) -> dict[str, Any]:
    """Run full modeling workflow after explicit human approval."""
    return run_audit_workflow(
        dataset_path=dataset_path,
        target_column=target_column,
        workflow_mode=WORKFLOW_MODE_HUMAN_APPROVED,
        human_review_decision=human_review_decision,
    )


if __name__ == "__main__":
    dataset_path = "data/sample/student_mark.csv"
    target_column = "Grade"

    output = run_audit_workflow_without_report(
        dataset_path=dataset_path,
        target_column=target_column,
    )

    printable_output = {
        key: value for key, value in output.items() if key != "audit_report"
    }

    print(printable_output)
