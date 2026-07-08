from __future__ import annotations

from typing import Any, TypedDict

import pandas as pd
from langgraph.graph import END, StateGraph

from src.audit.baseline_models import train_baseline_models
from src.audit.class_imbalance import detect_class_imbalance
from src.audit.data_quality import run_data_quality_audit
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


class AuditState(TypedDict, total=False):
    df: pd.DataFrame
    dataset_path: str
    target_column: str
    report_output_path: str

    profile: dict[str, Any]
    problem_type_result: dict[str, Any]
    problem_type: str
    data_quality: dict[str, Any]
    leakage: dict[str, Any]
    metric_recommendation: dict[str, Any]
    class_imbalance: dict[str, Any]
    baseline_results: dict[str, Any]
    mlflow_results: dict[str, Any]
    audit_report: str
    report_save_result: dict[str, Any]

    errors: list[str]
    warnings: list[str]


def load_dataset_node(state: AuditState) -> AuditState:
    logger.info("Workflow node started: load_dataset")

    state["df"] = load_dataset(state["dataset_path"])

    logger.info("Workflow node completed: load_dataset")
    return state


def profile_dataset_node(state: AuditState) -> AuditState:
    logger.info("Workflow node started: profile_dataset")

    state["profile"] = profile_dataset(
        df=state["df"],
        target_column=state["target_column"],
    )

    logger.info("Workflow node completed: profile_dataset")
    return state


def problem_type_node(state: AuditState) -> AuditState:
    logger.info("Workflow node started: problem_type")

    problem_result = detect_problem_type(
        df=state["df"],
        target_column=state["target_column"],
    )

    state["problem_type_result"] = problem_result
    state["problem_type"] = problem_result["problem_type"]

    logger.info("Workflow node completed: problem_type")
    return state


def data_quality_node(state: AuditState) -> AuditState:
    logger.info("Workflow node started: data_quality")

    state["data_quality"] = run_data_quality_audit(
        df=state["df"],
        target_column=state["target_column"],
    )

    logger.info("Workflow node completed: data_quality")
    return state


def leakage_node(state: AuditState) -> AuditState:
    logger.info("Workflow node started: leakage")

    state["leakage"] = run_leakage_check(
        df=state["df"],
        target_column=state["target_column"],
    )

    logger.info("Workflow node completed: leakage")
    return state


def class_imbalance_node(state: AuditState) -> AuditState:
    logger.info("Workflow node started: class_imbalance")

    state["class_imbalance"] = detect_class_imbalance(
        df=state["df"],
        target_column=state["target_column"],
        problem_type=state["problem_type"],
    )

    logger.info("Workflow node completed: class_imbalance")
    return state


def metric_recommendation_node(state: AuditState) -> AuditState:
    logger.info("Workflow node started: metric_recommendation")

    imbalance_severity = None

    if state.get("class_imbalance", {}).get("is_applicable"):
        imbalance_severity = state["class_imbalance"].get("imbalance_severity")

    state["metric_recommendation"] = recommend_metrics(
        problem_type=state["problem_type"],
        imbalance_severity=imbalance_severity,
    )

    logger.info("Workflow node completed: metric_recommendation")
    return state


def baseline_model_node(state: AuditState) -> AuditState:
    logger.info("Workflow node started: baseline_model")

    state["baseline_results"] = train_baseline_models(
        df=state["df"],
        target_column=state["target_column"],
        problem_type=state["problem_type"],
    )

    logger.info("Workflow node completed: baseline_model")
    return state


def mlflow_node(state: AuditState) -> AuditState:
    logger.info("Workflow node started: mlflow")

    try:
        enable_mlflow = str(
            get_config_value("mlflow.enabled", True)
        ).lower().strip() in {"true", "1", "yes", "y"}

        if not enable_mlflow:
            state["mlflow_results"] = {
                "enabled": False,
                "message": "MLflow tracking skipped because it is disabled in config.",
            }
            return state

        feature_df = state["df"].drop(columns=[state["target_column"]])
        sample_input = feature_df.head(5)

        state["mlflow_results"] = track_baseline_experiment(
            baseline_results=state["baseline_results"],
            sample_input=sample_input,
        )

    except Exception as error:
        warning = f"MLflow tracking failed but workflow continued: {error}"
        logger.warning(warning)

        state.setdefault("warnings", []).append(warning)
        state["mlflow_results"] = {
            "enabled": True,
            "success": False,
            "message": warning,
        }

    logger.info("Workflow node completed: mlflow")
    return state


def audit_report_node(state: AuditState) -> AuditState:
    logger.info("Workflow node started: audit_report")

    report = build_audit_report(state)

    output_path = state.get("report_output_path", "reports/audit_report.md")

    save_result = save_audit_report(
        report=report,
        output_path=output_path,
    )

    state["audit_report"] = report
    state["report_save_result"] = save_result

    logger.info("Workflow node completed: audit_report")
    return state


def build_audit_workflow():
    workflow = StateGraph(AuditState)
    
    workflow.add_node("load_dataset", load_dataset_node)
    workflow.add_node("profile_dataset", profile_dataset_node)
    workflow.add_node("problem_type", problem_type_node)
    workflow.add_node("data_quality", data_quality_node)
    workflow.add_node("leakage", leakage_node)
    workflow.add_node("class_imbalance", class_imbalance_node)
    workflow.add_node("metric_recommendation", metric_recommendation_node)
    workflow.add_node("baseline_model", baseline_model_node)
    workflow.add_node("mlflow", mlflow_node)
    workflow.add_node("audit_report", audit_report_node)

    workflow.set_entry_point("load_dataset")

    workflow.add_edge("load_dataset", "profile_dataset")
    workflow.add_edge("profile_dataset", "problem_type")
    workflow.add_edge("problem_type", "data_quality")
    workflow.add_edge("data_quality", "leakage")
    workflow.add_edge("leakage", "class_imbalance")
    workflow.add_edge("class_imbalance", "metric_recommendation")
    workflow.add_edge("metric_recommendation", "baseline_model")
    workflow.add_edge("baseline_model", "mlflow")
    workflow.add_edge("mlflow", "audit_report")
    workflow.add_edge("audit_report", END)

    return workflow.compile()


def run_audit_workflow(
    dataset_path: str,
    target_column: str,
    report_output_path: str = "reports/audit_report.md",
) -> AuditState:
    try:
        logger.info("Starting Agentic ML audit workflow")

        if dataset_path is None or str(dataset_path).strip() == "":
            raise AgentWorkflowError("Dataset path is required.")

        if target_column is None or str(target_column).strip() == "":
            raise AgentWorkflowError("Target column is required.")

        app = build_audit_workflow()

        initial_state: AuditState = {
            "dataset_path": dataset_path,
            "target_column": target_column,
            "report_output_path": report_output_path,
            "errors": [],
            "warnings": [],
        }

        final_state = app.invoke(initial_state)

        logger.info("Agentic ML audit workflow completed successfully")
        return final_state

    except AgentWorkflowError:
        raise

    except Exception as error:
        logger.error(f"Agentic ML audit workflow failed: {error}")
        raise AgentWorkflowError(
            "Agentic ML audit workflow failed.",
            error_detail=str(error),
        ) from error


def get_printable_workflow_output(state: AuditState) -> dict[str, Any]:
    """
    Remove dataframe and trained model objects for clean CLI/API output.
    """
    output = dict(state)

    output.pop("df", None)

    if "baseline_results" in output:
        output["baseline_results"] = dict(output["baseline_results"])
        output["baseline_results"].pop("trained_model_objects", None)

    return output


if __name__ == "__main__":
    output = run_audit_workflow(
        dataset_path="data/sample/student_mark.csv",
        target_column="Grade",
    )

    printable_output = get_printable_workflow_output(output)

    print("Problem Type:", printable_output["problem_type"])
    print("Recommended Metrics:", printable_output["metric_recommendation"])
    print("Best Model:", printable_output["baseline_results"]["best_model"])
    print("MLflow:", printable_output["mlflow_results"]["message"])
    print("Report:", printable_output["report_save_result"]["message"])
    print("Report Path:", printable_output["report_save_result"]["report_path"])