from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from groq import Groq

from src.utils.config import get_config_value, get_llm_config
from src.utils.exceptions import AuditCopilotException, LLMReportError
from src.utils.logger import get_logger


logger = get_logger(__name__)


REPORT_TITLE = "Agentic ML Audit Report"
DEFAULT_REPORT_PATH = "reports/audit_report.md"
DEFAULT_CONTEXT_PATH = "reports/audit_context.json"


SYSTEM_REPORT_PROMPT = (
    "You are a Senior Machine Learning Auditor. "
    "You only explain deterministic Python audit results. "
    "Never invent numbers. "
    "Never estimate values. "
    "Never calculate metrics. "
    "Never contradict the provided audit context. "
    "Always write in plain English only."
)


SYSTEM_CHAT_PROMPT = (
    "You are an ML Audit Copilot. Answer only from the completed audit context. "
    "Never invent numbers. Never perform new ML computations. "
    "Always reply in plain English, regardless of what language the user's question is written in."
)


def _safe_get(mapping: Dict[str, Any], key: str, default: Any = None) -> Any:
    value = mapping.get(key, default)
    return default if value is None else value


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.lower().strip() in {"true", "1", "yes", "y"}

    return bool(value)


def _json_safe(data: Any) -> Any:
    """
    Convert audit data into JSON-safe values and remove heavy runtime objects.
    """
    blocked_keys = {
        "trained_model_objects",
        "preprocessor",
        "pipeline",
        "model_object",
        "dataframe",
        "df",
        "x_train",
        "x_test",
        "y_train",
        "y_test",
    }

    if isinstance(data, dict):
        cleaned: Dict[str, Any] = {}
        for key, value in data.items():
            key_str = str(key)
            if key_str.lower() in blocked_keys:
                continue
            cleaned[key_str] = _json_safe(value)
        return cleaned

    if isinstance(data, list):
        return [_json_safe(item) for item in data]

    if isinstance(data, tuple):
        return [_json_safe(item) for item in data]

    if isinstance(data, (str, int, float, bool)) or data is None:
        return data

    return str(data)


def _get_groq_client(api_key: str) -> Groq:
    """
    Create Groq client. Timeout is handled by the request method when supported.
    """
    return Groq(api_key=api_key)


def _get_llm_runtime_config(default_max_tokens: int) -> Dict[str, Any]:
    llm_config = get_llm_config()

    return {
        "api_key": llm_config.get("api_key"),
        "model": llm_config.get("model", "openai/gpt-oss-120b"),
        "temperature": float(llm_config.get("temperature", 0.2)),
        "max_tokens": int(llm_config.get("max_tokens", default_max_tokens)),
        "timeout": int(llm_config.get("timeout", 120)),
    }


def _call_groq_chat(
    *,
    system_prompt: str,
    user_prompt: str,
    max_tokens_default: int,
) -> Optional[str]:
    """
    Call Groq chat completion safely. Returns None if unavailable or failed.
    """
    try:
        runtime = _get_llm_runtime_config(default_max_tokens=max_tokens_default)
        api_key = runtime.get("api_key")

        if not api_key:
            logger.warning("GROQ_API_KEY not found. Using deterministic fallback.")
            return None

        client = _get_groq_client(str(api_key))

        logger.info("Calling Groq model: %s", runtime["model"])

        response = client.chat.completions.create(
            model=str(runtime["model"]),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=float(runtime["temperature"]),
            max_tokens=int(runtime["max_tokens"]),
            timeout=int(runtime["timeout"]),
        )

        content = response.choices[0].message.content

        if not content or not content.strip():
            logger.warning("Groq returned empty content.")
            return None

        return content.strip()

    except TypeError:
        # Some Groq client versions may not support timeout as a request parameter.
        try:
            runtime = _get_llm_runtime_config(default_max_tokens=max_tokens_default)
            api_key = runtime.get("api_key")

            if not api_key:
                return None

            client = _get_groq_client(str(api_key))
            response = client.chat.completions.create(
                model=str(runtime["model"]),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=float(runtime["temperature"]),
                max_tokens=int(runtime["max_tokens"]),
            )

            content = response.choices[0].message.content
            return content.strip() if content and content.strip() else None

        except Exception as error:
            logger.warning("Groq call failed after retry without timeout: %s", error)
            return None

    except Exception as error:
        # Logging the exception type alongside the message makes failure
        # modes distinguishable in the logs — e.g. a groq.BadRequestError
        # with "model_decommissioned" (wrong/deprecated model name) looks
        # completely different from a groq.AuthenticationError (bad API
        # key) or a network timeout, even though the UI shows the same
        # generic fallback message for all three.
        logger.warning(
            "Groq call failed (%s). Using fallback if available. Error: %s",
            type(error).__name__,
            error,
        )
        return None


def extract_report_context(audit_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract a clean, JSON-safe audit context for report generation.

    The LLM receives only deterministic Python-generated audit results.
    Heavy objects such as fitted models, DataFrames, and sklearn pipelines are removed.
    """
    try:
        if not isinstance(audit_results, dict):
            raise LLMReportError("audit_results must be a dictionary.")

        profile = audit_results.get("profile", {}) or {}
        data_quality = audit_results.get("data_quality", {}) or {}
        leakage = audit_results.get("leakage", {}) or {}
        metric_recommendation = audit_results.get("metric_recommendation", {}) or {}
        class_imbalance = audit_results.get("class_imbalance", {}) or {}
        baseline_results = audit_results.get("baseline_results", {}) or {}
        mlflow_results = audit_results.get("mlflow_results", {}) or {}

        dataset_shape = profile.get("shape", {}) or {}
        rows = dataset_shape.get("rows", "N/A")
        columns = dataset_shape.get("columns", "N/A")

        feature_count: int | str = "N/A"
        if isinstance(columns, int) and columns > 0:
            feature_count = max(columns - 1, 0)

        target_summary = profile.get("target_summary", {}) or {}
        target_distribution = target_summary.get("distribution", {})
        if not target_distribution:
            target_distribution = profile.get("target_distribution", {}) or {}

        context = {
            "metadata": {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "project_name": get_config_value(
                    "project.name", "Agentic ML Audit Copilot"
                ),
                "project_version": get_config_value("project.version", "unknown"),
                "report_type": "ml_audit_report",
            },
            "target_column": audit_results.get("target_column", "N/A"),
            "problem_type": audit_results.get("problem_type", "N/A"),
            "dataset_overview": {
                "rows": rows,
                "columns": columns,
                "sample_count": rows,
                "feature_count": feature_count,
                "column_names": profile.get("columns", []),
                "numeric_columns": profile.get("numeric_columns", []),
                "categorical_columns": profile.get("categorical_columns", []),
                "datetime_columns": profile.get("datetime_columns", []),
                "boolean_columns": profile.get("boolean_columns", []),
                "other_columns": profile.get("other_columns", []),
                "memory_usage_mb": profile.get("memory_usage_mb", "N/A"),
                "duplicate_rows": profile.get("duplicate_rows", "N/A"),
                "duplicate_rows_percent": profile.get("duplicate_rows_percent", "N/A"),
                "target_summary": target_summary,
                "target_distribution": target_distribution,
                "profile_warnings": profile.get("warnings", []),
            },
            "data_quality": {
                "missing_values": data_quality.get(
                    "missing_values", profile.get("missing_values", {})
                ),
                "high_missing_columns": data_quality.get("high_missing_columns", []),
                "duplicate_rows": data_quality.get(
                    "duplicate_rows", profile.get("duplicate_rows", 0)
                ),
                "duplicate_rows_percent": data_quality.get(
                    "duplicate_rows_percent",
                    profile.get("duplicate_rows_percent", 0.0),
                ),
                "constant_columns": data_quality.get(
                    "constant_columns", profile.get("constant_columns", [])
                ),
                "near_constant_columns": data_quality.get(
                    "near_constant_columns", profile.get("near_constant_columns", [])
                ),
                "high_cardinality_columns": data_quality.get(
                    "high_cardinality_columns",
                    profile.get("high_cardinality_columns", []),
                ),
                "possible_id_columns": data_quality.get(
                    "possible_id_columns", profile.get("identifier_columns", [])
                ),
                "null_only_columns": data_quality.get(
                    "null_only_columns", profile.get("null_only_columns", [])
                ),
                "warnings": data_quality.get("warnings", profile.get("warnings", [])),
            },
            "leakage": {
                "target_column": leakage.get("target_column", "N/A"),
                "total_possible_leakage_risks": leakage.get(
                    "total_possible_leakage_risks", 0
                ),
                "name_based_risks": leakage.get("name_based_risks", []),
                "duplicate_target_risks": leakage.get("duplicate_target_risks", []),
                "numeric_correlation_risks": leakage.get(
                    "numeric_correlation_risks", []
                ),
                "classification_proxy_risks": leakage.get(
                    "classification_proxy_risks", []
                ),
                "all_risks": leakage.get("all_risks", []),
                "warning": leakage.get("warning", ""),
            },
            "metric_recommendation": {
                "recommended_metrics": metric_recommendation.get(
                    "recommended_metrics", []
                ),
                "primary_metric": metric_recommendation.get("primary_metric", "N/A"),
                "scoring_metric": metric_recommendation.get("scoring_metric", "N/A"),
                "secondary_metrics": metric_recommendation.get("secondary_metrics", []),
                "reason": metric_recommendation.get("reason", "N/A"),
                "notes": metric_recommendation.get("notes", []),
            },
            "class_imbalance": class_imbalance or {},
            "baseline_results": {
                "models_trained": baseline_results.get("models_trained", []),
                "results": baseline_results.get("results", {}),
                "best_model": baseline_results.get("best_model", {}),
                "message": baseline_results.get("message", ""),
            },
            "mlflow_results": {
                "enabled": mlflow_results.get("enabled", True),
                "experiment_name": mlflow_results.get("experiment_name", ""),
                "models_logged": mlflow_results.get("models_logged", []),
                "best_model": mlflow_results.get("best_model", {}),
                "logged_model_uri": mlflow_results.get("logged_model_uri", None),
                "message": mlflow_results.get("message", ""),
            },
        }

        return _json_safe(context)

    except LLMReportError:
        raise

    except Exception as error:
        logger.exception("Report context extraction failed.")
        raise LLMReportError(
            "Report context extraction failed.",
            error_detail=str(error),
        ) from error


def build_llm_prompt(report_context: Dict[str, Any]) -> str:
    """
    Build a strict, grounded prompt for Groq LLM report generation.
    """
    context_json = json.dumps(report_context, indent=2, default=str)

    return f"""
You are a Senior Machine Learning Auditor reviewing a dataset before model training.

You are given structured audit results generated by deterministic Python code.

Your responsibility:
- Explain the audit results.
- Summarize risks.
- Recommend practical next steps.
- Write a professional Markdown report.

CRITICAL RULES:
1. Do NOT invent numbers.
2. Do NOT estimate missing values.
3. Do NOT calculate new metrics.
4. Do NOT change any provided values.
5. Do NOT claim confirmed leakage.
6. If leakage is present, always call it "possible leakage risk".
7. Do NOT claim that the model is production-ready.
8. Do NOT recommend deployment based only on baseline models.
9. Do NOT copy raw JSON into the report.
10. Do NOT expose internal implementation details.
11. If a value is missing, write "Not available".
12. Baseline models are sanity-check models, not final optimized models.
13. Explain why each major finding matters for ML training.
14. Keep the report interview-friendly and practical.
15. Avoid generic phrases like "As an AI", "In conclusion", or "It is important to note".
16. Write the entire report in plain English only — regardless of any language used inside the structured audit results.

Writing style:
- Professional but clear.
- Use concise paragraphs.
- Use bullets only where useful.
- Use Markdown headings.
- Use bold text for important findings.
- Do not exceed 1,000 words.

Generate a Markdown report with these exact sections:

# Agentic ML Audit Report

## Executive Summary
## 1. Dataset Overview
## 2. Problem Type
## 3. Data Quality Audit
## 4. Possible Leakage Risk
## 5. Metric Recommendation
## 6. Class Imbalance Analysis
## 7. Baseline Model Benchmark
## 8. MLflow Tracking
## 9. Final Recommendations
## 10. Important Caveats

Structured audit results:
```json
{context_json}
```
""".strip()


def generate_report_with_groq(report_context: Dict[str, Any]) -> Optional[str]:
    """
    Generate a Markdown audit report using Groq LLM.

    Returns None when the LLM is unavailable, unconfigured, or fails.
    """
    prompt = build_llm_prompt(report_context)
    return _call_groq_chat(
        system_prompt=SYSTEM_REPORT_PROMPT,
        user_prompt=prompt,
        max_tokens_default=2000,
    )


def _format_list(values: Any) -> str:
    if not values:
        return "None"

    if isinstance(values, list):
        return ", ".join(str(value) for value in values) if values else "None"

    return str(values)


def _get_dataset_health(warnings: list[Any]) -> tuple[str, str]:
    if not warnings or warnings == ["No major basic data quality issues detected."]:
        return "Good", "No major basic data quality issues were detected."

    return "Needs Review", "Some data quality warnings were detected."


def _get_leakage_summary(leakage_count: int) -> tuple[str, str]:
    if leakage_count == 0:
        return "Low", "No possible leakage risks were detected."

    if leakage_count <= 3:
        return "Medium", "Some possible leakage risks were detected and should be reviewed."

    return "High", "Multiple possible leakage risks were detected and require careful review."


def build_fallback_report(report_context: Dict[str, Any]) -> str:
    """
    Build a deterministic Markdown report when the LLM is unavailable.
    """
    try:
        logger.info("Building fallback audit report")

        metadata = report_context.get("metadata", {}) or {}
        dataset = report_context.get("dataset_overview", {}) or {}
        data_quality = report_context.get("data_quality", {}) or {}
        leakage = report_context.get("leakage", {}) or {}
        metrics = report_context.get("metric_recommendation", {}) or {}
        imbalance = report_context.get("class_imbalance", {}) or {}
        baseline = report_context.get("baseline_results", {}) or {}
        mlflow = report_context.get("mlflow_results", {}) or {}

        leakage_count = int(leakage.get("total_possible_leakage_risks", 0) or 0)
        warnings = data_quality.get("warnings", []) or []
        best_model = baseline.get("best_model", {}) or {}

        primary_metric = metrics.get("primary_metric", "N/A")
        metric_reason = metrics.get("reason", "N/A")

        leakage_level, leakage_message = _get_leakage_summary(leakage_count)
        dataset_health, quality_message = _get_dataset_health(warnings)

        best_model_name = best_model.get("model_name") or best_model.get("model") or "N/A"
        best_model_score = best_model.get("score", "N/A")
        selection_metric = best_model.get("selection_metric", primary_metric)

        imbalance_severity = imbalance.get("imbalance_severity", "N/A")
        imbalance_warning = imbalance.get("warning", "N/A")
        imbalance_ratio = imbalance.get("imbalance_ratio", "N/A")

        models_trained = baseline.get("models_trained", [])
        models_text = _format_list(models_trained)

        recommended_metrics = _format_list(metrics.get("recommended_metrics", []))
        mlflow_models_logged = _format_list(mlflow.get("models_logged", []))

        return f"""# {REPORT_TITLE}

## Executive Summary
This fallback report was generated deterministically because the Groq LLM service was unavailable or unconfigured.

The dataset health is marked as **{dataset_health}**. The audit detected **{leakage_count} possible leakage risks**. The recommended primary metric is **{primary_metric}**. The best baseline model is **{best_model_name}** based on **{selection_metric}**.

- **Project:** {metadata.get("project_name", "Agentic ML Audit Copilot")}
- **Version:** {metadata.get("project_version", "N/A")}
- **Generated At:** {metadata.get("generated_at", "N/A")}

## 1. Dataset Overview
- **Total Rows:** {dataset.get("rows", "N/A")}
- **Total Columns:** {dataset.get("columns", "N/A")}
- **Feature Count:** {dataset.get("feature_count", "N/A")}
- **Target Column:** {report_context.get("target_column", "N/A")}
- **Duplicate Rows:** {dataset.get("duplicate_rows", "N/A")}
- **Duplicate Rows Percent:** {dataset.get("duplicate_rows_percent", "N/A")}
- **Memory Usage MB:** {dataset.get("memory_usage_mb", "N/A")}
- **Numeric Columns:** {len(dataset.get("numeric_columns", []))}
- **Categorical Columns:** {len(dataset.get("categorical_columns", []))}
- **Datetime Columns:** {len(dataset.get("datetime_columns", []))}

## 2. Problem Type
- **Detected Problem Type:** {report_context.get("problem_type", "N/A")}

This determines which preprocessing strategy, baseline models, and evaluation metrics should be used.

## 3. Data Quality Audit
- **Status:** {dataset_health}
- **Summary:** {quality_message}
- **Warnings:** {_format_list(warnings)}

Data quality should be reviewed before final model training because missing values, duplicate rows, constant columns, and ID-like columns can affect model reliability.

## 4. Possible Leakage Risk
- **Risk Level:** {leakage_level}
- **Total Possible Leakage Risks:** {leakage_count}
- **Summary:** {leakage_message}

These findings are possible leakage risks, not confirmed leakage. A human should verify whether flagged columns would be available at prediction time.

## 5. Metric Recommendation
- **Primary Metric:** {primary_metric}
- **Reason:** {metric_reason}
- **Recommended Metrics:** {recommended_metrics}

The primary metric should guide baseline comparison and future model tuning.

## 6. Class Imbalance Analysis
- **Imbalance Severity:** {imbalance_severity}
- **Imbalance Ratio:** {imbalance_ratio}
- **Warning:** {imbalance_warning}

If imbalance exists, accuracy alone may be misleading. Class-aware metrics such as Precision, Recall, F1 Score, ROC-AUC, or PR-AUC should be considered.

## 7. Baseline Model Benchmark
- **Models Trained:** {models_text}
- **Best Baseline Model:** {best_model_name}
- **Selection Metric:** {selection_metric}
- **Best Score:** {best_model_score}

These models are baseline sanity-checks. They are not final optimized models.

## 8. MLflow Tracking
- **Enabled:** {mlflow.get("enabled", "N/A")}
- **Experiment Name:** {mlflow.get("experiment_name", "N/A")}
- **Models Logged:** {mlflow_models_logged}
- **Logged Model URI:** {mlflow.get("logged_model_uri", "N/A")}
- **Status:** {mlflow.get("message", "N/A")}

MLflow tracking helps compare future experiments, tuned models, and metric changes.

## 9. Final Recommendations
1. Review all data quality warnings before final model training.
2. Manually inspect any possible leakage-risk columns.
3. Use **{primary_metric}** as the primary evaluation metric.
4. Treat baseline models as a starting point, not as final models.
5. Use MLflow runs to compare future tuned experiments.

## 10. Important Caveats
- This report does not claim confirmed data leakage.
- Baseline models are not optimized final models.
- This audit supports ML review before training, but human validation is still required.
"""

    except Exception as error:
        logger.exception("Fallback report generation failed.")
        return (
            f"# {REPORT_TITLE}\n\n"
            f"Error: Could not generate fallback audit report. Detail: {error}"
        )


def generate_final_report(audit_results: Dict[str, Any]) -> str:
    """
    Generate the final audit report.

    First extracts safe deterministic audit context, then tries Groq LLM report
    generation, and finally falls back to a deterministic Markdown report.
    """
    try:
        context = extract_report_context(audit_results)
        report = generate_report_with_groq(context)

        if report is None:
            report = build_fallback_report(context)

        return report

    except LLMReportError:
        raise

    except Exception as error:
        logger.exception("Final report generation failed.")
        raise LLMReportError(
            "Final report generation failed.",
            error_detail=str(error),
        ) from error


def build_audit_report(audit_results: Dict[str, Any]) -> str:
    """
    Backward-compatible wrapper used by workflow.py.
    """
    return generate_final_report(audit_results)


def build_section_explanation(section_name: str, section_data: Dict[str, Any]) -> str:
    """
    Build a short deterministic explanation for Streamlit sections.
    """
    if section_name == "metric_recommendation":
        primary_metric = section_data.get("primary_metric", "N/A")
        reason = section_data.get("reason", "N/A")
        return f"The recommended primary metric is **{primary_metric}**. {reason}"

    if section_name == "leakage":
        risk_count = section_data.get("total_possible_leakage_risks", 0)
        if risk_count == 0:
            return "No possible leakage risks were detected by the audit."
        return (
            f"The audit detected **{risk_count} possible leakage risks**. "
            "These should be manually reviewed before model training."
        )

    if section_name == "class_imbalance":
        severity = section_data.get("imbalance_severity", "N/A")
        ratio = section_data.get("imbalance_ratio", "N/A")
        return (
            f"The class imbalance severity is **{severity}** with an "
            f"imbalance ratio of **{ratio}**."
        )

    if section_name == "baseline_results":
        best_model = section_data.get("best_model", {}) or {}
        model_name = best_model.get("model_name", "N/A")
        score = best_model.get("score", "N/A")
        metric = best_model.get("selection_metric", "selected metric")
        return (
            f"The best baseline model is **{model_name}** with a {metric} "
            f"score of **{score}**. These are baseline sanity-check models, "
            "not final optimized models."
        )

    if section_name == "data_quality":
        warnings = section_data.get("warnings", []) or []
        if not warnings:
            return "No major basic data quality issues were detected."
        return f"The audit found **{len(warnings)} data quality warnings** that should be reviewed."

    if section_name == "mlflow_results":
        experiment_name = section_data.get("experiment_name", "N/A")
        message = section_data.get("message", "N/A")
        return f"MLflow experiment **{experiment_name}** status: {message}"

    return "Explanation not available for this section."


def save_report_context(
    report_context: Dict[str, Any],
    output_path: str | Path = DEFAULT_CONTEXT_PATH,
) -> Dict[str, Any]:
    """
    Save JSON-safe report context to disk.
    """
    try:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as file:
            json.dump(_json_safe(report_context), file, indent=2, default=str)

        return {
            "context_path": str(path),
            "message": "Audit context saved successfully.",
        }

    except Exception as error:
        logger.exception("Saving audit context failed.")
        raise LLMReportError(
            "Saving audit context failed.",
            error_detail=str(error),
        ) from error


def save_audit_report(
    report: str,
    output_path: str | Path | None = None,
    audit_results: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Save the generated audit report to disk.

    Uses config.yaml by default:
    - reports.default_report_path
    - reports.save_json
    """
    try:
        if not report or not str(report).strip():
            raise LLMReportError("Report content is empty and cannot be saved.")

        configured_path = get_config_value(
            "reports.default_report_path",
            DEFAULT_REPORT_PATH,
        )
        path = Path(output_path or configured_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(report), encoding="utf-8")

        result: Dict[str, Any] = {
            "report_path": str(path),
            "message": "Audit report saved successfully.",
        }

        save_json = _as_bool(get_config_value("reports.save_json", False))

        if save_json and audit_results is not None:
            context = extract_report_context(audit_results)
            json_path = path.with_suffix(".json")
            context_save_result = save_report_context(context, json_path)
            result["context_path"] = context_save_result["context_path"]

        return result

    except LLMReportError:
        raise

    except Exception as error:
        logger.exception("Saving audit report failed.")
        raise LLMReportError(
            "Saving audit report failed.",
            error_detail=str(error),
        ) from error


def ask_about_audit(
    audit_context: Dict[str, Any],
    user_question: str,
    chat_history: Optional[list[Dict[str, str]]] = None,
) -> Optional[str]:
    """
    Answer user questions using ONLY the completed audit results.

    This function is used by Streamlit audit chat after the audit has completed.
    It never performs new ML computations.
    """
    try:
        if not user_question or not user_question.strip():
            return None

        safe_context = extract_report_context(audit_context)
        context_json = json.dumps(safe_context, indent=2, default=str)

        history_text = ""
        if chat_history:
            cleaned_history = []
            for item in chat_history[-6:]:
                role = item.get("role", "user")
                content = item.get("content", "")
                if content:
                    cleaned_history.append(f"{role}: {content}")
            history_text = "\n".join(cleaned_history)

        prompt = f"""
You are an ML Audit Copilot.

The audit has already been completed by deterministic Python code.

Your job is ONLY to answer questions about the completed audit.

Rules:
- Never invent numbers.
- Never calculate new metrics.
- Never perform leakage detection yourself.
- Never perform preprocessing yourself.
- Never train or recommend a model not present in the audit.
- Never contradict the audit context.
- If the answer is unavailable in the audit context, clearly say so.
- Always reply in plain English, regardless of what language the question was asked in.
- Keep the answer concise and practical.
- Explain ML concepts in beginner-friendly language.

Recent Chat History:
{history_text or "No previous chat history."}

User Question:
{user_question}

Completed Audit Results:
```json
{context_json}
```
""".strip()

        return _call_groq_chat(
            system_prompt=SYSTEM_CHAT_PROMPT,
            user_prompt=prompt,
            max_tokens_default=700,
        )

    except Exception as error:
        logger.warning("Audit chat failed: %s", error)
        return None


if __name__ == "__main__":
    sample_results = {
        "target_column": "Outcome",
        "problem_type": "binary_classification",
        "profile": {
            "shape": {"rows": 768, "columns": 9},
            "columns": [
                "Pregnancies",
                "Glucose",
                "BloodPressure",
                "SkinThickness",
                "Insulin",
                "BMI",
                "DiabetesPedigreeFunction",
                "Age",
                "Outcome",
            ],
            "numeric_columns": [
                "Pregnancies",
                "Glucose",
                "BloodPressure",
                "SkinThickness",
                "Insulin",
                "BMI",
                "DiabetesPedigreeFunction",
                "Age",
            ],
            "categorical_columns": [],
            "datetime_columns": [],
            "duplicate_rows": 0,
            "target_summary": {
                "target_column": "Outcome",
                "distribution": {
                    "0": {"count": 500, "percent": 65.1},
                    "1": {"count": 268, "percent": 34.9},
                },
            },
        },
        "data_quality": {
            "missing_values": {},
            "high_missing_columns": [],
            "duplicate_rows": 0,
            "duplicate_rows_percent": 0.0,
            "constant_columns": [],
            "high_cardinality_columns": [],
            "possible_id_columns": [],
            "warnings": ["No major basic data quality issues detected."],
        },
        "leakage": {
            "target_column": "Outcome",
            "total_possible_leakage_risks": 0,
            "name_based_risks": [],
            "duplicate_target_risks": [],
            "numeric_correlation_risks": [],
            "classification_proxy_risks": [],
            "all_risks": [],
            "warning": (
                "These are possible leakage risks, not confirmed leakage. "
                "A human should review whether these columns would be available at prediction time."
            ),
        },
        "metric_recommendation": {
            "recommended_metrics": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "ROC-AUC",
                "PR-AUC",
            ],
            "primary_metric": "F1 Score",
            "reason": "Binary classification needs metrics beyond accuracy.",
        },
        "class_imbalance": {
            "problem_type": "binary_classification",
            "target_column": "Outcome",
            "is_applicable": True,
            "class_counts": {"0": 500, "1": 268},
            "class_percentages": {"0": 65.1, "1": 34.9},
            "majority_class": "0",
            "majority_count": 500,
            "minority_class": "1",
            "minority_count": 268,
            "imbalance_ratio": 1.87,
            "imbalance_severity": "moderate",
            "recommended_metrics": [
                "Precision",
                "Recall",
                "F1 Score",
                "PR-AUC",
                "ROC-AUC",
            ],
            "warning": "Moderate class imbalance detected. Accuracy alone may be misleading.",
        },
        "baseline_results": {
            "problem_type": "binary_classification",
            "target_column": "Outcome",
            "models_trained": ["Logistic Regression", "Random Forest Classifier"],
            "results": {
                "Logistic Regression": {
                    "accuracy": 0.7143,
                    "precision": 0.7065,
                    "recall": 0.7143,
                    "f1_score": 0.7084,
                    "roc_auc": 0.823,
                },
                "Random Forest Classifier": {
                    "accuracy": 0.7597,
                    "precision": 0.7546,
                    "recall": 0.7597,
                    "f1_score": 0.7555,
                    "roc_auc": 0.8147,
                },
            },
            "best_model": {
                "model_name": "Random Forest Classifier",
                "selection_metric": "f1_score",
                "score": 0.7555,
            },
            "message": "Baseline model training completed successfully.",
        },
        "mlflow_results": {
            "experiment_name": "agentic_ml_audit_baselines",
            "models_logged": ["Logistic Regression", "Random Forest Classifier"],
            "best_model": {
                "model_name": "Random Forest Classifier",
                "selection_metric": "f1_score",
                "score": 0.7555,
            },
            "message": "MLflow tracking completed successfully.",
        },
    }

    report_output = build_audit_report(sample_results)
    save_result = save_audit_report(report_output, audit_results=sample_results)

    print(report_output)
    print(save_result)