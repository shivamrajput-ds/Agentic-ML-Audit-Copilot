from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from groq import Groq

from src.utils.config import get_config_value, get_llm_config
from src.utils.exceptions import ReportGenerationError
from src.utils.logger import get_logger


logger = get_logger(__name__)


MAX_CONTEXT_CHARS = 18_000
MAX_CHAT_QUESTION_CHARS = 1_000


def as_bool(value: Any) -> bool:
    """
    Convert config values safely into boolean.
    """
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.lower().strip() in {"true", "1", "yes", "y"}

    return bool(value)


def json_safe(data: Any) -> Any:
    """
    Convert audit context into JSON-safe object.

    Removes model objects, dataframes, and runtime-only objects.
    """
    if isinstance(data, dict):
        cleaned: dict[str, Any] = {}

        blocked_keys = {
            "trained_model_objects",
            "runtime_objects",
            "df",
            "train_features",
            "test_features",
            "sample_features",
            "sample_target",
            "label_encoder",
        }

        for key, value in data.items():
            if key in blocked_keys:
                continue

            cleaned[str(key)] = json_safe(value)

        return cleaned

    if isinstance(data, list):
        return [json_safe(item) for item in data]

    if isinstance(data, tuple):
        return [json_safe(item) for item in data]

    if isinstance(data, (str, int, float, bool)) or data is None:
        return data

    return str(data)


def truncate_text(text: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """
    Truncate long context to avoid token overflow.
    """
    if len(text) <= max_chars:
        return text

    return (
        text[:max_chars]
        + "\n\n[TRUNCATED: audit context was longer than allowed budget]"
    )


def extract_report_context(audit_results: dict[str, Any]) -> dict[str, Any]:
    """
    Extract compact context for LLM report.

    Do not pass raw dataframe/model objects to the LLM.
    """
    safe = json_safe(audit_results)

    baseline = safe.get("baseline_results", {}) or {}
    leakage = safe.get("leakage", {}) or {}
    data_quality = safe.get("data_quality", {}) or {}
    explainability = safe.get("explainability", {}) or {}

    return {
        "project": {
            "name": get_config_value("project.name", "Agentic ML Audit Copilot"),
            "version": get_config_value("project.version", "1.0.0"),
            "philosophy": "Deterministic-first; LLM explains Python-generated audit results only.",
        },
        "target_column": safe.get("target_column"),
        "problem_type": safe.get("problem_type"),
        "audit_score": safe.get("audit_score"),
        "human_review": safe.get("human_review"),
        "execution_summary": safe.get("execution_summary"),
        "profile": safe.get("profile"),
        "problem_detection": safe.get("problem_detection"),
        "data_quality_summary": {
            "quality_score": data_quality.get("quality_score"),
            "duplicate_rows": data_quality.get("duplicate_rows"),
            "warnings": data_quality.get("warnings"),
            "recommended_actions": data_quality.get("recommended_actions"),
            "high_missing_columns": data_quality.get("high_missing_columns"),
            "possible_id_columns": data_quality.get("possible_id_columns"),
            "high_cardinality_columns": data_quality.get("high_cardinality_columns"),
        },
        "leakage_summary": {
            "target_column": leakage.get("target_column"),
            "total_possible_leakage_risks": leakage.get(
                "total_possible_leakage_risks"
            ),
            "overall_severity": leakage.get("overall_severity"),
            "risk_summary": leakage.get("risk_summary"),
            "all_risks": leakage.get("all_risks"),
            "recommended_actions": leakage.get("recommended_actions"),
            "warning": leakage.get("warning"),
        },
        "class_imbalance": safe.get("class_imbalance"),
        "metric_recommendation": safe.get("metric_recommendation"),
        "baseline_summary": {
            "models_trained": baseline.get("models_trained"),
            "results": baseline.get("results"),
            "best_model": baseline.get("best_model"),
            "evaluation_details": baseline.get("evaluation_details"),
            "note": baseline.get("note"),
        },
        "explainability_summary": {
            "available": explainability.get("available"),
            "best_model_name": explainability.get("best_model_name"),
            "model_type": explainability.get("model_type"),
            "summary": explainability.get("summary"),
            "builtin_feature_importance": explainability.get(
                "builtin_feature_importance"
            ),
            "shap": {
                "available": explainability.get("shap", {}).get("available")
                if isinstance(explainability.get("shap"), dict)
                else None,
                "method": explainability.get("shap", {}).get("method")
                if isinstance(explainability.get("shap"), dict)
                else None,
                "global_importance": explainability.get("shap", {}).get(
                    "global_importance"
                )
                if isinstance(explainability.get("shap"), dict)
                else None,
                "message": explainability.get("shap", {}).get("message")
                if isinstance(explainability.get("shap"), dict)
                else None,
            },
        },
        "mlflow_results": safe.get("mlflow_results"),
    }


def build_system_prompt() -> str:
    """
    Strict system prompt for grounded audit reporting.
    """
    return """
You are an ML audit report writer.

Rules:
1. Use ONLY the provided audit context.
2. Never invent metrics, scores, model names, columns, or risks.
3. If a value is missing, write "Not available in audit context".
4. Do not claim confirmed leakage. Say "possible leakage risk" unless explicitly confirmed by human review.
5. Explain that this is human-in-the-loop and final decisions require ML engineer review.
6. Keep the tone professional, concise, and technical.
7. Do not recommend hyperparameter tuning before data quality/leakage issues are reviewed.
8. Separate deterministic findings from LLM interpretation.
""".strip()


def build_report_user_prompt(context: dict[str, Any]) -> str:
    context_json = truncate_text(
        json.dumps(context, indent=2, default=str),
        max_chars=MAX_CONTEXT_CHARS,
    )

    return f"""
Generate a professional ML audit report in Markdown.

Required sections:
1. Executive Summary
2. Dataset & Target Overview
3. Data Quality Findings
4. Possible Leakage Risks
5. Class Imbalance / Target Distribution
6. Recommended Metrics
7. Baseline Model Results
8. Explainability Findings
9. Human-in-the-loop Review Checklist
10. Final Recommendations
11. Limitations

Important:
- Use only the context below.
- Do not fabricate values.
- Clearly label leakage as "possible leakage risk".
- Keep recommendations practical.

AUDIT_CONTEXT:
```json
{context_json}
```
""".strip()


def get_groq_client() -> Groq:
    """
    Create Groq client from config/env.
    """
    llm_config = get_llm_config()
    api_key = llm_config.get("api_key")

    if not api_key:
        raise ReportGenerationError(
            "GROQ_API_KEY is not configured. Cannot generate LLM report."
        )

    return Groq(api_key=str(api_key))


def call_groq(messages: list[dict[str, str]]) -> str:
    """
    Call Groq chat completion.
    """
    llm_config = get_llm_config()
    client = get_groq_client()

    response = client.chat.completions.create(
        model=str(llm_config.get("model", "llama-3.3-70b-versatile")),
        temperature=float(llm_config.get("temperature", 0.2)),
        max_tokens=int(llm_config.get("max_tokens", 2000)),
        messages=messages,
    )

    content = response.choices[0].message.content

    if not content:
        raise ReportGenerationError("LLM returned an empty response.")

    return str(content)


def build_deterministic_report(context: dict[str, Any]) -> str:
    """
    Fallback report when LLM is unavailable.
    """
    baseline = context.get("baseline_summary", {}) or {}
    best_model = baseline.get("best_model", {}) or {}
    leakage = context.get("leakage_summary", {}) or {}
    data_quality = context.get("data_quality_summary", {}) or {}
    metric = context.get("metric_recommendation", {}) or {}
    human_review = context.get("human_review", {}) or {}

    lines = [
        "# Agentic ML Audit Report",
        "",
        f"Generated at: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "## Executive Summary",
        f"- Target column: `{context.get('target_column', 'Not available')}`",
        f"- Problem type: `{context.get('problem_type', 'Not available')}`",
        f"- Audit score: `{(context.get('audit_score') or {}).get('score', 'Not available')}`",
        f"- Readiness: `{(context.get('audit_score') or {}).get('readiness', 'Not available')}`",
        "",
        "## Data Quality Findings",
        f"- Quality score: `{(data_quality.get('quality_score') or {}).get('score', 'Not available')}`",
        f"- Duplicate rows: `{data_quality.get('duplicate_rows', 'Not available')}`",
        f"- Warnings: `{data_quality.get('warnings', [])}`",
        "",
        "## Possible Leakage Risks",
        f"- Total possible leakage risks: `{leakage.get('total_possible_leakage_risks', 0)}`",
        f"- Overall severity: `{leakage.get('overall_severity', 'none')}`",
        "- Note: these are possible risks, not confirmed leakage.",
        "",
        "## Metric Recommendation",
        f"- Primary metric: `{metric.get('primary_metric', 'Not available')}`",
        f"- Reason: {metric.get('reason', 'Not available in audit context')}",
        "",
        "## Baseline Model Results",
        f"- Best model: `{best_model.get('model_name', 'Not available')}`",
        f"- Selection metric: `{best_model.get('selection_metric', 'Not available')}`",
        f"- Score: `{best_model.get('score', 'Not available')}`",
        "",
        "## Human-in-the-loop Review",
        f"- Requires human review: `{human_review.get('requires_human_review', 'Not available')}`",
        f"- Review items count: `{human_review.get('review_items_count', 'Not available')}`",
        "",
        "## Limitations",
        "- This report is based on deterministic audit checks.",
        "- Leakage findings are heuristic and require domain review.",
        "- Baseline models are sanity-check models, not final tuned models.",
    ]

    return "\n".join(lines)


def build_audit_report(audit_results: dict[str, Any]) -> str:
    """
    Build LLM audit report with deterministic fallback.
    """
    try:
        logger.info("Starting audit report generation")

        context = extract_report_context(audit_results)

        if not as_bool(get_config_value("llm.enabled", True)):
            logger.info("LLM report disabled. Using deterministic fallback.")
            return build_deterministic_report(context)

        messages = [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": build_report_user_prompt(context)},
        ]

        report = call_groq(messages)

        logger.info("Audit report generated successfully")
        return report

    except Exception as error:
        logger.warning("LLM report generation failed. Using fallback: %s", error)
        context = extract_report_context(audit_results)
        return build_deterministic_report(context)


def save_audit_report(
    report: str,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Save Markdown audit report.
    """
    try:
        if output_path is None:
            output_path = get_config_value(
                "reports.default_report_path",
                "reports/audit_report.md",
            )

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")

        return {
            "report_path": str(path),
            "message": "Audit report saved successfully.",
        }

    except Exception as error:
        logger.exception("Failed to save audit report.")
        raise ReportGenerationError(
            "Failed to save audit report.",
            error_detail=str(error),
        ) from error


def build_chat_context(audit_context: dict[str, Any]) -> dict[str, Any]:
    """
    Build compact context for audit Q&A.

    This avoids token overflow and context drift.
    """
    context = extract_report_context(audit_context)

    return {
        "target_column": context.get("target_column"),
        "problem_type": context.get("problem_type"),
        "audit_score": context.get("audit_score"),
        "human_review": context.get("human_review"),
        "data_quality_summary": context.get("data_quality_summary"),
        "leakage_summary": context.get("leakage_summary"),
        "metric_recommendation": context.get("metric_recommendation"),
        "baseline_summary": context.get("baseline_summary"),
        "explainability_summary": context.get("explainability_summary"),
    }


def build_chat_prompt(context: dict[str, Any], user_question: str) -> str:
    question = user_question.strip()[:MAX_CHAT_QUESTION_CHARS]

    context_json = truncate_text(
        json.dumps(context, indent=2, default=str),
        max_chars=MAX_CONTEXT_CHARS,
    )

    return f"""
Answer the user's question using ONLY the audit context.

Rules:
- Do not invent numbers, metrics, or columns.
- If the answer is not in the context, say it is not available.
- Do not claim confirmed leakage. Say possible leakage risk.
- Keep the answer concise and practical.

USER_QUESTION:
{question}

AUDIT_CONTEXT:
```json
{context_json}
```
""".strip()


def ask_about_audit(
    audit_context: dict[str, Any],
    user_question: str,
) -> str:
    """
    Ask LLM a grounded question about the current audit.

    Falls back to deterministic response if LLM is unavailable.
    """
    try:
        if not user_question or not user_question.strip():
            return "Please ask a non-empty question about the audit."

        context = build_chat_context(audit_context)

        if not as_bool(get_config_value("llm.enabled", True)):
            return fallback_audit_answer(context, user_question)

        messages = [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": build_chat_prompt(context, user_question)},
        ]

        return call_groq(messages)

    except Exception as error:
        logger.warning("Audit chat failed. Using fallback: %s", error)
        return fallback_audit_answer(build_chat_context(audit_context), user_question)


def fallback_audit_answer(context: dict[str, Any], user_question: str) -> str:
    """
    Simple deterministic fallback for common audit questions.
    """
    question = user_question.lower()
    leakage = context.get("leakage_summary", {}) or {}
    baseline = context.get("baseline_summary", {}) or {}
    metric = context.get("metric_recommendation", {}) or {}
    human_review = context.get("human_review", {}) or {}

    if "leak" in question:
        return (
            f"The audit found {leakage.get('total_possible_leakage_risks', 0)} "
            f"possible leakage risk(s). Severity: "
            f"{leakage.get('overall_severity', 'none')}. These are not confirmed "
            "leakage and require human review."
        )

    if "metric" in question:
        return (
            f"Recommended primary metric: {metric.get('primary_metric', 'Not available')}. "
            f"Reason: {metric.get('reason', 'Not available in audit context')}"
        )

    if "model" in question or "baseline" in question:
        best_model = baseline.get("best_model", {}) or {}
        return (
            f"Best baseline model: {best_model.get('model_name', 'Not available')}. "
            f"Selection metric: {best_model.get('selection_metric', 'Not available')}. "
            f"Score: {best_model.get('score', 'Not available')}."
        )

    if "ready" in question or "final" in question:
        return (
            f"Human review required: {human_review.get('requires_human_review', 'Not available')}. "
            "Final model readiness should be decided after reviewing leakage, data quality, "
            "and baseline results."
        )

    return (
        "I can answer questions about leakage, metrics, baseline models, data quality, "
        "and human review using the audit context."
    )


if __name__ == "__main__":
    sample_context = {
        "target_column": "Grade",
        "problem_type": "binary_classification",
        "audit_score": {"score": 82, "readiness": "needs_review"},
        "leakage": {
            "total_possible_leakage_risks": 1,
            "overall_severity": "medium",
        },
        "baseline_results": {
            "best_model": {
                "model_name": "Logistic Regression",
                "selection_metric": "f1_score",
                "score": 0.91,
            }
        },
    }

    print(build_audit_report(sample_context))
