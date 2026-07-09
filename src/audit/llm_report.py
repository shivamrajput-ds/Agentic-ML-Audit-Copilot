from __future__ import annotations

import hashlib
import html
import json
import time
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
DEFAULT_CACHE_DIR = "artifacts/report_cache"
PROMPT_VERSION = "v2.0"

TRUE_VALUES = {"true", "1", "yes", "y", "on"}

BLOCKED_CONTEXT_KEYS = {
    "trained_model_objects",
    "runtime_objects",
    "model_object",
    "df",
    "train_features",
    "test_features",
    "sample_features",
    "sample_target",
    "label_encoder",
}

SUSPICIOUS_UNGROUNDED_PATTERNS = [
    "99% accuracy",
    "100% accuracy",
    "production ready",
    "confirmed leakage",
    "guaranteed",
    "perfect model",
]


def as_bool(value: Any) -> bool:
    """Convert config/env style values into bool."""
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() in TRUE_VALUES

    return bool(value)


def safe_int(value: Any, default: int) -> int:
    """Read integer config values without crashing report generation."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float) -> float:
    """Read float config values without crashing report generation."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def json_safe(data: Any) -> Any:
    """
    Convert audit output into JSON-safe context.

    Heavy runtime objects are removed before sending anything to the LLM.
    """
    if isinstance(data, dict):
        cleaned: dict[str, Any] = {}

        for key, value in data.items():
            if str(key) in BLOCKED_CONTEXT_KEYS:
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
    """Keep prompt context inside a safe token budget."""
    if len(text) <= max_chars:
        return text

    return (
        text[:max_chars]
        + "\n\n[TRUNCATED: audit context was longer than allowed budget]"
    )


def get_report_config() -> dict[str, Any]:
    """
    Central config for report generation.

    All keys are optional. If config.yaml does not contain them, safe defaults are used.
    """
    return {
        "llm_enabled": as_bool(get_config_value("llm.enabled", True)),
        "cache_enabled": as_bool(get_config_value("llm.cache_enabled", True)),
        "cache_dir": str(get_config_value("llm.cache_dir", DEFAULT_CACHE_DIR)),
        "max_retries": safe_int(get_config_value("llm.max_retries", 3), 3),
        "retry_base_seconds": safe_float(
            get_config_value("llm.retry_base_seconds", 1.0),
            1.0,
        ),
        "hallucination_guard_enabled": as_bool(
            get_config_value("llm.hallucination_guard_enabled", True),
        ),
        "max_context_chars": safe_int(
            get_config_value("llm.max_context_chars", MAX_CONTEXT_CHARS),
            MAX_CONTEXT_CHARS,
        ),
        "max_chat_question_chars": safe_int(
            get_config_value("llm.max_chat_question_chars", MAX_CHAT_QUESTION_CHARS),
            MAX_CHAT_QUESTION_CHARS,
        ),
    }


def stable_context_hash(context: dict[str, Any]) -> str:
    """Build stable hash for report cache."""
    payload = json.dumps(context, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_cache_path(context: dict[str, Any], cache_dir: str) -> Path:
    """Return cache path for a context hash."""
    digest = stable_context_hash(context)
    return Path(cache_dir) / f"audit_report_{PROMPT_VERSION}_{digest}.md"


def read_cached_report(context: dict[str, Any], cache_dir: str) -> str | None:
    """Return cached report if available."""
    path = get_cache_path(context, cache_dir)

    if not path.exists():
        return None

    try:
        logger.info("Using cached audit report: %s", path)
        return path.read_text(encoding="utf-8")
    except OSError as error:
        logger.warning("Could not read cached report: %s", error)
        return None


def write_cached_report(context: dict[str, Any], report: str, cache_dir: str) -> None:
    """Save report cache. Cache failure should never fail the audit."""
    try:
        path = get_cache_path(context, cache_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")
    except OSError as error:
        logger.warning("Could not write cached report: %s", error)


def extract_report_context(audit_results: dict[str, Any]) -> dict[str, Any]:
    """
    Extract compact audit context for report/chat.

    Do not pass raw dataframes, sklearn models, or runtime objects to the LLM.
    """
    safe = json_safe(audit_results)

    baseline = safe.get("baseline_results", {}) or {}
    leakage = safe.get("leakage", {}) or {}
    data_quality = safe.get("data_quality", {}) or {}
    explainability = safe.get("explainability", {}) or {}

    if not isinstance(baseline, dict):
        baseline = {}

    if not isinstance(leakage, dict):
        leakage = {}

    if not isinstance(data_quality, dict):
        data_quality = {}

    if not isinstance(explainability, dict):
        explainability = {}

    shap_info = explainability.get("shap", {})
    if not isinstance(shap_info, dict):
        shap_info = {}

    return {
        "project": {
            "name": get_config_value("project.name", "Agentic ML Audit Copilot"),
            "version": get_config_value("project.version", "1.0.0"),
            "philosophy": (
                "Deterministic-first; LLM explains Python-generated audit results only."
            ),
            "prompt_version": PROMPT_VERSION,
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
            "duplicate_row_percent": data_quality.get("duplicate_row_percent"),
            "warnings": data_quality.get("warnings"),
            "recommended_actions": data_quality.get("recommended_actions"),
            "high_missing_columns": data_quality.get("high_missing_columns"),
            "possible_id_columns": data_quality.get("possible_id_columns"),
            "high_cardinality_columns": data_quality.get("high_cardinality_columns"),
            "findings": data_quality.get("findings"),
        },
        "leakage_summary": {
            "target_column": leakage.get("target_column"),
            "total_possible_leakage_risks": leakage.get("total_possible_leakage_risks"),
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
                "builtin_feature_importance",
            ),
            "shap": {
                "available": shap_info.get("available"),
                "method": shap_info.get("method"),
                "global_importance": shap_info.get("global_importance"),
                "message": shap_info.get("message"),
            },
        },
        "mlflow_results": safe.get("mlflow_results"),
    }


def build_system_prompt() -> str:
    """Strict prompt for grounded audit reporting."""
    return f"""
You are an ML audit report writer for a deterministic-first ML audit system.

Prompt version: {PROMPT_VERSION}

Rules:
1. Use ONLY the provided audit context.
2. Never invent metrics, scores, model names, column names, risks, or dataset properties.
3. If a value is missing, write "Not available in audit context".
4. Do not claim confirmed leakage. Say "possible leakage risk" unless explicitly confirmed by human review.
5. Separate deterministic findings from LLM interpretation.
6. Explain that final decisions require ML engineer review.
7. Do not recommend hyperparameter tuning before data quality and leakage issues are reviewed.
8. Keep the tone professional, concise, and technical.
9. Avoid marketing language. This is an audit report, not a sales page.
""".strip()


def build_report_user_prompt(context: dict[str, Any]) -> str:
    """Build user prompt for audit report generation."""
    config = get_report_config()
    context_json = truncate_text(
        json.dumps(context, indent=2, default=str),
        max_chars=int(config["max_context_chars"]),
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
- Mention that baseline models are sanity-check models, not final optimized models.

AUDIT_CONTEXT:
```json
{context_json}
```
""".strip()


def get_groq_client() -> Groq:
    """Create Groq client from config/env."""
    llm_config = get_llm_config()
    api_key = llm_config.get("api_key")

    if not api_key:
        raise ReportGenerationError(
            "GROQ_API_KEY is not configured. Cannot generate LLM report.",
        )

    return Groq(api_key=str(api_key))


def extract_usage(response: Any) -> dict[str, Any]:
    """Extract token usage if provider returns it."""
    usage = getattr(response, "usage", None)

    if usage is None:
        return {}

    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def call_groq_once(messages: list[dict[str, str]]) -> tuple[str, dict[str, Any]]:
    """Single Groq call. Retry is handled by call_groq()."""
    llm_config = get_llm_config()
    client = get_groq_client()

    started_at = time.perf_counter()

    response = client.chat.completions.create(
        model=str(llm_config.get("model", "llama-3.3-70b-versatile")),
        temperature=safe_float(llm_config.get("temperature", 0.2), 0.2),
        max_tokens=safe_int(llm_config.get("max_tokens", 2_000), 2_000),
        messages=messages,
    )

    latency_seconds = round(time.perf_counter() - started_at, 4)
    content = response.choices[0].message.content

    if not content:
        raise ReportGenerationError("LLM returned an empty response.")

    metadata = {
        "model": str(llm_config.get("model", "llama-3.3-70b-versatile")),
        "latency_seconds": latency_seconds,
        "usage": extract_usage(response),
    }

    logger.info(
        "LLM call completed. model=%s latency=%s usage=%s",
        metadata["model"],
        latency_seconds,
        metadata["usage"],
    )

    return str(content), metadata


def call_groq(messages: list[dict[str, str]]) -> str:
    """
    Call Groq with small retry/backoff.

    Return type remains str so existing workflow/UI code does not break.
    """
    config = get_report_config()
    max_retries = max(1, int(config["max_retries"]))
    base_seconds = max(0.1, float(config["retry_base_seconds"]))

    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            content, _metadata = call_groq_once(messages)
            return content
        except (AttributeError, TypeError, ValueError, OSError) as error:
            last_error = error

            if attempt >= max_retries:
                break

            sleep_seconds = base_seconds * (2 ** (attempt - 1))
            logger.warning(
                "LLM call failed. attempt=%s/%s retry_in=%ss error=%s",
                attempt,
                max_retries,
                sleep_seconds,
                error,
            )
            time.sleep(sleep_seconds)

    raise ReportGenerationError(
        "LLM call failed after retries.",
        error_detail=str(last_error),
    )


def contains_basic_hallucination_risk(report: str, context: dict[str, Any]) -> bool:
    """
    Lightweight hallucination guard.

    This does not prove correctness. It catches obvious unsafe claims before
    returning LLM output.
    """
    lowered = report.lower()
    context_text = json.dumps(context, default=str).lower()

    for pattern in SUSPICIOUS_UNGROUNDED_PATTERNS:
        if pattern in lowered and pattern not in context_text:
            logger.warning("Hallucination guard flagged phrase: %s", pattern)
            return True

    if "confirmed leakage" in lowered and "confirmed leakage" not in context_text:
        logger.warning("Hallucination guard flagged confirmed leakage claim.")
        return True

    return False


def build_deterministic_report(context: dict[str, Any]) -> str:
    """Build fallback Markdown report when LLM is unavailable or rejected."""
    baseline = context.get("baseline_summary", {}) or {}
    best_model = baseline.get("best_model", {}) or {}
    leakage = context.get("leakage_summary", {}) or {}
    data_quality = context.get("data_quality_summary", {}) or {}
    metric = context.get("metric_recommendation", {}) or {}
    human_review = context.get("human_review", {}) or {}
    explainability = context.get("explainability_summary", {}) or {}
    audit_score = context.get("audit_score") or {}

    if not isinstance(best_model, dict):
        best_model = {}

    if not isinstance(audit_score, dict):
        audit_score = {}

    lines = [
        "# Agentic ML Audit Report",
        "",
        f"Generated at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"Prompt version: `{PROMPT_VERSION}`",
        "",
        "## Executive Summary",
        f"- Target column: `{context.get('target_column', 'Not available')}`",
        f"- Problem type: `{context.get('problem_type', 'Not available')}`",
        f"- Audit score: `{audit_score.get('score', 'Not available')}`",
        f"- Readiness: `{audit_score.get('readiness', 'Not available')}`",
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
        f"- Scoring metric: `{metric.get('scoring_metric', 'Not available')}`",
        f"- Reason: {metric.get('reason', 'Not available in audit context')}",
        "",
        "## Baseline Model Results",
        f"- Best model: `{best_model.get('model_name', 'Not available')}`",
        f"- Selection metric: `{best_model.get('selection_metric', 'Not available')}`",
        f"- Score: `{best_model.get('score', 'Not available')}`",
        "",
        "## Explainability Findings",
        f"- Available: `{explainability.get('available', 'Not available')}`",
        f"- Best model name: `{explainability.get('best_model_name', 'Not available')}`",
        f"- Model type: `{explainability.get('model_type', 'Not available')}`",
        "",
        "## Human-in-the-loop Review",
        f"- Requires human review: `{human_review.get('requires_human_review', 'Not available')}`",
        f"- Review items count: `{human_review.get('review_items_count', 'Not available')}`",
        "",
        "## Limitations",
        "- This report is based on deterministic audit checks.",
        "- Leakage findings are heuristic and require domain review.",
        "- Baseline models are sanity-check models, not final tuned models.",
        "- LLM text is explanatory only; it does not perform ML computation.",
    ]

    return "\n".join(lines)


def build_audit_report(audit_results: dict[str, Any]) -> str:
    """
    Build audit report with LLM and deterministic fallback.

    Existing callers can keep using this function exactly as before.
    """
    try:
        logger.info("Starting audit report generation")

        config = get_report_config()
        context = extract_report_context(audit_results)

        if not bool(config["llm_enabled"]):
            logger.info("LLM report disabled. Using deterministic fallback.")
            return build_deterministic_report(context)

        if bool(config["cache_enabled"]):
            cached_report = read_cached_report(context, str(config["cache_dir"]))
            if cached_report:
                return cached_report

        messages = [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": build_report_user_prompt(context)},
        ]

        report = call_groq(messages)

        if bool(
            config["hallucination_guard_enabled"]
        ) and contains_basic_hallucination_risk(
            report=report,
            context=context,
        ):
            logger.warning(
                "LLM report rejected by hallucination guard. Using fallback."
            )
            return build_deterministic_report(context)

        if bool(config["cache_enabled"]):
            write_cached_report(context, report, str(config["cache_dir"]))

        logger.info("Audit report generated successfully")
        return report

    except (AttributeError, KeyError, TypeError, ValueError, OSError) as error:
        logger.warning("LLM report generation failed. Using fallback: %s", error)
        context = extract_report_context(audit_results)
        return build_deterministic_report(context)


def markdown_to_simple_html(markdown_text: str) -> str:
    """
    Small dependency-free Markdown-to-HTML fallback.

    This is intentionally simple. Use markdown/markdown2 package later if richer
    HTML is needed.
    """
    lines: list[str] = []

    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()

        if line.startswith("# "):
            lines.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            lines.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("### "):
            lines.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif line.startswith("- "):
            lines.append(f"<p>• {html.escape(line[2:])}</p>")
        elif not line.strip():
            lines.append("<br>")
        else:
            lines.append(f"<p>{html.escape(line)}</p>")

    return "\n".join(lines)


def save_audit_report(
    report: str,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Save audit report.

    Supported extensions:
    - .md  : raw Markdown
    - .html: simple HTML wrapper
    - .json: JSON with report metadata
    """
    try:
        if output_path is None:
            output_path = get_config_value(
                "reports.default_report_path",
                "reports/audit_report.md",
            )

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        extension = path.suffix.lower()

        if extension == ".html":
            html_body = markdown_to_simple_html(report)
            path.write_text(
                "\n".join(
                    [
                        "<!doctype html>",
                        "<html>",
                        "<head>",
                        "<meta charset='utf-8'>",
                        "<title>Agentic ML Audit Report</title>",
                        "<style>",
                        (
                            "body{font-family:Inter,Arial,sans-serif;max-width:980px;"
                            "margin:40px auto;line-height:1.6;color:#111827;}"
                        ),
                        (
                            "h1,h2,h3{color:#0f172a;} "
                            "code{background:#f1f5f9;padding:2px 5px;border-radius:4px;}"
                        ),
                        "</style>",
                        "</head>",
                        "<body>",
                        html_body,
                        "</body>",
                        "</html>",
                    ],
                ),
                encoding="utf-8",
            )
        elif extension == ".json":
            path.write_text(
                json.dumps(
                    {
                        "generated_at": datetime.now().isoformat(timespec="seconds"),
                        "prompt_version": PROMPT_VERSION,
                        "report_markdown": report,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        else:
            path.write_text(report, encoding="utf-8")

        return {
            "report_path": str(path),
            "format": extension.replace(".", "") or "md",
            "message": "Audit report saved successfully.",
        }

    except OSError as error:
        logger.exception("Failed to save audit report.")
        raise ReportGenerationError(
            "Failed to save audit report.",
            error_detail=str(error),
        ) from error


def build_chat_context(audit_context: dict[str, Any]) -> dict[str, Any]:
    """Build compact context for audit Q&A."""
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
    """Build grounded chat prompt from compact audit context."""
    config = get_report_config()
    question = user_question.strip()[: int(config["max_chat_question_chars"])]

    context_json = truncate_text(
        json.dumps(context, indent=2, default=str),
        max_chars=int(config["max_context_chars"]),
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
    """Ask a grounded question about the current audit."""
    try:
        if not user_question or not user_question.strip():
            return "Please ask a non-empty question about the audit."

        config = get_report_config()
        context = build_chat_context(audit_context)

        if not bool(config["llm_enabled"]):
            return fallback_audit_answer(context, user_question)

        messages = [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": build_chat_prompt(context, user_question)},
        ]

        answer = call_groq(messages)

        if bool(
            config["hallucination_guard_enabled"]
        ) and contains_basic_hallucination_risk(
            report=answer,
            context=context,
        ):
            return fallback_audit_answer(context, user_question)

        return answer

    except (AttributeError, KeyError, TypeError, ValueError, OSError) as error:
        logger.warning("Audit chat failed. Using fallback: %s", error)
        return fallback_audit_answer(build_chat_context(audit_context), user_question)


def fallback_audit_answer(context: dict[str, Any], user_question: str) -> str:
    """Deterministic fallback for common audit questions."""
    question = user_question.lower()
    leakage = context.get("leakage_summary", {}) or {}
    baseline = context.get("baseline_summary", {}) or {}
    metric = context.get("metric_recommendation", {}) or {}
    human_review = context.get("human_review", {}) or {}
    data_quality = context.get("data_quality_summary", {}) or {}

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

    if "quality" in question or "missing" in question:
        return (
            "Data quality score: "
            f"{(data_quality.get('quality_score') or {}).get('score', 'Not available')}. "
            f"Duplicate rows: {data_quality.get('duplicate_rows', 'Not available')}. "
            f"Warnings: {data_quality.get('warnings', [])}."
        )

    if "ready" in question or "final" in question:
        return (
            "Human review required: "
            f"{human_review.get('requires_human_review', 'Not available')}. "
            "Final readiness should be decided after reviewing leakage, data quality, "
            "class imbalance, explainability, and baseline results."
        )

    return (
        "I can answer questions about leakage, metrics, baseline models, data quality, "
        "explainability, and human review using the audit context."
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
            },
        },
    }

    print(build_audit_report(sample_context))
