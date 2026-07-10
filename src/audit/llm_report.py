from __future__ import annotations

import hashlib
import html
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.utils.config import get_config_value, get_llm_config
from src.utils.exceptions import ReportGenerationError
from src.utils.logger import get_logger

logger = get_logger(__name__)

MAX_CONTEXT_CHARS = 18_000
MAX_CHAT_QUESTION_CHARS = 1_000
DEFAULT_CACHE_DIR = "artifacts/report_cache"
PROMPT_VERSION = "v2.1"

TRUE_VALUES = {"true", "1", "yes", "y", "on"}
FALSE_VALUES = {"false", "0", "no", "n", "off"}

BLOCKED_CONTEXT_KEYS = {
    "trained_model_objects",
    "runtime_objects",
    "model_object",
    "df",
    "dataframe",
    "train_features",
    "test_features",
    "sample_features",
    "sample_target",
    "label_encoder",
    "preprocessor",
    "pipeline",
    "estimator",
}

SUSPICIOUS_UNGROUNDED_PATTERNS = [
    "99% accuracy",
    "100% accuracy",
    "production ready",
    "confirmed leakage",
    "guaranteed",
    "perfect model",
]

NEGATED_MARKETING_PHRASES = {
    "production ready": [
        "not production ready",
        "not yet production ready",
        "isn't production ready",
        "is not production ready",
    ],
    "confirmed leakage": [
        "not confirmed leakage",
        "not a confirmed leakage",
        "not confirmed as leakage",
        "possible leakage risk, not confirmed leakage",
    ],
}


def as_bool(value: Any) -> bool:
    """Convert config/env style values into bool."""
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in TRUE_VALUES:
            return True
        if normalized in FALSE_VALUES:
            return False

    return bool(value)


def safe_int(value: Any, default: int, *, minimum: int | None = None) -> int:
    """Convert a value to int with fallback and optional lower bound."""
    try:
        converted = int(value)
    except (KeyError, TypeError, ValueError, RuntimeError):
        converted = int(default)

    if minimum is not None:
        return max(minimum, converted)

    return converted


def safe_float(value: Any, default: float, *, minimum: float | None = None) -> float:
    """Convert a value to float with fallback and optional lower bound."""
    try:
        converted = float(value)
    except (KeyError, TypeError, ValueError, RuntimeError):
        converted = float(default)

    if not math.isfinite(converted):
        converted = float(default)

    if minimum is not None:
        return max(minimum, converted)

    return converted


def utc_timestamp() -> str:
    """Return an ISO timestamp suitable for report metadata."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def safe_config_str(path: str, default: str) -> str:
    """Read a config value as a non-empty string."""
    try:
        value = get_config_value(path, default)
    except (KeyError, TypeError, ValueError, RuntimeError):
        return default

    text_value = str(value).strip() if value is not None else ""
    return text_value or default


def _json_safe_scalar(value: Any) -> Any:
    """Convert scalar-like objects into JSON-safe values."""
    if value is None or isinstance(value, str | bool | int):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    try:
        # Handles numpy scalar values without importing numpy at module import time.
        if hasattr(value, "item") and callable(value.item):
            return _json_safe_scalar(value.item())
    except (AttributeError, TypeError, ValueError):
        pass

    try:
        if hasattr(value, "isoformat") and callable(value.isoformat):
            return str(value.isoformat())
    except (AttributeError, TypeError, ValueError):
        pass

    return str(value)


def json_safe(data: Any, *, max_list_items: int = 500) -> Any:
    """
    Convert audit output into JSON-safe context.

    Heavy runtime objects are removed before sending anything to the LLM.
    """
    if isinstance(data, dict):
        cleaned: dict[str, Any] = {}

        for key, value in data.items():
            key_name = str(key)
            if key_name in BLOCKED_CONTEXT_KEYS:
                continue
            cleaned[key_name] = json_safe(value, max_list_items=max_list_items)

        return cleaned

    if isinstance(data, list):
        return [
            json_safe(item, max_list_items=max_list_items)
            for item in data[:max_list_items]
        ]

    if isinstance(data, tuple | set):
        items = list(data)
        return [
            json_safe(item, max_list_items=max_list_items)
            for item in items[:max_list_items]
        ]

    # Avoid accidentally passing full pandas objects to the LLM.
    if hasattr(data, "to_dict") and data.__class__.__module__.startswith("pandas"):
        try:
            return json_safe(data.to_dict(), max_list_items=max_list_items)
        except (AttributeError, TypeError, ValueError):
            return str(data)

    return _json_safe_scalar(data)


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
    max_context_chars = safe_int(
        get_config_value("llm.max_context_chars", MAX_CONTEXT_CHARS),
        MAX_CONTEXT_CHARS,
        minimum=2_000,
    )
    max_chat_question_chars = safe_int(
        get_config_value("llm.max_chat_question_chars", MAX_CHAT_QUESTION_CHARS),
        MAX_CHAT_QUESTION_CHARS,
        minimum=100,
    )

    return {
        "llm_enabled": as_bool(get_config_value("llm.enabled", True)),
        "cache_enabled": as_bool(get_config_value("llm.cache_enabled", True)),
        "cache_dir": safe_config_str("llm.cache_dir", DEFAULT_CACHE_DIR),
        "max_retries": safe_int(get_config_value("llm.max_retries", 3), 3, minimum=1),
        "retry_base_seconds": safe_float(
            get_config_value("llm.retry_base_seconds", 1.0),
            1.0,
            minimum=0.1,
        ),
        "hallucination_guard_enabled": as_bool(
            get_config_value("llm.hallucination_guard_enabled", True),
        ),
        "max_context_chars": max_context_chars,
        "max_chat_question_chars": max_chat_question_chars,
        "cache_max_bytes": safe_int(
            get_config_value("llm.cache_max_bytes", 1_000_000),
            1_000_000,
            minimum=10_000,
        ),
    }


def stable_context_hash(context: dict[str, Any]) -> str:
    """Build stable hash for report cache."""
    payload = json.dumps(
        json_safe(context),
        sort_keys=True,
        default=str,
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_cache_path(context: dict[str, Any], cache_dir: str) -> Path:
    """Return cache path for a context hash."""
    digest = stable_context_hash(context)
    safe_version = PROMPT_VERSION.replace(".", "_")
    return Path(cache_dir) / f"audit_report_{safe_version}_{digest}.md"


def read_cached_report(context: dict[str, Any], cache_dir: str) -> str | None:
    """Return cached report if available."""
    path = get_cache_path(context, cache_dir)

    if not path.exists() or not path.is_file():
        return None

    try:
        config = get_report_config()
        if path.stat().st_size > int(config["cache_max_bytes"]):
            logger.warning("Cached report ignored because it is too large: %s", path)
            return None

        logger.info("Using cached audit report: %s", path)
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
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


def _dict_or_empty(value: Any) -> dict[str, Any]:
    """Return value if it is a dict, otherwise an empty dict."""
    return value if isinstance(value, dict) else {}


def extract_report_context(audit_results: dict[str, Any]) -> dict[str, Any]:
    """
    Extract compact audit context for report/chat.

    Do not pass raw dataframes, sklearn models, or runtime objects to the LLM.
    """
    safe = _dict_or_empty(json_safe(audit_results))

    baseline = _dict_or_empty(safe.get("baseline_results"))
    leakage = _dict_or_empty(safe.get("leakage"))
    data_quality = _dict_or_empty(safe.get("data_quality"))
    explainability = _dict_or_empty(safe.get("explainability"))
    shap_info = _dict_or_empty(explainability.get("shap"))

    return {
        "project": {
            "name": safe_config_str("project.name", "Agentic ML Audit Copilot"),
            "version": safe_config_str("project.version", "1.0.0"),
            "philosophy": (
                "Deterministic-first; LLM explains Python-generated audit results only."
            ),
            "prompt_version": PROMPT_VERSION,
        },
        "target_column": safe.get("target_column"),
        "problem_type": safe.get("problem_type"),
        "workflow_status": safe.get("workflow_status"),
        "audit_score": safe.get("audit_score"),
        "human_review": safe.get("human_review"),
        "execution_summary": safe.get("execution_summary"),
        "profile": safe.get("profile"),
        "problem_detection": safe.get("problem_detection"),
        "risk_aggregator": safe.get("risk_aggregator"),
        "decision_router": safe.get("decision_router"),
        "data_quality_summary": {
            "quality_score": data_quality.get("quality_score"),
            "target_quality": data_quality.get("target_quality"),
            "duplicate_rows": data_quality.get("duplicate_rows"),
            "duplicate_row_percent": data_quality.get("duplicate_row_percent"),
            "warnings": data_quality.get("warnings"),
            "recommended_actions": data_quality.get("recommended_actions"),
            "high_missing_columns": data_quality.get("high_missing_columns"),
            "null_only_columns": data_quality.get("null_only_columns"),
            "constant_columns": data_quality.get("constant_columns"),
            "near_constant_columns": data_quality.get("near_constant_columns"),
            "possible_id_columns": data_quality.get("possible_id_columns"),
            "high_cardinality_columns": data_quality.get("high_cardinality_columns"),
            "mixed_type_columns": data_quality.get("mixed_type_columns"),
            "infinite_values": data_quality.get("infinite_values"),
            "outlier_columns": data_quality.get("outlier_columns"),
            "findings": data_quality.get("findings"),
            "finding_summary": data_quality.get("finding_summary"),
        },
        "leakage_summary": {
            "target_column": leakage.get("target_column"),
            "total_possible_leakage_risks": leakage.get("total_possible_leakage_risks"),
            "overall_severity": leakage.get("overall_severity"),
            "risk_summary": leakage.get("risk_summary"),
            "risk_type_summary": leakage.get("risk_type_summary"),
            "review_columns": leakage.get("review_columns"),
            "all_risks": leakage.get("all_risks"),
            "recommended_actions": leakage.get("recommended_actions"),
            "warning": leakage.get("warning"),
            "leakage_policy": leakage.get("leakage_policy"),
        },
        "class_imbalance": safe.get("class_imbalance"),
        "metric_recommendation": safe.get("metric_recommendation"),
        "baseline_summary": {
            "models_trained": baseline.get("models_trained"),
            "results": baseline.get("results"),
            "best_model": baseline.get("best_model"),
            "evaluation_details": baseline.get("evaluation_details"),
            "note": baseline.get("note"),
            "warnings": baseline.get("warnings"),
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
        "optional_failures": safe.get("optional_failures"),
        "warnings": safe.get("warnings"),
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
4. Say "possible leakage risk" unless human review explicitly confirmed leakage.
5. Separate deterministic findings from LLM interpretation.
6. Explain that final decisions require ML engineer review.
7. Do not recommend hyperparameter tuning before data quality and leakage issues are reviewed.
8. Keep the tone professional, concise, and technical.
9. Avoid marketing language. This is an audit report, not a sales page.
10. Never say the project or model is production ready.
""".strip()


def build_report_user_prompt(context: dict[str, Any]) -> str:
    """Build user prompt for audit report generation."""
    config = get_report_config()
    context_json = truncate_text(
        json.dumps(
            json_safe(context),
            indent=2,
            default=str,
            ensure_ascii=False,
            allow_nan=False,
        ),
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
- Do not claim the model, dataset, or system is production ready.

AUDIT_CONTEXT:
```json
{context_json}
```
""".strip()


def get_groq_client() -> Any:
    """Create Groq client from config/env."""
    try:
        from groq import Groq
    except ImportError as error:
        raise ReportGenerationError(
            "groq package is not installed. Cannot generate LLM report.",
            error_detail=str(error),
        ) from error

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


def _extract_response_content(response: Any) -> str:
    """Extract text content from a Groq chat completion response."""
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as error:
        raise ReportGenerationError(
            "LLM response format was not recognized.",
            error_detail=str(error),
        ) from error

    if not content or not str(content).strip():
        raise ReportGenerationError("LLM returned an empty response.")

    return str(content).strip()


def call_groq_once(messages: list[dict[str, str]]) -> tuple[str, dict[str, Any]]:
    """Single Groq call. Retry is handled by call_groq()."""
    llm_config = get_llm_config()
    client = get_groq_client()
    model = str(llm_config.get("model", "llama-3.3-70b-versatile"))

    started_at = time.perf_counter()

    response = client.chat.completions.create(
        model=model,
        temperature=safe_float(llm_config.get("temperature", 0.2), 0.2, minimum=0.0),
        max_tokens=safe_int(llm_config.get("max_tokens", 2_000), 2_000, minimum=256),
        messages=messages,
    )

    latency_seconds = round(time.perf_counter() - started_at, 4)
    content = _extract_response_content(response)

    metadata = {
        "model": model,
        "latency_seconds": latency_seconds,
        "usage": extract_usage(response),
    }

    logger.info(
        "LLM call completed. model=%s latency=%s usage=%s",
        metadata["model"],
        latency_seconds,
        metadata["usage"],
    )

    return content, metadata


def call_groq(messages: list[dict[str, str]]) -> str:
    """
    Call Groq with small retry/backoff.

    Return type remains str so existing workflow/UI code does not break.
    """
    config = get_report_config()
    max_retries = int(config["max_retries"])
    base_seconds = float(config["retry_base_seconds"])

    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            content, _metadata = call_groq_once(messages)
            return content
        except Exception as error:  # noqa: BLE001 - provider exceptions vary by version.
            last_error = error

            if attempt >= max_retries:
                break

            sleep_seconds = base_seconds * (2 ** (attempt - 1))
            logger.warning(
                "LLM call failed. attempt=%s/%s retry_in=%ss error=%s",
                attempt,
                max_retries,
                round(sleep_seconds, 2),
                error,
            )
            time.sleep(sleep_seconds)

    raise ReportGenerationError(
        "LLM call failed after retries.",
        error_detail=str(last_error),
    )


def _is_phrase_negated(report_lower: str, phrase: str) -> bool:
    """Return True if a suspicious phrase appears in an explicitly negated form."""
    negations = NEGATED_MARKETING_PHRASES.get(phrase, [])
    return any(negated in report_lower for negated in negations)


def contains_basic_hallucination_risk(report: str, context: dict[str, Any]) -> bool:
    """
    Lightweight hallucination guard.

    This does not prove correctness. It catches obvious unsafe claims before
    returning LLM output.
    """
    lowered = report.lower()
    context_text = json.dumps(json_safe(context), default=str).lower()

    for pattern in SUSPICIOUS_UNGROUNDED_PATTERNS:
        pattern_lower = pattern.lower()
        if _is_phrase_negated(lowered, pattern_lower):
            continue

        if pattern_lower in lowered and pattern_lower not in context_text:
            logger.warning("Hallucination guard flagged phrase: %s", pattern)
            return True

    if "confirmed leakage" in lowered and "confirmed leakage" not in context_text:
        logger.warning("Hallucination guard flagged confirmed leakage claim.")
        return True

    return False


def _safe_nested_get(mapping: dict[str, Any], key: str, default: Any = None) -> Any:
    """Read dictionary key safely."""
    value = mapping.get(key, default)
    return default if value is None else value


def build_deterministic_report(context: dict[str, Any]) -> str:
    """Build fallback Markdown report when LLM is unavailable or rejected."""
    baseline = _dict_or_empty(context.get("baseline_summary"))
    best_model = _dict_or_empty(baseline.get("best_model"))
    leakage = _dict_or_empty(context.get("leakage_summary"))
    data_quality = _dict_or_empty(context.get("data_quality_summary"))
    metric = _dict_or_empty(context.get("metric_recommendation"))
    human_review = _dict_or_empty(context.get("human_review"))
    explainability = _dict_or_empty(context.get("explainability_summary"))
    audit_score = _dict_or_empty(context.get("audit_score"))
    quality_score = _dict_or_empty(data_quality.get("quality_score"))

    lines = [
        "# Agentic ML Audit Report",
        "",
        f"Generated at: `{utc_timestamp()}`",
        f"Prompt version: `{PROMPT_VERSION}`",
        "",
        "## Executive Summary",
        f"- Target column: `{context.get('target_column', 'Not available')}`",
        f"- Problem type: `{context.get('problem_type', 'Not available')}`",
        f"- Workflow status: `{context.get('workflow_status', 'Not available')}`",
        f"- Audit score: `{audit_score.get('score', 'Not available')}`",
        f"- Readiness: `{audit_score.get('readiness', 'Not available')}`",
        "",
        "## Data Quality Findings",
        f"- Quality score: `{quality_score.get('score', 'Not available')}`",
        f"- Duplicate rows: `{data_quality.get('duplicate_rows', 'Not available')}`",
        f"- Duplicate row percent: `{data_quality.get('duplicate_row_percent', 'Not available')}`",
        f"- Warnings: `{data_quality.get('warnings', [])}`",
        "",
        "## Possible Leakage Risks",
        (
            "- Total possible leakage risks: "
            f"`{leakage.get('total_possible_leakage_risks', 0)}`"
        ),
        f"- Overall severity: `{leakage.get('overall_severity', 'none')}`",
        "- Note: these are possible risks, not confirmed leakage.",
        "",
        "## Class Imbalance / Target Distribution",
        f"- Summary: `{context.get('class_imbalance', 'Not available')}`",
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
        (
            "- Requires human review: "
            f"`{human_review.get('requires_human_review', 'Not available')}`"
        ),
        (
            "- Review items count: "
            f"`{human_review.get('review_items_count', 'Not available')}`"
        ),
        "",
        "## Final Recommendations",
        "- Review data quality findings before trusting baseline scores.",
        "- Treat leakage results as possible risks and verify prediction-time availability.",
        "- Use baseline models as sanity checks, not final optimized models.",
        "- Document all human review decisions before final model development.",
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
    context = extract_report_context(audit_results)

    try:
        logger.info("Starting audit report generation")

        config = get_report_config()

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

    except (
        ReportGenerationError,
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        OSError,
    ) as error:
        logger.warning("LLM report generation failed. Using fallback: %s", error)
        return build_deterministic_report(context)


def markdown_to_simple_html(markdown_text: str) -> str:
    """
    Small dependency-free Markdown-to-HTML fallback.

    This is intentionally simple. Use markdown/markdown2 package later if richer
    HTML is needed.
    """
    lines: list[str] = []
    in_code_block = False

    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()

        if line.startswith("```"):
            if in_code_block:
                lines.append("</code></pre>")
            else:
                lines.append("<pre><code>")
            in_code_block = not in_code_block
            continue

        if in_code_block:
            lines.append(html.escape(line))
            continue

        if line.startswith("### "):
            lines.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif line.startswith("## "):
            lines.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("# "):
            lines.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("- "):
            lines.append(f"<p>• {html.escape(line[2:])}</p>")
        elif not line.strip():
            lines.append("<br>")
        else:
            lines.append(f"<p>{html.escape(line)}</p>")

    if in_code_block:
        lines.append("</code></pre>")

    return "\n".join(lines)


def _normalize_report_path(path: Path) -> tuple[Path, str]:
    """Return a path with supported extension and the normalized extension."""
    extension = path.suffix.lower()

    if extension in {".md", ".html", ".json"}:
        return path, extension

    if not extension:
        return path.with_suffix(".md"), ".md"

    return path.with_suffix(".md"), ".md"


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
        if not isinstance(report, str) or not report.strip():
            raise ReportGenerationError("Report content is empty and cannot be saved.")

        resolved_output_path: str | Path
        if output_path is None:
            resolved_output_path = safe_config_str(
                "reports.default_report_path",
                "reports/audit_report.md",
            )
        else:
            resolved_output_path = output_path

        path = Path(str(resolved_output_path)).expanduser()
        path, extension = _normalize_report_path(path)

        path.parent.mkdir(parents=True, exist_ok=True)

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
                            "code{background:#f1f5f9;padding:2px 5px;border-radius:4px;} "
                            "pre{background:#0f172a;color:#e5e7eb;padding:16px;"
                            "border-radius:8px;overflow-x:auto;}"
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
                        "generated_at": utc_timestamp(),
                        "prompt_version": PROMPT_VERSION,
                        "report_markdown": report,
                    },
                    indent=2,
                    ensure_ascii=False,
                    allow_nan=False,
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

    except ReportGenerationError:
        raise
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
        "workflow_status": context.get("workflow_status"),
        "audit_score": context.get("audit_score"),
        "human_review": context.get("human_review"),
        "data_quality_summary": context.get("data_quality_summary"),
        "leakage_summary": context.get("leakage_summary"),
        "class_imbalance": context.get("class_imbalance"),
        "metric_recommendation": context.get("metric_recommendation"),
        "baseline_summary": context.get("baseline_summary"),
        "explainability_summary": context.get("explainability_summary"),
        "risk_aggregator": context.get("risk_aggregator"),
        "decision_router": context.get("decision_router"),
    }


def build_chat_prompt(context: dict[str, Any], user_question: str) -> str:
    """Build grounded chat prompt from compact audit context."""
    config = get_report_config()
    question = user_question.strip()[: int(config["max_chat_question_chars"])]

    context_json = truncate_text(
        json.dumps(
            json_safe(context),
            indent=2,
            default=str,
            ensure_ascii=False,
            allow_nan=False,
        ),
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

    except (
        ReportGenerationError,
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        OSError,
    ) as error:
        logger.warning("Audit chat failed. Using fallback: %s", error)
        return fallback_audit_answer(build_chat_context(audit_context), user_question)


def fallback_audit_answer(context: dict[str, Any], user_question: str) -> str:
    """Deterministic fallback for common audit questions."""
    question = user_question.lower()
    leakage = _dict_or_empty(context.get("leakage_summary"))
    baseline = _dict_or_empty(context.get("baseline_summary"))
    metric = _dict_or_empty(context.get("metric_recommendation"))
    human_review = _dict_or_empty(context.get("human_review"))
    data_quality = _dict_or_empty(context.get("data_quality_summary"))
    explainability = _dict_or_empty(context.get("explainability_summary"))
    class_imbalance = _dict_or_empty(context.get("class_imbalance"))

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
        best_model = _dict_or_empty(baseline.get("best_model"))
        return (
            f"Best baseline model: {best_model.get('model_name', 'Not available')}. "
            f"Selection metric: {best_model.get('selection_metric', 'Not available')}. "
            f"Score: {best_model.get('score', 'Not available')}."
        )

    if "quality" in question or "missing" in question:
        quality_score = _dict_or_empty(data_quality.get("quality_score"))
        return (
            f"Data quality score: {quality_score.get('score', 'Not available')}. "
            f"Duplicate rows: {data_quality.get('duplicate_rows', 'Not available')}. "
            f"Warnings: {data_quality.get('warnings', [])}."
        )

    if "imbalance" in question or "class" in question:
        return (
            "Class imbalance severity: "
            f"{class_imbalance.get('imbalance_severity', 'Not available')}. "
            f"Warning: {class_imbalance.get('warning', 'Not available in audit context')}"
        )

    if "explain" in question or "shap" in question or "feature" in question:
        summary = _dict_or_empty(explainability.get("summary"))
        return (
            f"Explainability available: {explainability.get('available', 'Not available')}. "
            f"Top feature: {summary.get('top_feature', 'Not available')}. "
            f"Message: {summary.get('message', 'Not available in audit context')}"
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
        "class imbalance, explainability, and human review using the audit context."
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
