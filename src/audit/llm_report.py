from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from groq import Groq

from src.utils.config import get_llm_config
from src.utils.exceptions import AuditCopilotException
from src.utils.logger import get_logger


logger = get_logger(__name__)


def extract_report_context(audit_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract a clean, JSON-safe audit context for LLM report generation.

    This function removes heavy runtime objects such as DataFrames,
    fitted models, sklearn pipelines, and other non-serializable objects.
    The LLM receives only deterministic Python-generated audit results.
    """
    try:
        profile = audit_results.get("profile", {})
        data_quality = audit_results.get("data_quality", {})
        leakage = audit_results.get("leakage", {})
        metric_recommendation = audit_results.get("metric_recommendation", {})
        class_imbalance = audit_results.get("class_imbalance", {})
        baseline_results = audit_results.get("baseline_results", {})
        mlflow_results = audit_results.get("mlflow_results", {})

        dataset_shape = profile.get("shape", {})
        rows = dataset_shape.get("rows", "N/A")
        columns = dataset_shape.get("columns", "N/A")

        return {
            "generated_at": datetime.now().isoformat(),
            "target_column": audit_results.get("target_column", "N/A"),
            "problem_type": audit_results.get("problem_type", "N/A"),
            "dataset_overview": {
                "rows": rows,
                "columns": columns,
                "sample_count": rows,
                "feature_count": (
                    columns - 1
                    if isinstance(columns, int) and columns > 0
                    else "N/A"
                ),
                "column_names": profile.get("columns", []),
                "numeric_columns": profile.get("numeric_columns", []),
                "categorical_columns": profile.get("categorical_columns", []),
                "datetime_columns": profile.get("datetime_columns", []),
                "duplicate_rows": profile.get("duplicate_rows", "N/A"),
                "target_summary": profile.get("target_summary", {}),
                # BUG FIX: profiler.py never produces a top-level
                # "target_distribution" key — the real data lives at
                # profile["target_summary"]["distribution"]. Reading the
                # wrong key meant this was always an empty dict, so the
                # LLM report and fallback report silently lost the actual
                # class/value distribution of the target column.
                "target_distribution": profile.get("target_summary", {}).get(
                    "distribution", {}
                ),
            },
            "data_quality": {
                "missing_values": data_quality.get("missing_values", {}),
                "high_missing_columns": data_quality.get("high_missing_columns", []),
                "duplicate_rows": data_quality.get("duplicate_rows", 0),
                "duplicate_rows_percent": data_quality.get(
                    "duplicate_rows_percent", 0.0
                ),
                "constant_columns": data_quality.get("constant_columns", []),
                "high_cardinality_columns": data_quality.get(
                    "high_cardinality_columns", []
                ),
                "possible_id_columns": data_quality.get("possible_id_columns", []),
                "warnings": data_quality.get("warnings", []),
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
                "reason": metric_recommendation.get("reason", "N/A"),
            },
            "class_imbalance": class_imbalance or {},
            "baseline_results": {
                "models_trained": baseline_results.get("models_trained", []),
                "results": baseline_results.get("results", {}),
                "best_model": baseline_results.get("best_model", {}),
                "message": baseline_results.get("message", ""),
            },
            "mlflow_results": {
                "experiment_name": mlflow_results.get("experiment_name", ""),
                "models_logged": mlflow_results.get("models_logged", []),
                "best_model": mlflow_results.get("best_model", {}),
                "message": mlflow_results.get("message", ""),
            },
        }

    except Exception as e:
        logger.error(f"Report context extraction failed: {e}")
        raise AuditCopilotException(f"Report context extraction failed: {e}") from e


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
16. Write the entire report in plain English only — regardless of any
    language used inside the structured audit results (e.g. column
    names, values). Do not switch to Hindi, Hinglish, or any other
    language at any point.

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
    The caller should use the deterministic fallback report in that case.
    """
    try:
        llm_config = get_llm_config()

        api_key = llm_config.get("api_key")
        model = llm_config.get("model", "llama-3.3-70b-versatile")
        temperature = float(llm_config.get("temperature", 0.2))
        max_tokens = int(llm_config.get("max_tokens", 2000))

        if not api_key:
            logger.warning("GROQ_API_KEY not found. Using fallback report.")
            return None

        logger.info(f"Generating LLM audit report using Groq model: {model}")

        client = Groq(api_key=api_key)
        prompt = build_llm_prompt(report_context)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a Senior Machine Learning Auditor. "
                        "You only explain deterministic Python audit results. "
                        "Never invent numbers. "
                        "Never estimate values. "
                        "Never calculate metrics. "
                        "Never contradict the provided audit context. "
                        "Always write in plain English only."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        report = response.choices[0].message.content

        if not report or not report.strip():
            logger.warning("Groq returned an empty report. Using fallback report.")
            return None

        logger.info("Groq LLM audit report generated successfully")
        return report.strip()

    except Exception as e:
        logger.warning(f"Groq LLM report failed. Using fallback report. Error: {e}")
        return None


def build_fallback_report(report_context: Dict[str, Any]) -> str:
    """
    Build a deterministic Markdown report when the LLM is unavailable.
    """
    try:
        logger.info("Building fallback audit report")

        dataset = report_context.get("dataset_overview", {})
        data_quality = report_context.get("data_quality", {})
        leakage = report_context.get("leakage", {})
        metrics = report_context.get("metric_recommendation", {})
        imbalance = report_context.get("class_imbalance", {})
        baseline = report_context.get("baseline_results", {})
        mlflow = report_context.get("mlflow_results", {})

        leakage_count = leakage.get("total_possible_leakage_risks", 0)
        warnings = data_quality.get("warnings", [])
        best_model = baseline.get("best_model", {})

        primary_metric = metrics.get("primary_metric", "N/A")
        metric_reason = metrics.get("reason", "N/A")

        if leakage_count == 0:
            leakage_level = "Low"
            leakage_message = "No possible leakage risks were detected."
        elif leakage_count <= 3:
            leakage_level = "Medium"
            leakage_message = (
                "Some possible leakage risks were detected and should be reviewed."
            )
        else:
            leakage_level = "High"
            leakage_message = (
                "Multiple possible leakage risks were detected and require careful review."
            )

        if warnings == ["No major basic data quality issues detected."] or not warnings:
            dataset_health = "Good"
            quality_message = "No major basic data quality issues were detected."
        else:
            dataset_health = "Needs Review"
            quality_message = "Some data quality warnings were detected."

        best_model_name = (
            best_model.get("model_name")
            or best_model.get("model")
            or "N/A"
        )
        best_model_score = best_model.get("score", "N/A")
        selection_metric = best_model.get("selection_metric", primary_metric)

        imbalance_severity = imbalance.get("imbalance_severity", "N/A")
        imbalance_warning = imbalance.get("warning", "N/A")
        imbalance_ratio = imbalance.get("imbalance_ratio", "N/A")

        models_trained = baseline.get("models_trained", [])
        models_text = ", ".join(models_trained) if models_trained else "None"

        return f"""# Agentic ML Audit Report

## Executive Summary
This fallback report was generated deterministically because the Groq LLM service was unavailable or unconfigured.

The dataset health is marked as **{dataset_health}**. The audit detected **{leakage_count} possible leakage risks**. The recommended primary metric is **{primary_metric}**. The best baseline model is **{best_model_name}** based on **{selection_metric}**.

## 1. Dataset Overview
- **Total Rows:** {dataset.get("rows", "N/A")}
- **Total Columns:** {dataset.get("columns", "N/A")}
- **Feature Count:** {dataset.get("feature_count", "N/A")}
- **Target Column:** {report_context.get("target_column", "N/A")}
- **Duplicate Rows:** {dataset.get("duplicate_rows", "N/A")}
- **Numeric Columns:** {len(dataset.get("numeric_columns", []))}
- **Categorical Columns:** {len(dataset.get("categorical_columns", []))}
- **Datetime Columns:** {len(dataset.get("datetime_columns", []))}

## 2. Problem Type
- **Detected Problem Type:** {report_context.get("problem_type", "N/A")}

This determines which preprocessing strategy, baseline models, and evaluation metrics should be used.

## 3. Data Quality Audit
- **Status:** {dataset_health}
- **Summary:** {quality_message}
- **Warnings:** {", ".join(warnings) if warnings else "None"}

Data quality should be reviewed before final model training because missing values, duplicate rows, constant columns, and ID-like columns can affect model reliability.

## 4. Possible Leakage Risk
- **Risk Level:** {leakage_level}
- **Total Possible Leakage Risks:** {leakage_count}
- **Summary:** {leakage_message}

These findings are possible leakage risks, not confirmed leakage. A human should verify whether flagged columns would be available at prediction time.

## 5. Metric Recommendation
- **Primary Metric:** {primary_metric}
- **Reason:** {metric_reason}
- **Recommended Metrics:** {", ".join(metrics.get("recommended_metrics", [])) or "N/A"}

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
- **Experiment Name:** {mlflow.get("experiment_name", "N/A")}
- **Models Logged:** {", ".join(mlflow.get("models_logged", [])) or "N/A"}
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

    except Exception as e:
        logger.error(f"Fallback report generation failed: {e}")
        return (
            "# Agentic ML Audit Report\n\n"
            "Error: Could not generate fallback audit report."
        )


def generate_final_report(audit_results: Dict[str, Any]) -> str:
    """
    Generate the final audit report.

    The function first extracts safe deterministic audit context,
    then tries Groq LLM report generation, and finally falls back
    to a deterministic Markdown report if the LLM is unavailable.
    """
    context = extract_report_context(audit_results)

    report = generate_report_with_groq(context)

    if report is None:
        report = build_fallback_report(context)

    return report


def build_audit_report(audit_results: Dict[str, Any]) -> str:
    """
    Backward-compatible wrapper used by workflow.py.
    """
    return generate_final_report(audit_results)


def build_section_explanation(
    section_name: str,
    section_data: Dict[str, Any],
) -> str:
    """
    Build a small deterministic explanation for Streamlit sections.

    This helper is useful when the UI needs a short explanation but the
    full LLM report is not required.
    """
    if section_name == "metric_recommendation":
        primary_metric = section_data.get("primary_metric", "N/A")
        reason = section_data.get("reason", "N/A")
        return (
            f"The recommended primary metric is **{primary_metric}**. "
            f"{reason}"
        )

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
        best_model = section_data.get("best_model", {})
        model_name = best_model.get("model_name", "N/A")
        score = best_model.get("score", "N/A")
        metric = best_model.get("selection_metric", "selected metric")
        return (
            f"The best baseline model is **{model_name}** with a "
            f"{metric} score of **{score}**. These are baseline sanity-check "
            "models, not final optimized models."
        )

    return "Explanation not available for this section."


def save_audit_report(
    report: str,
    output_path: str = "reports/audit_report.md",
) -> Dict[str, Any]:
    """
    Save the generated audit report to disk.
    """
    try:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")

        return {
            "report_path": output_path,
            "message": "Audit report saved successfully.",
        }

    except Exception as e:
        logger.error(f"Saving audit report failed: {e}")
        raise AuditCopilotException(f"Saving audit report failed: {e}") from e


def ask_about_audit(
    audit_context: Dict[str, Any],
    user_question: str,
) -> Optional[str]:
    """
    Answer user questions using ONLY the completed audit results.

    This function is used by the Streamlit audit chat after the audit
    has completed. It never performs new ML computations.
    """
    try:
        if not user_question or not user_question.strip():
            return None

        llm_config = get_llm_config()

        api_key = llm_config.get("api_key")
        model = llm_config.get("model", "llama-3.3-70b-versatile")
        temperature = float(llm_config.get("temperature", 0.2))
        max_tokens = int(llm_config.get("max_tokens", 700))

        if not api_key:
            logger.warning("Groq API key not found. Audit chat unavailable.")
            return None

        safe_context = extract_report_context(audit_context)

        context_json = json.dumps(
            safe_context,
            indent=2,
            default=str,
        )

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
- Always reply in plain English, regardless of what language the
  question was asked in.
- Keep the answer concise and practical.
- Explain ML concepts in beginner-friendly language.

User Question:
{user_question}

Completed Audit Results:
{context_json}
"""

        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an ML Audit Copilot. Answer only from the "
                        "completed audit context. Never invent numbers. "
                        "Never perform new ML computations. Always reply "
                        "in plain English, regardless of what language the "
                        "user's question is written in."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        answer = response.choices[0].message.content

        if not answer or not answer.strip():
            logger.warning("Groq returned an empty answer for audit chat.")
            return None

        return answer.strip()

    except Exception as e:
        logger.error(f"Audit chat failed: {e}")
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
            "target_distribution": {"0": 500, "1": 268},
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
                "A human should review whether these columns would be "
                "available at prediction time."
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
            "warning": (
                "Moderate class imbalance detected. Accuracy alone may be "
                "misleading."
            ),
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

    print("Starting Groq LLM report test...")

    report_output = build_audit_report(sample_results)
    save_result = save_audit_report(report_output)

    print(report_output)
    print(save_result)