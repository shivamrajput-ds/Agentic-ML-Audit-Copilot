from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from src.utils.config import get_config_value
from src.utils.exceptions import InvalidTargetColumnError, LeakageDetectionError
from src.utils.logger import get_logger


logger = get_logger(__name__)


DEFAULT_TARGET_LIKE_KEYWORDS = [
    "target", "label", "outcome", "result", "final_result", "prediction",
    "predicted", "actual", "ground_truth", "grade", "score", "rank",
    "pass", "fail", "status", "class", "total",
]


def validate_inputs(df: pd.DataFrame, target_column: str) -> None:
    if df is None or df.empty:
        raise LeakageDetectionError("Dataset is empty.")

    if target_column is None or str(target_column).strip() == "":
        raise InvalidTargetColumnError("Target column is required.")

    if target_column not in df.columns:
        raise InvalidTargetColumnError(
            f"Target column '{target_column}' not found in dataset."
        )

    if df[target_column].dropna().empty:
        raise InvalidTargetColumnError(
            f"Target column '{target_column}' has only missing values."
        )


def get_target_like_keywords() -> list[str]:
    raw = get_config_value("leakage.target_like_keywords", DEFAULT_TARGET_LIKE_KEYWORDS)

    if not isinstance(raw, list):
        return DEFAULT_TARGET_LIKE_KEYWORDS

    return [str(item).lower().strip() for item in raw]


def get_thresholds() -> dict[str, float]:
    return {
        "high_correlation_threshold": float(
            get_config_value("leakage.high_correlation_threshold", 0.90)
        ),
        "classification_proxy_threshold": float(
            get_config_value("leakage.classification_proxy_threshold", 0.75)
        ),
        "duplicate_target_threshold": float(
            get_config_value("leakage.duplicate_target_threshold", 0.95)
        ),
        "mutual_information_threshold": float(
            get_config_value("leakage.mutual_information_threshold", 0.20)
        ),
        "id_unique_percent_threshold": float(
            get_config_value("audit.id_unique_percent_threshold", 95)
        ),
        "high_cardinality_threshold": float(
            get_config_value("audit.high_cardinality_threshold", 50)
        ),
    }


def safe_percent(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return round((float(numerator) / float(denominator)) * 100, 2)


def add_risk(
    risks: list[dict[str, Any]],
    column: str,
    risk_type: str,
    risk_level: str,
    reason: str,
    evidence: dict[str, Any] | None = None,
) -> None:
    risks.append(
        {
            "column": column,
            "risk_type": risk_type,
            "risk_level": risk_level,
            "is_confirmed_leakage": False,
            "requires_human_review": True,
            "reason": reason,
            "evidence": evidence or {},
        }
    )


def find_name_based_leakage_risks(
    df: pd.DataFrame,
    target_column: str,
) -> list[dict[str, Any]]:
    risks: list[dict[str, Any]] = []
    keywords = get_target_like_keywords()
    target_lower = str(target_column).lower().strip()

    for column in df.columns:
        if column == target_column:
            continue

        column_lower = str(column).lower().strip()

        matched_keywords = [
            keyword
            for keyword in keywords
            if (
                keyword == column_lower
                or column_lower.endswith(f"_{keyword}")
                or column_lower.startswith(f"{keyword}_")
                or f"_{keyword}_" in column_lower
            )
        ]

        target_name_overlap = (
            target_lower in column_lower
            or column_lower in target_lower
        )

        if matched_keywords or target_name_overlap:
            add_risk(
                risks=risks,
                column=column,
                risk_type="name_based_risk",
                risk_level="medium",
                reason=(
                    "Column name looks related to target/outcome. This is only a "
                    "possible leakage signal and requires prediction-time availability review."
                ),
                evidence={
                    "matched_keywords": matched_keywords,
                    "target_name_overlap": target_name_overlap,
                },
            )

    return risks


def find_direct_duplicate_target_columns(
    df: pd.DataFrame,
    target_column: str,
    threshold: float,
) -> list[dict[str, Any]]:
    risks: list[dict[str, Any]] = []
    target_as_str = df[target_column].fillna("__MISSING_TARGET__").astype(str)

    for column in df.columns:
        if column == target_column:
            continue

        feature_as_str = df[column].fillna("__MISSING_FEATURE__").astype(str)
        same_values_ratio = float(feature_as_str.eq(target_as_str).mean())

        if same_values_ratio >= threshold:
            add_risk(
                risks=risks,
                column=column,
                risk_type="duplicate_target_risk",
                risk_level="critical",
                reason=(
                    "Column values are almost identical to the target column. "
                    "This is a critical possible leakage risk."
                ),
                evidence={
                    "same_values_ratio": round(same_values_ratio, 4),
                    "threshold": threshold,
                },
            )

    return risks


def find_numeric_correlation_risks(
    df: pd.DataFrame,
    target_column: str,
    threshold: float,
) -> list[dict[str, Any]]:
    risks: list[dict[str, Any]] = []

    if not pd.api.types.is_numeric_dtype(df[target_column]):
        return risks

    numeric_df = df.select_dtypes(include=["number"]).replace([np.inf, -np.inf], np.nan)

    if target_column not in numeric_df.columns or numeric_df.shape[1] <= 1:
        return risks

    correlations = numeric_df.corr(numeric_only=True)[target_column].drop(
        labels=[target_column],
        errors="ignore",
    )

    for column, corr_value in correlations.items():
        if pd.isna(corr_value):
            continue

        abs_corr = abs(float(corr_value))

        if abs_corr >= threshold:
            level = "high" if abs_corr >= 0.98 else "medium"
            add_risk(
                risks=risks,
                column=column,
                risk_type="high_correlation_risk",
                risk_level=level,
                reason=(
                    "Column has very high correlation with the numeric target. "
                    "This may be valid signal, direct dependency, or possible leakage. "
                    "Confirm whether the feature is available before prediction time."
                ),
                evidence={
                    "correlation_with_target": round(float(corr_value), 4),
                    "absolute_correlation": round(abs_corr, 4),
                    "threshold": threshold,
                },
            )

    return risks


def prepare_numeric_matrix(
    df: pd.DataFrame,
    numeric_columns: list[str],
) -> pd.DataFrame:
    numeric_df = df[numeric_columns].replace([np.inf, -np.inf], np.nan)

    for column in numeric_df.columns:
        median = numeric_df[column].median()
        if pd.isna(median):
            median = 0
        numeric_df[column] = numeric_df[column].fillna(median)

    return numeric_df


def find_mutual_information_risks(
    df: pd.DataFrame,
    target_column: str,
    threshold: float,
) -> list[dict[str, Any]]:
    """
    Mutual information based target-proxy heuristic.

    This avoids arbitrary factorized-target Pearson correlation for classification.
    """
    risks: list[dict[str, Any]] = []

    feature_numeric_columns = (
        df.drop(columns=[target_column])
        .select_dtypes(include=["number"])
        .columns
        .tolist()
    )

    if not feature_numeric_columns:
        return risks

    valid_target_mask = df[target_column].notna()
    target = df.loc[valid_target_mask, target_column]

    if target.nunique(dropna=True) < 2:
        return risks

    X = prepare_numeric_matrix(
        df.loc[valid_target_mask],
        feature_numeric_columns,
    )

    try:
        if pd.api.types.is_numeric_dtype(target) and target.nunique(dropna=True) > 20:
            mi_scores = mutual_info_regression(
                X,
                pd.to_numeric(target, errors="coerce").fillna(target.median()),
                random_state=int(get_config_value("random_seed", 42)),
            )
            risk_type = "numeric_target_mutual_information_risk"
        else:
            y = target.astype(str)
            mi_scores = mutual_info_classif(
                X,
                y,
                random_state=int(get_config_value("random_seed", 42)),
            )
            risk_type = "classification_target_mutual_information_risk"

    except Exception as error:
        logger.warning("Mutual information leakage heuristic skipped: %s", error)
        return risks

    for column, score in zip(feature_numeric_columns, mi_scores):
        score_float = float(score)

        if score_float >= threshold:
            add_risk(
                risks=risks,
                column=column,
                risk_type=risk_type,
                risk_level="medium",
                reason=(
                    "Feature has high mutual information with the target. This can be a "
                    "valid strong predictor or a possible target proxy. Human review is required."
                ),
                evidence={
                    "mutual_information": round(score_float, 4),
                    "threshold": threshold,
                },
            )

    return risks


def find_classification_proxy_risks(
    df: pd.DataFrame,
    target_column: str,
    threshold: float,
) -> list[dict[str, Any]]:
    risks: list[dict[str, Any]] = []
    target = df[target_column]

    unique_classes = target.nunique(dropna=True)

    if unique_classes < 2 or unique_classes > 50:
        return risks

    numeric_columns = (
        df.drop(columns=[target_column])
        .select_dtypes(include=["number"])
        .columns
    )

    for column in numeric_columns:
        feature = df[column].replace([np.inf, -np.inf], np.nan)

        grouped_means = (
            pd.DataFrame({target_column: target, column: feature})
            .dropna()
            .groupby(target_column)[column]
            .mean()
        )

        if grouped_means.empty or grouped_means.isna().all():
            continue

        min_mean = float(grouped_means.min())
        max_mean = float(grouped_means.max())

        denominator = max(abs(max_mean), abs(min_mean), 1e-9)
        separation_ratio = abs(max_mean - min_mean) / denominator

        if separation_ratio >= threshold:
            add_risk(
                risks=risks,
                column=column,
                risk_type="target_proxy_separation_risk",
                risk_level="medium",
                reason=(
                    "Numeric feature values are strongly separated across target classes. "
                    "This may be valid predictive signal or possible target proxy."
                ),
                evidence={
                    "separation_ratio": round(float(separation_ratio), 4),
                    "threshold": threshold,
                    "class_means": {
                        str(key): round(float(value), 4)
                        for key, value in grouped_means.items()
                    },
                },
            )

    return risks


def find_high_cardinality_review_columns(
    df: pd.DataFrame,
    target_column: str,
    thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    risks: list[dict[str, Any]] = []

    for column in df.columns:
        if column == target_column:
            continue

        unique_count = int(df[column].nunique(dropna=True))
        unique_percent = safe_percent(unique_count, len(df))

        if (
            unique_count >= thresholds["high_cardinality_threshold"]
            and unique_percent >= thresholds["id_unique_percent_threshold"]
        ):
            add_risk(
                risks=risks,
                column=column,
                risk_type="high_cardinality_identifier_risk",
                risk_level="medium",
                reason=(
                    "Column has very high uniqueness and may be an identifier. "
                    "Identifiers can cause memorization and poor generalization."
                ),
                evidence={
                    "unique_count": unique_count,
                    "unique_percent": unique_percent,
                },
            )

    return risks


def summarize_risks(all_risks: list[dict[str, Any]]) -> dict[str, int]:
    summary = {
        "critical": 0,
        "high": 0,
        "medium": 0,
        "low": 0,
    }

    for risk in all_risks:
        level = str(risk.get("risk_level", "low")).lower()
        summary[level] = summary.get(level, 0) + 1

    return summary


def get_overall_severity(risk_summary: dict[str, int]) -> str:
    if risk_summary.get("critical", 0) > 0:
        return "critical"
    if risk_summary.get("high", 0) > 0:
        return "high"
    if risk_summary.get("medium", 0) > 0:
        return "medium"
    if risk_summary.get("low", 0) > 0:
        return "low"
    return "none"


def generate_recommended_actions(all_risks: list[dict[str, Any]]) -> list[str]:
    if not all_risks:
        return ["No obvious leakage indicators detected by current heuristics."]

    actions = [
        "Manually verify whether flagged columns are available at prediction time.",
        "Remove confirmed target-derived columns before model training.",
        "Compare baseline performance with and without flagged columns.",
        "Document human decisions for every flagged leakage risk.",
    ]

    risk_types = {risk.get("risk_type") for risk in all_risks}

    if "duplicate_target_risk" in risk_types:
        actions.insert(
            0,
            "Immediately review duplicate-target columns; they are likely invalid for modeling.",
        )

    if "high_correlation_risk" in risk_types:
        actions.append(
            "Investigate highly correlated numeric columns for formula-derived target leakage."
        )

    if (
        "classification_target_mutual_information_risk" in risk_types
        or "numeric_target_mutual_information_risk" in risk_types
    ):
        actions.append(
            "Review high mutual-information features as possible target proxies."
        )

    if "high_cardinality_identifier_risk" in risk_types:
        actions.append(
            "Drop or review identifier-like high-cardinality columns before one-hot encoding."
        )

    return actions


def run_leakage_check(
    df: pd.DataFrame,
    target_column: str,
) -> dict[str, Any]:
    """
    Run deterministic possible leakage-risk checks.

    This module reports possible leakage risks only.
    It never claims confirmed leakage.
    """
    try:
        logger.info("Starting leakage check")

        validate_inputs(df, target_column)

        thresholds = get_thresholds()

        name_based_risks = find_name_based_leakage_risks(df, target_column)

        duplicate_target_risks = find_direct_duplicate_target_columns(
            df=df,
            target_column=target_column,
            threshold=thresholds["duplicate_target_threshold"],
        )

        numeric_correlation_risks = find_numeric_correlation_risks(
            df=df,
            target_column=target_column,
            threshold=thresholds["high_correlation_threshold"],
        )

        mutual_information_risks = find_mutual_information_risks(
            df=df,
            target_column=target_column,
            threshold=thresholds["mutual_information_threshold"],
        )

        classification_proxy_risks = find_classification_proxy_risks(
            df=df,
            target_column=target_column,
            threshold=thresholds["classification_proxy_threshold"],
        )

        high_cardinality_review_risks = find_high_cardinality_review_columns(
            df=df,
            target_column=target_column,
            thresholds=thresholds,
        )

        all_risks = (
            name_based_risks
            + duplicate_target_risks
            + numeric_correlation_risks
            + mutual_information_risks
            + classification_proxy_risks
            + high_cardinality_review_risks
        )

        risk_summary = summarize_risks(all_risks)
        overall_severity = get_overall_severity(risk_summary)

        report: dict[str, Any] = {
            "target_column": target_column,
            "total_possible_leakage_risks": int(len(all_risks)),
            "overall_severity": overall_severity,
            "risk_summary": risk_summary,
            "thresholds": thresholds,
            "name_based_risks": name_based_risks,
            "duplicate_target_risks": duplicate_target_risks,
            "numeric_correlation_risks": numeric_correlation_risks,
            "mutual_information_risks": mutual_information_risks,
            "classification_proxy_risks": classification_proxy_risks,
            "high_cardinality_review_risks": high_cardinality_review_risks,
            "all_risks": sorted(
                all_risks,
                key=lambda risk: {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(
                    str(risk.get("risk_level", "low")).lower(),
                    99,
                ),
            ),
            "recommended_actions": generate_recommended_actions(all_risks),
            "requires_human_review": bool(all_risks),
            "warning": (
                "These are possible leakage risks, not confirmed leakage. "
                "A human should review whether flagged columns would be available at prediction time."
            ),
            "message": "Leakage check completed successfully.",
        }

        logger.info("Leakage check completed")
        return report

    except (LeakageDetectionError, InvalidTargetColumnError):
        raise

    except Exception as error:
        logger.exception("Leakage detection failed.")
        raise LeakageDetectionError(
            "Leakage detection failed.",
            error_detail=str(error),
        ) from error


if __name__ == "__main__":
    from src.audit.profiler import load_dataset

    dataset_path = "data/sample/student_mark.csv"
    target_column = "Grade"

    df = load_dataset(dataset_path)
    leakage_report = run_leakage_check(df, target_column)

    print(leakage_report)
