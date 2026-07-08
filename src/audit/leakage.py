from __future__ import annotations

from typing import Any

import pandas as pd

from src.utils.config import get_config_value
from src.utils.exceptions import InvalidTargetColumnError, LeakageDetectionError
from src.utils.logger import get_logger


logger = get_logger(__name__)


TARGET_LIKE_KEYWORDS = [
    "target",
    "label",
    "outcome",
    "result",
    "final_result",
    "prediction",
    "predicted",
    "actual",
    "ground_truth",
    "grade",
    "score",
    "rank",
    "pass",
    "fail",
    "status",
    "class",
    "total",
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


def find_name_based_leakage_risks(
    df: pd.DataFrame,
    target_column: str,
) -> list[dict[str, Any]]:
    risks: list[dict[str, Any]] = []
    target_lower = str(target_column).lower().strip()

    for column in df.columns:
        if column == target_column:
            continue

        column_lower = str(column).lower().strip()

        matched_keywords = [
            keyword
            for keyword in TARGET_LIKE_KEYWORDS
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
            risks.append(
                {
                    "column": column,
                    "risk_type": "name_based_risk",
                    "risk_level": "medium",
                    "matched_keywords": matched_keywords,
                    "reason": (
                        "Column name looks related to target/outcome. "
                        "Review whether this column is available before prediction time."
                    ),
                }
            )

    return risks


def find_direct_duplicate_target_columns(
    df: pd.DataFrame,
    target_column: str,
    threshold: float = 0.95,
) -> list[dict[str, Any]]:
    risks: list[dict[str, Any]] = []

    target_as_str = df[target_column].fillna("__MISSING_TARGET__").astype(str)

    for column in df.columns:
        if column == target_column:
            continue

        feature_as_str = df[column].fillna("__MISSING_FEATURE__").astype(str)
        same_values_ratio = float(feature_as_str.eq(target_as_str).mean())

        if same_values_ratio >= threshold:
            risks.append(
                {
                    "column": column,
                    "risk_type": "duplicate_target_risk",
                    "risk_level": "critical",
                    "same_values_ratio": float(round(same_values_ratio, 4)),
                    "reason": (
                        "Column values are almost identical to the target column. "
                        "This is a very strong possible leakage risk."
                    ),
                }
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

    numeric_df = df.select_dtypes(include=["number"])

    if target_column not in numeric_df.columns or numeric_df.shape[1] <= 1:
        return risks

    correlations = numeric_df.corr(numeric_only=True)[target_column].drop(
        labels=[target_column],
        errors="ignore",
    )

    for column, corr_value in correlations.items():
        if pd.isna(corr_value):
            continue

        if abs(corr_value) >= threshold:
            risks.append(
                {
                    "column": column,
                    "risk_type": "high_correlation_risk",
                    "risk_level": "high",
                    "correlation_with_target": float(round(corr_value, 4)),
                    "reason": (
                        "Column has very high correlation with the numeric target. "
                        "This may indicate leakage, target-derived features, or a direct dependency."
                    ),
                }
            )

    return risks


def find_encoded_target_correlation_risks(
    df: pd.DataFrame,
    target_column: str,
    threshold: float,
) -> list[dict[str, Any]]:
    """
    For classification targets, encode target labels and check numeric feature correlation.
    This is only a heuristic.
    """
    risks: list[dict[str, Any]] = []

    target = df[target_column].dropna()
    unique_count = target.nunique(dropna=True)

    if unique_count < 2 or unique_count > 20:
        return risks

    encoded_target = pd.Series(
        pd.factorize(df[target_column])[0],
        index=df.index,
    )

    numeric_columns = (
        df.drop(columns=[target_column])
        .select_dtypes(include=["number"])
        .columns
    )

    for column in numeric_columns:
        feature = df[column]

        valid_mask = feature.notna() & df[target_column].notna()

        if valid_mask.sum() < 2:
            continue

        corr_value = feature[valid_mask].corr(encoded_target[valid_mask])

        if pd.isna(corr_value):
            continue

        if abs(corr_value) >= threshold:
            risks.append(
                {
                    "column": column,
                    "risk_type": "encoded_target_correlation_risk",
                    "risk_level": "medium",
                    "correlation_with_encoded_target": float(round(corr_value, 4)),
                    "reason": (
                        "Numeric feature is highly correlated with encoded target classes. "
                        "This may be a valid strong feature or a possible target proxy."
                    ),
                }
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

    if unique_classes < 2 or unique_classes > 20:
        return risks

    numeric_columns = (
        df.drop(columns=[target_column])
        .select_dtypes(include=["number"])
        .columns
    )

    for column in numeric_columns:
        grouped_means = df.groupby(target_column, dropna=True)[column].mean()

        if grouped_means.empty or grouped_means.isna().all():
            continue

        min_mean = grouped_means.min()
        max_mean = grouped_means.max()

        denominator = max(abs(max_mean), abs(min_mean), 1e-9)
        separation_ratio = abs(max_mean - min_mean) / denominator

        if separation_ratio >= threshold:
            risks.append(
                {
                    "column": column,
                    "risk_type": "target_proxy_risk",
                    "risk_level": "medium",
                    "separation_ratio": float(round(separation_ratio, 4)),
                    "reason": (
                        "Numeric feature values are strongly separated across target classes. "
                        "This may be a valid predictive feature or a possible target proxy."
                    ),
                }
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
        if level not in summary:
            summary[level] = 0
        summary[level] += 1

    return summary


def generate_recommended_actions(all_risks: list[dict[str, Any]]) -> list[str]:
    if not all_risks:
        return ["No obvious leakage indicators detected by basic heuristics."]

    actions = [
        "Manually verify whether flagged columns are available at prediction time.",
        "Remove confirmed target-derived columns before training.",
        "Compare baseline performance with and without flagged columns.",
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

    if "name_based_risk" in risk_types:
        actions.append(
            "Review target-like column names carefully; names alone are not proof of leakage."
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
        validate_inputs(df, target_column)

        correlation_threshold = float(
            get_config_value("leakage.high_correlation_threshold", 0.90)
        )
        proxy_threshold = float(
            get_config_value("leakage.classification_proxy_threshold", 0.75)
        )
        duplicate_threshold = float(
            get_config_value("leakage.duplicate_target_threshold", 0.95)
        )
        encoded_corr_threshold = float(
            get_config_value("leakage.encoded_target_correlation_threshold", 0.90)
        )

        name_based_risks = find_name_based_leakage_risks(df, target_column)

        duplicate_target_risks = find_direct_duplicate_target_columns(
            df,
            target_column,
            threshold=duplicate_threshold,
        )

        numeric_correlation_risks = find_numeric_correlation_risks(
            df,
            target_column,
            threshold=correlation_threshold,
        )

        encoded_target_correlation_risks = find_encoded_target_correlation_risks(
            df,
            target_column,
            threshold=encoded_corr_threshold,
        )

        classification_proxy_risks = find_classification_proxy_risks(
            df,
            target_column,
            threshold=proxy_threshold,
        )

        all_risks = (
            name_based_risks
            + duplicate_target_risks
            + numeric_correlation_risks
            + encoded_target_correlation_risks
            + classification_proxy_risks
        )

        report: dict[str, Any] = {
            "target_column": target_column,
            "total_possible_leakage_risks": int(len(all_risks)),
            "risk_summary": summarize_risks(all_risks),
            "thresholds": {
                "high_correlation_threshold": correlation_threshold,
                "classification_proxy_threshold": proxy_threshold,
                "duplicate_target_threshold": duplicate_threshold,
                "encoded_target_correlation_threshold": encoded_corr_threshold,
            },
            "name_based_risks": name_based_risks,
            "duplicate_target_risks": duplicate_target_risks,
            "numeric_correlation_risks": numeric_correlation_risks,
            "encoded_target_correlation_risks": encoded_target_correlation_risks,
            "classification_proxy_risks": classification_proxy_risks,
            "all_risks": all_risks,
            "recommended_actions": generate_recommended_actions(all_risks),
            "warning": (
                "These are possible leakage risks, not confirmed leakage. "
                "A human should review whether these columns would be available at prediction time."
            ),
        }

        logger.info("Leakage check completed")
        return report

    except (LeakageDetectionError, InvalidTargetColumnError):
        raise

    except Exception as error:
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