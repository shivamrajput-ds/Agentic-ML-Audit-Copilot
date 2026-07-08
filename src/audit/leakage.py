from typing import Any

import pandas as pd

from src.utils.config import get_config_value
from src.utils.exceptions import LeakageDetectionError, InvalidTargetColumnError
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
]



def validate_inputs(df: pd.DataFrame, target_column: str) -> None:
    """
    Validate dataset and target column before leakage checks.
    """
    if df.empty:
        raise LeakageDetectionError("Dataset is empty.")

    if not target_column:
        raise InvalidTargetColumnError("Target column is required.")

    if target_column not in df.columns:
        raise InvalidTargetColumnError(
            f"Target column '{target_column}' not found in dataset."
        )


def find_name_based_leakage_risks(
    df: pd.DataFrame,
    target_column: str,
) -> list[dict[str, Any]]:
    """
    Find columns whose names look related to the target or outcome.
    """
    risks: list[dict[str, Any]] = []
    target_lower = target_column.lower()

    for column in df.columns:
        if column == target_column:
            continue

        column_lower = column.lower()

        matched_keywords = [
    keyword
    for keyword in TARGET_LIKE_KEYWORDS
    if keyword == column_lower
    or column_lower.endswith(f"_{keyword}")
    or column_lower.startswith(f"{keyword}_")
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
                    "reason": (
                        "Column name looks related to target/outcome "
                        "or contains target-like keywords."
                    ),
                    "matched_keywords": matched_keywords,
                }
            )

    return risks


def find_direct_duplicate_target_columns(
    df: pd.DataFrame,
    target_column: str,
) -> list[dict[str, Any]]:
    """
    Find columns whose values are almost identical to the target column.
    """
    risks: list[dict[str, Any]] = []

    target_as_str = df[target_column].fillna("__MISSING_TARGET__").astype(str)

    for column in df.columns:
        if column == target_column:
            continue

        feature_as_str = df[column].fillna("__MISSING_FEATURE__").astype(str)
        same_values_ratio = feature_as_str.eq(target_as_str).mean()

        if same_values_ratio >= 0.95:
            risks.append(
                {
                    "column": column,
                    "risk_type": "duplicate_target_risk",
                    "risk_level": "high",
                    "same_values_ratio": float(round(same_values_ratio, 4)),
                    "reason": (
                        "Column values are almost identical to the target column. "
                        "This is a strong possible leakage risk."
                    ),
                }
            )

    return risks


def find_numeric_correlation_risks(
    df: pd.DataFrame,
    target_column: str,
    threshold: float,
) -> list[dict[str, Any]]:
    """
    Find numeric columns highly correlated with a numeric target.
    """
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
                        "This may indicate possible leakage or direct dependency."
                    ),
                }
            )

    return risks


def find_classification_proxy_risks(
    df: pd.DataFrame,
    target_column: str,
    threshold: float,
) -> list[dict[str, Any]]:
    """
    Find numeric features that strongly separate target classes.

    This is only a heuristic. A flagged column may be a valid predictive
    feature or a possible proxy leakage risk.
    """
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
        grouped_means = df.groupby(target_column)[column].mean(numeric_only=True)

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


def run_leakage_check(
    df: pd.DataFrame,
    target_column: str,
) -> dict[str, Any]:
    """
    Run deterministic possible leakage-risk checks.

    The module reports possible leakage risks only.
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

        name_based_risks = find_name_based_leakage_risks(
            df=df,
            target_column=target_column,
        )

        duplicate_target_risks = find_direct_duplicate_target_columns(
            df=df,
            target_column=target_column,
        )

        numeric_correlation_risks = find_numeric_correlation_risks(
            df=df,
            target_column=target_column,
            threshold=correlation_threshold,
        )

        classification_proxy_risks = find_classification_proxy_risks(
            df=df,
            target_column=target_column,
            threshold=proxy_threshold,
        )

        all_risks = (
            name_based_risks
            + duplicate_target_risks
            + numeric_correlation_risks
            + classification_proxy_risks
        )

        report = {
            "target_column": target_column,
            "total_possible_leakage_risks": len(all_risks),
            "name_based_risks": name_based_risks,
            "duplicate_target_risks": duplicate_target_risks,
            "numeric_correlation_risks": numeric_correlation_risks,
            "classification_proxy_risks": classification_proxy_risks,
            "all_risks": all_risks,
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
            "Leakage detection failed",
            error_detail=str(error),
        ) from error


if __name__ == "__main__":
    from src.audit.profiler import load_dataset

    dataset_path = "data/sample/student_mark.csv"
    target_column = "Grade"

    df = load_dataset(dataset_path)
    leakage_report = run_leakage_check(df, target_column)

    print(leakage_report)