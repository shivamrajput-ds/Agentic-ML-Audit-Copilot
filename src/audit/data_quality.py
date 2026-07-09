from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import get_config_value
from src.utils.exceptions import DataQualityError, InvalidTargetColumnError
from src.utils.logger import get_logger

logger = get_logger(__name__)

SEVERITY_RANK = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
    "info": 4,
}


def get_float_config(path: str, default: float) -> float:
    """Read float config values with a safe fallback."""
    try:
        return float(get_config_value(path, default))
    except (TypeError, ValueError):
        return default


def validate_inputs(df: pd.DataFrame, target_column: str) -> None:
    """Validate inputs for data quality audit."""
    if df is None or df.empty:
        raise DataQualityError("Input dataframe is empty.")

    if target_column is None or not str(target_column).strip():
        raise InvalidTargetColumnError("Target column is required.")

    if target_column not in df.columns:
        raise InvalidTargetColumnError(
            f"Target column '{target_column}' not found in dataset.",
        )

    if len(df.columns) <= 1:
        raise DataQualityError("Dataset must contain at least one feature column.")


def get_thresholds() -> dict[str, float]:
    """Read thresholds from config.yaml."""
    return {
        "high_missing_threshold": get_float_config(
            "audit.high_missing_threshold",
            50.0,
        ),
        "warning_missing_threshold": get_float_config(
            "missing_values.warning_threshold",
            20.0,
        ),
        "high_cardinality_threshold": get_float_config(
            "audit.high_cardinality_threshold",
            50.0,
        ),
        "id_unique_percent_threshold": get_float_config(
            "audit.id_unique_percent_threshold",
            95.0,
        ),
        "near_constant_threshold": get_float_config(
            "audit.near_constant_threshold",
            95.0,
        ),
        "rare_value_threshold_percent": get_float_config(
            "data_quality.rare_value_threshold_percent",
            1.0,
        ),
        "iqr_multiplier": get_float_config("outliers.iqr_multiplier", 1.5),
    }


def safe_percent(numerator: float, denominator: float) -> float:
    """Calculate percentage safely."""
    if denominator == 0:
        return 0.0

    return round((numerator / denominator) * 100, 2)


def add_finding(
    findings: list[dict[str, Any]],
    severity: str,
    category: str,
    message: str,
    column: str | None = None,
    evidence: dict[str, Any] | None = None,
    recommendation: str | None = None,
) -> None:
    """Add a standardized finding."""
    findings.append(
        {
            "severity": severity,
            "category": category,
            "column": column,
            "message": message,
            "evidence": evidence or {},
            "recommendation": recommendation,
            "requires_human_review": severity in {"critical", "high", "medium"},
        },
    )


def sort_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort findings by severity."""
    return sorted(
        findings,
        key=lambda item: SEVERITY_RANK.get(str(item.get("severity", "info")), 99),
    )


def analyze_missing_values(
    df: pd.DataFrame,
    target_column: str,
    thresholds: dict[str, float],
    findings: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Analyze missing values for all columns."""
    missing_values: dict[str, dict[str, Any]] = {}

    for column in df.columns:
        missing_count = int(df[column].isna().sum())
        missing_percent = safe_percent(missing_count, len(df))

        if missing_count > 0:
            missing_values[str(column)] = {
                "missing_count": missing_count,
                "missing_percent": missing_percent,
            }

        if column == target_column:
            if missing_percent > 0:
                add_finding(
                    findings=findings,
                    severity="high",
                    category="target_missing_values",
                    column=str(column),
                    message=(
                        f"Target column has {missing_percent}% missing values. "
                        "Rows with missing target cannot be used for supervised training."
                    ),
                    evidence={
                        "missing_count": missing_count,
                        "missing_percent": missing_percent,
                    },
                    recommendation="Drop rows with missing target before model training.",
                )
            continue

        if missing_percent >= thresholds["high_missing_threshold"]:
            add_finding(
                findings=findings,
                severity="high",
                category="high_missing_values",
                column=str(column),
                message=f"Column has high missing values: {missing_percent}%.",
                evidence={
                    "missing_count": missing_count,
                    "missing_percent": missing_percent,
                },
                recommendation=(
                    "Review whether this feature is useful. Consider dropping it "
                    "or using domain-specific imputation."
                ),
            )

        elif missing_percent >= thresholds["warning_missing_threshold"]:
            add_finding(
                findings=findings,
                severity="medium",
                category="moderate_missing_values",
                column=str(column),
                message=f"Column has moderate missing values: {missing_percent}%.",
                evidence={
                    "missing_count": missing_count,
                    "missing_percent": missing_percent,
                },
                recommendation="Use appropriate imputation and monitor missingness pattern.",
            )

    return missing_values


def detect_duplicate_rows(
    df: pd.DataFrame,
    findings: list[dict[str, Any]],
) -> int:
    """Detect duplicate rows."""
    duplicate_rows = int(df.duplicated().sum())
    duplicate_percent = safe_percent(duplicate_rows, len(df))

    if duplicate_rows > 0:
        severity = "medium" if duplicate_percent < 5 else "high"
        add_finding(
            findings=findings,
            severity=severity,
            category="duplicate_rows",
            message=(
                f"Dataset contains {duplicate_rows} duplicate rows "
                f"({duplicate_percent}%)."
            ),
            evidence={
                "duplicate_rows": duplicate_rows,
                "duplicate_percent": duplicate_percent,
            },
            recommendation=(
                "Review duplicates and remove them if they are not valid repeated "
                "observations."
            ),
        )

    return duplicate_rows


def detect_duplicate_columns(
    df: pd.DataFrame,
    findings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Detect duplicate columns by exact value equality."""
    duplicate_columns: list[dict[str, Any]] = []
    columns = list(df.columns)

    for index, col_a in enumerate(columns):
        for col_b in columns[index + 1 :]:
            try:
                if df[col_a].equals(df[col_b]):
                    record = {
                        "column_a": str(col_a),
                        "column_b": str(col_b),
                    }
                    duplicate_columns.append(record)
                    add_finding(
                        findings=findings,
                        severity="medium",
                        category="duplicate_columns",
                        column=str(col_b),
                        message=f"Column '{col_b}' is an exact duplicate of '{col_a}'.",
                        evidence=record,
                        recommendation="Drop one of the duplicate columns before modeling.",
                    )
            except (TypeError, ValueError):
                continue

    return duplicate_columns


def detect_constant_columns(
    df: pd.DataFrame,
    target_column: str,
    findings: list[dict[str, Any]],
) -> list[str]:
    """Detect constant feature columns."""
    constant_columns: list[str] = []

    for column in df.columns:
        if column == target_column:
            continue

        unique_count = int(df[column].nunique(dropna=False))

        if unique_count <= 1:
            constant_columns.append(str(column))
            add_finding(
                findings=findings,
                severity="medium",
                category="constant_column",
                column=str(column),
                message="Column has only one unique value and provides no predictive signal.",
                evidence={"unique_count": unique_count},
                recommendation="Drop this column before training.",
            )

    return constant_columns


def detect_near_constant_columns(
    df: pd.DataFrame,
    target_column: str,
    thresholds: dict[str, float],
    findings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Detect near-constant feature columns."""
    near_constant_columns: list[dict[str, Any]] = []

    for column in df.columns:
        if column == target_column:
            continue

        value_counts = df[column].value_counts(dropna=False)

        if value_counts.empty:
            continue

        dominant_count = int(value_counts.iloc[0])
        dominant_percent = safe_percent(dominant_count, len(df))

        if dominant_percent >= thresholds["near_constant_threshold"]:
            record = {
                "column": str(column),
                "dominant_value": str(value_counts.index[0]),
                "dominant_count": dominant_count,
                "dominant_percent": dominant_percent,
            }
            near_constant_columns.append(record)
            add_finding(
                findings=findings,
                severity="low",
                category="near_constant_column",
                column=str(column),
                message=(
                    "Column is near-constant. Dominant value appears in "
                    f"{dominant_percent}% rows."
                ),
                evidence=record,
                recommendation=(
                    "Review usefulness; near-constant columns often add little signal."
                ),
            )

    return near_constant_columns


def detect_high_cardinality_columns(
    df: pd.DataFrame,
    target_column: str,
    thresholds: dict[str, float],
    findings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Detect high-cardinality categorical columns."""
    high_cardinality_columns: list[dict[str, Any]] = []
    feature_df = df.drop(columns=[target_column])

    categorical_columns = feature_df.select_dtypes(
        include=["object", "category", "string"],
    ).columns

    for column in categorical_columns:
        unique_count = int(df[column].nunique(dropna=True))
        unique_percent = safe_percent(unique_count, len(df))

        if unique_count >= thresholds["high_cardinality_threshold"]:
            record = {
                "column": str(column),
                "unique_count": unique_count,
                "unique_percent": unique_percent,
            }
            high_cardinality_columns.append(record)

            severity = (
                "high"
                if unique_percent >= thresholds["id_unique_percent_threshold"]
                else "medium"
            )

            add_finding(
                findings=findings,
                severity=severity,
                category="high_cardinality_column",
                column=str(column),
                message=(
                    f"Column has high cardinality: {unique_count} unique values "
                    f"({unique_percent}%)."
                ),
                evidence=record,
                recommendation=(
                    "Avoid naive one-hot encoding if cardinality is high. "
                    "Consider dropping ID-like columns or using target-independent "
                    "encodings."
                ),
            )

    return high_cardinality_columns


def column_name_suggests_id(column: str) -> bool:
    """Detect identifier-like column names with safe token/suffix matching."""
    normalized = (
        str(column)
        .lower()
        .strip()
        .replace("-", "_")
        .replace(" ", "_")
        .replace(".", "_")
    )

    exact_names = {
        "id",
        "uuid",
        "guid",
        "identifier",
        "serial",
        "roll",
        "roll_no",
        "zipcode",
        "zip",
        "phone",
        "mobile",
        "email",
    }

    suffixes = (
        "_id",
        "_uuid",
        "_guid",
        "_identifier",
        "_serial",
        "_roll",
        "_roll_no",
        "_zipcode",
        "_zip",
        "_phone",
        "_mobile",
        "_email",
    )

    return normalized in exact_names or normalized.endswith(suffixes)


def detect_possible_id_columns(
    df: pd.DataFrame,
    target_column: str,
    thresholds: dict[str, float],
    findings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Detect possible ID/identifier columns."""
    possible_id_columns: list[dict[str, Any]] = []

    for column in df.columns:
        if column == target_column:
            continue

        unique_count = int(df[column].nunique(dropna=True))
        unique_percent = safe_percent(unique_count, len(df))

        name_suggests_id = column_name_suggests_id(str(column))
        uniqueness_suggests_id = (
            unique_percent >= thresholds["id_unique_percent_threshold"]
        )

        if name_suggests_id or uniqueness_suggests_id:
            record = {
                "column": str(column),
                "unique_count": unique_count,
                "unique_percent": unique_percent,
                "name_suggests_id": name_suggests_id,
                "uniqueness_suggests_id": uniqueness_suggests_id,
            }
            possible_id_columns.append(record)
            add_finding(
                findings=findings,
                severity="high" if uniqueness_suggests_id else "medium",
                category="possible_id_column",
                column=str(column),
                message=(
                    "Column may be an identifier and can cause memorization or "
                    "high-dimensional noise."
                ),
                evidence=record,
                recommendation=(
                    "Review this column. Drop it unless it has a legitimate, "
                    "prediction-time meaning."
                ),
            )

    return possible_id_columns


def detect_mixed_type_columns(
    df: pd.DataFrame,
    target_column: str,
    findings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Detect object columns with mixed Python value types."""
    mixed_type_columns: list[dict[str, Any]] = []
    object_columns = df.select_dtypes(include=["object"]).columns

    for column in object_columns:
        if column == target_column:
            continue

        type_counts = (
            df[column]
            .dropna()
            .map(lambda value: type(value).__name__)
            .value_counts()
            .to_dict()
        )

        if len(type_counts) > 1:
            record = {
                "column": str(column),
                "type_counts": {
                    str(key): int(value) for key, value in type_counts.items()
                },
            }
            mixed_type_columns.append(record)
            add_finding(
                findings=findings,
                severity="medium",
                category="mixed_type_column",
                column=str(column),
                message="Column contains mixed Python value types.",
                evidence=record,
                recommendation="Standardize this column before modeling.",
            )

    return mixed_type_columns


def detect_infinite_values(
    df: pd.DataFrame,
    target_column: str,
    findings: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Detect np.inf and -np.inf in numeric columns."""
    infinite_values: dict[str, dict[str, Any]] = {}
    numeric_columns = df.select_dtypes(include=["number"]).columns

    for column in numeric_columns:
        series = df[column]
        pos_inf_count = int(np.isposinf(series).sum())
        neg_inf_count = int(np.isneginf(series).sum())
        total_inf_count = pos_inf_count + neg_inf_count

        if total_inf_count > 0:
            record = {
                "positive_infinity_count": pos_inf_count,
                "negative_infinity_count": neg_inf_count,
                "total_infinity_count": total_inf_count,
            }
            infinite_values[str(column)] = record
            severity = "high" if column == target_column else "medium"
            add_finding(
                findings=findings,
                severity=severity,
                category="infinite_values",
                column=str(column),
                message=(
                    "Column contains infinite values that can break "
                    "preprocessing/model training."
                ),
                evidence=record,
                recommendation="Replace infinite values with NaN before imputation/modeling.",
            )

    return infinite_values


def detect_outlier_columns(
    df: pd.DataFrame,
    target_column: str,
    thresholds: dict[str, float],
    findings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Detect numeric outliers using IQR."""
    if not bool(get_config_value("outliers.enabled", True)):
        return []

    outlier_columns: list[dict[str, Any]] = []

    numeric_columns = (
        df.drop(columns=[target_column]).select_dtypes(include=["number"]).columns
    )

    for column in numeric_columns:
        series = df[column].replace([np.inf, -np.inf], np.nan).dropna()

        if series.empty or series.nunique() <= 1:
            continue

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1

        if iqr == 0:
            continue

        lower_bound = q1 - thresholds["iqr_multiplier"] * iqr
        upper_bound = q3 + thresholds["iqr_multiplier"] * iqr

        outlier_mask = (series < lower_bound) | (series > upper_bound)
        outlier_count = int(outlier_mask.sum())
        outlier_percent = safe_percent(outlier_count, len(series))

        if outlier_count > 0:
            record = {
                "column": str(column),
                "outlier_count": outlier_count,
                "outlier_percent": outlier_percent,
                "lower_bound": round(float(lower_bound), 4),
                "upper_bound": round(float(upper_bound), 4),
                "method": "iqr",
            }
            outlier_columns.append(record)

            severity = "medium" if outlier_percent >= 5 else "low"
            add_finding(
                findings=findings,
                severity=severity,
                category="outliers",
                column=str(column),
                message=(
                    f"Column has {outlier_count} possible IQR outliers "
                    f"({outlier_percent}%)."
                ),
                evidence=record,
                recommendation=(
                    "Review outliers. Do not remove automatically unless domain "
                    "context supports it."
                ),
            )

    return outlier_columns


def analyze_target_quality(
    df: pd.DataFrame,
    target_column: str,
    findings: list[dict[str, Any]],
) -> dict[str, Any]:
    """Analyze target column quality."""
    target = df[target_column]
    missing_count = int(target.isna().sum())
    missing_percent = safe_percent(missing_count, len(df))
    unique_count = int(target.nunique(dropna=True))

    target_quality = {
        "target_column": target_column,
        "dtype": str(target.dtype),
        "missing_count": missing_count,
        "missing_percent": missing_percent,
        "unique_count": unique_count,
        "is_numeric": bool(pd.api.types.is_numeric_dtype(target)),
    }

    if unique_count < 2:
        add_finding(
            findings=findings,
            severity="critical",
            category="invalid_target",
            column=target_column,
            message="Target column has fewer than 2 unique non-null values.",
            evidence=target_quality,
            recommendation="Choose a valid target column with at least 2 classes/values.",
        )

    return target_quality


def calculate_quality_score(findings: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Calculate quality score from findings.

    This is a triage score, not a statistical guarantee.
    """
    score = 100.0

    penalty_map = {
        "critical": 35,
        "high": 15,
        "medium": 7,
        "low": 2,
        "info": 0,
    }

    penalties: list[dict[str, Any]] = []

    for finding in findings:
        severity = str(finding.get("severity", "info")).lower()
        penalty = penalty_map.get(severity, 0)

        if penalty <= 0:
            continue

        score -= penalty
        penalties.append(
            {
                "category": finding.get("category"),
                "column": finding.get("column"),
                "severity": severity,
                "penalty": penalty,
            },
        )

    final_score = max(0.0, min(100.0, score))

    if final_score >= 90:
        health_label = "good"
    elif final_score >= 75:
        health_label = "needs_review"
    elif final_score >= 50:
        health_label = "poor"
    else:
        health_label = "critical"

    return {
        "score": round(final_score, 2),
        "health_label": health_label,
        "penalties": penalties,
        "note": (
            "Quality score is a practical triage score, not a guarantee of model "
            "readiness."
        ),
    }


def generate_warnings(findings: list[dict[str, Any]]) -> list[str]:
    """Generate human-readable warnings."""
    if not findings:
        return ["No major basic data quality issues detected."]

    warnings: list[str] = []

    for finding in sort_findings(findings):
        severity = str(finding.get("severity", "info")).upper()
        column = finding.get("column")
        message = finding.get("message")

        if column:
            warnings.append(f"[{severity}] {column}: {message}")
        else:
            warnings.append(f"[{severity}] {message}")

    return warnings


def generate_recommended_actions(findings: list[dict[str, Any]]) -> list[str]:
    """Generate deduplicated recommended actions."""
    actions: list[str] = []

    for finding in sort_findings(findings):
        recommendation = finding.get("recommendation")
        if recommendation and recommendation not in actions:
            actions.append(str(recommendation))

    if not actions:
        return ["No major data quality preprocessing action required."]

    return actions


def build_column_quality_summary(
    df: pd.DataFrame,
    target_column: str,
    findings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build column-wise quality summary."""
    finding_count_by_column: dict[str, int] = {}

    for finding in findings:
        column = finding.get("column")
        if column:
            column_name = str(column)
            finding_count_by_column[column_name] = (
                finding_count_by_column.get(column_name, 0) + 1
            )

    summary: list[dict[str, Any]] = []

    for column in df.columns:
        missing_count = int(df[column].isna().sum())
        unique_count = int(df[column].nunique(dropna=True))

        summary.append(
            {
                "column": str(column),
                "dtype": str(df[column].dtype),
                "is_target": column == target_column,
                "missing_count": missing_count,
                "missing_percent": safe_percent(missing_count, len(df)),
                "unique_count": unique_count,
                "unique_percent": safe_percent(unique_count, len(df)),
                "finding_count": finding_count_by_column.get(str(column), 0),
            },
        )

    return summary


def run_data_quality_audit(
    df: pd.DataFrame,
    target_column: str,
) -> dict[str, Any]:
    """
    Run deterministic data quality audit.

    This module reports data quality risks only. It does not automatically mutate data.
    """
    try:
        logger.info("Starting data quality audit")

        validate_inputs(df, target_column)

        thresholds = get_thresholds()
        findings: list[dict[str, Any]] = []

        target_quality = analyze_target_quality(df, target_column, findings)
        missing_values = analyze_missing_values(df, target_column, thresholds, findings)
        duplicate_rows = detect_duplicate_rows(df, findings)
        duplicate_columns = detect_duplicate_columns(df, findings)
        constant_columns = detect_constant_columns(df, target_column, findings)
        near_constant_columns = detect_near_constant_columns(
            df,
            target_column,
            thresholds,
            findings,
        )
        high_cardinality_columns = detect_high_cardinality_columns(
            df,
            target_column,
            thresholds,
            findings,
        )
        possible_id_columns = detect_possible_id_columns(
            df,
            target_column,
            thresholds,
            findings,
        )
        mixed_type_columns = detect_mixed_type_columns(df, target_column, findings)
        infinite_values = detect_infinite_values(df, target_column, findings)
        outlier_columns = detect_outlier_columns(
            df,
            target_column,
            thresholds,
            findings,
        )

        null_only_columns = [
            column
            for column, values in missing_values.items()
            if values["missing_percent"] == 100
        ]

        high_missing_columns = [
            {
                "column": column,
                **values,
            }
            for column, values in missing_values.items()
            if values["missing_percent"] >= thresholds["high_missing_threshold"]
        ]

        sorted_findings = sort_findings(findings)
        quality_score = calculate_quality_score(sorted_findings)
        warnings = generate_warnings(sorted_findings)
        recommended_actions = generate_recommended_actions(sorted_findings)
        column_quality_summary = build_column_quality_summary(
            df,
            target_column,
            sorted_findings,
        )

        report: dict[str, Any] = {
            "target_column": target_column,
            "total_rows": int(len(df)),
            "total_columns": int(len(df.columns)),
            "quality_score": quality_score,
            "target_quality": target_quality,
            "duplicate_rows": duplicate_rows,
            "duplicate_row_percent": safe_percent(duplicate_rows, len(df)),
            "duplicate_columns": duplicate_columns,
            "missing_values": missing_values,
            "high_missing_columns": high_missing_columns,
            "null_only_columns": null_only_columns,
            "constant_columns": constant_columns,
            "near_constant_columns": near_constant_columns,
            "high_cardinality_columns": high_cardinality_columns,
            "possible_id_columns": possible_id_columns,
            "mixed_type_columns": mixed_type_columns,
            "infinite_values": infinite_values,
            "outlier_columns": outlier_columns,
            "findings": sorted_findings,
            "column_quality_summary": column_quality_summary,
            "warnings": warnings,
            "recommended_actions": recommended_actions,
            "thresholds": thresholds,
            "message": "Data quality audit completed successfully.",
        }

        logger.info("Data quality audit completed successfully")
        return report

    except (DataQualityError, InvalidTargetColumnError):
        raise
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        pd.errors.ParserError,
    ) as error:
        logger.exception("Data quality audit failed.")
        raise DataQualityError(
            "Data quality audit failed.",
            error_detail=str(error),
        ) from error


if __name__ == "__main__":
    sample_df = pd.DataFrame(
        {
            "student_id": [1, 2, 3, 4, 5, 5],
            "age": [18, 19, 20, np.inf, 22, 22],
            "gender": ["M", "F", "M", None, "F", "F"],
            "constant_col": ["x", "x", "x", "x", "x", "x"],
            "grade": ["A", "B", "A", "B", "A", "A"],
        },
    )

    output = run_data_quality_audit(sample_df, "grade")
    print(output)
