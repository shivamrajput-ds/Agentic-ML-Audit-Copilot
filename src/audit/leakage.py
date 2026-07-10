from __future__ import annotations

import re
from difflib import get_close_matches
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from src.utils.config import get_config_value
from src.utils.exceptions import InvalidTargetColumnError, LeakageDetectionError
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_TARGET_LIKE_KEYWORDS = [
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

RISK_SEVERITY_ORDER = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
}

TRUE_VALUES = {"true", "1", "yes", "y", "on"}
FALSE_VALUES = {"false", "0", "no", "n", "off"}


# -----------------------------------------------------------------------------
# Config / utility helpers
# -----------------------------------------------------------------------------


def as_bool(value: Any, default: bool = False) -> bool:
    """Convert config/env-like values safely into boolean."""
    if isinstance(value, bool):
        return value

    if value is None:
        return default

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in TRUE_VALUES:
            return True
        if normalized in FALSE_VALUES:
            return False
        return default

    return bool(value)


def get_float_config(
    path: str,
    default: float,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    """Read float config values with safe fallback and optional bounds."""
    try:
        value = float(get_config_value(path, default))
    except (KeyError, TypeError, ValueError, RuntimeError):
        logger.warning("Invalid float config for %s. Using default=%s", path, default)
        value = float(default)

    if not np.isfinite(value):
        logger.warning(
            "Non-finite float config for %s. Using default=%s", path, default
        )
        value = float(default)

    if min_value is not None:
        value = max(min_value, value)

    if max_value is not None:
        value = min(max_value, value)

    return value


def get_int_config(
    path: str,
    default: int,
    *,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
    """Read int config values with safe fallback and optional bounds."""
    try:
        value = int(get_config_value(path, default))
    except (KeyError, TypeError, ValueError, RuntimeError):
        logger.warning("Invalid int config for %s. Using default=%s", path, default)
        value = int(default)

    if min_value is not None:
        value = max(min_value, value)

    if max_value is not None:
        value = min(max_value, value)

    return value


def safe_percent(numerator: int | float, denominator: int | float) -> float:
    """Calculate percentage safely."""
    try:
        if denominator == 0:
            return 0.0
        value = (float(numerator) / float(denominator)) * 100
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0

    if not np.isfinite(value):
        return 0.0

    return round(float(value), 2)


def safe_float(value: Any, digits: int = 4) -> float | None:
    """Convert numeric values to JSON-safe floats."""
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(output):
        return None

    return round(output, digits)


def nunique_safely(series: pd.Series, dropna: bool = True) -> int:
    """Return unique count safely for columns with unhashable object values."""
    try:
        return int(series.nunique(dropna=dropna))
    except (TypeError, ValueError):
        safe_series = series.map(lambda value: None if pd.isna(value) else str(value))
        return int(safe_series.nunique(dropna=dropna))


def normalize_column_name(column: str) -> str:
    """Normalize column name for token-safe matching."""
    return (
        str(column)
        .lower()
        .strip()
        .replace("-", "_")
        .replace(" ", "_")
        .replace(".", "_")
    )


def tokenize_column_name(column: str) -> list[str]:
    """Split a column name into stable lowercase tokens."""
    normalized = normalize_column_name(column)
    return [token for token in re.split(r"_+|\W+", normalized) if token]


def is_string_like_series(series: pd.Series) -> bool:
    """Return True for object/string/category columns."""
    dtype = series.dtype
    return bool(
        pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
        or isinstance(dtype, pd.CategoricalDtype)
    )


def column_name_suggests_identifier(column: str) -> bool:
    """Detect identifier-like column names using conservative token matching."""
    normalized = normalize_column_name(column)

    exact_names = {
        "id",
        "uuid",
        "guid",
        "identifier",
        "serial",
        "record_id",
        "row_id",
        "user_id",
        "customer_id",
        "student_id",
        "roll_no",
        "email",
        "phone",
        "mobile",
        "zipcode",
        "zip",
    }
    suffixes = (
        "_id",
        "_uuid",
        "_guid",
        "_identifier",
        "_serial",
        "_email",
        "_phone",
        "_mobile",
        "_zipcode",
        "_zip",
        "_roll_no",
    )

    return normalized in exact_names or normalized.endswith(suffixes)


def resolve_target_column(df: pd.DataFrame, target_column: str) -> str:
    """Resolve target column while being friendly to accidental whitespace/case issues."""
    requested = str(target_column).strip()

    if requested in df.columns:
        return requested

    normalized_requested = requested.lower()
    case_matches = [
        str(column)
        for column in df.columns
        if str(column).strip().lower() == normalized_requested
    ]

    if len(case_matches) == 1:
        return case_matches[0]

    close_matches = get_close_matches(
        requested, [str(column) for column in df.columns], n=3
    )
    suggestion = f" Did you mean: {close_matches}?" if close_matches else ""

    raise InvalidTargetColumnError(
        f"Target column '{target_column}' not found in dataset.{suggestion}",
    )


# -----------------------------------------------------------------------------
# Input / threshold helpers
# -----------------------------------------------------------------------------


def validate_inputs(df: pd.DataFrame, target_column: str) -> str:
    """Validate inputs for leakage checks and return resolved target column."""
    if df is None or df.empty:
        raise LeakageDetectionError("Dataset is empty.")

    if target_column is None or not str(target_column).strip():
        raise InvalidTargetColumnError("Target column is required.")

    if df.columns.duplicated().any():
        duplicate_columns = [
            str(column) for column in df.columns[df.columns.duplicated()]
        ]
        raise LeakageDetectionError(
            f"Duplicate column names found. Leakage checks require unique columns: {duplicate_columns}",
        )

    resolved_target = resolve_target_column(df, target_column)

    if df[resolved_target].dropna().empty:
        raise InvalidTargetColumnError(
            f"Target column '{resolved_target}' has only missing values.",
        )

    if nunique_safely(df[resolved_target], dropna=True) < 2:
        raise InvalidTargetColumnError(
            f"Target column '{resolved_target}' must contain at least 2 unique non-null values.",
        )

    return resolved_target


def get_target_like_keywords() -> list[str]:
    """Read target-like keywords from config."""
    raw_keywords = get_config_value(
        "leakage.target_like_keywords",
        DEFAULT_TARGET_LIKE_KEYWORDS,
    )

    if isinstance(raw_keywords, str):
        candidates = raw_keywords.split(",")
    elif isinstance(raw_keywords, list):
        candidates = raw_keywords
    else:
        candidates = DEFAULT_TARGET_LIKE_KEYWORDS

    cleaned = [
        normalize_column_name(str(item)) for item in candidates if str(item).strip()
    ]

    return cleaned or DEFAULT_TARGET_LIKE_KEYWORDS


def get_thresholds() -> dict[str, float]:
    """Read leakage thresholds from config with conservative bounds."""
    return {
        "high_correlation_threshold": get_float_config(
            "leakage.high_correlation_threshold",
            0.90,
            min_value=0.0,
            max_value=1.0,
        ),
        "classification_proxy_threshold": get_float_config(
            "leakage.classification_proxy_threshold",
            0.75,
            min_value=0.0,
        ),
        "duplicate_target_threshold": get_float_config(
            "leakage.duplicate_target_threshold",
            0.95,
            min_value=0.0,
            max_value=1.0,
        ),
        "mutual_information_threshold": get_float_config(
            "leakage.mutual_information_threshold",
            0.20,
            min_value=0.0,
        ),
        "id_unique_percent_threshold": get_float_config(
            "audit.id_unique_percent_threshold",
            95.0,
            min_value=0.0,
            max_value=100.0,
        ),
        "high_cardinality_threshold": get_float_config(
            "audit.high_cardinality_threshold",
            50.0,
            min_value=1.0,
        ),
        "min_compare_rows": float(
            get_int_config("leakage.min_compare_rows", 10, min_value=2),
        ),
        "mutual_information_max_rows": float(
            get_int_config(
                "leakage.mutual_information_max_rows", 10_000, min_value=100
            ),
        ),
    }


# -----------------------------------------------------------------------------
# Risk record helpers
# -----------------------------------------------------------------------------


def add_risk(
    risks: list[dict[str, Any]],
    column: str,
    risk_type: str,
    risk_level: str,
    reason: str,
    evidence: dict[str, Any] | None = None,
) -> None:
    """Add a standardized possible leakage risk."""
    risks.append(
        {
            "column": str(column),
            "risk_type": risk_type,
            "risk_level": str(risk_level).lower(),
            "is_confirmed_leakage": False,
            "requires_human_review": True,
            "reason": reason,
            "evidence": evidence or {},
        },
    )


def sort_risks(all_risks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort risks by severity for stable reporting."""
    return sorted(
        all_risks,
        key=lambda risk: (
            RISK_SEVERITY_ORDER.get(str(risk.get("risk_level", "low")).lower(), 99),
            str(risk.get("column", "")),
            str(risk.get("risk_type", "")),
        ),
    )


def summarize_risks(all_risks: list[dict[str, Any]]) -> dict[str, int]:
    """Summarize leakage risks by severity level."""
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


def summarize_risk_types(all_risks: list[dict[str, Any]]) -> dict[str, int]:
    """Summarize leakage risks by risk type."""
    summary: dict[str, int] = {}

    for risk in all_risks:
        risk_type = str(risk.get("risk_type", "unknown"))
        summary[risk_type] = summary.get(risk_type, 0) + 1

    return dict(sorted(summary.items()))


def get_overall_severity(risk_summary: dict[str, int]) -> str:
    """Return overall leakage severity from risk summary."""
    if risk_summary.get("critical", 0) > 0:
        return "critical"

    if risk_summary.get("high", 0) > 0:
        return "high"

    if risk_summary.get("medium", 0) > 0:
        return "medium"

    if risk_summary.get("low", 0) > 0:
        return "low"

    return "none"


# -----------------------------------------------------------------------------
# Leakage heuristics
# -----------------------------------------------------------------------------


def find_name_based_leakage_risks(
    df: pd.DataFrame,
    target_column: str,
) -> list[dict[str, Any]]:
    """Find possible leakage risks based on target-like column names."""
    risks: list[dict[str, Any]] = []
    keywords = set(get_target_like_keywords())
    target_normalized = normalize_column_name(target_column)
    target_tokens = set(tokenize_column_name(target_column))

    for column in df.columns:
        if column == target_column:
            continue

        column_name = str(column)
        column_normalized = normalize_column_name(column_name)
        column_tokens = set(tokenize_column_name(column_name))

        matched_keywords = sorted(column_tokens.intersection(keywords))
        target_name_overlap = bool(
            target_tokens
            and column_tokens
            and target_tokens.intersection(column_tokens)
        )

        explicit_target_relation = (
            column_normalized.endswith(f"_{target_normalized}")
            or column_normalized.startswith(f"{target_normalized}_")
            or target_normalized in column_tokens
        )

        if matched_keywords or target_name_overlap or explicit_target_relation:
            risk_level = "high" if explicit_target_relation else "medium"
            add_risk(
                risks=risks,
                column=column_name,
                risk_type="name_based_risk",
                risk_level=risk_level,
                reason=(
                    "Column name looks related to target/outcome. This is only a "
                    "possible leakage signal and requires prediction-time availability review."
                ),
                evidence={
                    "matched_keywords": matched_keywords,
                    "target_name_overlap": target_name_overlap,
                    "explicit_target_relation": explicit_target_relation,
                },
            )

    return risks


def find_direct_duplicate_target_columns(
    df: pd.DataFrame,
    target_column: str,
    threshold: float,
    min_compare_rows: int,
) -> list[dict[str, Any]]:
    """Find columns whose values are almost identical to the target."""
    risks: list[dict[str, Any]] = []
    target = df[target_column]

    for column in df.columns:
        if column == target_column:
            continue

        feature = df[column]
        compare_mask = target.notna() & feature.notna()
        compare_rows = int(compare_mask.sum())

        if compare_rows < min_compare_rows:
            continue

        target_as_str = target.loc[compare_mask].astype(str)
        feature_as_str = feature.loc[compare_mask].astype(str)
        same_values_ratio = float(feature_as_str.eq(target_as_str).mean())

        if same_values_ratio >= threshold:
            add_risk(
                risks=risks,
                column=str(column),
                risk_type="duplicate_target_risk",
                risk_level="critical",
                reason=(
                    "Column values are almost identical to the target column on non-null rows. "
                    "This is a critical possible leakage risk."
                ),
                evidence={
                    "same_values_ratio": round(same_values_ratio, 4),
                    "threshold": threshold,
                    "compared_rows": compare_rows,
                    "coverage_percent": safe_percent(compare_rows, len(df)),
                },
            )

    return risks


def find_numeric_correlation_risks(
    df: pd.DataFrame,
    target_column: str,
    threshold: float,
    min_compare_rows: int,
) -> list[dict[str, Any]]:
    """Find numeric columns highly correlated with a numeric target."""
    risks: list[dict[str, Any]] = []

    if not pd.api.types.is_numeric_dtype(df[target_column]):
        return risks

    numeric_df = df.select_dtypes(include=["number"]).replace([np.inf, -np.inf], np.nan)

    if target_column not in numeric_df.columns or numeric_df.shape[1] <= 1:
        return risks

    target = numeric_df[target_column]

    for column in numeric_df.columns:
        if column == target_column:
            continue

        pair = pd.DataFrame({"target": target, "feature": numeric_df[column]}).dropna()

        if len(pair) < min_compare_rows:
            continue

        if (
            pair["target"].nunique(dropna=True) <= 1
            or pair["feature"].nunique(dropna=True) <= 1
        ):
            continue

        corr_value = pair["feature"].corr(pair["target"])

        if pd.isna(corr_value):
            continue

        abs_corr = abs(float(corr_value))

        if abs_corr >= threshold:
            risk_level = "high" if abs_corr >= 0.98 else "medium"
            add_risk(
                risks=risks,
                column=str(column),
                risk_type="high_correlation_risk",
                risk_level=risk_level,
                reason=(
                    "Column has very high correlation with the numeric target. "
                    "This may be valid signal, direct dependency, or possible leakage. "
                    "Confirm whether the feature is available before prediction time."
                ),
                evidence={
                    "correlation_with_target": round(float(corr_value), 4),
                    "absolute_correlation": round(abs_corr, 4),
                    "threshold": threshold,
                    "compared_rows": int(len(pair)),
                },
            )

    return risks


def prepare_numeric_matrix(
    df: pd.DataFrame,
    numeric_columns: list[str],
) -> pd.DataFrame:
    """Prepare numeric feature matrix for mutual information checks."""
    numeric_df = df[numeric_columns].replace([np.inf, -np.inf], np.nan).copy()

    for column in numeric_df.columns:
        median = numeric_df[column].median()
        if pd.isna(median):
            median = 0
        numeric_df[column] = numeric_df[column].fillna(median)

    return numeric_df


def _prepare_mutual_information_target(target: pd.Series) -> pd.Series:
    """Prepare numeric target for mutual information regression."""
    numeric_target = pd.to_numeric(target, errors="coerce")
    median = numeric_target.median()

    if pd.isna(median):
        median = 0

    return numeric_target.fillna(median)


def _drop_constant_numeric_features(features: pd.DataFrame) -> pd.DataFrame:
    """Remove constant numeric columns before mutual-information checks."""
    keep_columns = [
        column
        for column in features.columns
        if nunique_safely(features[column], dropna=True) > 1
    ]
    return features[keep_columns]


def _sample_for_mutual_information(
    features: pd.DataFrame,
    target: pd.Series,
    max_rows: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Sample mutual-information inputs for performance on large datasets."""
    if len(features) <= max_rows:
        return features, target

    random_state = get_int_config("random_seed", 42)
    sampled_index = features.sample(n=max_rows, random_state=random_state).index
    return features.loc[sampled_index], target.loc[sampled_index]


def find_mutual_information_risks(
    df: pd.DataFrame,
    target_column: str,
    threshold: float,
    max_rows: int,
) -> list[dict[str, Any]]:
    """
    Find high mutual-information target-proxy risks.

    This avoids arbitrary factorized-target Pearson correlation for classification.
    Mutual information is a heuristic, not proof of leakage.
    """
    if not as_bool(get_config_value("leakage.mutual_information_enabled", True), True):
        return []

    risks: list[dict[str, Any]] = []

    feature_numeric_columns = (
        df.drop(columns=[target_column])
        .select_dtypes(include=["number"])
        .columns.tolist()
    )

    if not feature_numeric_columns:
        return risks

    valid_target_mask = df[target_column].notna()
    target = df.loc[valid_target_mask, target_column]

    if nunique_safely(target, dropna=True) < 2:
        return risks

    features = prepare_numeric_matrix(
        df.loc[valid_target_mask],
        feature_numeric_columns,
    )
    features = _drop_constant_numeric_features(features)

    if features.empty:
        return risks

    features, target = _sample_for_mutual_information(
        features, target, max_rows=max_rows
    )

    try:
        random_state = get_int_config("random_seed", 42)

        if (
            pd.api.types.is_numeric_dtype(target)
            and nunique_safely(target, dropna=True) > 20
        ):
            mi_scores = mutual_info_regression(
                features,
                _prepare_mutual_information_target(target),
                random_state=random_state,
            )
            risk_type = "numeric_target_mutual_information_risk"
        else:
            mi_scores = mutual_info_classif(
                features,
                target.astype(str),
                random_state=random_state,
            )
            risk_type = "classification_target_mutual_information_risk"

    except (TypeError, ValueError) as error:
        logger.warning("Mutual information leakage heuristic skipped: %s", error)
        return risks

    for column, score in zip(features.columns, mi_scores, strict=False):
        score_float = float(score)

        if not np.isfinite(score_float):
            continue

        if score_float >= threshold:
            add_risk(
                risks=risks,
                column=str(column),
                risk_type=risk_type,
                risk_level="medium",
                reason=(
                    "Feature has high mutual information with the target. This can be a "
                    "valid strong predictor or a possible target proxy. Human review is required."
                ),
                evidence={
                    "mutual_information": round(score_float, 4),
                    "threshold": threshold,
                    "rows_used": int(len(features)),
                },
            )

    return risks


def find_classification_proxy_risks(
    df: pd.DataFrame,
    target_column: str,
    threshold: float,
    min_compare_rows: int,
) -> list[dict[str, Any]]:
    """Find numeric features strongly separated across target classes."""
    risks: list[dict[str, Any]] = []
    target = df[target_column]

    unique_classes = nunique_safely(target, dropna=True)

    if unique_classes < 2 or unique_classes > 50:
        return risks

    numeric_columns = (
        df.drop(columns=[target_column]).select_dtypes(include=["number"]).columns
    )

    for column in numeric_columns:
        feature = df[column].replace([np.inf, -np.inf], np.nan)

        paired = pd.DataFrame({target_column: target, column: feature}).dropna()

        if len(paired) < min_compare_rows:
            continue

        grouped = paired.groupby(target_column)[column]
        grouped_means = grouped.mean()
        grouped_counts = grouped.size()

        if (
            grouped_means.empty
            or grouped_means.isna().all()
            or grouped_means.shape[0] < 2
        ):
            continue

        if int(grouped_counts.min()) < 2:
            continue

        min_mean = float(grouped_means.min())
        max_mean = float(grouped_means.max())

        denominator = max(abs(max_mean), abs(min_mean), 1e-9)
        separation_ratio = abs(max_mean - min_mean) / denominator

        if separation_ratio >= threshold:
            add_risk(
                risks=risks,
                column=str(column),
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
                    "class_counts": {
                        str(key): int(value) for key, value in grouped_counts.items()
                    },
                    "compared_rows": int(len(paired)),
                },
            )

    return risks


def find_high_cardinality_review_columns(
    df: pd.DataFrame,
    target_column: str,
    thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    """Find high-cardinality columns that may behave like identifiers."""
    risks: list[dict[str, Any]] = []

    for column in df.columns:
        if column == target_column:
            continue

        series = df[column]
        unique_count = nunique_safely(series, dropna=True)
        unique_percent = safe_percent(unique_count, len(df))
        name_suggests_id = column_name_suggests_identifier(str(column))
        uniqueness_suggests_id = bool(
            is_string_like_series(series)
            and unique_count >= thresholds["high_cardinality_threshold"]
            and unique_percent >= thresholds["id_unique_percent_threshold"]
        )

        if name_suggests_id or uniqueness_suggests_id:
            add_risk(
                risks=risks,
                column=str(column),
                risk_type="high_cardinality_identifier_risk",
                risk_level="medium" if name_suggests_id else "low",
                reason=(
                    "Column has identifier-like naming or very high string/cardinality uniqueness. "
                    "Identifiers can cause memorization and poor generalization."
                ),
                evidence={
                    "unique_count": unique_count,
                    "unique_percent": unique_percent,
                    "name_suggests_identifier": name_suggests_id,
                    "uniqueness_suggests_identifier": uniqueness_suggests_id,
                },
            )

    return risks


# -----------------------------------------------------------------------------
# Reporting helpers
# -----------------------------------------------------------------------------


def generate_recommended_actions(all_risks: list[dict[str, Any]]) -> list[str]:
    """Generate recommended human-review actions for leakage risks."""
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
            "Investigate highly correlated numeric columns for formula-derived target leakage.",
        )

    if (
        "classification_target_mutual_information_risk" in risk_types
        or "numeric_target_mutual_information_risk" in risk_types
    ):
        actions.append(
            "Review high mutual-information features as possible target proxies.",
        )

    if "high_cardinality_identifier_risk" in risk_types:
        actions.append(
            "Drop or review identifier-like high-cardinality columns before one-hot encoding.",
        )

    deduped: list[str] = []
    for action in actions:
        if action not in deduped:
            deduped.append(action)

    return deduped


def build_review_columns(all_risks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build compact column-wise review summary for UI/HITL."""
    by_column: dict[str, dict[str, Any]] = {}

    for risk in all_risks:
        column = str(risk.get("column", ""))
        if not column:
            continue

        entry = by_column.setdefault(
            column,
            {
                "column": column,
                "highest_risk_level": "low",
                "risk_types": [],
                "risk_count": 0,
                "requires_human_review": True,
            },
        )

        entry["risk_count"] += 1
        risk_type = str(risk.get("risk_type", "unknown"))
        if risk_type not in entry["risk_types"]:
            entry["risk_types"].append(risk_type)

        current_level = str(entry["highest_risk_level"])
        new_level = str(risk.get("risk_level", "low"))
        if RISK_SEVERITY_ORDER.get(new_level, 99) < RISK_SEVERITY_ORDER.get(
            current_level, 99
        ):
            entry["highest_risk_level"] = new_level

    return sorted(
        by_column.values(),
        key=lambda item: (
            RISK_SEVERITY_ORDER.get(str(item.get("highest_risk_level", "low")), 99),
            str(item.get("column", "")),
        ),
    )


def build_leakage_warning(total_risks: int) -> str:
    """Return standard leakage warning text."""
    if total_risks == 0:
        return (
            "No obvious leakage indicators were detected by current heuristics. "
            "This does not guarantee the dataset is leakage-free."
        )

    return (
        "These are possible leakage risks, not confirmed leakage. "
        "A human should review whether flagged columns would be available at prediction time."
    )


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


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

        resolved_target = validate_inputs(df, target_column)
        thresholds = get_thresholds()
        min_compare_rows = int(thresholds["min_compare_rows"])
        mi_max_rows = int(thresholds["mutual_information_max_rows"])

        name_based_risks = find_name_based_leakage_risks(df, resolved_target)

        duplicate_target_risks = find_direct_duplicate_target_columns(
            df=df,
            target_column=resolved_target,
            threshold=thresholds["duplicate_target_threshold"],
            min_compare_rows=min_compare_rows,
        )

        numeric_correlation_risks = find_numeric_correlation_risks(
            df=df,
            target_column=resolved_target,
            threshold=thresholds["high_correlation_threshold"],
            min_compare_rows=min_compare_rows,
        )

        mutual_information_risks = find_mutual_information_risks(
            df=df,
            target_column=resolved_target,
            threshold=thresholds["mutual_information_threshold"],
            max_rows=mi_max_rows,
        )

        classification_proxy_risks = find_classification_proxy_risks(
            df=df,
            target_column=resolved_target,
            threshold=thresholds["classification_proxy_threshold"],
            min_compare_rows=min_compare_rows,
        )

        high_cardinality_review_risks = find_high_cardinality_review_columns(
            df=df,
            target_column=resolved_target,
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

        sorted_all_risks = sort_risks(all_risks)
        risk_summary = summarize_risks(sorted_all_risks)
        risk_type_summary = summarize_risk_types(sorted_all_risks)
        overall_severity = get_overall_severity(risk_summary)
        review_columns = build_review_columns(sorted_all_risks)

        report: dict[str, Any] = {
            "target_column": resolved_target,
            "total_possible_leakage_risks": int(len(sorted_all_risks)),
            "overall_severity": overall_severity,
            "risk_summary": risk_summary,
            "risk_type_summary": risk_type_summary,
            "thresholds": thresholds,
            "name_based_risks": name_based_risks,
            "duplicate_target_risks": duplicate_target_risks,
            "numeric_correlation_risks": numeric_correlation_risks,
            "mutual_information_risks": mutual_information_risks,
            "classification_proxy_risks": classification_proxy_risks,
            "high_cardinality_review_risks": high_cardinality_review_risks,
            "review_columns": review_columns,
            "all_risks": sorted_all_risks,
            "recommended_actions": generate_recommended_actions(sorted_all_risks),
            "requires_human_review": bool(sorted_all_risks),
            "leakage_policy": {
                "deterministic_only": True,
                "confirmed_leakage_claimed": False,
                "human_review_required_for_flagged_columns": bool(sorted_all_risks),
            },
            "warning": build_leakage_warning(len(sorted_all_risks)),
            "message": "Leakage check completed successfully.",
        }

        logger.info(
            "Leakage check completed. risks=%s severity=%s",
            report["total_possible_leakage_risks"],
            overall_severity,
        )
        return report

    except (LeakageDetectionError, InvalidTargetColumnError):
        raise
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        pd.errors.ParserError,
    ) as error:
        logger.exception("Leakage detection failed.")
        raise LeakageDetectionError(
            "Leakage detection failed.",
            error_detail=str(error),
        ) from error


if __name__ == "__main__":
    from src.audit.profiler import load_dataset

    dataset_path = "data/sample/student_mark.csv"
    target_column = "Grade"

    dataframe = load_dataset(dataset_path)
    leakage_report = run_leakage_check(dataframe, target_column)

    print(leakage_report)
