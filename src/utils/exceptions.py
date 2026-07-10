from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


class AuditCopilotException(Exception):
    """
    Base exception for Agentic ML Audit Copilot.

    The exception keeps three concerns separate:
    - message: clean user-facing message for API/UI responses
    - error_detail: optional technical detail for logs/debugging
    - metadata: optional JSON-safe structured context

    Backward compatibility:
    Existing usage like AuditCopilotException("msg", error_detail="detail")
    continues to work exactly as before.
    """

    status_code: int = 500
    default_error_code: str = "audit_copilot_error"

    def __init__(
        self,
        message: str,
        error_detail: str | None = None,
        *,
        error_code: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        self.message = str(message).strip() or "An unexpected audit error occurred."
        self.error_detail = str(error_detail).strip() if error_detail else None
        self.error_code = str(error_code or self.default_error_code).strip()
        self.metadata = self._json_safe_metadata(metadata or {})
        self.cause = cause

        full_message = self.message
        if self.error_detail:
            full_message = f"{self.message} | Detail: {self.error_detail}"

        super().__init__(full_message)

    @staticmethod
    def _json_safe_value(value: Any) -> Any:
        """Convert arbitrary values into JSON-safe primitives."""
        if value is None:
            return None

        if isinstance(value, bool | str | int):
            return value

        if isinstance(value, float):
            return value if math.isfinite(value) else None

        try:
            if hasattr(value, "item") and callable(value.item):
                return AuditCopilotException._json_safe_value(value.item())
        except (AttributeError, TypeError, ValueError):
            pass

        try:
            if hasattr(value, "isoformat") and callable(value.isoformat):
                return str(value.isoformat())
        except (AttributeError, TypeError, ValueError):
            pass

        if isinstance(value, Mapping):
            return {
                str(key): AuditCopilotException._json_safe_value(item)
                for key, item in value.items()
            }

        if isinstance(value, list | tuple | set):
            return [AuditCopilotException._json_safe_value(item) for item in value]

        return str(value)

    @classmethod
    def _json_safe_metadata(cls, metadata: Mapping[str, Any]) -> dict[str, Any]:
        """Return JSON-safe metadata for API/UI/logging."""
        return {
            str(key): cls._json_safe_value(value) for key, value in metadata.items()
        }

    @classmethod
    def from_exception(
        cls,
        message: str,
        error: BaseException,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> AuditCopilotException:
        """Create a project exception from an underlying exception."""
        return cls(
            message=message,
            error_detail=str(error),
            metadata=metadata,
            cause=error,
        )

    def to_dict(self, *, include_metadata: bool = True) -> dict[str, Any]:
        """Convert the exception into a safe API/UI response dictionary."""
        payload: dict[str, Any] = {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "error_detail": self.error_detail,
            "status_code": int(self.status_code),
        }

        if include_metadata and self.metadata:
            payload["metadata"] = self.metadata

        return payload

    def to_log_dict(self) -> dict[str, Any]:
        """Convert the exception into a structured logging dictionary."""
        payload = self.to_dict(include_metadata=True)

        if self.cause is not None:
            payload["cause_type"] = self.cause.__class__.__name__
            payload["cause_message"] = str(self.cause)

        return payload

    def user_message(self) -> str:
        """Return a clean message for UI/API display."""
        return self.message

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"status_code={self.status_code!r}, "
            f"error_code={self.error_code!r})"
        )


class ConfigError(AuditCopilotException):
    """Raised when configuration loading or validation fails."""

    status_code = 500
    default_error_code = "config_error"


class InvalidDatasetError(AuditCopilotException):
    """Raised when the dataset is missing, empty, unreadable, or invalid."""

    status_code = 400
    default_error_code = "invalid_dataset"


class InvalidTargetColumnError(AuditCopilotException):
    """Raised when the selected target column does not exist or is unusable."""

    status_code = 400
    default_error_code = "invalid_target_column"


class DataQualityError(AuditCopilotException):
    """Raised when data quality audit fails unexpectedly."""

    status_code = 500
    default_error_code = "data_quality_error"


class ProblemTypeDetectionError(AuditCopilotException):
    """Raised when problem type detection fails unexpectedly."""

    status_code = 500
    default_error_code = "problem_type_detection_error"


class LeakageDetectionError(AuditCopilotException):
    """Raised when leakage detection fails unexpectedly."""

    status_code = 500
    default_error_code = "leakage_detection_error"


class MetricRecommendationError(AuditCopilotException):
    """Raised when metric recommendation fails unexpectedly."""

    status_code = 500
    default_error_code = "metric_recommendation_error"


class ClassImbalanceError(AuditCopilotException):
    """Raised when class imbalance detection fails unexpectedly."""

    status_code = 500
    default_error_code = "class_imbalance_error"


class PreprocessingError(AuditCopilotException):
    """Raised when preprocessing pipeline creation or transformation fails."""

    status_code = 500
    default_error_code = "preprocessing_error"


class ModelTrainingError(AuditCopilotException):
    """Raised when baseline model training or evaluation fails."""

    status_code = 500
    default_error_code = "model_training_error"


class MLflowTrackingError(AuditCopilotException):
    """Raised when MLflow experiment tracking fails."""

    status_code = 500
    default_error_code = "mlflow_tracking_error"


class ExplainabilityError(AuditCopilotException):
    """Raised when model explainability generation fails."""

    status_code = 500
    default_error_code = "explainability_error"


class ReportGenerationError(AuditCopilotException):
    """Raised when audit report generation or saving fails."""

    status_code = 500
    default_error_code = "report_generation_error"


class LLMReportError(AuditCopilotException):
    """Raised when LLM-based report explanation or Q&A fails."""

    status_code = 500
    default_error_code = "llm_report_error"


class AgentWorkflowError(AuditCopilotException):
    """Raised when LangGraph workflow execution fails."""

    status_code = 500
    default_error_code = "agent_workflow_error"


__all__ = [
    "AgentWorkflowError",
    "AuditCopilotException",
    "ClassImbalanceError",
    "ConfigError",
    "DataQualityError",
    "ExplainabilityError",
    "InvalidDatasetError",
    "InvalidTargetColumnError",
    "LLMReportError",
    "LeakageDetectionError",
    "MetricRecommendationError",
    "MLflowTrackingError",
    "ModelTrainingError",
    "PreprocessingError",
    "ProblemTypeDetectionError",
    "ReportGenerationError",
]
