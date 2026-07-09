from __future__ import annotations

from typing import Any


class AuditCopilotException(Exception):
    """
    Base exception for Agentic ML Audit Copilot.

    Keeps a clean user-facing message and an optional technical detail.
    API/UI layers can safely convert this exception to a dictionary.
    """

    status_code: int = 500

    def __init__(self, message: str, error_detail: str | None = None):
        self.message = str(message)
        self.error_detail = str(error_detail) if error_detail else None

        full_message = (
            f"{self.message} | Detail: {self.error_detail}"
            if self.error_detail
            else self.message
        )

        super().__init__(full_message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_detail": self.error_detail,
            "status_code": self.status_code,
        }

    def user_message(self) -> str:
        """
        Return a clean message for UI/API display.
        """
        return self.message


class ConfigError(AuditCopilotException):
    """Raised when configuration loading or validation fails."""

    status_code = 500


class InvalidDatasetError(AuditCopilotException):
    """Raised when the dataset is missing, empty, unreadable, or invalid."""

    status_code = 400


class InvalidTargetColumnError(AuditCopilotException):
    """Raised when the selected target column does not exist or is unusable."""

    status_code = 400


class DataQualityError(AuditCopilotException):
    """Raised when data quality audit fails unexpectedly."""

    status_code = 500


class ProblemTypeDetectionError(AuditCopilotException):
    """Raised when problem type detection fails unexpectedly."""

    status_code = 500


class LeakageDetectionError(AuditCopilotException):
    """Raised when leakage detection fails unexpectedly."""

    status_code = 500


class MetricRecommendationError(AuditCopilotException):
    """Raised when metric recommendation fails unexpectedly."""

    status_code = 500


class ClassImbalanceError(AuditCopilotException):
    """Raised when class imbalance detection fails unexpectedly."""

    status_code = 500


class PreprocessingError(AuditCopilotException):
    """Raised when preprocessing pipeline creation or transformation fails."""

    status_code = 500


class ModelTrainingError(AuditCopilotException):
    """Raised when baseline model training or evaluation fails."""

    status_code = 500


class MLflowTrackingError(AuditCopilotException):
    """Raised when MLflow experiment tracking fails."""

    status_code = 500


class ReportGenerationError(AuditCopilotException):
    """Raised when audit report generation or saving fails."""

    status_code = 500


class LLMReportError(AuditCopilotException):
    """Raised when LLM-based report explanation or Q&A fails."""

    status_code = 500


class AgentWorkflowError(AuditCopilotException):
    """Raised when LangGraph workflow execution fails."""

    status_code = 500
