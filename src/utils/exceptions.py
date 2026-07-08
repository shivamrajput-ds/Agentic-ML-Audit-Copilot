from __future__ import annotations


class AuditCopilotException(Exception):
    """
    Base exception for Agentic ML Audit Copilot.
    """

    def __init__(self, message: str, error_detail: str | None = None):
        self.message = message
        self.error_detail = error_detail

        full_message = (
            f"{message} | Detail: {error_detail}"
            if error_detail
            else message
        )

        super().__init__(full_message)

    def to_dict(self) -> dict[str, str | None]:
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_detail": self.error_detail,
        }


class ConfigError(AuditCopilotException):
    """Raised when configuration loading or validation fails."""


class InvalidDatasetError(AuditCopilotException):
    """Raised when the dataset is missing, empty, unreadable, or invalid."""


class InvalidTargetColumnError(AuditCopilotException):
    """Raised when the selected target column does not exist or is unusable."""


class DataQualityError(AuditCopilotException):
    """Raised when data quality audit fails unexpectedly."""


class ProblemTypeDetectionError(AuditCopilotException):
    """Raised when problem type detection fails unexpectedly."""


class LeakageDetectionError(AuditCopilotException):
    """Raised when leakage detection fails unexpectedly."""


class MetricRecommendationError(AuditCopilotException):
    """Raised when metric recommendation fails unexpectedly."""


class ClassImbalanceError(AuditCopilotException):
    """Raised when class imbalance detection fails unexpectedly."""


class PreprocessingError(AuditCopilotException):
    """Raised when preprocessing pipeline creation or transformation fails."""


class ModelTrainingError(AuditCopilotException):
    """Raised when baseline model training or evaluation fails."""


class MLflowTrackingError(AuditCopilotException):
    """Raised when MLflow experiment tracking fails."""


class ReportGenerationError(AuditCopilotException):
    """Raised when audit report generation or saving fails."""


class LLMReportError(AuditCopilotException):
    """Raised when LLM-based report explanation or Q&A fails."""


class AgentWorkflowError(AuditCopilotException):
    """Raised when LangGraph workflow execution fails."""