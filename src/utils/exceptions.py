class AuditCopilotException(Exception):
    """
    Base exception for Agentic ML Audit Copilot.

    Use this for project-specific errors instead of raising
    generic exceptions across modules.
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


class PreprocessingError(AuditCopilotException):
    """Raised when preprocessing pipeline creation or transformation fails."""


class ModelTrainingError(AuditCopilotException):
    """Raised when baseline model training or evaluation fails."""


class MLflowTrackingError(AuditCopilotException):
    """Raised when MLflow experiment tracking fails."""


class ReportGenerationError(AuditCopilotException):
    """Raised when audit report generation fails."""


class AgentWorkflowError(AuditCopilotException):
    """Raised when LangGraph workflow execution fails."""