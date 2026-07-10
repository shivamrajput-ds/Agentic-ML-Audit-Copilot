from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from src.utils.config import get_config_value, get_project_root

_LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_DEFAULT_LOG_FILE = "logs/app.log"
_DEFAULT_MAX_BYTES = 5_000_000
_DEFAULT_BACKUP_COUNT = 5
_DEFAULT_LOG_LEVEL = logging.INFO
_DEFAULT_LOGGER_NAME = "agentic_ml_audit_copilot"

_TRUE_VALUES = {"true", "1", "yes", "y", "on"}
_FALSE_VALUES = {"false", "0", "no", "n", "off"}
_HANDLER_KIND_ATTR = "_agentic_ml_audit_handler_kind"


class _SafeLoggerConfig:
    """Small config wrapper so logging setup never crashes the application."""

    @staticmethod
    def get(key_path: str, default: Any = None) -> Any:
        try:
            return get_config_value(key_path, default)
        except Exception:  # noqa: BLE001 - logger setup must be failure-safe.
            return default


def as_bool(value: Any, default: bool = False) -> bool:
    """Convert common config values safely into boolean."""
    if isinstance(value, bool):
        return value

    if value is None:
        return default

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_VALUES:
            return True
        if normalized in _FALSE_VALUES:
            return False
        return default

    return bool(value)


def safe_int(
    value: Any,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Convert config values to int with a safe fallback and optional bounds."""
    try:
        parsed = int(value)
    except (KeyError, TypeError, ValueError, RuntimeError):
        parsed = int(default)

    if minimum is not None:
        parsed = max(minimum, parsed)

    if maximum is not None:
        parsed = min(maximum, parsed)

    return parsed


def get_log_level() -> int:
    """Read log level from config and fall back to INFO if invalid."""
    raw_level = _SafeLoggerConfig.get("logging.log_level", "INFO")

    if isinstance(raw_level, int):
        return raw_level

    level_name = str(raw_level).strip().upper()
    level = getattr(logging, level_name, _DEFAULT_LOG_LEVEL)

    return level if isinstance(level, int) else _DEFAULT_LOG_LEVEL


def build_formatter() -> logging.Formatter:
    """Build a consistent formatter for console and file logs."""
    raw_format = _SafeLoggerConfig.get("logging.format", _LOG_FORMAT)
    raw_date_format = _SafeLoggerConfig.get("logging.date_format", _DATE_FORMAT)

    log_format = str(raw_format).strip() or _LOG_FORMAT
    date_format = str(raw_date_format).strip() or _DATE_FORMAT

    return logging.Formatter(fmt=log_format, datefmt=date_format)


def _mark_handler(handler: logging.Handler, kind: str) -> None:
    """Mark project-created handlers to avoid duplicate setup on reruns."""
    setattr(handler, _HANDLER_KIND_ATTR, kind)


def _get_handler_kind(handler: logging.Handler) -> str | None:
    """Return project handler kind when present."""
    kind = getattr(handler, _HANDLER_KIND_ATTR, None)
    return str(kind) if kind is not None else None


def _has_project_handler(target_logger: logging.Logger, kind: str) -> bool:
    """Return True when this logger already has a project handler of this kind."""
    return any(_get_handler_kind(handler) == kind for handler in target_logger.handlers)


def _resolve_log_path(log_file: str | Path) -> Path:
    """Resolve configured log path relative to project root when needed."""
    path = Path(str(log_file)).expanduser()

    if path.is_absolute():
        return path

    try:
        return get_project_root() / path
    except Exception:  # noqa: BLE001 - fallback keeps logging robust.
        return Path.cwd() / path


def _close_handler(handler: logging.Handler) -> None:
    """Close a logging handler safely."""
    try:
        handler.close()
    except Exception:  # noqa: BLE001 - logger cleanup must be failure-safe.
        pass


def add_console_handler(
    target_logger: logging.Logger,
    log_level: int,
    formatter: logging.Formatter,
) -> None:
    """Add console logging once."""
    if _has_project_handler(target_logger, "console"):
        return

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    _mark_handler(console_handler, "console")
    target_logger.addHandler(console_handler)


def add_file_handler(
    target_logger: logging.Logger,
    log_level: int,
    formatter: logging.Formatter,
) -> None:
    """Add rotating file logging once."""
    if _has_project_handler(target_logger, "file"):
        return

    raw_log_file = _SafeLoggerConfig.get("logging.log_file", _DEFAULT_LOG_FILE)
    log_file = str(raw_log_file).strip() or _DEFAULT_LOG_FILE

    max_bytes = safe_int(
        _SafeLoggerConfig.get("logging.max_bytes", _DEFAULT_MAX_BYTES),
        _DEFAULT_MAX_BYTES,
        minimum=1_024,
    )
    backup_count = safe_int(
        _SafeLoggerConfig.get("logging.backup_count", _DEFAULT_BACKUP_COUNT),
        _DEFAULT_BACKUP_COUNT,
        minimum=0,
    )

    log_path = _resolve_log_path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        filename=log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    _mark_handler(file_handler, "file")
    target_logger.addHandler(file_handler)


def _update_existing_handlers(
    target_logger: logging.Logger,
    log_level: int,
    formatter: logging.Formatter,
) -> None:
    """Update existing handler levels and formatters during reruns/tests."""
    for handler in target_logger.handlers:
        handler.setLevel(log_level)

        if not isinstance(handler, logging.NullHandler):
            handler.setFormatter(formatter)


def _remove_disabled_project_handlers(
    target_logger: logging.Logger,
    *,
    console_logging: bool,
    file_logging: bool,
) -> None:
    """Remove project handlers when config disables them during tests/reruns."""
    handlers_to_remove: list[logging.Handler] = []

    for handler in target_logger.handlers:
        kind = _get_handler_kind(handler)

        if kind == "console" and not console_logging:
            handlers_to_remove.append(handler)
        elif kind == "file" and not file_logging:
            handlers_to_remove.append(handler)
        elif kind == "null" and (console_logging or file_logging):
            handlers_to_remove.append(handler)

    for handler in handlers_to_remove:
        target_logger.removeHandler(handler)
        _close_handler(handler)


def _ensure_null_handler(target_logger: logging.Logger) -> None:
    """Add a null handler when all output handlers are disabled/unavailable."""
    if target_logger.handlers:
        return

    null_handler = logging.NullHandler()
    _mark_handler(null_handler, "null")
    target_logger.addHandler(null_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Create and return a configured project logger.

    Supports config-driven log level, optional console/file logging, rotating file
    logs, duplicate handler prevention, test reruns, and safe fallback. Logger
    setup must never crash the application.
    """
    logger_name = str(name).strip() if name is not None else ""
    logger_name = logger_name or _DEFAULT_LOGGER_NAME

    target_logger = logging.getLogger(logger_name)
    log_level = get_log_level()
    formatter = build_formatter()

    console_logging = as_bool(
        _SafeLoggerConfig.get("logging.console_logging", True),
        default=True,
    )
    file_logging = as_bool(
        _SafeLoggerConfig.get("logging.file_logging", True),
        default=True,
    )
    propagate = as_bool(
        _SafeLoggerConfig.get("logging.propagate", False),
        default=False,
    )

    target_logger.setLevel(log_level)
    target_logger.propagate = propagate

    _update_existing_handlers(target_logger, log_level, formatter)
    _remove_disabled_project_handlers(
        target_logger,
        console_logging=console_logging,
        file_logging=file_logging,
    )

    if file_logging:
        try:
            add_file_handler(target_logger, log_level, formatter)
        except Exception:  # noqa: BLE001 - logging setup should never crash app.
            pass

    if console_logging:
        try:
            add_console_handler(target_logger, log_level, formatter)
        except Exception:  # noqa: BLE001 - logging setup should never crash app.
            pass

    _ensure_null_handler(target_logger)
    return target_logger


def reset_logger(name: str) -> None:
    """
    Remove and close handlers for one logger.

    Useful in tests after calling reload_config(). This does not affect the public
    get_logger() API and is safe to ignore in normal application code.
    """
    logger_name = str(name).strip() if name is not None else ""
    logger_name = logger_name or _DEFAULT_LOGGER_NAME
    target_logger = logging.getLogger(logger_name)

    for handler in list(target_logger.handlers):
        target_logger.removeHandler(handler)
        _close_handler(handler)


__all__ = [
    "get_logger",
    "reset_logger",
]
