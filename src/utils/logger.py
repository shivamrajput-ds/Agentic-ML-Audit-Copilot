from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from src.utils.config import get_config_value


_LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | "
    "%(filename)s:%(lineno)d | %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def as_bool(value: Any) -> bool:
    """
    Convert config values safely into boolean.
    """
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.lower().strip() in {"true", "1", "yes", "y", "on"}

    return bool(value)


def get_log_level() -> int:
    """
    Read log level from config and fall back to INFO if invalid.
    """
    level_name = str(get_config_value("logging.log_level", "INFO")).upper().strip()
    level = getattr(logging, level_name, None)

    if isinstance(level, int):
        return level

    return logging.INFO


def build_formatter() -> logging.Formatter:
    """
    Build a consistent formatter for console and file logs.
    """
    return logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)


def has_handler(logger: logging.Logger, handler_type: type[logging.Handler]) -> bool:
    """
    Check whether the logger already has a handler of this type.
    """
    return any(isinstance(handler, handler_type) for handler in logger.handlers)


def add_console_handler(
    logger: logging.Logger,
    log_level: int,
    formatter: logging.Formatter,
) -> None:
    """
    Add console logging once.
    """
    if has_handler(logger, logging.StreamHandler):
        return

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def add_file_handler(
    logger: logging.Logger,
    log_level: int,
    formatter: logging.Formatter,
) -> None:
    """
    Add rotating file logging once.
    """
    if has_handler(logger, RotatingFileHandler):
        return

    log_file = str(get_config_value("logging.log_file", "logs/app.log"))
    max_bytes = int(get_config_value("logging.max_bytes", 5_000_000))
    backup_count = int(get_config_value("logging.backup_count", 5))

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        filename=log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Create and return a configured project logger.

    Supports:
    - config-driven log level
    - console logging on/off
    - rotating file logging on/off
    - duplicate handler prevention
    - safe fallback when logging config is missing or invalid
    """
    logger = logging.getLogger(name)
    log_level = get_log_level()

    logger.setLevel(log_level)
    logger.propagate = False

    # If handlers already exist, only update their level and return.
    # This prevents duplicate logs during Streamlit reruns/tests.
    if logger.handlers:
        for handler in logger.handlers:
            handler.setLevel(log_level)
        return logger

    formatter = build_formatter()

    console_logging = as_bool(get_config_value("logging.console_logging", True))
    file_logging = as_bool(get_config_value("logging.file_logging", True))

    if file_logging:
        try:
            add_file_handler(logger, log_level, formatter)
        except Exception:
            # Logging setup should never crash the application.
            # Console handler below still gives visibility if enabled.
            pass

    if console_logging:
        add_console_handler(logger, log_level, formatter)

    if not logger.handlers:
        logger.addHandler(logging.NullHandler())

    return logger
