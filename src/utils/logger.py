import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.utils.config import get_config_value


def get_logger(name: str) -> logging.Logger:
    """
    Create and return a configured project logger.

    Supports console logging, rotating file logging,
    config-driven settings, and duplicate handler prevention.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    log_level_name = str(get_config_value("logging.log_level", "INFO")).upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    log_file = get_config_value("logging.log_file", "logs/app.log")
    max_bytes = int(get_config_value("logging.max_bytes", 5_000_000))
    backup_count = int(get_config_value("logging.backup_count", 3))

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.setLevel(log_level)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt=(
            "%(asctime)s | %(levelname)s | %(name)s | "
            "%(filename)s:%(lineno)d | %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        filename=log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger