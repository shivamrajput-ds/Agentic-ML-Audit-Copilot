from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from src.utils.exceptions import ConfigError

ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT_DIR / "config.yaml"
ENV_PATH = ROOT_DIR / ".env"

TRUE_VALUES = {"true", "1", "yes", "y", "on"}
FALSE_VALUES = {"false", "0", "no", "n", "off"}

load_dotenv(dotenv_path=ENV_PATH)


@lru_cache(maxsize=1)
def load_config(config_path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """
    Load config.yaml once and return it as a dictionary.

    Use reload_config() in tests or after changing config at runtime.
    """
    path = Path(config_path)

    if not path.is_absolute():
        path = ROOT_DIR / path

    try:
        if not path.exists():
            raise ConfigError(f"Config file not found: {path}")

        with path.open(encoding="utf-8") as file:
            config = yaml.safe_load(file)

        if config is None:
            return {}

        if not isinstance(config, dict):
            raise ConfigError("config.yaml must contain a valid YAML dictionary.")

        return config

    except ConfigError:
        raise
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(
            "Failed to load configuration.",
            error_detail=str(exc),
        ) from exc


def reload_config() -> None:
    """Clear cached config and reload .env."""
    load_config.cache_clear()
    load_dotenv(dotenv_path=ENV_PATH, override=True)


def get_config_value(key_path: str, default: Any = None) -> Any:
    """
    Get nested config value using dot notation.

    Example:
        get_config_value("modeling.test_size", 0.2)
    """
    if not str(key_path).strip():
        return default

    value: Any = load_config()

    for key in str(key_path).split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]

    return value


def get_env_value(key: str, default: Any = None) -> Any:
    """Get environment variable from .env or system environment."""
    normalized_key = str(key).strip()

    if not normalized_key:
        return default

    value = os.getenv(normalized_key)

    if value is None or not value.strip():
        return default

    return value


def to_bool(value: Any, default: bool = False) -> bool:
    """Convert config/env values into bool without surprising string behavior."""
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


def get_bool_config(key_path: str, default: bool = False) -> bool:
    """Get boolean config value safely."""
    value = get_config_value(key_path, default)
    return to_bool(value, default=default)


def get_int_config(key_path: str, default: int) -> int:
    """Read an integer config value with safe fallback."""
    value = get_config_value(key_path, default)

    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def get_float_config(key_path: str, default: float) -> float:
    """Read a float config value with safe fallback."""
    value = get_config_value(key_path, default)

    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def get_list_config(key_path: str, default: list[Any] | None = None) -> list[Any]:
    """Read a list config value safely."""
    fallback = [] if default is None else default
    value = get_config_value(key_path, fallback)

    if isinstance(value, list):
        return value

    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]

    return fallback


def get_groq_api_key() -> str | None:
    """Get Groq API key from environment variables."""
    api_key = get_env_value("GROQ_API_KEY")

    if api_key is None:
        return None

    normalized_api_key = str(api_key).strip()
    return normalized_api_key or None


def get_llm_config() -> dict[str, Any]:
    """Get LLM configuration from config.yaml and environment variables."""
    return {
        "provider": str(get_config_value("llm.provider", "groq")),
        "model": str(get_config_value("llm.model", "llama-3.3-70b-versatile")),
        "temperature": get_float_config("llm.temperature", 0.2),
        "max_tokens": get_int_config("llm.max_tokens", 2_000),
        "timeout": get_int_config("llm.timeout", 120),
        "api_key": get_groq_api_key(),
        "enabled": get_bool_config("llm.enabled", True),
        "cache_enabled": get_bool_config("llm.cache_enabled", True),
        "max_retries": get_int_config("llm.max_retries", 3),
    }


def get_project_root() -> Path:
    """Return repository root path."""
    return ROOT_DIR
