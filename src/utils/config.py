from __future__ import annotations

import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency fallback

    def load_dotenv(*args: Any, **kwargs: Any) -> bool:
        """Fallback when python-dotenv is not installed."""
        _ = args, kwargs
        return False


from src.utils.exceptions import ConfigError

ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT_DIR / "config.yaml"
ENV_PATH = ROOT_DIR / ".env"

TRUE_VALUES = {"true", "1", "yes", "y", "on"}
FALSE_VALUES = {"false", "0", "no", "n", "off"}


load_dotenv(dotenv_path=ENV_PATH)


def resolve_config_path(config_path: str | Path) -> Path:
    """Resolve config path relative to the project root when needed."""
    if config_path is None or not str(config_path).strip():
        raise ConfigError("Config path is required.")

    path = Path(str(config_path)).expanduser()

    if not path.is_absolute():
        path = ROOT_DIR / path

    return path.resolve()


@lru_cache(maxsize=8)
def load_config(config_path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """
    Load a YAML config file and return it as a dictionary.

    The default project config is cached for performance. Use reload_config()
    in tests or after changing config values at runtime.
    """
    path = resolve_config_path(config_path)

    try:
        if not path.exists():
            raise ConfigError(f"Config file not found: {path}")

        if not path.is_file():
            raise ConfigError(f"Config path is not a file: {path}")

        with path.open(encoding="utf-8") as file:
            config = yaml.safe_load(file)

        if config is None:
            return {}

        if not isinstance(config, dict):
            raise ConfigError("config.yaml must contain a valid YAML dictionary.")

        return {str(key): value for key, value in config.items()}

    except ConfigError:
        raise
    except (OSError, yaml.YAMLError) as error:
        raise ConfigError(
            "Failed to load configuration.",
            error_detail=str(error),
        ) from error


def reload_config() -> None:
    """Clear cached config and reload environment variables from .env."""
    load_config.cache_clear()
    load_dotenv(dotenv_path=ENV_PATH, override=True)


def get_config_value(key_path: str, default: Any = None) -> Any:
    """
    Get a nested config value using dot notation.

    Example:
        get_config_value("modeling.test_size", 0.2)
    """
    if key_path is None or not str(key_path).strip():
        return default

    try:
        value: Any = load_config()
    except ConfigError:
        return default

    for raw_key in str(key_path).split("."):
        key = raw_key.strip()
        if not key:
            return default

        if not isinstance(value, dict) or key not in value:
            return default

        value = value[key]

    return default if value is None else value


def get_config_section(
    key_path: str,
    default: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a config section as a dictionary."""
    fallback = {} if default is None else dict(default)
    value = get_config_value(key_path, fallback)

    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}

    return fallback


def get_env_value(key: str, default: Any = None) -> Any:
    """Get environment variable from .env or system environment."""
    normalized_key = str(key).strip() if key is not None else ""

    if not normalized_key:
        return default

    value = os.getenv(normalized_key)

    if value is None or not str(value).strip():
        return default

    return value.strip() if isinstance(value, str) else value


def get_env_first(keys: list[str] | tuple[str, ...], default: Any = None) -> Any:
    """Return the first available environment value from candidate keys."""
    for key in keys:
        value = get_env_value(key, None)
        if value is not None:
            return value

    return default


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

    if isinstance(value, int | float):
        if isinstance(value, float) and not math.isfinite(value):
            return default
        return bool(value)

    return bool(value)


def get_bool_config(key_path: str, default: bool = False) -> bool:
    """Get boolean config value safely."""
    value = get_config_value(key_path, default)
    return to_bool(value, default=default)


def safe_int(value: Any, default: int, minimum: int | None = None) -> int:
    """Convert value to int with optional lower bound."""
    try:
        parsed = int(value)
    except (KeyError, TypeError, ValueError, RuntimeError):
        parsed = int(default)

    if minimum is not None:
        return max(minimum, parsed)

    return parsed


def safe_float(value: Any, default: float, minimum: float | None = None) -> float:
    """Convert value to float with optional lower bound and finite-value guard."""
    try:
        parsed = float(value)
    except (KeyError, TypeError, ValueError, RuntimeError):
        parsed = float(default)

    if not math.isfinite(parsed):
        parsed = float(default)

    if minimum is not None:
        return max(float(minimum), parsed)

    return parsed


def get_int_config(key_path: str, default: int, minimum: int | None = None) -> int:
    """Read an integer config value with safe fallback."""
    value = get_config_value(key_path, default)
    return safe_int(value=value, default=default, minimum=minimum)


def get_float_config(
    key_path: str,
    default: float,
    minimum: float | None = None,
) -> float:
    """Read a float config value with safe fallback."""
    value = get_config_value(key_path, default)
    return safe_float(value=value, default=default, minimum=minimum)


def get_list_config(key_path: str, default: list[Any] | None = None) -> list[Any]:
    """Read a list config value safely."""
    fallback = [] if default is None else list(default)
    value = get_config_value(key_path, fallback)

    if isinstance(value, list):
        return value

    if isinstance(value, tuple | set):
        return list(value)

    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]

    return fallback


def get_str_config(key_path: str, default: str = "") -> str:
    """Read a string config value safely."""
    value = get_config_value(key_path, default)

    if value is None:
        return default

    normalized = str(value).strip()
    return normalized if normalized else default


def get_path_config(key_path: str, default: str | Path) -> Path:
    """Read a path config value and resolve relative paths from project root."""
    raw_path = get_config_value(key_path, default)

    if raw_path is None or not str(raw_path).strip():
        raw_path = default

    path = Path(str(raw_path)).expanduser()

    if not path.is_absolute():
        path = ROOT_DIR / path

    return path.resolve()


def get_groq_api_key() -> str | None:
    """Get Groq API key from environment variables."""
    api_key = get_env_value("GROQ_API_KEY")

    if api_key is None:
        return None

    normalized_api_key = str(api_key).strip()
    return normalized_api_key or None


def get_llm_config() -> dict[str, Any]:
    """Get LLM configuration from config.yaml and environment variables."""
    provider = str(
        get_env_first(("LLM_PROVIDER",), get_config_value("llm.provider", "groq")),
    )
    model = str(
        get_env_first(
            ("GROQ_MODEL", "LLM_MODEL"),
            get_config_value("llm.model", "llama-3.3-70b-versatile"),
        ),
    )

    temperature = safe_float(
        get_env_first(("LLM_TEMPERATURE",), get_config_value("llm.temperature", 0.2)),
        default=0.2,
        minimum=0.0,
    )
    temperature = min(2.0, temperature)

    max_tokens = safe_int(
        get_env_first(("LLM_MAX_TOKENS",), get_config_value("llm.max_tokens", 2_000)),
        default=2_000,
        minimum=1,
    )

    timeout = safe_int(
        get_env_first(("LLM_TIMEOUT",), get_config_value("llm.timeout", 120)),
        default=120,
        minimum=1,
    )

    max_retries = safe_int(
        get_env_first(("LLM_MAX_RETRIES",), get_config_value("llm.max_retries", 3)),
        default=3,
        minimum=0,
    )

    return {
        "provider": provider.strip().lower() or "groq",
        "model": model.strip() or "llama-3.3-70b-versatile",
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout,
        "api_key": get_groq_api_key(),
        "enabled": to_bool(
            get_env_first(("LLM_ENABLED",), get_config_value("llm.enabled", True)),
            default=True,
        ),
        "cache_enabled": to_bool(
            get_env_first(
                ("LLM_CACHE_ENABLED",),
                get_config_value("llm.cache_enabled", True),
            ),
            default=True,
        ),
        "max_retries": max_retries,
    }


def get_project_root() -> Path:
    """Return repository root path."""
    return ROOT_DIR
