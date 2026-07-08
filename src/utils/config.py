from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT_DIR / "config.yaml"
ENV_PATH = ROOT_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)


@lru_cache(maxsize=1)
def load_config(config_path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """
    Load config.yaml once and return it as a dictionary.
    """
    path = Path(config_path)

    if not path.is_absolute():
        path = ROOT_DIR / path

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        return {}

    if not isinstance(config, dict):
        raise ValueError("config.yaml must contain a valid YAML dictionary.")

    return config


def reload_config() -> None:
    """
    Clear cached config. Useful in tests.
    """
    load_config.cache_clear()


def get_config_value(key_path: str, default: Any = None) -> Any:
    """
    Get nested config value using dot notation.
    """
    if not key_path:
        return default

    value: Any = load_config()

    for key in key_path.split("."):
        if not isinstance(value, dict) or key not in value:
            return default

        value = value[key]

    return value


def get_env_value(key: str, default: Any = None) -> Any:
    """
    Get environment variable from .env or system environment.
    """
    if not key:
        return default

    return os.getenv(key, default)


def get_bool_config(key_path: str, default: bool = False) -> bool:
    value = get_config_value(key_path, default)

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.lower().strip() in {"true", "1", "yes", "y"}

    return bool(value)


def get_groq_api_key() -> str | None:
    """
    Get Groq API key from environment variables.
    """
    api_key = get_env_value("GROQ_API_KEY")

    if api_key is None or str(api_key).strip() == "":
        return None

    return str(api_key)


def get_llm_config() -> dict[str, Any]:
    """
    Get LLM configuration from config.yaml and environment variables.
    """
    return {
        "provider": get_config_value("llm.provider", "groq"),
        "model": get_config_value("llm.model", "llama-3.3-70b-versatile"),
        "temperature": float(get_config_value("llm.temperature", 0.2)),
        "max_tokens": int(get_config_value("llm.max_tokens", 2000)),
        "api_key": get_groq_api_key(),
    }