from functools import lru_cache
from pathlib import Path
from typing import Any
import os

import yaml
from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT_DIR / "config.yaml"
ENV_PATH = ROOT_DIR / ".env"


load_dotenv(dotenv_path=ENV_PATH)


@lru_cache(maxsize=1)
def load_config(config_path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """
    Load config.yaml once and return it as a Python dictionary.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config or {}


def get_config_value(key_path: str, default: Any = None) -> Any:
    """
    Get a nested config value using dot notation.

    Example:
        get_config_value("modeling.random_state")
    """
    config = load_config()
    value: Any = config

    for key in key_path.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]

    return value


def get_env_value(key: str, default: Any = None) -> Any:
    """
    Get an environment variable from .env or system environment.
    """
    return os.getenv(key, default)


def get_groq_api_key() -> str | None:
    """
    Get Groq API key from environment variables.
    """
    return get_env_value("GROQ_API_KEY")


def get_llm_config() -> dict[str, Any]:
    """
    Get LLM configuration from config.yaml and environment variables.
    """
    return {
        "provider": get_config_value("llm.provider", "groq"),
        "model": get_config_value("llm.model", "llama-3.3-70b-versatile"),
        "temperature": get_config_value("llm.temperature", 0.2),
        "max_tokens": get_config_value("llm.max_tokens", 2000),
        "api_key": get_groq_api_key(),
    }