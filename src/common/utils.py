from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import yaml


def _load_env_file() -> None:
    """Load .env file if exists"""
    env_path = Path(".env")
    if not env_path.exists():
        # Try project root
        env_path = Path(__file__).parent.parent.parent / ".env"

    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    # Only set if not already in environment
                    if key not in os.environ:
                        os.environ[key] = value


def _substitute_env_vars(value: Any) -> Any:
    """Substitute environment variables in config values.

    Supports syntax: ${VAR_NAME} or ${VAR_NAME:default}
    """
    if isinstance(value, str):
        pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'

        def replace(match):
            var_name = match.group(1)
            default = match.group(2) if match.group(2) is not None else ""
            return os.environ.get(var_name, default)

        return re.sub(pattern, replace, value)
    elif isinstance(value, dict):
        return {k: _substitute_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_substitute_env_vars(item) for item in value]
    return value


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, data: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=True, default=str)


def read_yaml(path: str | Path) -> dict[str, Any]:
    # Load .env file first
    _load_env_file()

    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    # Substitute environment variables
    return _substitute_env_vars(config)


def write_yaml(path: str | Path, data: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target
