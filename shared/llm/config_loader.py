"""YAML configuration loader for agents, crawlers, and scheduler settings."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from shared.logging import get_logger

logger = get_logger("config_loader")


class ConfigKeyError(KeyError):
    """Raised when a requested key is not found in a config YAML file."""


def _resolve_config_dir() -> Path:
    """Resolve config directory from env var or by walking up from this file."""
    env_path = os.environ.get("CONFIG_DIR")
    if env_path:
        return Path(env_path)
    current = Path(__file__).parent
    for _ in range(6):
        candidate = current / "config"
        if candidate.is_dir():
            return candidate
        current = current.parent
    raise RuntimeError("Cannot locate config/ directory. Set CONFIG_DIR env var.")


class ConfigLoader:
    """Loads and caches YAML configuration files."""

    def __init__(self, config_dir: Path) -> None:
        self._config_dir = config_dir
        self._cache: dict[str, dict] = {}

    def _load_yaml(self, relative_path: str) -> dict:
        """Load a YAML file relative to config_dir, with caching."""
        if relative_path in self._cache:
            return self._cache[relative_path]
        full_path = self._config_dir / relative_path
        if not full_path.is_file():
            raise FileNotFoundError(f"Config file not found: {full_path}")
        with open(full_path) as f:
            data = yaml.safe_load(f)
        self._cache[relative_path] = data
        return data

    def get_model(self, task_key: str) -> str:
        """Return model name for a task key from agents/models.yaml."""
        config = self._load_yaml("agents/models.yaml")
        try:
            return config["models"][task_key]
        except KeyError:
            raise ConfigKeyError(f"Model key '{task_key}' not found in models.yaml")

    def get_temperature(self, key: str = "default") -> float:
        """Return temperature value from agents/models.yaml."""
        config = self._load_yaml("agents/models.yaml")
        try:
            return float(config["temperature"][key])
        except KeyError:
            raise ConfigKeyError(f"Temperature key '{key}' not found in models.yaml")

    def get_max_tokens(self, task_key: str) -> int:
        """Return max_tokens for a task key from agents/models.yaml."""
        config = self._load_yaml("agents/models.yaml")
        try:
            return int(config["max_tokens"][task_key])
        except KeyError:
            raise ConfigKeyError(f"Max tokens key '{task_key}' not found in models.yaml")

    def get_threshold(self, key: str) -> float:
        """Return threshold value from agents/thresholds.yaml.

        Supports dot-notation: get_threshold("confidence.min_recommendation")
        """
        config = self._load_yaml("agents/thresholds.yaml")
        keys = key.split(".")
        current = config
        for k in keys:
            try:
                current = current[k]
            except (KeyError, TypeError):
                raise ConfigKeyError(f"Threshold key '{key}' not found in thresholds.yaml")
        return float(current)

    def get_sources(self) -> dict:
        """Return crawler sources from crawlers/sources.yaml."""
        return self._load_yaml("crawlers/sources.yaml")

    def get_schedules(self) -> dict:
        """Return schedules from scheduler/schedules.yaml."""
        return self._load_yaml("scheduler/schedules.yaml")


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_loader: ConfigLoader | None = None


def get_config_loader() -> ConfigLoader:
    """Return the module-level singleton ConfigLoader."""
    global _default_loader
    if _default_loader is None:
        _default_loader = ConfigLoader(_resolve_config_dir())
    return _default_loader


def reset_config_loader() -> None:
    """Reset the singleton — useful for test isolation."""
    global _default_loader
    _default_loader = None


def get_model(task_key: str) -> str:
    """Module-level convenience: get model name for task_key."""
    return get_config_loader().get_model(task_key)


def get_threshold(key: str) -> float:
    """Module-level convenience: get threshold value."""
    return get_config_loader().get_threshold(key)


def get_sources() -> dict:
    """Module-level convenience: get crawler sources."""
    return get_config_loader().get_sources()


def get_schedules() -> dict:
    """Module-level convenience: get scheduler schedules."""
    return get_config_loader().get_schedules()
