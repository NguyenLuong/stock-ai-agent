"""Tests for shared.llm.config_loader — YAML config loader."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from shared.llm.config_loader import ConfigKeyError, ConfigLoader, reset_config_loader


@pytest.fixture(autouse=True)
def _reset_loader():
    reset_config_loader()
    yield
    reset_config_loader()


def _write_models_yaml(config_dir: Path) -> None:
    agents_dir = config_dir / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "models": {
            "triage": "gpt-4o-mini",
            "morning_briefing": "gpt-4o-mini",
            "orchestrator_synthesis": "gpt-4o",
            "terminal_chat": "gpt-4o",
        },
        "temperature": {
            "default": 0.3,
            "creative": 0.7,
            "deterministic": 0.0,
        },
        "max_tokens": {
            "summary": 500,
            "analysis": 2000,
        },
    }
    (agents_dir / "models.yaml").write_text(yaml.dump(data))


def _write_thresholds_yaml(config_dir: Path) -> None:
    agents_dir = config_dir / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "confidence": {
            "min_recommendation": 0.7,
            "alert_trigger": 0.85,
        },
        "alerts": {
            "price_change_pct": 5.0,
        },
    }
    (agents_dir / "thresholds.yaml").write_text(yaml.dump(data))


class TestConfigLoaderGetModel:
    def test_get_model_triage(self, tmp_path: Path):
        _write_models_yaml(tmp_path)
        loader = ConfigLoader(tmp_path)
        assert loader.get_model("triage") == "gpt-4o-mini"

    def test_get_model_orchestrator_synthesis(self, tmp_path: Path):
        _write_models_yaml(tmp_path)
        loader = ConfigLoader(tmp_path)
        assert loader.get_model("orchestrator_synthesis") == "gpt-4o"

    def test_config_key_error_invalid_model(self, tmp_path: Path):
        _write_models_yaml(tmp_path)
        loader = ConfigLoader(tmp_path)
        with pytest.raises(ConfigKeyError, match="nonexistent"):
            loader.get_model("nonexistent")


class TestConfigLoaderGetThreshold:
    def test_get_threshold_dot_notation(self, tmp_path: Path):
        _write_thresholds_yaml(tmp_path)
        loader = ConfigLoader(tmp_path)
        assert loader.get_threshold("confidence.min_recommendation") == 0.7

    def test_get_threshold_key_error(self, tmp_path: Path):
        _write_thresholds_yaml(tmp_path)
        loader = ConfigLoader(tmp_path)
        with pytest.raises(ConfigKeyError, match="nonexistent"):
            loader.get_threshold("nonexistent.key")


class TestConfigLoaderCache:
    def test_cache_reads_file_once(self, tmp_path: Path):
        _write_models_yaml(tmp_path)
        loader = ConfigLoader(tmp_path)

        result1 = loader.get_model("triage")
        result2 = loader.get_model("orchestrator_synthesis")

        assert result1 == "gpt-4o-mini"
        assert result2 == "gpt-4o"
        # Only one file loaded (models.yaml cached)
        assert "agents/models.yaml" in loader._cache


class TestConfigLoaderTemperatureAndTokens:
    def test_get_temperature_default(self, tmp_path: Path):
        _write_models_yaml(tmp_path)
        loader = ConfigLoader(tmp_path)
        assert loader.get_temperature("default") == 0.3

    def test_get_max_tokens(self, tmp_path: Path):
        _write_models_yaml(tmp_path)
        loader = ConfigLoader(tmp_path)
        assert loader.get_max_tokens("analysis") == 2000


class TestConfigKeyErrorMessages:
    def test_model_key_error_message_format(self, tmp_path: Path):
        _write_models_yaml(tmp_path)
        loader = ConfigLoader(tmp_path)
        with pytest.raises(ConfigKeyError, match="Model key 'bad_key' not found in models.yaml"):
            loader.get_model("bad_key")

    def test_threshold_key_error_message_format(self, tmp_path: Path):
        _write_thresholds_yaml(tmp_path)
        loader = ConfigLoader(tmp_path)
        with pytest.raises(ConfigKeyError, match="Threshold key 'bad.key' not found in thresholds.yaml"):
            loader.get_threshold("bad.key")


class TestModuleLevelConvenienceFunctions:
    def test_get_model_module_level(self, tmp_path: Path, monkeypatch):
        _write_models_yaml(tmp_path)
        monkeypatch.setenv("CONFIG_DIR", str(tmp_path))

        from shared.llm.config_loader import get_model
        assert get_model("triage") == "gpt-4o-mini"
        assert get_model("orchestrator_synthesis") == "gpt-4o"

    def test_get_threshold_module_level(self, tmp_path: Path, monkeypatch):
        _write_thresholds_yaml(tmp_path)
        monkeypatch.setenv("CONFIG_DIR", str(tmp_path))

        from shared.llm.config_loader import get_threshold
        assert get_threshold("confidence.min_recommendation") == 0.7

    def test_get_sources_module_level(self, tmp_path: Path, monkeypatch):
        crawlers_dir = tmp_path / "crawlers"
        crawlers_dir.mkdir(parents=True)
        (crawlers_dir / "sources.yaml").write_text(
            yaml.dump({"sources": [{"name": "cafef", "url": "https://cafef.vn"}]})
        )
        monkeypatch.setenv("CONFIG_DIR", str(tmp_path))

        from shared.llm.config_loader import get_sources
        result = get_sources()
        assert "sources" in result

    def test_get_schedules_module_level(self, tmp_path: Path, monkeypatch):
        scheduler_dir = tmp_path / "scheduler"
        scheduler_dir.mkdir(parents=True)
        (scheduler_dir / "schedules.yaml").write_text(
            yaml.dump({"morning_briefing": "0 8 * * 1-5"})
        )
        monkeypatch.setenv("CONFIG_DIR", str(tmp_path))

        from shared.llm.config_loader import get_schedules
        result = get_schedules()
        assert result["morning_briefing"] == "0 8 * * 1-5"
