"""Tests for shared.logging.setup — structlog JSON config."""

import json

import structlog

from shared.logging.setup import configure_logging, get_logger


class TestConfigureLogging:
    def setup_method(self):
        """Reset structlog before each test."""
        structlog.reset_defaults()

    def test_json_output_has_required_fields(self, capsys):
        configure_logging(env="production")
        log = get_logger("test")
        log.info("test_event", key="value")

        output = capsys.readouterr().out.strip()
        data = json.loads(output)
        assert "timestamp" in data
        assert data["component"] == "test"
        assert data["level"] == "info"
        assert data["message"] == "test_event"
        assert "context" in data
        assert data["context"]["key"] == "value"

    def test_api_key_filtered_from_output(self, capsys):
        configure_logging(env="production")
        log = get_logger("test")
        log.info("sensitive", api_key="sk-test-secret-key", token="abc123")

        output = capsys.readouterr().out.strip()
        data = json.loads(output)
        assert "api_key" not in data
        assert "token" not in data
        assert "sk-test" not in output
        # context may not exist if all keys were filtered
        if "context" in data:
            assert "api_key" not in data["context"]
            assert "token" not in data["context"]

    def test_password_and_secret_filtered(self, capsys):
        configure_logging(env="production")
        log = get_logger("test")
        log.info("creds", password="hunter2", secret="top-secret")

        output = capsys.readouterr().out.strip()
        data = json.loads(output)
        assert "password" not in data
        assert "secret" not in data

    def test_nested_sensitive_key_filtered(self, capsys):
        configure_logging(env="production")
        log = get_logger("test")
        log.info("nested", data={"api_key": "sk-nested-secret", "safe": "ok"})

        output = capsys.readouterr().out.strip()
        data = json.loads(output)
        assert "sk-nested-secret" not in output
        nested = data.get("context", {}).get("data", {})
        assert "api_key" not in nested
        assert nested.get("safe") == "ok"

    def test_dev_mode_uses_console_renderer(self):
        configure_logging(env="development")
        log = get_logger("dev_test")
        # Should not raise — console renderer works
        log.info("dev_event")
