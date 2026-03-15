"""Tests for schedule config loader — YAML parsing, cron validation, enabled/disabled."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from services.scheduler.config_loader import (
    ScheduleEntry,
    _validate_cron,
    load_schedules,
)


class TestValidateCron:
    """Tests for cron expression validation."""

    def test_valid_5_field_cron(self):
        """Standard 5-field cron expression passes validation."""
        _validate_cron("0 6,12,18 * * *", "test_flow")

    def test_invalid_too_few_fields(self):
        """Cron with fewer than 5 fields raises ValueError."""
        with pytest.raises(ValueError, match="expected 5 fields"):
            _validate_cron("0 6 * *", "bad_flow")

    def test_invalid_too_many_fields(self):
        """Cron with more than 5 fields raises ValueError."""
        with pytest.raises(ValueError, match="expected 5 fields"):
            _validate_cron("0 6 * * * *", "bad_flow")

    def test_empty_cron_raises(self):
        """Empty cron string raises ValueError."""
        with pytest.raises(ValueError):
            _validate_cron("", "empty_flow")

    def test_weekday_cron(self):
        """Weekday-specific cron passes validation."""
        _validate_cron("0 7 * * 1-5", "weekday_flow")

    def test_out_of_range_hour_raises(self):
        """Hour value > 23 raises ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            _validate_cron("0 25 * * *", "bad_hour")

    def test_out_of_range_minute_raises(self):
        """Minute value > 59 raises ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            _validate_cron("60 6 * * *", "bad_minute")

    def test_non_numeric_value_raises(self):
        """Non-numeric cron value raises ValueError."""
        with pytest.raises(ValueError, match="not a valid value"):
            _validate_cron("abc 6 * * *", "bad_value")

    def test_valid_step_cron(self):
        """Cron with step (*/5) passes validation."""
        _validate_cron("*/5 * * * *", "step_flow")

    def test_valid_comma_separated(self):
        """Cron with comma-separated values passes validation."""
        _validate_cron("0 6,12,18 * * *", "comma_flow")

    def test_out_of_range_day_of_month_raises(self):
        """Day of month = 0 raises ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            _validate_cron("0 6 0 * *", "bad_dom")


class TestLoadSchedules:
    """Tests for loading schedules from YAML."""

    def test_loads_enabled_flows(self, tmp_path: Path):
        """Only enabled flows are returned."""
        config = tmp_path / "schedules.yaml"
        config.write_text(dedent("""\
            flows:
              active_flow:
                cron: "0 6 * * *"
                description: "Active"
                enabled: true
              disabled_flow:
                cron: "0 2 * * *"
                description: "Disabled"
                enabled: false
        """))

        entries = load_schedules(config)

        assert len(entries) == 1
        assert entries[0].name == "active_flow"
        assert entries[0].cron == "0 6 * * *"
        assert entries[0].description == "Active"
        assert entries[0].enabled is True

    def test_file_not_found_raises(self, tmp_path: Path):
        """Missing config file raises FileNotFoundError."""
        missing = tmp_path / "missing.yaml"
        with pytest.raises(FileNotFoundError):
            load_schedules(missing)

    def test_invalid_cron_raises(self, tmp_path: Path):
        """Invalid cron in config raises ValueError at load time."""
        config = tmp_path / "schedules.yaml"
        config.write_text(dedent("""\
            flows:
              bad_flow:
                cron: "invalid cron"
                description: "Bad"
                enabled: true
        """))

        with pytest.raises(ValueError, match="expected 5 fields"):
            load_schedules(config)

    def test_empty_flows_returns_empty_list(self, tmp_path: Path):
        """Config with no flows section returns empty list."""
        config = tmp_path / "schedules.yaml"
        config.write_text("flows:\n")

        entries = load_schedules(config)
        assert entries == []

    def test_multiple_enabled_flows(self, tmp_path: Path):
        """Multiple enabled flows are all returned."""
        config = tmp_path / "schedules.yaml"
        config.write_text(dedent("""\
            flows:
              flow_a:
                cron: "0 6,12,18 * * *"
                description: "Flow A"
                enabled: true
              flow_b:
                cron: "0 2 * * *"
                description: "Flow B"
                enabled: true
              flow_c:
                cron: "0 7 * * 1-5"
                description: "Flow C"
                enabled: true
        """))

        entries = load_schedules(config)
        assert len(entries) == 3
        names = [e.name for e in entries]
        assert "flow_a" in names
        assert "flow_b" in names
        assert "flow_c" in names

    def test_schedule_entry_is_frozen(self, tmp_path: Path):
        """ScheduleEntry is immutable."""
        config = tmp_path / "schedules.yaml"
        config.write_text(dedent("""\
            flows:
              test_flow:
                cron: "0 6 * * *"
                description: "Test"
                enabled: true
        """))

        entries = load_schedules(config)
        with pytest.raises(AttributeError):
            entries[0].name = "changed"  # type: ignore[misc]

    def test_defaults_enabled_true(self, tmp_path: Path):
        """Flow without explicit enabled defaults to enabled=True."""
        config = tmp_path / "schedules.yaml"
        config.write_text(dedent("""\
            flows:
              default_flow:
                cron: "0 6 * * *"
                description: "Default enabled"
        """))

        entries = load_schedules(config)
        assert len(entries) == 1
        assert entries[0].enabled is True
