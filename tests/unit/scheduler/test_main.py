"""Tests for scheduler main.py — serve setup with mocked Prefect."""

from __future__ import annotations

import signal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.scheduler.config_loader import ScheduleEntry
from services.scheduler.main import _build_deployments, _handle_signal, FLOW_REGISTRY, SCHEDULE_TO_FLOW


MOCK_SCHEDULES = [
    ScheduleEntry(
        name="crawler_vietstock",
        cron="0 6,12,18 * * *",
        description="Vietstock news crawler",
        enabled=True,
    ),
    ScheduleEntry(
        name="crawler_cafef",
        cron="0 6,12,18 * * *",
        description="CafeF news crawler",
        enabled=True,
    ),
    ScheduleEntry(
        name="data_cleanup",
        cron="0 2 * * *",
        description="Data cleanup",
        enabled=True,
    ),
]


class TestBuildDeployments:
    """Tests for _build_deployments."""

    def test_deduplicates_data_pipeline(self):
        """Multiple crawler schedules produce only one data-pipeline deployment."""
        deployments = _build_deployments(MOCK_SCHEDULES)

        deployment_names = [d.name for d in deployments]
        assert deployment_names.count("data-pipeline") == 1

    def test_data_cleanup_maps_to_own_flow(self):
        """data_cleanup schedule produces a data-cleanup deployment."""
        deployments = _build_deployments(MOCK_SCHEDULES)

        deployment_names = [d.name for d in deployments]
        assert "data-cleanup" in deployment_names

    def test_total_deployments(self):
        """Two deployments total: data-pipeline + data-cleanup."""
        deployments = _build_deployments(MOCK_SCHEDULES)
        assert len(deployments) == 2

    def test_skips_unknown_flows(self):
        """Schedules not in SCHEDULE_TO_FLOW are skipped."""
        unknown = [
            ScheduleEntry(
                name="morning_briefing",
                cron="0 7 * * 1-5",
                description="Epic 4 scope",
                enabled=True,
            ),
        ]

        deployments = _build_deployments(unknown)
        assert len(deployments) == 0

    def test_uses_first_cron_for_grouped_flows(self):
        """data_pipeline uses cron from the first matching schedule entry."""
        deployments = _build_deployments(MOCK_SCHEDULES)

        dp_deployment = [d for d in deployments if d.name == "data-pipeline"][0]
        assert dp_deployment.schedules  # has schedules

    def test_empty_schedules(self):
        """No schedules returns empty list."""
        deployments = _build_deployments([])
        assert deployments == []


class TestFlowRegistry:
    """Tests for flow registry and schedule mapping."""

    def test_all_mapped_flows_exist_in_registry(self):
        """Every flow key in SCHEDULE_TO_FLOW has a corresponding FLOW_REGISTRY entry."""
        for flow_key in set(SCHEDULE_TO_FLOW.values()):
            assert flow_key in FLOW_REGISTRY, f"Missing registry entry for {flow_key}"

    def test_registry_contains_expected_flows(self):
        """Registry has data_pipeline and data_cleanup."""
        assert "data_pipeline" in FLOW_REGISTRY
        assert "data_cleanup" in FLOW_REGISTRY


class TestMain:
    """Tests for the main() entry point."""

    @patch("services.scheduler.main.serve", new_callable=AsyncMock)
    @patch("services.scheduler.main.load_schedules")
    @patch("services.scheduler.main.configure_logging")
    def test_main_calls_serve(self, mock_logging, mock_load, mock_serve):
        """main() loads schedules and calls serve with deployments."""
        mock_load.return_value = MOCK_SCHEDULES

        from services.scheduler.main import main

        main()

        mock_logging.assert_called_once()
        mock_load.assert_called_once()
        mock_serve.assert_called_once()
        # serve should be called with 2 deployments
        assert len(mock_serve.call_args.args) == 2

    @patch("services.scheduler.main.serve", new_callable=AsyncMock)
    @patch("services.scheduler.main.load_schedules")
    @patch("services.scheduler.main.configure_logging")
    def test_main_no_schedules(self, mock_logging, mock_load, mock_serve):
        """main() returns early when no schedules found."""
        mock_load.return_value = []

        from services.scheduler.main import main

        main()

        mock_serve.assert_not_called()


class TestHandleSignal:
    """Tests for _handle_signal graceful shutdown."""

    def test_sigterm_raises_keyboard_interrupt(self):
        """SIGTERM handler raises KeyboardInterrupt for clean asyncio shutdown."""
        with pytest.raises(KeyboardInterrupt):
            _handle_signal(signal.SIGTERM, None)

    def test_sigint_raises_keyboard_interrupt(self):
        """SIGINT handler raises KeyboardInterrupt."""
        with pytest.raises(KeyboardInterrupt):
            _handle_signal(signal.SIGINT, None)
