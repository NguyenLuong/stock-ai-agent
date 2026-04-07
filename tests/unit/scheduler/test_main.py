"""Tests for scheduler main.py — serve setup with mocked Prefect."""

from __future__ import annotations

import signal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.scheduler.config_loader import ScheduleEntry
from services.scheduler.main import _build_deployments, _handle_signal, FLOW_REGISTRY, SCHEDULE_TO_FLOW


MOCK_SCHEDULES = [
    ScheduleEntry(
        name="news_crawl",
        cron="0 23,5,11 * * *",
        description="News crawl + embedding",
        enabled=True,
    ),
    ScheduleEntry(
        name="stock_pipeline",
        cron="0 10 * * 1-5",
        description="Stock crawl + technical indicators",
        enabled=True,
    ),
    ScheduleEntry(
        name="data_cleanup",
        cron="0 19 * * *",
        description="Data cleanup",
        enabled=True,
    ),
]


class TestBuildDeployments:
    """Tests for _build_deployments."""

    def test_each_schedule_creates_own_deployment(self):
        """Each schedule entry creates a separate deployment."""
        deployments = _build_deployments(MOCK_SCHEDULES)

        deployment_names = [d.name for d in deployments]
        assert "news-crawl" in deployment_names
        assert "stock-pipeline" in deployment_names
        assert "data-cleanup" in deployment_names

    def test_total_deployments(self):
        """One deployment per schedule entry."""
        deployments = _build_deployments(MOCK_SCHEDULES)
        assert len(deployments) == 3

    def test_skips_unknown_flows(self):
        """Schedules not in SCHEDULE_TO_FLOW are skipped."""
        unknown = [
            ScheduleEntry(
                name="unknown_future_flow",
                cron="0 7 * * 1-5",
                description="Future scope",
                enabled=True,
            ),
        ]

        deployments = _build_deployments(unknown)
        assert len(deployments) == 0

    def test_each_deployment_has_correct_cron(self):
        """Each deployment uses its own cron schedule."""
        deployments = _build_deployments(MOCK_SCHEDULES)

        by_name = {d.name: d for d in deployments}
        assert by_name["news-crawl"].schedules
        assert by_name["stock-pipeline"].schedules
        assert by_name["data-cleanup"].schedules

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
        """Registry has news_crawl, stock_pipeline, and data_cleanup."""
        assert "news_crawl" in FLOW_REGISTRY
        assert "stock_pipeline" in FLOW_REGISTRY
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
        # serve should be called with 3 deployments (one per schedule entry)
        assert len(mock_serve.call_args.args) == 3

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
