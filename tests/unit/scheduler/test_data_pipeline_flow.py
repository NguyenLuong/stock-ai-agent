"""Tests for data pipeline flow — orchestration, error handling per step."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from services.scheduler.flows.data_pipeline import (
    data_cleanup_flow,
    data_pipeline_flow,
    PIPELINE_STEPS,
)


class TestDataPipelineFlow:
    """Tests for the data_pipeline_flow."""

    @patch("services.scheduler.flows.data_pipeline.trigger_pipeline", new_callable=AsyncMock)
    async def test_all_steps_succeed(self, mock_trigger):
        """All steps succeed — summary shows all successful, 0 failed."""
        mock_trigger.return_value = {"status": "ok"}

        result = await data_pipeline_flow.fn()

        total = len(PIPELINE_STEPS)
        assert result["successful_steps"] == total
        assert result["failed_steps"] == 0
        assert result["total_steps"] == total
        assert len(result["errors"]) == 0
        assert mock_trigger.call_count == total

    @patch("services.scheduler.flows.data_pipeline.trigger_pipeline", new_callable=AsyncMock)
    async def test_one_step_fails_others_continue(self, mock_trigger):
        """If crawl fails, remaining steps still run."""
        mock_trigger.side_effect = [
            Exception("network timeout"),  # crawl fails
            {"status": "ok"},              # stock-crawl succeeds
            {"status": "ok"},              # technical-indicators succeeds
            {"status": "ok"},              # embedding succeeds
            {"status": "ok"},              # lifecycle succeeds
        ]

        result = await data_pipeline_flow.fn()

        assert result["successful_steps"] == len(PIPELINE_STEPS) - 1
        assert result["failed_steps"] == 1
        assert "crawl: network timeout" in result["errors"][0]
        assert result["steps"]["crawl"]["status"] == "failed"
        assert result["steps"]["stock-crawl"]["status"] == "success"
        assert result["steps"]["technical-indicators"]["status"] == "success"
        assert result["steps"]["embedding"]["status"] == "success"
        assert result["steps"]["lifecycle"]["status"] == "success"

    @patch("services.scheduler.flows.data_pipeline.trigger_pipeline", new_callable=AsyncMock)
    async def test_all_steps_fail(self, mock_trigger):
        """All steps fail — summary reflects total failure."""
        mock_trigger.side_effect = Exception("service down")

        result = await data_pipeline_flow.fn()

        total = len(PIPELINE_STEPS)
        assert result["successful_steps"] == 0
        assert result["failed_steps"] == total
        assert len(result["errors"]) == total

    @patch("services.scheduler.flows.data_pipeline.trigger_pipeline", new_callable=AsyncMock)
    async def test_duration_tracked(self, mock_trigger):
        """Total duration is tracked in the summary."""
        mock_trigger.return_value = {"status": "ok"}

        result = await data_pipeline_flow.fn()

        assert "total_duration_seconds" in result
        assert isinstance(result["total_duration_seconds"], float)
        assert result["total_duration_seconds"] >= 0

    @patch("services.scheduler.flows.data_pipeline.trigger_pipeline", new_callable=AsyncMock)
    async def test_step_results_include_duration(self, mock_trigger):
        """Each step result includes its own duration."""
        mock_trigger.return_value = {"status": "ok"}

        result = await data_pipeline_flow.fn()

        for step_name, _desc in PIPELINE_STEPS:
            assert "duration_seconds" in result["steps"][step_name]

    @patch("services.scheduler.flows.data_pipeline.trigger_pipeline", new_callable=AsyncMock)
    async def test_uses_app_url_env(self, mock_trigger):
        """Trigger pipeline is called with the correct app_url."""
        mock_trigger.return_value = {"status": "ok"}

        await data_pipeline_flow.fn()

        for call in mock_trigger.call_args_list:
            assert "app_url" in call.kwargs or len(call.args) >= 1

    @patch("services.scheduler.flows.data_pipeline.trigger_pipeline", new_callable=AsyncMock)
    async def test_started_at_is_iso_string(self, mock_trigger):
        """Summary includes an ISO formatted started_at timestamp."""
        mock_trigger.return_value = {"status": "ok"}

        result = await data_pipeline_flow.fn()

        assert "started_at" in result
        assert "T" in result["started_at"]  # ISO format


class TestDataCleanupFlow:
    """Tests for the data_cleanup_flow."""

    @patch("services.scheduler.flows.data_pipeline.trigger_pipeline", new_callable=AsyncMock)
    async def test_cleanup_success(self, mock_trigger):
        """Successful lifecycle trigger returns status ok."""
        mock_trigger.return_value = {"status": "ok", "summarized_count": 10}

        result = await data_cleanup_flow.fn()

        assert result["status"] == "success"
        assert "duration_seconds" in result
        assert result["result"]["status"] == "ok"
        mock_trigger.assert_called_once()

    @patch("services.scheduler.flows.data_pipeline.trigger_pipeline", new_callable=AsyncMock)
    async def test_cleanup_failure(self, mock_trigger):
        """Failed lifecycle trigger returns status failed with error."""
        mock_trigger.side_effect = Exception("connection refused")

        result = await data_cleanup_flow.fn()

        assert result["status"] == "failed"
        assert "connection refused" in result["error"]
