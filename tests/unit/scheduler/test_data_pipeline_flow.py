"""Tests for data pipeline flows — orchestration, error handling per step."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from services.scheduler.flows.data_pipeline import (
    data_cleanup_flow,
    news_crawl_flow,
    stock_pipeline_flow,
    NEWS_CRAWL_STEPS,
    STOCK_STEPS,
)


class TestNewsCrawlFlow:
    """Tests for the news_crawl_flow (crawl → embedding)."""

    @patch("services.scheduler.flows.data_pipeline.trigger_pipeline", new_callable=AsyncMock)
    async def test_all_steps_succeed(self, mock_trigger):
        """All steps succeed — summary shows all successful, 0 failed."""
        mock_trigger.return_value = {"status": "ok"}

        result = await news_crawl_flow.fn()

        total = len(NEWS_CRAWL_STEPS)
        assert result["successful_steps"] == total
        assert result["failed_steps"] == 0
        assert result["total_steps"] == total
        assert len(result["errors"]) == 0
        assert mock_trigger.call_count == total

    @patch("services.scheduler.flows.data_pipeline.trigger_pipeline", new_callable=AsyncMock)
    async def test_crawl_fails_embedding_still_runs(self, mock_trigger):
        """If crawl fails, embedding still runs."""
        mock_trigger.side_effect = [
            Exception("network timeout"),  # crawl fails
            {"status": "ok"},              # embedding succeeds
        ]

        result = await news_crawl_flow.fn()

        assert result["successful_steps"] == 1
        assert result["failed_steps"] == 1
        assert "crawl: network timeout" in result["errors"][0]
        assert result["steps"]["crawl"]["status"] == "failed"
        assert result["steps"]["embedding"]["status"] == "success"

    @patch("services.scheduler.flows.data_pipeline.trigger_pipeline", new_callable=AsyncMock)
    async def test_all_steps_fail(self, mock_trigger):
        """All steps fail — summary reflects total failure."""
        mock_trigger.side_effect = Exception("service down")

        result = await news_crawl_flow.fn()

        total = len(NEWS_CRAWL_STEPS)
        assert result["successful_steps"] == 0
        assert result["failed_steps"] == total
        assert len(result["errors"]) == total

    @patch("services.scheduler.flows.data_pipeline.trigger_pipeline", new_callable=AsyncMock)
    async def test_duration_tracked(self, mock_trigger):
        """Total duration is tracked in the summary."""
        mock_trigger.return_value = {"status": "ok"}

        result = await news_crawl_flow.fn()

        assert "total_duration_seconds" in result
        assert isinstance(result["total_duration_seconds"], float)
        assert result["total_duration_seconds"] >= 0

    @patch("services.scheduler.flows.data_pipeline.trigger_pipeline", new_callable=AsyncMock)
    async def test_only_calls_crawl_and_embedding(self, mock_trigger):
        """news_crawl_flow only triggers crawl and embedding endpoints."""
        mock_trigger.return_value = {"status": "ok"}

        await news_crawl_flow.fn()

        called_endpoints = [call.kwargs["endpoint"] for call in mock_trigger.call_args_list]
        assert called_endpoints == ["crawl", "embedding"]


class TestStockPipelineFlow:
    """Tests for the stock_pipeline_flow (stock-crawl → technical-indicators)."""

    @patch("services.scheduler.flows.data_pipeline.trigger_pipeline", new_callable=AsyncMock)
    async def test_all_steps_succeed(self, mock_trigger):
        """All steps succeed."""
        mock_trigger.return_value = {"status": "ok"}

        result = await stock_pipeline_flow.fn()

        total = len(STOCK_STEPS)
        assert result["successful_steps"] == total
        assert result["failed_steps"] == 0
        assert mock_trigger.call_count == total

    @patch("services.scheduler.flows.data_pipeline.trigger_pipeline", new_callable=AsyncMock)
    async def test_stock_crawl_fails_indicators_still_run(self, mock_trigger):
        """If stock-crawl fails, technical-indicators still runs."""
        mock_trigger.side_effect = [
            Exception("timeout"),   # stock-crawl fails
            {"status": "ok"},       # technical-indicators succeeds
        ]

        result = await stock_pipeline_flow.fn()

        assert result["successful_steps"] == 1
        assert result["failed_steps"] == 1
        assert result["steps"]["stock-crawl"]["status"] == "failed"
        assert result["steps"]["technical-indicators"]["status"] == "success"

    @patch("services.scheduler.flows.data_pipeline.trigger_pipeline", new_callable=AsyncMock)
    async def test_only_calls_stock_and_indicators(self, mock_trigger):
        """stock_pipeline_flow only triggers stock-crawl and technical-indicators."""
        mock_trigger.return_value = {"status": "ok"}

        await stock_pipeline_flow.fn()

        called_endpoints = [call.kwargs["endpoint"] for call in mock_trigger.call_args_list]
        assert called_endpoints == ["stock-crawl", "technical-indicators"]


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
