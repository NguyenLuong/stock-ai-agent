"""Tests for /internal/trigger/stock-crawl endpoint."""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.app.routers.internal import router

app = FastAPI()
app.include_router(router)
client = TestClient(app)

TRIGGER_HEADER = {"X-Trigger-Source": "prefect-scheduler"}


@dataclass
class MockStockCrawlResult:
    total_tickers: int = 10
    success_count: int = 8
    failed_count: int = 2
    initial_count: int = 3
    incremental_count: int = 5
    skipped_count: int = 0
    rows_inserted: int = 150
    duration_seconds: float = 12.5
    skipped_reason: str | None = None
    errors: list[dict] = field(default_factory=lambda: [{"ticker": "BAD", "error": "API error"}])


class TestTriggerStockCrawl:
    """Tests for /internal/trigger/stock-crawl endpoint."""

    def test_missing_header_returns_403(self) -> None:
        """Missing trigger source header returns 403."""
        response = client.post("/internal/trigger/stock-crawl")
        assert response.status_code == 403

    def test_wrong_header_returns_403(self) -> None:
        """Wrong trigger source header returns 403."""
        response = client.post(
            "/internal/trigger/stock-crawl",
            headers={"X-Trigger-Source": "unknown"},
        )
        assert response.status_code == 403

    @patch(
        "market_data.stock_crawl_manager.run_stock_crawl",
        new_callable=AsyncMock,
    )
    def test_stock_crawl_success(self, mock_crawl) -> None:
        """Successful stock crawl returns status ok with result summary."""
        mock_crawl.return_value = MockStockCrawlResult()

        response = client.post("/internal/trigger/stock-crawl", headers=TRIGGER_HEADER)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["result"]["total_tickers"] == 10
        assert data["result"]["success_count"] == 8
        assert data["result"]["initial_count"] == 3
        assert data["result"]["incremental_count"] == 5
        assert data["result"]["rows_inserted"] == 150
        assert data["duration_seconds"] >= 0

    @patch(
        "market_data.stock_crawl_manager.run_stock_crawl",
        new_callable=AsyncMock,
    )
    def test_stock_crawl_non_trading_day(self, mock_crawl) -> None:
        """Non-trading day result includes skipped_reason."""
        mock_crawl.return_value = MockStockCrawlResult(
            success_count=0,
            failed_count=0,
            initial_count=0,
            incremental_count=0,
            skipped_count=10,
            rows_inserted=0,
            skipped_reason="non_trading_day",
            errors=[],
        )

        response = client.post("/internal/trigger/stock-crawl", headers=TRIGGER_HEADER)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["result"]["skipped_reason"] == "non_trading_day"

    @patch(
        "market_data.stock_crawl_manager.run_stock_crawl",
        new_callable=AsyncMock,
    )
    def test_stock_crawl_failure(self, mock_crawl) -> None:
        """Exception in stock crawl returns status failed."""
        mock_crawl.side_effect = Exception("Config file missing")

        response = client.post("/internal/trigger/stock-crawl", headers=TRIGGER_HEADER)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "failed"
        assert "Config file missing" in data["error"]
