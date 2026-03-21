"""Tests for /internal/trigger/technical-indicators endpoint."""

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
class MockIndicatorResult:
    total_tickers: int = 10
    success_count: int = 8
    failed_count: int = 2
    rows_inserted: int = 150
    indicators_calculated: int = 100
    duration_seconds: float = 5.0
    skipped_reason: str | None = None
    errors: list[dict] = field(default_factory=lambda: [{"ticker": "BAD", "error": "DB error"}])


class TestTriggerTechnicalIndicators:
    """Tests for /internal/trigger/technical-indicators endpoint."""

    def test_missing_header_returns_403(self) -> None:
        """Missing trigger source header returns 403."""
        response = client.post("/internal/trigger/technical-indicators")
        assert response.status_code == 403

    def test_wrong_header_returns_403(self) -> None:
        """Wrong trigger source header returns 403."""
        response = client.post(
            "/internal/trigger/technical-indicators",
            headers={"X-Trigger-Source": "unknown"},
        )
        assert response.status_code == 403

    @patch(
        "market_data.indicator_manager.run_indicator_calculation",
        new_callable=AsyncMock,
    )
    def test_success(self, mock_calc) -> None:
        """Successful calculation returns status ok with result summary."""
        mock_calc.return_value = MockIndicatorResult()

        response = client.post(
            "/internal/trigger/technical-indicators",
            headers=TRIGGER_HEADER,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["result"]["total_tickers"] == 10
        assert data["result"]["success_count"] == 8
        assert data["result"]["rows_inserted"] == 150
        assert data["result"]["indicators_calculated"] == 100
        assert data["duration_seconds"] >= 0

    @patch(
        "market_data.indicator_manager.run_indicator_calculation",
        new_callable=AsyncMock,
    )
    def test_non_trading_day(self, mock_calc) -> None:
        """Non-trading day result includes skipped_reason."""
        mock_calc.return_value = MockIndicatorResult(
            success_count=0,
            failed_count=0,
            rows_inserted=0,
            indicators_calculated=0,
            skipped_reason="non_trading_day",
            errors=[],
        )

        response = client.post(
            "/internal/trigger/technical-indicators",
            headers=TRIGGER_HEADER,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["result"]["skipped_reason"] == "non_trading_day"

    @patch(
        "market_data.indicator_manager.run_indicator_calculation",
        new_callable=AsyncMock,
    )
    def test_failure(self, mock_calc) -> None:
        """Exception returns status failed."""
        mock_calc.side_effect = Exception("Config file missing")

        response = client.post(
            "/internal/trigger/technical-indicators",
            headers=TRIGGER_HEADER,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "failed"
        assert "Config file missing" in data["error"]
