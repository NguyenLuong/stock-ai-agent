"""Tests for POST /internal/trigger/morning-briefing endpoint."""

from __future__ import annotations

import sys
from datetime import date, datetime, timezone
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.app.routers.internal import router

app = FastAPI()
app.include_router(router)

# Attach a mock telegram_bot to app.state so tests can control it
app.state.telegram_bot = None

client = TestClient(app)

TRIGGER_HEADER = {"X-Trigger-Source": "prefect-scheduler"}

# Module path prefix for patching top-level imports in internal.py
_P = "services.app.routers.internal"


def _mock_now_utc(year, month, day, hour=12, minute=0):
    """Return a timezone-aware UTC datetime."""
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


SAMPLE_MARKET_RESULT = {
    "market_sentiment": "bullish",
    "affected_sectors": ["banking", "steel"],
    "key_events": ["FED giữ lãi suất"],
    "top_picks": [
        {"ticker": "HPG", "signal": "uptrend", "confidence": 0.8, "summary": "Tích cực"},
    ],
    "market_summary": "Thị trường tăng điểm nhờ banking dẫn dắt.",
    "stale_warnings": [],
    "unavailable_warnings": [],
    "disclaimer": "Đây là phân tích tham khảo.",
    "generated_at": "2026-04-07T06:00:00Z",
}

SAMPLE_GRAPH_STATE = {
    "market_result": SAMPLE_MARKET_RESULT,
    "failed_steps": [],
}


# ---------------------------------------------------------------------------
# Fixtures for mocking lazy imports used inside trigger_morning_briefing()
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_graph():
    """Mock the morning_briefing_graph module in sys.modules."""
    mock = MagicMock()
    mock.morning_briefing_graph = MagicMock()
    mod_key = "services.app.agents.morning_briefing_graph"
    original = sys.modules.get(mod_key)
    sys.modules[mod_key] = mock
    yield mock.morning_briefing_graph
    if original is None:
        sys.modules.pop(mod_key, None)
    else:
        sys.modules[mod_key] = original


@pytest.fixture()
def mock_format():
    """Mock the briefing_formatter module in sys.modules."""
    mock = MagicMock()
    mock.format_morning_briefing = MagicMock(return_value="📊 Morning Briefing")
    mod_key = "telegram.formatters.briefing_formatter"
    original = sys.modules.get(mod_key)
    sys.modules[mod_key] = mock
    yield mock.format_morning_briefing
    if original is None:
        sys.modules.pop(mod_key, None)
    else:
        sys.modules[mod_key] = original


@pytest.fixture()
def mock_save():
    """Mock the recommendation_repo module in sys.modules."""
    mock = MagicMock()
    mock.save_recommendation = AsyncMock(return_value=MagicMock())
    mod_key = "shared.db.repositories.recommendation_repo"
    original = sys.modules.get(mod_key)
    sys.modules[mod_key] = mock
    yield mock.save_recommendation
    if original is None:
        sys.modules.pop(mod_key, None)
    else:
        sys.modules[mod_key] = original


@pytest.fixture()
def mock_rec_model():
    """Mock the RecommendationCreate model module."""
    mod_key = "shared.models.recommendation"
    original = sys.modules.get(mod_key)
    # Try to use the real module since it's a simple Pydantic model
    try:
        from shared.models.recommendation import RecommendationCreate  # noqa: F401
    except ImportError:
        mock = MagicMock()
        sys.modules[mod_key] = mock
    yield
    if original is not None:
        sys.modules[mod_key] = original


class TestMorningBriefingTriggerSource:
    """X-Trigger-Source validation for morning-briefing."""

    def test_missing_header_returns_403(self):
        response = client.post("/internal/trigger/morning-briefing")
        assert response.status_code == 403

    def test_wrong_header_returns_403(self):
        response = client.post(
            "/internal/trigger/morning-briefing",
            headers={"X-Trigger-Source": "unknown"},
        )
        assert response.status_code == 403


class TestMorningBriefingNonTradingDay:
    """AC #4: Non-trading day → skip."""

    @patch(f"{_P}.now_utc")
    @patch(f"{_P}._load_variable_holidays", return_value=[])
    def test_saturday_skipped(self, _mock_holidays, mock_now):
        """Saturday (weekday=5) → skipped."""
        mock_now.return_value = _mock_now_utc(2026, 4, 4, 23, 0)  # Saturday
        response = client.post("/internal/trigger/morning-briefing", headers=TRIGGER_HEADER)
        data = response.json()
        assert data["status"] == "skipped"
        assert data["reason"] == "non_trading_day"

    @patch(f"{_P}.now_utc")
    @patch(f"{_P}._load_variable_holidays", return_value=[])
    def test_sunday_skipped(self, _mock_holidays, mock_now):
        """Sunday (weekday=6) → skipped."""
        mock_now.return_value = _mock_now_utc(2026, 4, 5, 23, 0)  # Sunday
        response = client.post("/internal/trigger/morning-briefing", headers=TRIGGER_HEADER)
        assert response.json()["status"] == "skipped"

    @patch(f"{_P}.now_utc")
    @patch(f"{_P}._load_variable_holidays", return_value=[])
    def test_vn_holiday_new_year_skipped(self, _mock_holidays, mock_now):
        """January 1 (New Year) → skipped."""
        mock_now.return_value = _mock_now_utc(2026, 1, 1, 23, 0)  # Thursday
        response = client.post("/internal/trigger/morning-briefing", headers=TRIGGER_HEADER)
        assert response.json()["status"] == "skipped"

    @patch(f"{_P}.now_utc")
    @patch(f"{_P}._load_variable_holidays", return_value=[])
    def test_vn_holiday_apr30_skipped(self, _mock_holidays, mock_now):
        """April 30 (VN Liberation Day) → skipped."""
        mock_now.return_value = _mock_now_utc(2026, 4, 30, 23, 0)  # Thursday
        response = client.post("/internal/trigger/morning-briefing", headers=TRIGGER_HEADER)
        assert response.json()["status"] == "skipped"

    @patch(f"{_P}.now_utc")
    @patch(f"{_P}._load_variable_holidays", return_value=[])
    def test_vn_holiday_may1_skipped(self, _mock_holidays, mock_now):
        """May 1 (International Labour Day) → skipped."""
        mock_now.return_value = _mock_now_utc(2026, 5, 1, 23, 0)  # Friday
        response = client.post("/internal/trigger/morning-briefing", headers=TRIGGER_HEADER)
        assert response.json()["status"] == "skipped"

    @patch(f"{_P}.now_utc")
    @patch(f"{_P}._load_variable_holidays", return_value=[])
    def test_vn_holiday_sep2_skipped(self, _mock_holidays, mock_now):
        """September 2 (National Day) → skipped."""
        mock_now.return_value = _mock_now_utc(2026, 9, 2, 23, 0)  # Wednesday
        response = client.post("/internal/trigger/morning-briefing", headers=TRIGGER_HEADER)
        assert response.json()["status"] == "skipped"

    @patch(f"{_P}.now_utc")
    @patch(f"{_P}._load_variable_holidays")
    def test_variable_holiday_skipped(self, mock_holidays, mock_now):
        """Variable holiday (Tet) → skipped."""
        mock_now.return_value = _mock_now_utc(2026, 2, 17, 23, 0)  # Tuesday
        mock_holidays.return_value = [date(2026, 2, 17)]
        response = client.post("/internal/trigger/morning-briefing", headers=TRIGGER_HEADER)
        assert response.json()["status"] == "skipped"


class TestMorningBriefingTradingDay:
    """AC #2, #3: Full pipeline on trading day."""

    @patch(f"{_P}._load_watchlist", return_value=["HPG", "VNM", "FPT"])
    @patch(f"{_P}._load_variable_holidays", return_value=[])
    @patch(f"{_P}.now_utc")
    def test_successful_pipeline(
        self, mock_now, _mock_holidays, _mock_watchlist,
        mock_graph, mock_format, mock_save, mock_rec_model,
    ):
        """Trading day → graph invoked once → response ok."""
        mock_now.return_value = _mock_now_utc(2026, 4, 7, 23, 0)  # Tuesday
        mock_graph.ainvoke = AsyncMock(return_value=SAMPLE_GRAPH_STATE)

        response = client.post("/internal/trigger/morning-briefing", headers=TRIGGER_HEADER)
        data = response.json()

        assert data["status"] == "ok"
        assert data["affected_sectors"] == ["banking", "steel"]
        assert data["top_picks_count"] == 1
        assert data["telegram"] == "skipped"  # telegram_bot is None

        # Graph called exactly once with correct params
        mock_graph.ainvoke.assert_called_once()
        call_arg = mock_graph.ainvoke.call_args[0][0]
        assert call_arg["analysis_date"] == "2026-04-07"
        assert call_arg["watchlist"] == ["HPG", "VNM", "FPT"]

    @patch(f"{_P}._load_watchlist", return_value=["HPG"])
    @patch(f"{_P}._load_variable_holidays", return_value=[])
    @patch(f"{_P}.now_utc")
    def test_market_result_none_returns_failed(
        self, mock_now, _mock_holidays, _mock_watchlist,
        mock_graph, mock_format, mock_save, mock_rec_model,
    ):
        """market_result is None → returns failed."""
        mock_now.return_value = _mock_now_utc(2026, 4, 7, 23, 0)
        mock_graph.ainvoke = AsyncMock(return_value={"market_result": None, "failed_steps": []})

        response = client.post("/internal/trigger/morning-briefing", headers=TRIGGER_HEADER)
        data = response.json()

        assert data["status"] == "failed"
        assert "no result" in data["error"]

    @patch(f"{_P}._load_watchlist", return_value=["HPG"])
    @patch(f"{_P}._load_variable_holidays", return_value=[])
    @patch(f"{_P}.now_utc")
    def test_telegram_bot_none_still_saves_db(
        self, mock_now, _mock_holidays, _mock_watchlist,
        mock_graph, mock_format, mock_save, mock_rec_model,
    ):
        """telegram_bot is None → DB save still happens, telegram=skipped."""
        mock_now.return_value = _mock_now_utc(2026, 4, 7, 23, 0)
        mock_graph.ainvoke = AsyncMock(return_value=SAMPLE_GRAPH_STATE)

        response = client.post("/internal/trigger/morning-briefing", headers=TRIGGER_HEADER)
        data = response.json()

        assert data["telegram"] == "skipped"
        mock_save.assert_called_once()

    @patch(f"{_P}._load_watchlist", return_value=["HPG"])
    @patch(f"{_P}._load_variable_holidays", return_value=[])
    @patch(f"{_P}.now_utc")
    def test_graph_exception_returns_failed(
        self, mock_now, _mock_holidays, _mock_watchlist,
        mock_graph, mock_format, mock_save, mock_rec_model,
    ):
        """morning_briefing_graph raises exception → status failed."""
        mock_now.return_value = _mock_now_utc(2026, 4, 7, 23, 0)
        mock_graph.ainvoke = AsyncMock(side_effect=Exception("LLM timeout"))

        response = client.post("/internal/trigger/morning-briefing", headers=TRIGGER_HEADER)
        data = response.json()

        assert data["status"] == "failed"
        assert "LLM timeout" in data["error"]

    @patch(f"{_P}.now_utc")
    @patch(f"{_P}._load_variable_holidays", return_value=[])
    @patch(f"{_P}._load_watchlist", return_value=[])
    def test_empty_watchlist_returns_failed(self, _mock_watchlist, _mock_holidays, mock_now):
        """No tickers in watchlist → status failed."""
        mock_now.return_value = _mock_now_utc(2026, 4, 7, 23, 0)

        response = client.post("/internal/trigger/morning-briefing", headers=TRIGGER_HEADER)
        data = response.json()

        assert data["status"] == "failed"
        assert "No tickers" in data["error"]

    @patch(f"{_P}._load_watchlist", return_value=["HPG"])
    @patch(f"{_P}._load_variable_holidays", return_value=[])
    @patch(f"{_P}.now_utc")
    def test_failed_steps_returns_partial(
        self, mock_now, _mock_holidays, _mock_watchlist,
        mock_graph, mock_format, mock_save, mock_rec_model,
    ):
        """failed_steps non-empty → status partial even when Telegram ok."""
        mock_now.return_value = _mock_now_utc(2026, 4, 7, 23, 0)
        state_with_failures = {
            **SAMPLE_GRAPH_STATE,
            "failed_steps": ["fundamental_analysis"],
        }
        mock_graph.ainvoke = AsyncMock(return_value=state_with_failures)

        response = client.post("/internal/trigger/morning-briefing", headers=TRIGGER_HEADER)
        data = response.json()

        assert data["status"] == "partial"
        assert "fundamental_analysis" in data["errors"]

    @patch(f"{_P}._load_watchlist", return_value=["HPG"])
    @patch(f"{_P}._load_variable_holidays", return_value=[])
    @patch(f"{_P}.now_utc")
    def test_top_picks_in_response(
        self, mock_now, _mock_holidays, _mock_watchlist,
        mock_graph, mock_format, mock_save, mock_rec_model,
    ):
        """top_picks populated → response contains top_picks_count > 0."""
        mock_now.return_value = _mock_now_utc(2026, 4, 7, 23, 0)
        mock_graph.ainvoke = AsyncMock(return_value=SAMPLE_GRAPH_STATE)

        response = client.post("/internal/trigger/morning-briefing", headers=TRIGGER_HEADER)
        data = response.json()

        assert data["top_picks_count"] == 1

    @patch(f"{_P}._load_watchlist", return_value=["HPG"])
    @patch(f"{_P}._load_variable_holidays", return_value=[])
    @patch(f"{_P}.now_utc")
    def test_telegram_delivered(
        self, mock_now, _mock_holidays, _mock_watchlist,
        mock_graph, mock_format, mock_save, mock_rec_model,
    ):
        """Telegram bot present and sends successfully → telegram=delivered."""
        mock_now.return_value = _mock_now_utc(2026, 4, 7, 23, 0)
        mock_graph.ainvoke = AsyncMock(return_value=SAMPLE_GRAPH_STATE)

        # Set up telegram bot mock on app state
        mock_bot = MagicMock()
        mock_bot.sender.send_message = AsyncMock(return_value=[12345])
        app.state.telegram_bot = mock_bot

        try:
            response = client.post("/internal/trigger/morning-briefing", headers=TRIGGER_HEADER)
            data = response.json()

            assert data["telegram"] == "delivered"
            mock_bot.sender.send_message.assert_called_once()
        finally:
            app.state.telegram_bot = None
