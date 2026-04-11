"""Tests for shared.db.repositories.recommendation_repo."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.models.recommendation import RecommendationCreate


def _make_rec(**overrides) -> RecommendationCreate:
    """Helper to create a RecommendationCreate with sensible defaults."""
    defaults = {
        "type": "morning_briefing",
        "ticker_symbol": "MARKET",
        "content": "Thị trường tăng điểm nhờ banking.",
        "confidence_score": None,
        "risk_level": None,
        "agents_used": ["market_context", "technical_analysis", "fundamental_analysis"],
        "agents_failed": [],
        "data_sources": {"top_picks": [{"ticker": "HPG"}]},
    }
    defaults.update(overrides)
    return RecommendationCreate(**defaults)


@pytest.mark.asyncio
@patch("shared.db.repositories.recommendation_repo.get_async_session")
async def test_save_recommendation_returns_uuid(mock_get_session):
    """save_recommendation inserts into DB and returns UUID."""
    expected_id = uuid.uuid4()

    mock_session = AsyncMock()

    def _add_side_effect(obj):
        obj.id = expected_id

    mock_session.add = MagicMock(side_effect=_add_side_effect)
    mock_session.commit = AsyncMock()
    mock_session.refresh = AsyncMock()

    mock_ctx = AsyncMock()
    mock_ctx.__aenter__.return_value = mock_session
    mock_ctx.__aexit__.return_value = None
    mock_get_session.return_value = mock_ctx

    from shared.db.repositories.recommendation_repo import save_recommendation

    rec = _make_rec()
    result = await save_recommendation(rec)

    assert result == expected_id
    mock_session.add.assert_called_once()
    mock_session.commit.assert_called_once()
    mock_session.refresh.assert_called_once()


@pytest.mark.asyncio
@patch("shared.db.repositories.recommendation_repo.get_async_session")
async def test_save_recommendation_maps_fields_correctly(mock_get_session):
    """All fields from RecommendationCreate are passed to ORM constructor."""
    expected_id = uuid.uuid4()

    mock_session = AsyncMock()

    captured_obj = {}

    def _add_side_effect(obj):
        obj.id = expected_id
        captured_obj["type"] = obj.type
        captured_obj["ticker_symbol"] = obj.ticker_symbol
        captured_obj["content"] = obj.content
        captured_obj["data_sources"] = obj.data_sources

    mock_session.add = MagicMock(side_effect=_add_side_effect)
    mock_session.commit = AsyncMock()
    mock_session.refresh = AsyncMock()

    mock_ctx = AsyncMock()
    mock_ctx.__aenter__.return_value = mock_session
    mock_ctx.__aexit__.return_value = None
    mock_get_session.return_value = mock_ctx

    from shared.db.repositories.recommendation_repo import save_recommendation

    rec = _make_rec(
        type="morning_briefing",
        ticker_symbol="MARKET",
        content="Test content",
        data_sources={"top_picks": [{"ticker": "VNM"}]},
    )
    await save_recommendation(rec)

    assert captured_obj["type"] == "morning_briefing"
    assert captured_obj["ticker_symbol"] == "MARKET"
    assert captured_obj["content"] == "Test content"
    assert captured_obj["data_sources"] == {"top_picks": [{"ticker": "VNM"}]}
