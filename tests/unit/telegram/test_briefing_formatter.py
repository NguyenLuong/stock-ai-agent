"""Tests for briefing_formatter — market_result → Telegram message."""

from __future__ import annotations

import pytest
from unittest.mock import patch

from services.app.telegram.formatters.briefing_formatter import (
    format_morning_briefing,
    _signal_emoji,
    _sentiment_emoji,
    _escape_md,
)


@pytest.fixture
def sample_market_result():
    return {
        "market_sentiment": "bullish",
        "affected_sectors": ["banking", "steel"],
        "key_events": ["Ngân hàng trung ương giữ lãi suất ổn định"],
        "top_picks": [
            {
                "ticker": "HPG",
                "signal": "uptrend",
                "confidence": 0.85,
                "summary": "Strong uptrend with high volume",
            },
            {
                "ticker": "VCB",
                "signal": "sideways",
                "confidence": 0.6,
                "summary": "Sideways movement",
            },
        ],
        "market_summary": "Thị trường tích cực với ngành thép dẫn đầu",
        "stale_warnings": [],
        "unavailable_warnings": [],
        "disclaimer": "Thông tin chỉ mang tính chất tham khảo, không phải khuyến nghị đầu tư.",
        "generated_at": "2026-04-01T06:30:00Z",
    }


class TestFormatMorningBriefing:
    def test_contains_morning_briefing_header(self, sample_market_result):
        result = format_morning_briefing(sample_market_result)
        assert "MORNING BRIEFING" in result

    def test_contains_sentiment(self, sample_market_result):
        result = format_morning_briefing(sample_market_result)
        assert "BULLISH" in result

    def test_contains_sectors(self, sample_market_result):
        result = format_morning_briefing(sample_market_result)
        assert "banking" in result
        assert "steel" in result

    def test_contains_top_picks(self, sample_market_result):
        result = format_morning_briefing(sample_market_result)
        assert "HPG" in result
        assert "VCB" in result
        assert "TOP PICKS" in result

    def test_contains_summary(self, sample_market_result):
        result = format_morning_briefing(sample_market_result)
        assert "Thị trường tích cực" in result

    def test_contains_key_events(self, sample_market_result):
        result = format_morning_briefing(sample_market_result)
        assert "lãi suất ổn định" in result

    def test_stale_warnings_appear(self, sample_market_result):
        sample_market_result["stale_warnings"] = ["Dữ liệu đã cũ hơn 4 giờ"]
        result = format_morning_briefing(sample_market_result)
        assert "Dữ liệu đã cũ hơn 4 giờ" in result

    def test_unavailable_warnings_appear(self, sample_market_result):
        sample_market_result["unavailable_warnings"] = ["Technical analysis unavailable: VNM"]
        result = format_morning_briefing(sample_market_result)
        assert "Technical analysis unavailable: VNM" in result

    def test_no_top_picks_shows_calm_message(self, sample_market_result):
        sample_market_result["top_picks"] = []
        result = format_morning_briefing(sample_market_result)
        assert "không có tín hiệu nổi bật" in result.lower()

    def test_contains_disclaimer(self, sample_market_result):
        result = format_morning_briefing(sample_market_result)
        assert "tham khảo" in result

    def test_empty_sectors(self, sample_market_result):
        sample_market_result["affected_sectors"] = []
        result = format_morning_briefing(sample_market_result)
        assert "MORNING BRIEFING" in result


class TestHelperFunctions:
    def test_signal_emoji_uptrend(self):
        assert _signal_emoji("uptrend") == "🟢"

    def test_signal_emoji_downtrend(self):
        assert _signal_emoji("downtrend") == "🔴"

    def test_signal_emoji_sideways(self):
        assert _signal_emoji("sideways") == "🟡"

    def test_sentiment_emoji_bullish(self):
        assert _sentiment_emoji("bullish") == "📈"

    def test_sentiment_emoji_bearish(self):
        assert _sentiment_emoji("bearish") == "📉"

    def test_sentiment_emoji_neutral(self):
        assert _sentiment_emoji("neutral") == "➡️"

    def test_escape_md_special_chars(self):
        assert _escape_md("hello *world* _foo_ `bar`") == r"hello \*world\* \_foo\_ \`bar\`"

    def test_escape_md_no_special(self):
        assert _escape_md("plain text") == "plain text"
