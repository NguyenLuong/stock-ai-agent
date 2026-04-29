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


class TestFormatAbortedPipeline:
    @pytest.fixture
    def aborted_no_sectors(self):
        return {
            "pipeline_status": "aborted",
            "abort_reason": "no_sectors_identified",
            "market_sentiment": "neutral",
            "affected_sectors": [],
            "key_events": ["Ngân hàng trung ương giữ lãi suất ổn định"],
            "market_summary": "Thị trường bình lặng, không có biến động lớn",
            "top_picks": [],
            "stale_warnings": [],
            "unavailable_warnings": [],
            "disclaimer": "Thông tin chỉ mang tính chất tham khảo, không phải khuyến nghị đầu tư.",
            "generated_at": "2026-04-01T06:30:00Z",
        }

    def test_aborted_no_sectors_identified_message(self, aborted_no_sectors):
        result = format_morning_briefing(aborted_no_sectors)
        assert "Market Context không xác định được ngành bị ảnh hưởng" in result
        assert "TOP PICKS" not in result
        assert "Không có tín hiệu nổi bật" not in result

    def test_aborted_sectors_not_in_watchlist_message(self, aborted_no_sectors):
        aborted_no_sectors["abort_reason"] = "sectors_not_in_watchlist"
        result = format_morning_briefing(aborted_no_sectors)
        assert "Các ngành bị ảnh hưởng không có trong danh mục theo dõi" in result
        assert "TOP PICKS" not in result

    def test_aborted_includes_macro_summary(self, aborted_no_sectors):
        result = format_morning_briefing(aborted_no_sectors)
        assert "TÓM TẮT VĨ MÔ" in result
        assert "Thị trường bình lặng" in result

    def test_aborted_includes_key_events(self, aborted_no_sectors):
        result = format_morning_briefing(aborted_no_sectors)
        assert "SỰ KIỆN CHÍNH" in result
        assert "lãi suất" in result

    def test_aborted_includes_disclaimer(self, aborted_no_sectors):
        result = format_morning_briefing(aborted_no_sectors)
        assert "tham khảo" in result

    def test_aborted_unknown_reason_default_text(self, aborted_no_sectors):
        aborted_no_sectors["abort_reason"] = "some_unknown_reason"
        result = format_morning_briefing(aborted_no_sectors)
        assert "Không thể xác định tín hiệu thị trường" in result

    def test_aborted_no_summary_no_summary_section(self, aborted_no_sectors):
        aborted_no_sectors["market_summary"] = ""
        result = format_morning_briefing(aborted_no_sectors)
        assert "TÓM TẮT VĨ MÔ" not in result


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
