"""Tests for briefing_formatter — synthesis_result → Telegram message."""

from __future__ import annotations

import pytest
from unittest.mock import patch

from services.app.telegram.formatters.briefing_formatter import (
    format_morning_briefing,
    _parse_confidence_value,
    _confidence_emoji,
    _parse_recommendation,
)

SAMPLE_TICKER = "HPG"


@pytest.fixture
def sample_synthesis_result():
    return {
        "synthesis": "HPG đang trong xu hướng tăng. Khuyến nghị mua với độ tin cậy cao. P/E: 8.5 [source: vnstock, 2026-03-30T06:00:00Z]",
        "conflict_resolution": "",
        "conflicts": [],
        "data_sources": [{"agent": "technical_analysis", "source": "vnstock", "data_as_of": "2026-03-30T06:00:00Z"}],
        "stale_warnings": [],
        "unavailable_warnings": [],
        "confidence_display": "🟢 Cao (75%)",
        "disclaimer": "Đây là tham khảo từ AI, không phải khuyến nghị mua/bán chính thức",
        "generated_at": "2026-03-30T06:30:00Z",
        "risk_assessment": {
            "level": "low",
            "reasoning": "độ tin cậy tốt (75%), không có dấu hiệu rủi ro cao",
            "confidence_score": 0.75,
            "failed_agents": [],
        },
        "stop_loss_suggestion": 28000.0,
        "integrity_violations": [],
        "traceability_warnings": [],
        "null_fields": [],
    }


class TestFormatMorningBriefing:
    def test_contains_ticker(self, sample_synthesis_result):
        result = format_morning_briefing(sample_synthesis_result, ticker=SAMPLE_TICKER)
        assert "HPG" in result

    def test_confidence_high_green_emoji(self, sample_synthesis_result):
        """confidence >=0.70 → 🟢 appears exactly once in confidence line (no duplicate)."""
        sample_synthesis_result["confidence_display"] = "🟢 Cao (75%)"
        result = format_morning_briefing(sample_synthesis_result, ticker=SAMPLE_TICKER)
        assert "🟢" in result
        # Verify no duplicate emoji: "🟢 Độ tin cậy: 🟢 Cao (75%)" would be wrong
        assert "Độ tin cậy: 🟢 Cao (75%)" in result

    def test_confidence_medium_yellow_emoji(self, sample_synthesis_result):
        """confidence 0.40-0.69 → 🟡 (from confidence_display, not prepended)."""
        sample_synthesis_result["confidence_display"] = "🟡 Trung bình (55%)"
        result = format_morning_briefing(sample_synthesis_result, ticker=SAMPLE_TICKER)
        assert "Độ tin cậy: 🟡 Trung bình (55%)" in result

    def test_confidence_low_red_emoji(self, sample_synthesis_result):
        """confidence <0.40 → 🔴 (from confidence_display, not prepended)."""
        sample_synthesis_result["confidence_display"] = "🔴 Thấp (25%)"
        result = format_morning_briefing(sample_synthesis_result, ticker=SAMPLE_TICKER)
        assert "Độ tin cậy: 🔴 Thấp (25%)" in result

    def test_stale_warnings_appear(self, sample_synthesis_result):
        sample_synthesis_result["stale_warnings"] = ["Dữ liệu technical_analysis đã cũ hơn 4 giờ"]
        result = format_morning_briefing(sample_synthesis_result, ticker=SAMPLE_TICKER)
        assert "Dữ liệu technical_analysis đã cũ hơn 4 giờ" in result

    def test_stop_loss_present(self, sample_synthesis_result):
        """stop_loss_suggestion = 28000.0 → "Stop-loss" in output."""
        sample_synthesis_result["stop_loss_suggestion"] = 28000.0
        result = format_morning_briefing(sample_synthesis_result, ticker=SAMPLE_TICKER)
        assert "Stop-loss" in result

    def test_stop_loss_none_absent(self, sample_synthesis_result):
        """stop_loss_suggestion = None → "Stop-loss" NOT in output."""
        sample_synthesis_result["stop_loss_suggestion"] = None
        result = format_morning_briefing(sample_synthesis_result, ticker=SAMPLE_TICKER)
        assert "Stop-loss" not in result

    def test_integrity_violations_in_alerts(self, sample_synthesis_result):
        sample_synthesis_result["integrity_violations"] = ["Phát hiện xung đột dữ liệu giữa các agent"]
        result = format_morning_briefing(sample_synthesis_result, ticker=SAMPLE_TICKER)
        assert "Phát hiện xung đột dữ liệu giữa các agent" in result


class TestHelperFunctions:
    def test_parse_confidence_75(self):
        assert _parse_confidence_value("🟢 Cao (75%)") == 0.75

    def test_parse_confidence_no_match(self):
        assert _parse_confidence_value("unknown") == 0.5

    def test_confidence_emoji_high(self):
        assert _confidence_emoji(0.75) == "🟢"

    def test_confidence_emoji_medium(self):
        assert _confidence_emoji(0.55) == "🟡"

    def test_confidence_emoji_low(self):
        assert _confidence_emoji(0.30) == "🔴"

    def test_parse_recommendation_mua(self):
        assert _parse_recommendation("khuyến nghị mua HPG") == "MUA"

    def test_parse_recommendation_ban(self):
        assert _parse_recommendation("nên bán cổ phiếu") == "BÁN"

    def test_parse_recommendation_default(self):
        assert _parse_recommendation("no keywords here") == "THEO DÕI"
