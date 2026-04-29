"""Tests for Morning Briefing pipeline nodes."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from services.app.agents.morning_briefing.nodes import (
    _extract_key_events,
    _infer_sentiment,
    _normalize_sectors,
    morning_market_context_node,
    sector_filter_node,
    technical_batch_node,
    fundamental_batch_node,
    morning_synthesis_node,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_state(**overrides) -> dict:
    """Minimal MorningBriefingState."""
    s: dict = {
        "analysis_date": "2026-04-01",
        "watchlist": ["HPG", "VNM", "VCB", "MBB"],
    }
    s.update(overrides)
    return s


_PATCH_MC = "services.app.agents.morning_briefing.nodes.market_context_node"
_PATCH_TA = "services.app.agents.morning_briefing.nodes.technical_analysis_node"
_PATCH_FA = "services.app.agents.morning_briefing.nodes.fundamental_analysis_node"
_PATCH_YAML_OPEN = "builtins.open"


# ---------------------------------------------------------------------------
# Pure function tests
# ---------------------------------------------------------------------------


class TestNormalizeSectors:
    def test_vietnamese_to_english_keys(self):
        raw = ["ngân hàng", "thép"]
        result = _normalize_sectors(raw)
        assert "banking" in result
        assert "steel" in result

    def test_energy_maps_to_oil_gas(self):
        result = _normalize_sectors(["năng lượng"])
        assert "oil_gas" in result

    def test_empty_input(self):
        assert _normalize_sectors([]) == []

    def test_real_estate_maps(self):
        assert _normalize_sectors(["bất động sản"]) == ["real_estate"]

    def test_no_match(self):
        assert _normalize_sectors(["ngành không tồn tại xyz"]) == []

    def test_deduplication(self):
        raw = ["ngân hàng", "banking", "tín dụng"]
        result = _normalize_sectors(raw)
        assert result.count("banking") == 1


class TestInferSentiment:
    def test_bullish(self):
        assert _infer_sentiment("Thị trường tăng mạnh, triển vọng tích cực") == "bullish"

    def test_bearish(self):
        assert _infer_sentiment("Giảm mạnh, rủi ro bán tháo") == "bearish"

    def test_neutral(self):
        assert _infer_sentiment("Thị trường giao dịch bình thường") == "neutral"

    def test_tied_keywords_high_confidence_bullish(self):
        # "tăng" (bull) + "giảm" (bear) = tied, confidence > 0.6 → bullish
        assert _infer_sentiment("Thị trường tăng rồi giảm nhẹ", confidence=0.8) == "bullish"

    def test_tied_keywords_low_confidence_neutral(self):
        # tied but low confidence → neutral
        assert _infer_sentiment("Thị trường tăng rồi giảm nhẹ", confidence=0.3) == "neutral"


class TestExtractKeyEvents:
    def test_extracts_lines(self):
        text = "Headline short\n- Sự kiện quan trọng: lãi suất tăng 0.5%\n- GDP tăng trưởng mạnh hơn kỳ vọng"
        events = _extract_key_events(text)
        assert len(events) == 2

    def test_max_events(self):
        text = "\n".join(f"- Event number {i} lãi suất tăng enough length to pass" for i in range(10))
        events = _extract_key_events(text, max_events=3)
        assert len(events) == 3

    def test_prioritises_event_keywords(self):
        text = (
            "Đây là một dòng bình thường không có gì đặc biệt ở đây\n"
            "Lãi suất tăng 0.5% ảnh hưởng ngành ngân hàng mạnh\n"
            "Giá dầu giảm mạnh hôm qua ảnh hưởng xuất khẩu\n"
            "Một dòng dài đủ ký tự nhưng không có keyword gì cả"
        )
        events = _extract_key_events(text, max_events=2)
        assert any("lãi suất" in e.lower() for e in events)
        assert any("dầu" in e.lower() or "xuất khẩu" in e.lower() for e in events)

    def test_multi_paragraph_real_world(self):
        text = (
            "Tổng quan thị trường ngày 01/04/2026.\n"
            "\n"
            "Ngân hàng Nhà nước giữ lãi suất điều hành ổn định.\n"
            "FDI vào Việt Nam tăng 15% so với cùng kỳ năm ngoái.\n"
            "Thị trường giao dịch bình thường, thanh khoản ổn định.\n"
            "- Giá thép Hòa Phát tăng 3% nhờ nhu cầu xuất khẩu.\n"
            "- CPI tháng 3 tăng nhẹ 0.2% so với tháng trước.\n"
        )
        events = _extract_key_events(text, max_events=4)
        assert len(events) == 4
        # Keyword lines should be prioritised
        event_text = " ".join(events).lower()
        assert "lãi suất" in event_text or "fdi" in event_text


# ---------------------------------------------------------------------------
# Node 1: morning_market_context_node
# ---------------------------------------------------------------------------


class TestMorningMarketContextNode:
    @pytest.mark.asyncio
    @patch(_PATCH_MC, new_callable=AsyncMock)
    async def test_success_with_sectors(self, mock_mc):
        mock_mc.return_value = {
            "market_summary": {
                "macro_summary": "Ngân hàng trung ương tăng lãi suất, ảnh hưởng ngành thép",
                "stock_summary": "Cổ phiếu ngân hàng giảm điểm",
                "affected_sectors": ["ngân hàng", "thép"],
                "confidence": 0.75,
                "data_as_of": "2026-04-01T06:00:00Z",
                "sources": ["vietstock", "cafef"],
            },
        }
        result = await morning_market_context_node(_base_state())
        assert "banking" in result["affected_sectors"]
        assert "steel" in result["affected_sectors"]
        assert result["market_sentiment"] in ("bullish", "bearish", "neutral")
        assert result["market_summary"] is not None

    @pytest.mark.asyncio
    @patch(_PATCH_MC, new_callable=AsyncMock)
    async def test_market_context_fail_returns_empty(self, mock_mc):
        mock_mc.side_effect = Exception("LLM timeout")
        result = await morning_market_context_node(_base_state())
        assert result["affected_sectors"] == []
        assert result["market_sentiment"] == "neutral"
        assert result["key_events"] == []
        assert "market_context" in result["failed_steps"]

    @pytest.mark.asyncio
    @patch(_PATCH_MC, new_callable=AsyncMock)
    async def test_market_summary_none(self, mock_mc):
        mock_mc.return_value = {"market_summary": None}
        result = await morning_market_context_node(_base_state())
        assert result["affected_sectors"] == []
        assert "market_context" in result["failed_steps"]


# ---------------------------------------------------------------------------
# Node 2: sector_filter_node
# ---------------------------------------------------------------------------

_YAML_CONTENT = """
groups:
  banking:
    enabled: true
    sector: "banking"
    tickers: [VCB, MBB, TCB]
  steel:
    enabled: true
    sector: "steel"
    tickers: [HPG, HSG]
  disabled_group:
    enabled: false
    sector: "other"
    tickers: [XXX]
"""

_YAML_WITH_MAX = """
groups:
  banking:
    enabled: true
    sector: "banking"
    max_tickers: 2
    tickers: [VCB, BID, CTG, TCB, MBB, ACB, VPB, TPB, STB, HDB]
  steel:
    enabled: true
    sector: "steel"
    tickers: [HPG, HSG, NKG]
"""


class TestSectorFilterNode:
    @pytest.mark.asyncio
    async def test_filter_banking_sector(self, tmp_path):
        config_file = tmp_path / "crawlers" / "stock_tickers.yaml"
        config_file.parent.mkdir(parents=True)
        config_file.write_text(_YAML_CONTENT)

        state = _base_state(affected_sectors=["banking"])
        with patch.dict("os.environ", {"CONFIG_DIR": str(tmp_path)}):
            result = await sector_filter_node(state)
        assert set(result["filtered_tickers"]) == {"VCB", "MBB", "TCB"}

    @pytest.mark.asyncio
    async def test_empty_sectors_aborts_pipeline(self, tmp_path):
        config_file = tmp_path / "crawlers" / "stock_tickers.yaml"
        config_file.parent.mkdir(parents=True)
        config_file.write_text(_YAML_CONTENT)

        state = _base_state(affected_sectors=[])
        with patch.dict("os.environ", {"CONFIG_DIR": str(tmp_path)}):
            result = await sector_filter_node(state)
        assert result["filtered_tickers"] == []
        assert result["pipeline_aborted"] is True
        assert result["abort_reason"] == "no_sectors_identified"

    @pytest.mark.asyncio
    async def test_nonexistent_sector_aborts_pipeline(self, tmp_path):
        config_file = tmp_path / "crawlers" / "stock_tickers.yaml"
        config_file.parent.mkdir(parents=True)
        config_file.write_text(_YAML_CONTENT)

        state = _base_state(affected_sectors=["nonexistent"])
        with patch.dict("os.environ", {"CONFIG_DIR": str(tmp_path)}):
            result = await sector_filter_node(state)
        assert result["filtered_tickers"] == []
        assert result["pipeline_aborted"] is True
        assert result["abort_reason"] == "sectors_not_in_watchlist"

    @pytest.mark.asyncio
    async def test_max_tickers_caps_group(self, tmp_path):
        config_file = tmp_path / "crawlers" / "stock_tickers.yaml"
        config_file.parent.mkdir(parents=True)
        config_file.write_text(_YAML_WITH_MAX)

        state = _base_state(affected_sectors=["banking"])
        with patch.dict("os.environ", {"CONFIG_DIR": str(tmp_path)}):
            result = await sector_filter_node(state)
        # banking group has 10 tickers but max_tickers=2 → only first 2
        assert result["filtered_tickers"] == ["VCB", "BID"]
        assert "pipeline_aborted" not in result or not result.get("pipeline_aborted")

    @pytest.mark.asyncio
    async def test_max_tickers_none_takes_all(self, tmp_path):
        config_file = tmp_path / "crawlers" / "stock_tickers.yaml"
        config_file.parent.mkdir(parents=True)
        config_file.write_text(_YAML_WITH_MAX)

        state = _base_state(affected_sectors=["steel"])
        with patch.dict("os.environ", {"CONFIG_DIR": str(tmp_path)}):
            result = await sector_filter_node(state)
        # steel group has no max_tickers → all 3 tickers
        assert result["filtered_tickers"] == ["HPG", "HSG", "NKG"]

    @pytest.mark.asyncio
    async def test_disabled_group_excluded(self, tmp_path):
        config_file = tmp_path / "crawlers" / "stock_tickers.yaml"
        config_file.parent.mkdir(parents=True)
        config_file.write_text(_YAML_CONTENT)

        state = _base_state(affected_sectors=["other"])
        with patch.dict("os.environ", {"CONFIG_DIR": str(tmp_path)}):
            result = await sector_filter_node(state)
        # Disabled group excluded → abort with sectors_not_in_watchlist
        assert result["filtered_tickers"] == []
        assert result.get("pipeline_aborted") is True

    @pytest.mark.asyncio
    async def test_config_not_found_fallback(self):
        # config missing is a distinct edge case — still returns full watchlist
        state = _base_state(affected_sectors=["banking"])
        with patch.dict("os.environ", {"CONFIG_DIR": "/nonexistent/path"}):
            result = await sector_filter_node(state)
        assert result["filtered_tickers"] == ["HPG", "VNM", "VCB", "MBB"]


# ---------------------------------------------------------------------------
# Node 3: technical_batch_node
# ---------------------------------------------------------------------------


class TestTechnicalBatchNode:
    @pytest.mark.asyncio
    @patch(_PATCH_TA, new_callable=AsyncMock)
    async def test_two_tickers_one_fail(self, mock_ta):
        mock_ta.side_effect = [
            {"technical_analysis": {"signals": {"trend": "uptrend", "momentum": "bullish"}, "confidence": 0.8}},
            Exception("timeout"),
        ]
        state = _base_state(filtered_tickers=["HPG", "VNM"], analysis_date="2026-04-01")
        result = await technical_batch_node(state)
        assert len(result["technical_results"]) == 2
        assert result["technical_results"][0]["failed"] is False
        assert result["technical_results"][1]["failed"] is True
        assert "HPG" in result["notable_tickers"]

    @pytest.mark.asyncio
    @patch(_PATCH_TA, new_callable=AsyncMock)
    async def test_all_fail_fallback_takes_first_two(self, mock_ta):
        """When no notable tickers emerge, fallback to first 2 filtered tickers."""
        mock_ta.side_effect = Exception("all fail")
        state = _base_state(filtered_tickers=["HPG", "VNM", "VCB", "MBB"])
        result = await technical_batch_node(state)
        assert result["notable_tickers"] == ["HPG", "VNM"]
        assert "technical_batch" in result["failed_steps"]

    @pytest.mark.asyncio
    @patch(_PATCH_TA, new_callable=AsyncMock)
    async def test_all_sideways_fallback_takes_first_two(self, mock_ta):
        """All tickers sideways+neutral → notable empty → fallback to first 2."""
        mock_ta.return_value = {
            "technical_analysis": {
                "signals": {"trend": "sideways", "momentum": "neutral"},
                "confidence": 0.5,
            },
        }
        state = _base_state(filtered_tickers=["HPG", "VNM", "VCB", "MBB"])
        result = await technical_batch_node(state)
        assert result["notable_tickers"] == ["HPG", "VNM"]

    @pytest.mark.asyncio
    @patch(_PATCH_TA, new_callable=AsyncMock)
    async def test_one_notable_no_fallback(self, mock_ta):
        """1 notable ticker exists → no fallback, keep as is."""
        mock_ta.side_effect = [
            {"technical_analysis": {"signals": {"trend": "uptrend", "momentum": "bullish"}, "confidence": 0.8}},
            {"technical_analysis": {"signals": {"trend": "sideways", "momentum": "neutral"}, "confidence": 0.5}},
            {"technical_analysis": {"signals": {"trend": "sideways", "momentum": "neutral"}, "confidence": 0.5}},
        ]
        state = _base_state(filtered_tickers=["HPG", "VNM", "VCB"])
        result = await technical_batch_node(state)
        assert result["notable_tickers"] == ["HPG"]

    @pytest.mark.asyncio
    @patch(_PATCH_TA, new_callable=AsyncMock)
    async def test_notable_vs_not_notable(self, mock_ta):
        mock_ta.side_effect = [
            {"technical_analysis": {"signals": {"trend": "uptrend", "momentum": "bullish"}, "confidence": 0.8}},
            {"technical_analysis": {"signals": {"trend": "sideways", "momentum": "neutral"}, "confidence": 0.5}},
        ]
        state = _base_state(filtered_tickers=["HPG", "VNM"])
        result = await technical_batch_node(state)
        assert "HPG" in result["notable_tickers"]
        assert "VNM" not in result["notable_tickers"]

    @pytest.mark.asyncio
    @patch(_PATCH_TA, new_callable=AsyncMock)
    async def test_sideways_but_bullish_momentum_is_notable(self, mock_ta):
        mock_ta.return_value = {
            "technical_analysis": {"signals": {"trend": "sideways", "momentum": "bullish"}, "confidence": 0.6},
        }
        state = _base_state(filtered_tickers=["HPG"])
        result = await technical_batch_node(state)
        assert "HPG" in result["notable_tickers"]


# ---------------------------------------------------------------------------
# Node 4: fundamental_batch_node
# ---------------------------------------------------------------------------


class TestFundamentalBatchNode:
    @pytest.mark.asyncio
    @patch(_PATCH_FA, new_callable=AsyncMock)
    async def test_only_calls_for_notable_tickers(self, mock_fa):
        mock_fa.return_value = {"fundamental_analysis": {"bctc_summary": "OK"}}
        state = _base_state(notable_tickers=["HPG", "VCB"])
        result = await fundamental_batch_node(state)
        assert mock_fa.call_count == 2
        assert len(result["fundamental_results"]) == 2

    @pytest.mark.asyncio
    @patch(_PATCH_FA, new_callable=AsyncMock)
    async def test_failure_recorded(self, mock_fa):
        mock_fa.side_effect = Exception("DB error")
        state = _base_state(notable_tickers=["HPG"])
        result = await fundamental_batch_node(state)
        assert result["fundamental_results"][0]["failed"] is True
        assert "fundamental_batch" in result["failed_steps"]


# ---------------------------------------------------------------------------
# Node 5: morning_synthesis_node
# ---------------------------------------------------------------------------


class TestMorningSynthesisNode:
    def _make_state(self, **overrides):
        s = {
            "market_summary": {
                "macro_summary": "Macro analysis text",
                "stock_summary": "Stock summary",
                "affected_sectors": ["ngân hàng"],
                "confidence": 0.7,
            },
            "market_sentiment": "bullish",
            "affected_sectors": ["banking"],
            "key_events": ["Event 1"],
            "technical_results": [
                {
                    "ticker": "HPG",
                    "technical_analysis": {
                        "signals": {"trend": "uptrend", "momentum": "bullish"},
                        "confidence": 0.85,
                        "indicator_summary": "Strong uptrend with high volume",
                    },
                    "failed": False,
                },
                {
                    "ticker": "VNM",
                    "technical_analysis": {
                        "signals": {"trend": "sideways", "momentum": "neutral"},
                        "confidence": 0.55,
                        "indicator_summary": "Sideways movement",
                    },
                    "failed": False,
                },
            ],
            "fundamental_results": [
                {"ticker": "HPG", "fundamental_analysis": {"bctc_summary": "Good"}, "failed": False},
            ],
            "failed_steps": [],
        }
        s.update(overrides)
        return s

    @pytest.mark.asyncio
    async def test_top_picks_only_with_both_tech_and_fundamental(self):
        state = self._make_state()
        result = await morning_synthesis_node(state)
        mr = result["market_result"]
        tickers_in_picks = [p["ticker"] for p in mr["top_picks"]]
        assert "HPG" in tickers_in_picks
        # VNM has technical but no fundamental → excluded
        assert "VNM" not in tickers_in_picks

    @pytest.mark.asyncio
    async def test_empty_top_picks_gives_calm_message(self):
        state = self._make_state(
            technical_results=[{"ticker": "HPG", "technical_analysis": None, "failed": True}],
            fundamental_results=[],
        )
        result = await morning_synthesis_node(state)
        mr = result["market_result"]
        assert mr["top_picks"] == []
        assert "bình lặng" in mr["market_summary"]

    @pytest.mark.asyncio
    async def test_top_picks_sorted_by_confidence_max_5(self):
        tech_results = [
            {
                "ticker": f"T{i}",
                "technical_analysis": {
                    "signals": {"trend": "uptrend"},
                    "confidence": 0.5 + i * 0.05,
                    "indicator_summary": f"Summary {i}",
                },
                "failed": False,
            }
            for i in range(7)
        ]
        fund_results = [
            {"ticker": f"T{i}", "fundamental_analysis": {"bctc_summary": "OK"}, "failed": False}
            for i in range(7)
        ]
        state = self._make_state(technical_results=tech_results, fundamental_results=fund_results)
        result = await morning_synthesis_node(state)
        picks = result["market_result"]["top_picks"]
        assert len(picks) <= 5
        confidences = [p["confidence"] for p in picks]
        assert confidences == sorted(confidences, reverse=True)

    @pytest.mark.asyncio
    async def test_unavailable_warnings(self):
        state = self._make_state(
            technical_results=[{"ticker": "HPG", "technical_analysis": None, "failed": True}],
            fundamental_results=[{"ticker": "HPG", "fundamental_analysis": None, "failed": True}],
        )
        result = await morning_synthesis_node(state)
        warnings = result["market_result"]["unavailable_warnings"]
        assert len(warnings) == 2

    @pytest.mark.asyncio
    async def test_market_result_has_required_keys(self):
        state = self._make_state()
        result = await morning_synthesis_node(state)
        mr = result["market_result"]
        required_keys = [
            "market_sentiment", "affected_sectors", "key_events",
            "top_picks", "market_summary", "stale_warnings",
            "unavailable_warnings", "disclaimer", "generated_at",
        ]
        for key in required_keys:
            assert key in mr, f"Missing key: {key}"

    @pytest.mark.asyncio
    async def test_aborted_pipeline_yields_aborted_market_result(self):
        state = {
            "pipeline_aborted": True,
            "abort_reason": "no_sectors_identified",
            "market_sentiment": "neutral",
            "affected_sectors": [],
            "key_events": ["Event 1"],
            "market_summary": {"macro_summary": "Macro text"},
        }
        result = await morning_synthesis_node(state)
        mr = result["market_result"]
        assert mr["pipeline_status"] == "aborted"
        assert mr["abort_reason"] == "no_sectors_identified"
        assert mr["top_picks"] == []
        assert mr["stale_warnings"] == []
        assert mr["unavailable_warnings"] == []
        assert mr["market_summary"] == "Macro text"

    @pytest.mark.asyncio
    async def test_stale_warnings_populated_for_old_data(self):
        """data_as_of > 4h ago → stale warning generated."""
        from datetime import datetime, timedelta, timezone

        old_ts = (datetime.now(timezone.utc) - timedelta(hours=6)).isoformat()
        state = self._make_state(
            technical_results=[
                {
                    "ticker": "HPG",
                    "technical_analysis": {
                        "signals": {"trend": "uptrend"},
                        "confidence": 0.8,
                        "data_as_of": old_ts,
                    },
                    "failed": False,
                },
            ],
            fundamental_results=[
                {"ticker": "HPG", "fundamental_analysis": {"bctc_summary": "OK", "data_as_of": old_ts}, "failed": False},
            ],
        )
        result = await morning_synthesis_node(state)
        assert len(result["market_result"]["stale_warnings"]) >= 1
        assert any("HPG" in w for w in result["market_result"]["stale_warnings"])
