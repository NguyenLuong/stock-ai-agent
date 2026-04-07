"""Integration tests for the Morning Briefing pipeline (node sequence).

langgraph is a runtime dependency (Docker container) not available in unit test env,
so we test the 5-node pipeline directly by calling nodes in sequence.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from services.app.agents.morning_briefing.nodes import (
    morning_market_context_node,
    sector_filter_node,
    technical_batch_node,
    fundamental_batch_node,
    morning_synthesis_node,
)


_PATCH_MC = "services.app.agents.morning_briefing.nodes.market_context_node"
_PATCH_TA = "services.app.agents.morning_briefing.nodes.technical_analysis_node"
_PATCH_FA = "services.app.agents.morning_briefing.nodes.fundamental_analysis_node"


def _mock_market_context_result():
    return {
        "market_summary": {
            "macro_summary": "Ngân hàng trung ương giữ lãi suất ổn định, triển vọng tích cực",
            "stock_summary": "Ngành thép tăng mạnh nhờ nhu cầu xuất khẩu",
            "affected_sectors": ["ngân hàng", "thép"],
            "confidence": 0.75,
            "data_as_of": "2026-04-01T06:00:00Z",
            "sources": ["vietstock", "cafef"],
        },
    }


def _mock_technical_result(ticker: str, trend: str = "uptrend"):
    return {
        "technical_analysis": {
            "signals": {"trend": trend, "momentum": "bullish" if trend == "uptrend" else "neutral"},
            "confidence": 0.8 if trend == "uptrend" else 0.5,
            "indicator_summary": f"{ticker} technical analysis",
        },
    }


def _mock_fundamental_result(ticker: str):
    return {
        "fundamental_analysis": {
            "bctc_summary": f"{ticker} BCTC OK",
            "confidence": 0.7,
        },
    }


_YAML_CONTENT = """
groups:
  banking:
    enabled: true
    sector: "banking"
    tickers: [VCB, MBB]
  steel:
    enabled: true
    sector: "steel"
    tickers: [HPG, HSG]
"""


class TestMorningBriefingPipeline:
    """Integration tests — run all 5 nodes sequentially (equivalent to LangGraph graph)."""

    @pytest.mark.asyncio
    @patch(_PATCH_FA, new_callable=AsyncMock)
    @patch(_PATCH_TA, new_callable=AsyncMock)
    @patch(_PATCH_MC, new_callable=AsyncMock)
    async def test_full_pipeline_happy_path(self, mock_mc, mock_ta, mock_fa, tmp_path):
        mock_mc.return_value = _mock_market_context_result()

        def ta_side_effect(state):
            ticker = state.get("ticker", "")
            if ticker == "HPG":
                return _mock_technical_result("HPG", "uptrend")
            return _mock_technical_result(ticker, "sideways")

        mock_ta.side_effect = ta_side_effect
        mock_fa.side_effect = lambda state: _mock_fundamental_result(state.get("ticker", ""))

        config_file = tmp_path / "crawlers" / "stock_tickers.yaml"
        config_file.parent.mkdir(parents=True)
        config_file.write_text(_YAML_CONTENT)

        # Initial state
        state: dict = {
            "analysis_date": "2026-04-01",
            "watchlist": ["HPG", "VNM", "VCB", "MBB"],
        }

        with patch.dict("os.environ", {"CONFIG_DIR": str(tmp_path)}):
            # Step 1: Market Context
            r1 = await morning_market_context_node(state)
            state.update(r1)

            # Step 2: Sector Filter
            r2 = await sector_filter_node(state)
            state.update(r2)

            # Step 3: Technical Batch
            r3 = await technical_batch_node(state)
            state.update(r3)

            # Step 4: Fundamental Batch
            r4 = await fundamental_batch_node(state)
            state.update(r4)

            # Step 5: Synthesis
            r5 = await morning_synthesis_node(state)
            state.update(r5)

        # Validate final state
        assert state.get("market_result") is not None
        mr = state["market_result"]
        assert mr["market_sentiment"] in ("bullish", "bearish", "neutral")
        assert isinstance(mr["top_picks"], list)
        assert isinstance(mr["affected_sectors"], list)
        assert "banking" in state["affected_sectors"]
        assert "steel" in state["affected_sectors"]
        assert mr.get("generated_at") is not None
        assert mr.get("disclaimer") is not None

    @pytest.mark.asyncio
    @patch(_PATCH_FA, new_callable=AsyncMock)
    @patch(_PATCH_TA, new_callable=AsyncMock)
    @patch(_PATCH_MC, new_callable=AsyncMock)
    async def test_pipeline_with_market_context_failure(self, mock_mc, mock_ta, mock_fa, tmp_path):
        """Pipeline continues even when market_context fails — fallback to full watchlist."""
        mock_mc.side_effect = Exception("LLM unavailable")
        mock_ta.return_value = _mock_technical_result("HPG", "uptrend")
        mock_fa.return_value = _mock_fundamental_result("HPG")

        config_file = tmp_path / "crawlers" / "stock_tickers.yaml"
        config_file.parent.mkdir(parents=True)
        config_file.write_text(_YAML_CONTENT)

        state: dict = {
            "analysis_date": "2026-04-01",
            "watchlist": ["HPG", "VNM", "VCB", "MBB"],
        }

        with patch.dict("os.environ", {"CONFIG_DIR": str(tmp_path)}):
            r1 = await morning_market_context_node(state)
            state.update(r1)

            r2 = await sector_filter_node(state)
            state.update(r2)

            r3 = await technical_batch_node(state)
            state.update(r3)

            r4 = await fundamental_batch_node(state)
            state.update(r4)

            r5 = await morning_synthesis_node(state)
            state.update(r5)

        assert "market_context" in state.get("failed_steps", [])
        assert state.get("market_result") is not None
        # Fallback: full watchlist used since no sectors identified
        assert state["filtered_tickers"] == ["HPG", "VNM", "VCB", "MBB"]
