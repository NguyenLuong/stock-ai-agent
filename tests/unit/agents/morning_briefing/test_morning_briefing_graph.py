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
    async def test_pipeline_aborts_when_market_context_fails(self, mock_mc, mock_ta, mock_fa, tmp_path):
        """When market_context fails → no affected_sectors → sector_filter aborts the pipeline.
        Technical/fundamental batches must NOT run; synthesis returns aborted market_result."""
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

            # Simulate conditional routing: when aborted, skip TA/FA entirely
            if not state.get("pipeline_aborted"):
                r3 = await technical_batch_node(state)
                state.update(r3)

                r4 = await fundamental_batch_node(state)
                state.update(r4)

            r5 = await morning_synthesis_node(state)
            state.update(r5)

        assert "market_context" in state.get("failed_steps", [])
        assert state.get("pipeline_aborted") is True
        assert state.get("abort_reason") == "no_sectors_identified"
        assert state["filtered_tickers"] == []
        # TA/FA skipped → mocks never called
        mock_ta.assert_not_called()
        mock_fa.assert_not_called()
        mr = state["market_result"]
        assert mr["pipeline_status"] == "aborted"
        assert mr["top_picks"] == []

    def test_conditional_routing_aborted_goes_to_synthesis(self):
        """Verify _route_after_sector_filter conditional routing function.

        langgraph is not available in the unit test environment, so we re-import
        the routing function via importlib with a stubbed langgraph module.
        """
        import sys
        import types
        import importlib

        # Stub langgraph.graph if not installed
        if "langgraph" not in sys.modules:
            fake_langgraph = types.ModuleType("langgraph")
            fake_graph = types.ModuleType("langgraph.graph")
            fake_graph.END = "__END__"

            class _StubStateGraph:
                def __init__(self, *_, **__):
                    pass

                def add_node(self, *_, **__):
                    pass

                def set_entry_point(self, *_, **__):
                    pass

                def add_edge(self, *_, **__):
                    pass

                def add_conditional_edges(self, *_, **__):
                    pass

                def compile(self):
                    return None

            fake_graph.StateGraph = _StubStateGraph
            sys.modules["langgraph"] = fake_langgraph
            sys.modules["langgraph.graph"] = fake_graph

        mod = importlib.import_module(
            "services.app.agents.morning_briefing_graph",
        )
        route_fn = mod._route_after_sector_filter

        assert route_fn({"pipeline_aborted": True}) == "synthesis"
        assert route_fn({"pipeline_aborted": False}) == "technical_batch"
        assert route_fn({}) == "technical_batch"
