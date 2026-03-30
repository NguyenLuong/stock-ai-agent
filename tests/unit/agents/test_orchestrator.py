"""Tests for Orchestrator — dispatch, confidence, formatter, synthesis, and graph."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers & fixtures
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _old_iso(hours: int = 6) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()


def _market_summary(**overrides) -> dict:
    d = {
        "macro_summary": "Thị trường ổn định",
        "stock_summary": "HPG tăng nhẹ",
        "affected_sectors": ["thép"],
        "confidence": 0.75,
        "data_as_of": _now_iso(),
        "sources": ["vietstock", "cafef"],
    }
    d.update(overrides)
    return d


def _technical_analysis(**overrides) -> dict:
    d = {
        "indicator_summary": "RSI trung tính",
        "pattern_summary": "Double bottom",
        "signals": {
            "trend": "uptrend",
            "momentum": "bullish",
            "volatility": "normal",
            "volume_confirmation": True,
        },
        "support_levels": [25000.0],
        "resistance_levels": [28000.0],
        "confidence": 0.80,
        "data_as_of": _now_iso(),
        "data_source": "calculated",
    }
    d.update(overrides)
    return d


def _fundamental_analysis(**overrides) -> dict:
    d = {
        "bctc_summary": "BCTC tích cực",
        "ratio_comparison": "PE thấp hơn ngành",
        "company_ratios": {
            "pe": 8.5, "pb": 1.2, "roe": 18.0, "eps": 3500.0, "eps_growth_yoy": 15.0,
        },
        "sector_ratios": {"pe": 12.0, "pb": 1.5, "roe": 14.0, "eps": 2800.0},
        "sector_name": "Thép",
        "signals": {
            "valuation": "undervalued",
            "profitability": "strong",
            "financial_health": "healthy",
            "growth": "growing",
        },
        "confidence": 0.85,
        "data_as_of": _now_iso(),
        "data_source": "vnstock",
    }
    d.update(overrides)
    return d


def _base_state(**overrides) -> dict:
    s = {
        "ticker": "HPG",
        "analysis_type": "morning_briefing",
        "analysis_date": "2026-03-29",
        "watchlist": ["HPG", "VNM"],
    }
    s.update(overrides)
    return s


def _full_state(**overrides) -> dict:
    s = _base_state()
    s["market_summary"] = _market_summary()
    s["technical_analysis"] = _technical_analysis()
    s["fundamental_analysis"] = _fundamental_analysis()
    s["failed_agents"] = []
    s.update(overrides)
    return s


# Shared patch targets — patch on the source module so getattr() picks it up
_PATCH_MARKET = "services.app.agents.market_context.node.market_context_node"
_PATCH_TECHNICAL = "services.app.agents.technical_analysis.node.technical_analysis_node"
_PATCH_FUNDAMENTAL = "services.app.agents.fundamental_analysis.node.fundamental_analysis_node"
_PATCH_LLM_CLIENT = "services.app.agents.orchestrator.formatter._get_llm_client"
_PATCH_LOAD_PROMPT = "services.app.agents.orchestrator.formatter.load_prompt"
_PATCH_CONFIG = "services.app.agents.orchestrator.formatter.get_config_loader"


def _setup_config_mock(mock_config_cls):
    cfg = MagicMock()
    cfg.get_model.return_value = "gpt-4o"
    cfg.get_temperature.return_value = 0.7
    mock_config_cls.return_value = cfg
    return cfg


def _setup_prompt_mock(mock_load):
    rendered = MagicMock()
    rendered.text = "rendered prompt"
    rendered.model_key = "orchestrator_synthesis"
    mock_load.return_value = rendered
    return rendered


# ===========================================================================
# Unit Tests: calculate_confidence
# ===========================================================================


class TestCalculateConfidence:
    """Test confidence scoring with various agent success/failure scenarios."""

    def test_all_agents_ok_high_agreement(self):
        from services.app.agents.orchestrator.confidence import calculate_confidence

        state = _full_state()
        score = calculate_confidence(state)
        assert 0.5 <= score <= 1.0, f"Expected high score, got {score}"

    def test_one_agent_failed(self):
        from services.app.agents.orchestrator.confidence import calculate_confidence

        state = _full_state(
            failed_agents=["market_context"],
            market_summary=None,
        )
        score = calculate_confidence(state)
        # Penalty of 0.20 applied
        assert 0.0 <= score <= 0.80, f"Expected reduced score, got {score}"

    def test_two_agents_failed(self):
        from services.app.agents.orchestrator.confidence import calculate_confidence

        state = _full_state(
            failed_agents=["market_context", "technical_analysis"],
            market_summary=None,
            technical_analysis=None,
        )
        score = calculate_confidence(state)
        # Penalty of 0.40 applied
        assert 0.0 <= score <= 0.60, f"Expected low score, got {score}"

    def test_all_agents_failed(self):
        from services.app.agents.orchestrator.confidence import calculate_confidence

        state = _full_state(
            failed_agents=["market_context", "technical_analysis", "fundamental_analysis"],
            market_summary=None,
            technical_analysis=None,
            fundamental_analysis=None,
        )
        score = calculate_confidence(state)
        assert score == 0.0

    def test_stale_data_reduces_freshness(self):
        from services.app.agents.orchestrator.confidence import calculate_confidence

        state_fresh = _full_state()
        state_stale = _full_state(
            market_summary=_market_summary(data_as_of=_old_iso(6)),
            technical_analysis=_technical_analysis(data_as_of=_old_iso(6)),
            fundamental_analysis=_fundamental_analysis(data_as_of=_old_iso(6)),
        )
        score_fresh = calculate_confidence(state_fresh)
        score_stale = calculate_confidence(state_stale)
        assert score_fresh > score_stale


class TestConfidenceDisplay:
    def test_green(self):
        from services.app.agents.orchestrator.confidence import confidence_display
        assert "🟢" in confidence_display(0.75)

    def test_yellow(self):
        from services.app.agents.orchestrator.confidence import confidence_display
        assert "🟡" in confidence_display(0.55)

    def test_red(self):
        from services.app.agents.orchestrator.confidence import confidence_display
        assert "🔴" in confidence_display(0.30)


# ===========================================================================
# Unit Tests: format_agent_output_for_prompt
# ===========================================================================


class TestFormatAgentOutput:
    """Test output formatting for each agent type."""

    def test_format_market_context(self):
        from services.app.agents.orchestrator.formatter import format_agent_output_for_prompt

        result = format_agent_output_for_prompt(_market_summary(), "market_context")
        assert "Vĩ mô" in result
        assert "Cổ phiếu" in result
        assert "thép" in result

    def test_format_technical(self):
        from services.app.agents.orchestrator.formatter import format_agent_output_for_prompt

        result = format_agent_output_for_prompt(_technical_analysis(), "technical_analysis")
        assert "Chỉ báo" in result
        assert "Mô hình giá" in result
        assert "uptrend" in result

    def test_format_fundamental(self):
        from services.app.agents.orchestrator.formatter import format_agent_output_for_prompt

        result = format_agent_output_for_prompt(_fundamental_analysis(), "fundamental_analysis")
        assert "BCTC" in result
        assert "undervalued" in result

    def test_format_none_output(self):
        from services.app.agents.orchestrator.formatter import format_agent_output_for_prompt

        result = format_agent_output_for_prompt(None, "technical_analysis")
        assert "không khả dụng" in result


# ===========================================================================
# Unit Tests: Conflict detection
# ===========================================================================


class TestConflictDetection:
    """Test conflict detection between technical and fundamental signals."""

    def test_bullish_tech_bearish_fundamental(self):
        from services.app.agents.orchestrator.formatter import detect_conflicts

        state = _full_state(
            technical_analysis=_technical_analysis(
                signals={"trend": "uptrend", "momentum": "bullish", "volatility": "normal", "volume_confirmation": True}
            ),
            fundamental_analysis=_fundamental_analysis(
                signals={"valuation": "overvalued", "profitability": "weak", "financial_health": "risky", "growth": "declining"}
            ),
        )
        conflicts = detect_conflicts(state)
        assert len(conflicts) >= 1
        assert any("Trend vs Valuation" in c["topic"] for c in conflicts)

    def test_same_direction_no_conflict(self):
        from services.app.agents.orchestrator.formatter import detect_conflicts

        state = _full_state()  # Both bullish by default
        conflicts = detect_conflicts(state)
        assert len(conflicts) == 0

    def test_missing_agent_no_conflict(self):
        from services.app.agents.orchestrator.formatter import detect_conflicts

        state = _full_state(
            failed_agents=["technical_analysis"],
            technical_analysis=None,
        )
        conflicts = detect_conflicts(state)
        assert len(conflicts) == 0


# ===========================================================================
# Unit Tests: Staleness warnings
# ===========================================================================


class TestStalenessWarnings:
    """Test staleness warning generation for stale agent data."""

    @patch(_PATCH_CONFIG)
    @patch(_PATCH_LOAD_PROMPT)
    @patch(_PATCH_LLM_CLIENT)
    async def test_stale_data_produces_warnings(self, mock_llm_cls, mock_load, mock_config):
        from services.app.agents.orchestrator.formatter import synthesize_node

        _setup_config_mock(mock_config)
        _setup_prompt_mock(mock_load)
        llm = AsyncMock()
        llm.call.return_value = "KHUYẾN NGHỊ: THEO DÕI"
        mock_llm_cls.return_value = llm

        state = _full_state(
            market_summary=_market_summary(data_as_of=_old_iso(6)),
            technical_analysis=_technical_analysis(data_as_of=_old_iso(8)),
        )
        result = await synthesize_node(state)

        warnings = result["synthesis_result"]["stale_warnings"]
        assert len(warnings) >= 2
        assert any("market_context" in w for w in warnings)
        assert any("technical_analysis" in w for w in warnings)

    @patch(_PATCH_CONFIG)
    @patch(_PATCH_LOAD_PROMPT)
    @patch(_PATCH_LLM_CLIENT)
    async def test_fresh_data_no_warnings(self, mock_llm_cls, mock_load, mock_config):
        from services.app.agents.orchestrator.formatter import synthesize_node

        _setup_config_mock(mock_config)
        _setup_prompt_mock(mock_load)
        llm = AsyncMock()
        llm.call.return_value = "KHUYẾN NGHỊ: MUA"
        mock_llm_cls.return_value = llm

        state = _full_state()  # All data_as_of is _now_iso() → fresh
        result = await synthesize_node(state)

        warnings = result["synthesis_result"]["stale_warnings"]
        assert len(warnings) == 0


# ===========================================================================
# Unit Tests: dispatch_and_collect
# ===========================================================================


class TestDispatchAndCollect:
    """Test parallel agent dispatch with success and failure scenarios."""

    @patch(_PATCH_FUNDAMENTAL, new_callable=AsyncMock)
    @patch(_PATCH_TECHNICAL, new_callable=AsyncMock)
    @patch(_PATCH_MARKET, new_callable=AsyncMock)
    async def test_all_agents_success(self, mock_market, mock_tech, mock_fund):
        from services.app.agents.orchestrator.node import dispatch_and_collect

        mock_market.return_value = {"market_summary": _market_summary()}
        mock_tech.return_value = {"technical_analysis": _technical_analysis()}
        mock_fund.return_value = {"fundamental_analysis": _fundamental_analysis()}

        result = await dispatch_and_collect(_base_state())

        assert "market_summary" in result
        assert "technical_analysis" in result
        assert "fundamental_analysis" in result
        assert result["failed_agents"] == []

    @patch(_PATCH_FUNDAMENTAL, new_callable=AsyncMock)
    @patch(_PATCH_TECHNICAL, new_callable=AsyncMock)
    @patch(_PATCH_MARKET, new_callable=AsyncMock)
    async def test_one_agent_fails(self, mock_market, mock_tech, mock_fund):
        from services.app.agents.orchestrator.node import dispatch_and_collect

        mock_market.side_effect = RuntimeError("API error")
        mock_tech.return_value = {"technical_analysis": _technical_analysis()}
        mock_fund.return_value = {"fundamental_analysis": _fundamental_analysis()}

        result = await dispatch_and_collect(_base_state())

        assert "market_context" in result["failed_agents"]
        assert "technical_analysis" in result
        assert "fundamental_analysis" in result

    @patch(_PATCH_FUNDAMENTAL, new_callable=AsyncMock)
    @patch(_PATCH_TECHNICAL, new_callable=AsyncMock)
    @patch(_PATCH_MARKET, new_callable=AsyncMock)
    async def test_all_agents_fail(self, mock_market, mock_tech, mock_fund):
        from services.app.agents.orchestrator.node import dispatch_and_collect

        mock_market.side_effect = RuntimeError("fail")
        mock_tech.side_effect = RuntimeError("fail")
        mock_fund.side_effect = RuntimeError("fail")

        result = await dispatch_and_collect(_base_state())

        assert len(result["failed_agents"]) == 3


# ===========================================================================
# Integration Tests: synthesize_node
# ===========================================================================


class TestSynthesizeNode:
    """Test synthesis node with mocked LLM calls."""

    @patch(_PATCH_CONFIG)
    @patch(_PATCH_LOAD_PROMPT)
    @patch(_PATCH_LLM_CLIENT)
    async def test_happy_path(self, mock_llm_cls, mock_load, mock_config):
        from services.app.agents.orchestrator.formatter import synthesize_node

        _setup_config_mock(mock_config)
        _setup_prompt_mock(mock_load)
        llm = AsyncMock()
        llm.call.return_value = "KHUYẾN NGHỊ: MUA\nTIN CẬY: 0.8"
        mock_llm_cls.return_value = llm

        state = _full_state()
        result = await synthesize_node(state)

        assert result["error"] is None
        assert result["synthesis_result"] is not None
        assert result["confidence_score"] > 0
        assert result["synthesis_result"]["disclaimer"] is not None

    @patch(_PATCH_CONFIG)
    @patch(_PATCH_LOAD_PROMPT)
    @patch(_PATCH_LLM_CLIENT)
    async def test_all_agents_failed_no_llm_call(self, mock_llm_cls, mock_load, mock_config):
        from services.app.agents.orchestrator.formatter import synthesize_node

        state = _full_state(
            failed_agents=["market_context", "technical_analysis", "fundamental_analysis"],
            market_summary=None,
            technical_analysis=None,
            fundamental_analysis=None,
        )
        result = await synthesize_node(state)

        assert result["synthesis_result"] is None
        assert result["confidence_score"] == 0.0
        assert result["error"] is not None
        # LLM should NOT have been called
        mock_llm_cls.assert_not_called()

    @patch(_PATCH_CONFIG)
    @patch(_PATCH_LOAD_PROMPT)
    @patch(_PATCH_LLM_CLIENT)
    async def test_partial_failure_synthesis(self, mock_llm_cls, mock_load, mock_config):
        from services.app.agents.orchestrator.formatter import synthesize_node

        _setup_config_mock(mock_config)
        _setup_prompt_mock(mock_load)
        llm = AsyncMock()
        llm.call.return_value = "KHUYẾN NGHỊ: THEO DÕI"
        mock_llm_cls.return_value = llm

        state = _full_state(
            failed_agents=["market_context"],
            market_summary=None,
        )
        result = await synthesize_node(state)

        assert result["error"] is None
        assert result["synthesis_result"] is not None
        assert "Market Context" in str(result["synthesis_result"]["unavailable_warnings"])

    @patch(_PATCH_CONFIG)
    @patch(_PATCH_LOAD_PROMPT)
    @patch(_PATCH_LLM_CLIENT)
    async def test_conflict_detection_triggers_resolution(self, mock_llm_cls, mock_load, mock_config):
        from services.app.agents.orchestrator.formatter import synthesize_node

        _setup_config_mock(mock_config)
        _setup_prompt_mock(mock_load)
        llm = AsyncMock()
        llm.call.return_value = "Resolved conflict"
        mock_llm_cls.return_value = llm

        state = _full_state(
            technical_analysis=_technical_analysis(
                signals={"trend": "uptrend", "momentum": "bullish", "volatility": "normal", "volume_confirmation": True}
            ),
            fundamental_analysis=_fundamental_analysis(
                signals={"valuation": "overvalued", "profitability": "weak", "financial_health": "risky", "growth": "declining"}
            ),
        )
        result = await synthesize_node(state)

        assert result["synthesis_result"]["conflict_resolution"] is not None
        assert len(result["synthesis_result"]["conflicts"]) >= 1

    @patch(_PATCH_CONFIG)
    @patch(_PATCH_LOAD_PROMPT)
    @patch(_PATCH_LLM_CLIENT)
    async def test_llm_failure_returns_error(self, mock_llm_cls, mock_load, mock_config):
        from shared.llm.client import LLMCallError
        from services.app.agents.orchestrator.formatter import synthesize_node

        _setup_config_mock(mock_config)
        _setup_prompt_mock(mock_load)
        llm = AsyncMock()
        llm.call.side_effect = LLMCallError("API error", RuntimeError("timeout"))
        mock_llm_cls.return_value = llm

        state = _full_state()
        result = await synthesize_node(state)

        assert result["synthesis_result"] is None
        assert "failed" in result["error"].lower()


# ===========================================================================
# Integration Tests: Full graph
# ===========================================================================


class TestOrchestratorPipeline:
    """Integration tests — dispatch → synthesize pipeline (equivalent to LangGraph graph).

    langgraph is a runtime dependency (Docker container) not available in unit test env,
    so we test the two-node pipeline directly.
    """

    @patch(_PATCH_CONFIG)
    @patch(_PATCH_LOAD_PROMPT)
    @patch(_PATCH_LLM_CLIENT)
    @patch(_PATCH_FUNDAMENTAL, new_callable=AsyncMock)
    @patch(_PATCH_TECHNICAL, new_callable=AsyncMock)
    @patch(_PATCH_MARKET, new_callable=AsyncMock)
    async def test_full_pipeline_happy_path(
        self, mock_market, mock_tech, mock_fund,
        mock_llm_cls, mock_load, mock_config,
    ):
        from services.app.agents.orchestrator.node import dispatch_and_collect
        from services.app.agents.orchestrator.formatter import synthesize_node

        mock_market.return_value = {"market_summary": _market_summary()}
        mock_tech.return_value = {"technical_analysis": _technical_analysis()}
        mock_fund.return_value = {"fundamental_analysis": _fundamental_analysis()}

        _setup_config_mock(mock_config)
        _setup_prompt_mock(mock_load)
        llm = AsyncMock()
        llm.call.return_value = "KHUYẾN NGHỊ: MUA"
        mock_llm_cls.return_value = llm

        state = _base_state()
        dispatch_result = await dispatch_and_collect(state)
        state.update(dispatch_result)
        result = await synthesize_node(state)

        assert result.get("synthesis_result") is not None
        assert result.get("confidence_score") is not None
        assert result.get("error") is None

    @patch(_PATCH_CONFIG)
    @patch(_PATCH_LOAD_PROMPT)
    @patch(_PATCH_LLM_CLIENT)
    @patch(_PATCH_FUNDAMENTAL, new_callable=AsyncMock)
    @patch(_PATCH_TECHNICAL, new_callable=AsyncMock)
    @patch(_PATCH_MARKET, new_callable=AsyncMock)
    async def test_full_pipeline_partial_failure(
        self, mock_market, mock_tech, mock_fund,
        mock_llm_cls, mock_load, mock_config,
    ):
        from services.app.agents.orchestrator.node import dispatch_and_collect
        from services.app.agents.orchestrator.formatter import synthesize_node

        mock_market.side_effect = RuntimeError("API down")
        mock_tech.return_value = {"technical_analysis": _technical_analysis()}
        mock_fund.return_value = {"fundamental_analysis": _fundamental_analysis()}

        _setup_config_mock(mock_config)
        _setup_prompt_mock(mock_load)
        llm = AsyncMock()
        llm.call.return_value = "KHUYẾN NGHỊ: THEO DÕI"
        mock_llm_cls.return_value = llm

        state = _base_state()
        dispatch_result = await dispatch_and_collect(state)
        state.update(dispatch_result)
        result = await synthesize_node(state)

        assert "market_context" in state.get("failed_agents", [])
        assert result.get("synthesis_result") is not None

    @patch(_PATCH_FUNDAMENTAL, new_callable=AsyncMock)
    @patch(_PATCH_TECHNICAL, new_callable=AsyncMock)
    @patch(_PATCH_MARKET, new_callable=AsyncMock)
    async def test_full_pipeline_total_failure(
        self, mock_market, mock_tech, mock_fund,
    ):
        from services.app.agents.orchestrator.node import dispatch_and_collect
        from services.app.agents.orchestrator.formatter import synthesize_node

        mock_market.side_effect = RuntimeError("fail")
        mock_tech.side_effect = RuntimeError("fail")
        mock_fund.side_effect = RuntimeError("fail")

        state = _base_state()
        dispatch_result = await dispatch_and_collect(state)
        state.update(dispatch_result)
        result = await synthesize_node(state)

        assert result.get("synthesis_result") is None
        assert result.get("error") is not None
        assert result.get("confidence_score") == 0.0

    @patch(_PATCH_CONFIG)
    @patch(_PATCH_LOAD_PROMPT)
    @patch(_PATCH_LLM_CLIENT)
    @patch(_PATCH_FUNDAMENTAL, new_callable=AsyncMock)
    @patch(_PATCH_TECHNICAL, new_callable=AsyncMock)
    @patch(_PATCH_MARKET, new_callable=AsyncMock)
    async def test_full_pipeline_under_2min(
        self, mock_market, mock_tech, mock_fund,
        mock_llm_cls, mock_load, mock_config,
    ):
        """Verify pipeline completes well under the 120s timeout."""
        import asyncio
        import time

        from services.app.agents.orchestrator.node import dispatch_and_collect
        from services.app.agents.orchestrator.formatter import synthesize_node

        mock_market.return_value = {"market_summary": _market_summary()}
        mock_tech.return_value = {"technical_analysis": _technical_analysis()}
        mock_fund.return_value = {"fundamental_analysis": _fundamental_analysis()}

        _setup_config_mock(mock_config)
        _setup_prompt_mock(mock_load)
        llm = AsyncMock()
        llm.call.return_value = "KHUYẾN NGHỊ: MUA"
        mock_llm_cls.return_value = llm

        start = time.monotonic()

        async def run_pipeline():
            state = _base_state()
            dispatch_result = await dispatch_and_collect(state)
            state.update(dispatch_result)
            return await synthesize_node(state)

        result = await asyncio.wait_for(run_pipeline(), timeout=120)
        duration = time.monotonic() - start

        assert duration < 120
        assert result.get("synthesis_result") is not None
