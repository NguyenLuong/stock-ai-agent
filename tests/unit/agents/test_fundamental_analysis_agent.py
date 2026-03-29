"""Tests for Fundamental Analysis Agent node and helper functions."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.app.agents.fundamental_analysis.node import (
    _build_ratio_for_prompt,
    _calc_confidence,
    _determine_growth,
    _determine_profitability,
    _determine_valuation,
    fundamental_analysis_node,
)
from shared.llm.client import LLMCallError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_company_ratios_raw() -> dict[str, Decimal]:
    """Sample raw ratio dict as returned from DB."""
    return {
        "pe_ratio": Decimal("12.5"),
        "pb_ratio": Decimal("1.8"),
        "roe": Decimal("15.2"),
        "eps": Decimal("3500"),
        "eps_growth_yoy": Decimal("0.12"),
    }


def _sample_sector_avg() -> dict[str, Decimal | None]:
    return {
        "pe": Decimal("10.3"),
        "pb": Decimal("1.5"),
        "roe": Decimal("13.8"),
        "eps": Decimal("2800"),
    }


def _sample_peers() -> list[dict]:
    return [
        {"ticker": "HSG", "pe": 8.5, "pb": 1.2, "roe": 12.0},
        {"ticker": "NKG", "pe": 7.0, "pb": 0.9, "roe": 10.5},
    ]


def _base_state() -> dict:
    return {
        "ticker": "HPG",
        "analysis_type": "morning_briefing",
        "analysis_date": "2026-03-29",
    }


# ---------------------------------------------------------------------------
# _build_ratio_for_prompt
# ---------------------------------------------------------------------------

class TestBuildRatioForPrompt:
    def test_full_ratios(self) -> None:
        result = _build_ratio_for_prompt(_sample_company_ratios_raw())
        assert result["pe"] == 12.5
        assert result["pb"] == 1.8
        assert result["roe"] == 15.2
        assert result["eps"] == 3500.0
        assert result["eps_growth_yoy"] == 0.12

    def test_partial_ratios_none_handling(self) -> None:
        raw = {"pe_ratio": Decimal("10.0"), "roe": None}
        result = _build_ratio_for_prompt(raw)
        assert result["pe"] == 10.0
        assert result["pb"] is None
        assert result["roe"] is None
        assert result["eps"] is None
        assert result["eps_growth_yoy"] is None


# ---------------------------------------------------------------------------
# Signal determination
# ---------------------------------------------------------------------------

class TestDetermineValuation:
    def test_undervalued(self) -> None:
        assert _determine_valuation(8.0, 12.0) == "undervalued"

    def test_overvalued(self) -> None:
        assert _determine_valuation(15.0, 10.0) == "overvalued"

    def test_fair(self) -> None:
        assert _determine_valuation(10.0, 10.0) == "fair"

    def test_none_company_pe(self) -> None:
        assert _determine_valuation(None, 10.0) == "fair"

    def test_none_sector_pe(self) -> None:
        assert _determine_valuation(10.0, None) == "fair"


class TestDetermineProfitability:
    def test_strong(self) -> None:
        assert _determine_profitability(20.0) == "strong"

    def test_weak(self) -> None:
        assert _determine_profitability(5.0) == "weak"

    def test_average(self) -> None:
        assert _determine_profitability(12.0) == "average"

    def test_none(self) -> None:
        assert _determine_profitability(None) == "average"


class TestDetermineGrowth:
    def test_growing(self) -> None:
        assert _determine_growth(0.15) == "growing"

    def test_declining(self) -> None:
        assert _determine_growth(-0.20) == "declining"

    def test_stable(self) -> None:
        assert _determine_growth(0.05) == "stable"

    def test_none(self) -> None:
        assert _determine_growth(None) == "stable"


# ---------------------------------------------------------------------------
# _calc_confidence
# ---------------------------------------------------------------------------

class TestCalcConfidence:
    def test_full_data_both_phases(self) -> None:
        company = {"pe": 12.5, "pb": 1.8, "roe": 15.2, "eps": 3500.0, "eps_growth_yoy": 0.12}
        sector = _sample_sector_avg()
        data_as_of = datetime.now(timezone.utc) - timedelta(days=30)
        result = _calc_confidence(
            company, sector, phase1_ok=True, phase2_ok=True,
            data_as_of=data_as_of, peer_count=3,
        )
        # base(0.50) + 5 ratios(0.15) + peers>=2(0.10) + both phases(0.10) + fresh(0.05) = 0.90
        assert result == 0.90

    def test_no_data_one_phase(self) -> None:
        company = {"pe": None, "pb": None, "roe": None, "eps": None, "eps_growth_yoy": None}
        sector = {"pe": None, "pb": None, "roe": None, "eps": None}
        result = _calc_confidence(
            company, sector, phase1_ok=True, phase2_ok=False,
            data_as_of=None, peer_count=0,
        )
        # base(0.50) + 0 ratios + 0 peers + 1 phase(0.05) = 0.55
        assert result == 0.55

    def test_max_cap_at_095(self) -> None:
        company = {"pe": 12.5, "pb": 1.8, "roe": 15.2, "eps": 3500.0, "eps_growth_yoy": 0.12}
        sector = _sample_sector_avg()
        data_as_of = datetime.now(timezone.utc)
        result = _calc_confidence(
            company, sector, phase1_ok=True, phase2_ok=True,
            data_as_of=data_as_of, peer_count=5,
        )
        assert result <= 0.95


# ---------------------------------------------------------------------------
# fundamental_analysis_node — integration-style tests with mocks
# ---------------------------------------------------------------------------

_NODE_MODULE = "services.app.agents.fundamental_analysis.node"


class TestNodePhase1Success:
    @pytest.mark.asyncio
    @patch(f"{_NODE_MODULE}.get_config_loader")
    @patch(f"{_NODE_MODULE}.LLMClient")
    @patch(f"{_NODE_MODULE}.load_prompt")
    @patch(f"{_NODE_MODULE}.get_peer_ratios", new_callable=AsyncMock)
    @patch(f"{_NODE_MODULE}.get_sector_average_ratios", new_callable=AsyncMock)
    @patch(f"{_NODE_MODULE}.get_sector_for_ticker")
    @patch(f"{_NODE_MODULE}.get_latest_financial_ratios", new_callable=AsyncMock)
    async def test_bctc_summary_in_output(
        self,
        mock_get_ratios,
        mock_get_sector,
        mock_get_sector_avg,
        mock_get_peers,
        mock_load_prompt,
        mock_llm_cls,
        mock_get_config,
    ) -> None:
        mock_get_ratios.return_value = (
            _sample_company_ratios_raw(),
            datetime(2026, 3, 1, tzinfo=timezone.utc),
        )
        mock_get_sector.return_value = ("Thép", ["HPG", "HSG", "NKG"])
        mock_get_sector_avg.return_value = _sample_sector_avg()
        mock_get_peers.return_value = _sample_peers()

        mock_prompt = MagicMock()
        mock_prompt.text = "test prompt"
        mock_prompt.model_key = "triage"
        mock_load_prompt.return_value = mock_prompt

        mock_config = MagicMock()
        mock_config.get_model.return_value = "gpt-4o-mini"
        mock_config.get_temperature.return_value = 0.3
        mock_get_config.return_value = mock_config

        mock_llm = AsyncMock()
        mock_llm.call.return_value = "BCTC analysis result"
        mock_llm_cls.return_value = mock_llm

        result = await fundamental_analysis_node(_base_state())

        assert result["fundamental_analysis"] is not None
        assert result["fundamental_analysis"]["bctc_summary"] == "BCTC analysis result"


class TestNodePhase2Success:
    @pytest.mark.asyncio
    @patch(f"{_NODE_MODULE}.get_config_loader")
    @patch(f"{_NODE_MODULE}.LLMClient")
    @patch(f"{_NODE_MODULE}.load_prompt")
    @patch(f"{_NODE_MODULE}.get_peer_ratios", new_callable=AsyncMock)
    @patch(f"{_NODE_MODULE}.get_sector_average_ratios", new_callable=AsyncMock)
    @patch(f"{_NODE_MODULE}.get_sector_for_ticker")
    @patch(f"{_NODE_MODULE}.get_latest_financial_ratios", new_callable=AsyncMock)
    async def test_ratio_comparison_in_output(
        self,
        mock_get_ratios,
        mock_get_sector,
        mock_get_sector_avg,
        mock_get_peers,
        mock_load_prompt,
        mock_llm_cls,
        mock_get_config,
    ) -> None:
        mock_get_ratios.return_value = (
            _sample_company_ratios_raw(),
            datetime(2026, 3, 1, tzinfo=timezone.utc),
        )
        mock_get_sector.return_value = ("Thép", ["HPG", "HSG", "NKG"])
        mock_get_sector_avg.return_value = _sample_sector_avg()
        mock_get_peers.return_value = _sample_peers()

        mock_prompt = MagicMock()
        mock_prompt.text = "test prompt"
        mock_prompt.model_key = "triage"
        mock_load_prompt.return_value = mock_prompt

        mock_config = MagicMock()
        mock_config.get_model.return_value = "gpt-4o-mini"
        mock_config.get_temperature.return_value = 0.3
        mock_get_config.return_value = mock_config

        mock_llm = AsyncMock()
        mock_llm.call.return_value = "Ratio comparison result"
        mock_llm_cls.return_value = mock_llm

        result = await fundamental_analysis_node(_base_state())

        assert result["fundamental_analysis"]["ratio_comparison"] == "Ratio comparison result"


class TestNodeCombine:
    @pytest.mark.asyncio
    @patch(f"{_NODE_MODULE}.get_config_loader")
    @patch(f"{_NODE_MODULE}.LLMClient")
    @patch(f"{_NODE_MODULE}.load_prompt")
    @patch(f"{_NODE_MODULE}.get_peer_ratios", new_callable=AsyncMock)
    @patch(f"{_NODE_MODULE}.get_sector_average_ratios", new_callable=AsyncMock)
    @patch(f"{_NODE_MODULE}.get_sector_for_ticker")
    @patch(f"{_NODE_MODULE}.get_latest_financial_ratios", new_callable=AsyncMock)
    async def test_output_has_all_fields(
        self,
        mock_get_ratios,
        mock_get_sector,
        mock_get_sector_avg,
        mock_get_peers,
        mock_load_prompt,
        mock_llm_cls,
        mock_get_config,
    ) -> None:
        mock_get_ratios.return_value = (
            _sample_company_ratios_raw(),
            datetime(2026, 3, 1, tzinfo=timezone.utc),
        )
        mock_get_sector.return_value = ("Thép", ["HPG", "HSG", "NKG"])
        mock_get_sector_avg.return_value = _sample_sector_avg()
        mock_get_peers.return_value = _sample_peers()

        mock_prompt = MagicMock()
        mock_prompt.text = "test prompt"
        mock_prompt.model_key = "triage"
        mock_load_prompt.return_value = mock_prompt

        mock_config = MagicMock()
        mock_config.get_model.return_value = "gpt-4o-mini"
        mock_config.get_temperature.return_value = 0.3
        mock_get_config.return_value = mock_config

        mock_llm = AsyncMock()
        mock_llm.call.return_value = "LLM output"
        mock_llm_cls.return_value = mock_llm

        result = await fundamental_analysis_node(_base_state())

        fa = result["fundamental_analysis"]
        assert fa is not None
        assert "signals" in fa
        assert "company_ratios" in fa
        assert "sector_ratios" in fa
        assert "confidence" in fa
        assert "data_as_of" in fa
        assert "sector_name" in fa
        assert fa["sector_name"] == "Thép"
        assert fa["signals"]["valuation"] in ("undervalued", "overvalued", "fair")
        assert fa["signals"]["profitability"] in ("strong", "weak", "average")
        assert fa["signals"]["financial_health"] == "neutral"
        assert fa["signals"]["growth"] in ("growing", "declining", "stable")
        assert 0.0 <= fa["confidence"] <= 0.95


class TestNodeVnstockFallback:
    @pytest.mark.asyncio
    @patch(f"{_NODE_MODULE}.get_config_loader")
    @patch(f"{_NODE_MODULE}.LLMClient")
    @patch(f"{_NODE_MODULE}.load_prompt")
    @patch(f"{_NODE_MODULE}.get_peer_ratios", new_callable=AsyncMock)
    @patch(f"{_NODE_MODULE}.get_sector_average_ratios", new_callable=AsyncMock)
    @patch(f"{_NODE_MODULE}.get_sector_for_ticker")
    @patch(f"{_NODE_MODULE}.save_financial_ratios", new_callable=AsyncMock)
    @patch(f"{_NODE_MODULE}.VnstockClient")
    @patch(f"{_NODE_MODULE}.get_latest_financial_ratios", new_callable=AsyncMock)
    async def test_fallback_success(
        self,
        mock_get_ratios,
        mock_vnstock_cls,
        mock_save_ratios,
        mock_get_sector,
        mock_get_sector_avg,
        mock_get_peers,
        mock_load_prompt,
        mock_llm_cls,
        mock_get_config,
    ) -> None:
        # First call: empty, second call (after vnstock save): has data
        mock_get_ratios.side_effect = [
            ({}, None),
            (_sample_company_ratios_raw(), datetime(2026, 3, 1, tzinfo=timezone.utc)),
        ]

        mock_client = MagicMock()
        mock_df = MagicMock()
        mock_df.attrs = {"data_source": "vnstock"}
        mock_client.get_financial_ratios.return_value = mock_df
        mock_vnstock_cls.return_value = mock_client
        mock_save_ratios.return_value = 1

        mock_get_sector.return_value = ("Thép", ["HPG", "HSG"])
        mock_get_sector_avg.return_value = _sample_sector_avg()
        mock_get_peers.return_value = _sample_peers()

        mock_prompt = MagicMock()
        mock_prompt.text = "prompt"
        mock_prompt.model_key = "triage"
        mock_load_prompt.return_value = mock_prompt

        mock_config = MagicMock()
        mock_config.get_model.return_value = "gpt-4o-mini"
        mock_config.get_temperature.return_value = 0.3
        mock_get_config.return_value = mock_config

        mock_llm = AsyncMock()
        mock_llm.call.return_value = "LLM output"
        mock_llm_cls.return_value = mock_llm

        result = await fundamental_analysis_node(_base_state())

        assert result["fundamental_analysis"] is not None
        mock_save_ratios.assert_awaited_once()

    @pytest.mark.asyncio
    @patch(f"{_NODE_MODULE}.get_config_loader")
    @patch(f"{_NODE_MODULE}.LLMClient")
    @patch(f"{_NODE_MODULE}.load_prompt")
    @patch(f"{_NODE_MODULE}.get_peer_ratios", new_callable=AsyncMock)
    @patch(f"{_NODE_MODULE}.get_sector_average_ratios", new_callable=AsyncMock)
    @patch(f"{_NODE_MODULE}.get_sector_for_ticker")
    @patch(f"{_NODE_MODULE}.save_financial_ratios", new_callable=AsyncMock)
    @patch(f"{_NODE_MODULE}.VnstockClient")
    @patch(f"{_NODE_MODULE}.get_latest_financial_ratios", new_callable=AsyncMock)
    async def test_fallback_also_fails(
        self,
        mock_get_ratios,
        mock_vnstock_cls,
        mock_save_ratios,
        mock_get_sector,
        mock_get_sector_avg,
        mock_get_peers,
        mock_load_prompt,
        mock_llm_cls,
        mock_get_config,
    ) -> None:
        # Both calls return empty
        mock_get_ratios.side_effect = [({}, None), ({}, None)]

        mock_client = MagicMock()
        mock_client.get_financial_ratios.side_effect = Exception("API down")
        mock_vnstock_cls.return_value = mock_client

        result = await fundamental_analysis_node(_base_state())

        assert result["fundamental_analysis"] is None
        assert "fundamental_analysis" in result["failed_agents"]


class TestNodeLLMFailurePhase1:
    @pytest.mark.asyncio
    @patch(f"{_NODE_MODULE}.get_config_loader")
    @patch(f"{_NODE_MODULE}.LLMClient")
    @patch(f"{_NODE_MODULE}.load_prompt")
    @patch(f"{_NODE_MODULE}.get_peer_ratios", new_callable=AsyncMock)
    @patch(f"{_NODE_MODULE}.get_sector_average_ratios", new_callable=AsyncMock)
    @patch(f"{_NODE_MODULE}.get_sector_for_ticker")
    @patch(f"{_NODE_MODULE}.get_latest_financial_ratios", new_callable=AsyncMock)
    async def test_phase2_still_runs(
        self,
        mock_get_ratios,
        mock_get_sector,
        mock_get_sector_avg,
        mock_get_peers,
        mock_load_prompt,
        mock_llm_cls,
        mock_get_config,
    ) -> None:
        mock_get_ratios.return_value = (
            _sample_company_ratios_raw(),
            datetime(2026, 3, 1, tzinfo=timezone.utc),
        )
        mock_get_sector.return_value = ("Thép", ["HPG", "HSG"])
        mock_get_sector_avg.return_value = _sample_sector_avg()
        mock_get_peers.return_value = _sample_peers()

        mock_prompt = MagicMock()
        mock_prompt.text = "prompt"
        mock_prompt.model_key = "triage"
        mock_load_prompt.return_value = mock_prompt

        mock_config = MagicMock()
        mock_config.get_model.return_value = "gpt-4o-mini"
        mock_config.get_temperature.return_value = 0.3
        mock_get_config.return_value = mock_config

        # Phase 1 fails, Phase 2 succeeds
        mock_llm = AsyncMock()
        mock_llm.call.side_effect = [
            LLMCallError("Rate limit"),
            "Ratio comparison OK",
        ]
        mock_llm_cls.return_value = mock_llm

        result = await fundamental_analysis_node(_base_state())

        fa = result["fundamental_analysis"]
        assert fa is not None
        assert fa["bctc_summary"] is None
        assert fa["ratio_comparison"] == "Ratio comparison OK"


class TestNodeLLMFailurePhase2:
    @pytest.mark.asyncio
    @patch(f"{_NODE_MODULE}.get_config_loader")
    @patch(f"{_NODE_MODULE}.LLMClient")
    @patch(f"{_NODE_MODULE}.load_prompt")
    @patch(f"{_NODE_MODULE}.get_peer_ratios", new_callable=AsyncMock)
    @patch(f"{_NODE_MODULE}.get_sector_average_ratios", new_callable=AsyncMock)
    @patch(f"{_NODE_MODULE}.get_sector_for_ticker")
    @patch(f"{_NODE_MODULE}.get_latest_financial_ratios", new_callable=AsyncMock)
    async def test_bctc_analysis_preserved(
        self,
        mock_get_ratios,
        mock_get_sector,
        mock_get_sector_avg,
        mock_get_peers,
        mock_load_prompt,
        mock_llm_cls,
        mock_get_config,
    ) -> None:
        mock_get_ratios.return_value = (
            _sample_company_ratios_raw(),
            datetime(2026, 3, 1, tzinfo=timezone.utc),
        )
        mock_get_sector.return_value = ("Thép", ["HPG", "HSG"])
        mock_get_sector_avg.return_value = _sample_sector_avg()
        mock_get_peers.return_value = _sample_peers()

        mock_prompt = MagicMock()
        mock_prompt.text = "prompt"
        mock_prompt.model_key = "triage"
        mock_load_prompt.return_value = mock_prompt

        mock_config = MagicMock()
        mock_config.get_model.return_value = "gpt-4o-mini"
        mock_config.get_temperature.return_value = 0.3
        mock_get_config.return_value = mock_config

        # Phase 1 succeeds, Phase 2 fails
        mock_llm = AsyncMock()
        mock_llm.call.side_effect = [
            "BCTC analysis OK",
            LLMCallError("Timeout"),
        ]
        mock_llm_cls.return_value = mock_llm

        result = await fundamental_analysis_node(_base_state())

        fa = result["fundamental_analysis"]
        assert fa is not None
        assert fa["bctc_summary"] == "BCTC analysis OK"
        assert fa["ratio_comparison"] is None


class TestNodeBothPhasesFail:
    @pytest.mark.asyncio
    @patch(f"{_NODE_MODULE}.get_config_loader")
    @patch(f"{_NODE_MODULE}.LLMClient")
    @patch(f"{_NODE_MODULE}.load_prompt")
    @patch(f"{_NODE_MODULE}.get_peer_ratios", new_callable=AsyncMock)
    @patch(f"{_NODE_MODULE}.get_sector_average_ratios", new_callable=AsyncMock)
    @patch(f"{_NODE_MODULE}.get_sector_for_ticker")
    @patch(f"{_NODE_MODULE}.get_latest_financial_ratios", new_callable=AsyncMock)
    async def test_returns_none_and_failed_agents(
        self,
        mock_get_ratios,
        mock_get_sector,
        mock_get_sector_avg,
        mock_get_peers,
        mock_load_prompt,
        mock_llm_cls,
        mock_get_config,
    ) -> None:
        mock_get_ratios.return_value = (
            _sample_company_ratios_raw(),
            datetime(2026, 3, 1, tzinfo=timezone.utc),
        )
        mock_get_sector.return_value = ("Thép", ["HPG", "HSG"])
        mock_get_sector_avg.return_value = _sample_sector_avg()
        mock_get_peers.return_value = _sample_peers()

        mock_prompt = MagicMock()
        mock_prompt.text = "prompt"
        mock_prompt.model_key = "triage"
        mock_load_prompt.return_value = mock_prompt

        mock_config = MagicMock()
        mock_config.get_model.return_value = "gpt-4o-mini"
        mock_config.get_temperature.return_value = 0.3
        mock_get_config.return_value = mock_config

        mock_llm = AsyncMock()
        mock_llm.call.side_effect = LLMCallError("Down")
        mock_llm_cls.return_value = mock_llm

        result = await fundamental_analysis_node(_base_state())

        assert result["fundamental_analysis"] is None
        assert "fundamental_analysis" in result["failed_agents"]


class TestNodeMissingSectorData:
    @pytest.mark.asyncio
    @patch(f"{_NODE_MODULE}.get_config_loader")
    @patch(f"{_NODE_MODULE}.LLMClient")
    @patch(f"{_NODE_MODULE}.load_prompt")
    @patch(f"{_NODE_MODULE}.get_peer_ratios", new_callable=AsyncMock)
    @patch(f"{_NODE_MODULE}.get_sector_average_ratios", new_callable=AsyncMock)
    @patch(f"{_NODE_MODULE}.get_sector_for_ticker")
    @patch(f"{_NODE_MODULE}.get_latest_financial_ratios", new_callable=AsyncMock)
    async def test_agent_works_without_sector_data(
        self,
        mock_get_ratios,
        mock_get_sector,
        mock_get_sector_avg,
        mock_get_peers,
        mock_load_prompt,
        mock_llm_cls,
        mock_get_config,
    ) -> None:
        mock_get_ratios.return_value = (
            _sample_company_ratios_raw(),
            datetime(2026, 3, 1, tzinfo=timezone.utc),
        )
        mock_get_sector.return_value = ("Không xác định", ["HPG"])
        mock_get_sector_avg.return_value = {
            "pe": None, "pb": None, "roe": None, "eps": None,
        }
        mock_get_peers.return_value = []

        mock_prompt = MagicMock()
        mock_prompt.text = "prompt"
        mock_prompt.model_key = "triage"
        mock_load_prompt.return_value = mock_prompt

        mock_config = MagicMock()
        mock_config.get_model.return_value = "gpt-4o-mini"
        mock_config.get_temperature.return_value = 0.3
        mock_get_config.return_value = mock_config

        mock_llm = AsyncMock()
        mock_llm.call.return_value = "Analysis"
        mock_llm_cls.return_value = mock_llm

        result = await fundamental_analysis_node(_base_state())

        fa = result["fundamental_analysis"]
        assert fa is not None
        assert fa["sector_ratios"]["pe"] is None
        assert fa["sector_ratios"]["pb"] is None


class TestNodeLogging:
    @pytest.mark.asyncio
    @patch(f"{_NODE_MODULE}.logger")
    @patch(f"{_NODE_MODULE}.get_config_loader")
    @patch(f"{_NODE_MODULE}.LLMClient")
    @patch(f"{_NODE_MODULE}.load_prompt")
    @patch(f"{_NODE_MODULE}.get_peer_ratios", new_callable=AsyncMock)
    @patch(f"{_NODE_MODULE}.get_sector_average_ratios", new_callable=AsyncMock)
    @patch(f"{_NODE_MODULE}.get_sector_for_ticker")
    @patch(f"{_NODE_MODULE}.get_latest_financial_ratios", new_callable=AsyncMock)
    async def test_structured_log_events(
        self,
        mock_get_ratios,
        mock_get_sector,
        mock_get_sector_avg,
        mock_get_peers,
        mock_load_prompt,
        mock_llm_cls,
        mock_get_config,
        mock_logger,
    ) -> None:
        mock_get_ratios.return_value = (
            _sample_company_ratios_raw(),
            datetime(2026, 3, 1, tzinfo=timezone.utc),
        )
        mock_get_sector.return_value = ("Thép", ["HPG", "HSG"])
        mock_get_sector_avg.return_value = _sample_sector_avg()
        mock_get_peers.return_value = _sample_peers()

        mock_prompt = MagicMock()
        mock_prompt.text = "prompt"
        mock_prompt.model_key = "triage"
        mock_load_prompt.return_value = mock_prompt

        mock_config = MagicMock()
        mock_config.get_model.return_value = "gpt-4o-mini"
        mock_config.get_temperature.return_value = 0.3
        mock_get_config.return_value = mock_config

        mock_llm = AsyncMock()
        mock_llm.call.return_value = "LLM output"
        mock_llm_cls.return_value = mock_llm

        await fundamental_analysis_node(_base_state())

        # Verify key log events were called
        log_events = [call.args[0] for call in mock_logger.info.call_args_list]
        assert "agent_started" in log_events
        assert "ratio_retrieval_started" in log_events
        assert "ratio_retrieval_completed" in log_events
        assert "sector_lookup_started" in log_events
        assert "sector_lookup_completed" in log_events
        assert "sector_avg_retrieval_completed" in log_events
        assert "agent_completed" in log_events
