"""Tests for Technical Analysis Agent node and helper functions."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from services.app.agents.technical_analysis.node import (
    MIN_OHLCV_ROWS,
    _build_indicator_dicts,
    _build_ohlcv_for_prompt,
    _calc_confidence,
    _calculate_support_resistance,
    _determine_trend,
    technical_analysis_node,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _full_raw_indicators() -> dict[str, Decimal]:
    """Sample full set of raw indicator values from DB."""
    return {
        "RSI_14": Decimal("45.2"),
        "SMA_20": Decimal("28600"),
        "SMA_50": Decimal("27800"),
        "SMA_200": Decimal("26500"),
        "MACD_LINE": Decimal("1.23"),
        "MACD_SIGNAL": Decimal("0.98"),
        "MACD_HISTOGRAM": Decimal("0.25"),
        "BB_UPPER": Decimal("30000"),
        "BB_MIDDLE": Decimal("28500"),
        "BB_LOWER": Decimal("27000"),
        "VOLUME_AVG_20": Decimal("5000000"),
        "VOLUME_CURRENT": Decimal("8000000"),
        "VOLUME_RATIO": Decimal("1.6"),
        "ATR_14": Decimal("800"),
        "DC_UPPER": Decimal("30500"),
        "DC_MIDDLE": Decimal("28750"),
        "DC_LOWER": Decimal("27000"),
        "RS_VNINDEX": Decimal("1.15"),
    }


def _make_ohlcv_df(n: int = 50) -> pd.DataFrame:
    """Create sample OHLCV DataFrame with n rows."""
    base_price = 28000.0
    rows = []
    for i in range(n):
        price = base_price + (i * 100) + ((-1) ** i * 200)
        rows.append({
            "time": datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=i),
            "open": price,
            "high": price + 500,
            "low": price - 500,
            "close": price + 100,
            "volume": 5000000.0 + i * 10000,
        })
    return pd.DataFrame(rows)


def _base_state(**overrides) -> dict:
    """Return a minimal TechnicalAnalysisState dict."""
    s: dict = {
        "ticker": "HPG",
        "analysis_type": "morning_briefing",
        "analysis_date": "2026-03-28",
    }
    s.update(overrides)
    return s


# Shared patch targets
_PATCH_GET_INDICATORS = "services.app.agents.technical_analysis.node.get_latest_indicators"
_PATCH_GET_PRICES = "services.app.agents.technical_analysis.node.get_stock_prices_df"
_PATCH_LLM_CLIENT = "services.app.agents.technical_analysis.node.LLMClient"
_PATCH_LOAD_PROMPT = "services.app.agents.technical_analysis.node.load_prompt"
_PATCH_CONFIG = "services.app.agents.technical_analysis.node.get_config_loader"
_PATCH_CALC_INDICATORS = "services.app.agents.technical_analysis.node.calculate_indicators"


def _setup_config_mock(mock_config_cls):
    cfg = MagicMock()
    cfg.get_model.return_value = "gpt-4o-mini"
    cfg.get_temperature.return_value = 0.3
    mock_config_cls.return_value = cfg
    return cfg


def _setup_prompt_mock(mock_load):
    rendered = MagicMock()
    rendered.text = "rendered prompt"
    rendered.model_key = "triage"
    mock_load.return_value = rendered
    return rendered


# ---------------------------------------------------------------------------
# Test _build_indicator_dicts
# ---------------------------------------------------------------------------


class TestBuildIndicatorDicts:
    def test_full_indicators(self) -> None:
        """Full raw indicators produce all 8 groups with correct values."""
        result = _build_indicator_dicts(_full_raw_indicators())

        assert result["rsi"] == {"value": 45.2, "period": 14}
        assert result["sma"]["sma_20"] == 28600.0
        assert result["sma"]["sma_50"] == 27800.0
        assert result["sma"]["sma_200"] == 26500.0
        assert result["macd"]["macd"] == 1.23
        assert result["macd"]["signal"] == 0.98
        assert result["macd"]["histogram"] == 0.25
        assert result["bollinger"]["upper"] == 30000.0
        assert result["volume_analysis"]["ratio"] == 1.6
        assert result["atr"] == {"value": 800.0, "period": 14}
        assert result["donchian"]["upper"] == 30500.0
        assert result["relative_strength"]["value"] == 1.15

    def test_partial_indicators(self) -> None:
        """Partial indicators — missing groups are None, present groups have values."""
        partial = {"RSI_14": Decimal("55.0"), "SMA_20": Decimal("28000")}
        result = _build_indicator_dicts(partial)

        assert result["rsi"] == {"value": 55.0, "period": 14}
        assert result["sma"] == {"sma_20": 28000.0}
        assert result["macd"] is None
        assert result["bollinger"] is None
        assert result["volume_analysis"] is None
        assert result["atr"] is None
        assert result["donchian"] is None
        assert result["relative_strength"] is None

    def test_empty_indicators(self) -> None:
        """Empty input returns all 8 groups as None."""
        result = _build_indicator_dicts({})
        for group in ("rsi", "macd", "bollinger", "sma", "volume_analysis", "atr", "donchian", "relative_strength"):
            assert result[group] is None

    def test_all_eight_keys_present(self) -> None:
        """All 8 required keys always present regardless of input."""
        result = _build_indicator_dicts({"RSI_14": Decimal("50")})
        assert len(result) == 8
        expected_keys = {"rsi", "macd", "bollinger", "sma", "volume_analysis", "atr", "donchian", "relative_strength"}
        assert set(result.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Test _determine_trend
# ---------------------------------------------------------------------------


class TestDetermineTrend:
    def test_uptrend(self) -> None:
        sma = {"sma_20": 30000, "sma_50": 28000, "sma_200": 26000}
        assert _determine_trend(sma, 31000) == "uptrend"

    def test_downtrend(self) -> None:
        sma = {"sma_20": 26000, "sma_50": 28000, "sma_200": 30000}
        assert _determine_trend(sma, 25000) == "downtrend"

    def test_sideways(self) -> None:
        sma = {"sma_20": 28000, "sma_50": 30000, "sma_200": 26000}
        assert _determine_trend(sma, 28000) == "sideways"

    def test_none_sma(self) -> None:
        assert _determine_trend(None, 28000) == "sideways"

    def test_missing_keys(self) -> None:
        sma = {"sma_20": 28000}  # Missing sma_50, sma_200
        assert _determine_trend(sma, 28000) == "sideways"


# ---------------------------------------------------------------------------
# Test _calculate_support_resistance
# ---------------------------------------------------------------------------


class TestCalculateSupportResistance:
    def test_returns_valid_levels(self) -> None:
        df = _make_ohlcv_df(50)
        support, resistance = _calculate_support_resistance(df)

        assert len(support) > 0
        assert len(resistance) > 0
        assert len(support) <= 3
        assert len(resistance) <= 3
        assert support == sorted(support)
        assert resistance == sorted(resistance)

    def test_short_data_fallback(self) -> None:
        """Data shorter than window uses fallback min/max."""
        df = _make_ohlcv_df(10)
        support, resistance = _calculate_support_resistance(df, window=20)

        assert len(support) >= 1
        assert len(resistance) >= 1
        assert support[0] == float(df["low"].min())
        assert resistance[0] == float(df["high"].max())


# ---------------------------------------------------------------------------
# Test _build_ohlcv_for_prompt
# ---------------------------------------------------------------------------


class TestBuildOhlcvForPrompt:
    def test_returns_last_30_rows(self) -> None:
        df = _make_ohlcv_df(50)
        result = _build_ohlcv_for_prompt(df, last_n=30)

        assert len(result) == 30
        assert isinstance(result[0], dict)
        assert set(result[0].keys()) == {"date", "open", "high", "low", "close", "volume"}

    def test_returns_all_when_fewer_rows(self) -> None:
        df = _make_ohlcv_df(10)
        result = _build_ohlcv_for_prompt(df, last_n=30)

        assert len(result) == 10

    def test_values_are_float(self) -> None:
        df = _make_ohlcv_df(5)
        result = _build_ohlcv_for_prompt(df)

        for row in result:
            assert isinstance(row["open"], float)
            assert isinstance(row["close"], float)
            assert isinstance(row["volume"], float)


# ---------------------------------------------------------------------------
# Test _calc_confidence
# ---------------------------------------------------------------------------


class TestCalcConfidence:
    def test_base_score(self) -> None:
        """Empty indicators, few rows → base score only."""
        empty_dicts = {g: None for g in ("rsi", "macd", "bollinger", "sma", "volume_analysis", "atr", "donchian", "relative_strength")}
        score = _calc_confidence(empty_dicts, 30, phase1_ok=False, phase2_ok=False)
        assert score == 0.55  # base 0.50 + 0.05 (30 rows)

    def test_max_score_capped(self) -> None:
        """Full indicators + large data + fresh → capped at 0.95."""
        full = _build_indicator_dicts(_full_raw_indicators())
        score = _calc_confidence(
            full, 300,
            phase1_ok=True, phase2_ok=True,
            data_as_of=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        assert score <= 0.95

    def test_ohlcv_depth_scoring(self) -> None:
        empty_dicts = {g: None for g in ("rsi", "macd", "bollinger", "sma", "volume_analysis", "atr", "donchian", "relative_strength")}
        s30 = _calc_confidence(empty_dicts, 30, phase1_ok=False, phase2_ok=False)
        s50 = _calc_confidence(empty_dicts, 50, phase1_ok=False, phase2_ok=False)
        s200 = _calc_confidence(empty_dicts, 200, phase1_ok=False, phase2_ok=False)
        assert s30 < s50 < s200

    def test_phase_success_scoring(self) -> None:
        empty_dicts = {g: None for g in ("rsi", "macd", "bollinger", "sma", "volume_analysis", "atr", "donchian", "relative_strength")}
        both_fail = _calc_confidence(empty_dicts, 30, phase1_ok=False, phase2_ok=False)
        one_ok = _calc_confidence(empty_dicts, 30, phase1_ok=True, phase2_ok=False)
        both_ok = _calc_confidence(empty_dicts, 30, phase1_ok=True, phase2_ok=True)
        assert both_fail < one_ok < both_ok


# ---------------------------------------------------------------------------
# Test technical_analysis_node — Phase 1 success
# ---------------------------------------------------------------------------


class TestNodePhase1Success:
    @patch(_PATCH_CONFIG)
    @patch(_PATCH_LOAD_PROMPT)
    @patch(_PATCH_LLM_CLIENT)
    @patch(_PATCH_GET_PRICES)
    @patch(_PATCH_GET_INDICATORS)
    async def test_phase1_produces_indicator_summary(
        self, mock_indicators, mock_prices, mock_llm_cls, mock_load, mock_config,
    ) -> None:
        dt = datetime(2026, 3, 28, tzinfo=timezone.utc)
        mock_indicators.return_value = (_full_raw_indicators(), dt)
        mock_prices.return_value = _make_ohlcv_df(50)

        mock_llm = AsyncMock()
        mock_llm.call.return_value = "RSI at 45 — neutral momentum"
        mock_llm_cls.return_value = mock_llm

        _setup_prompt_mock(mock_load)
        _setup_config_mock(mock_config)

        result = await technical_analysis_node(_base_state())

        assert result["technical_analysis"] is not None
        assert result["technical_analysis"]["indicator_summary"] == "RSI at 45 — neutral momentum"


# ---------------------------------------------------------------------------
# Test technical_analysis_node — Phase 2 success
# ---------------------------------------------------------------------------


class TestNodePhase2Success:
    @patch(_PATCH_CONFIG)
    @patch(_PATCH_LOAD_PROMPT)
    @patch(_PATCH_LLM_CLIENT)
    @patch(_PATCH_GET_PRICES)
    @patch(_PATCH_GET_INDICATORS)
    async def test_phase2_produces_pattern_summary(
        self, mock_indicators, mock_prices, mock_llm_cls, mock_load, mock_config,
    ) -> None:
        dt = datetime(2026, 3, 28, tzinfo=timezone.utc)
        mock_indicators.return_value = (_full_raw_indicators(), dt)
        mock_prices.return_value = _make_ohlcv_df(50)

        mock_llm = AsyncMock()
        # Phase 1 returns indicator summary, Phase 2 returns pattern summary
        mock_llm.call.side_effect = [
            "RSI at 45 — neutral",
            "Ascending triangle forming",
        ]
        mock_llm_cls.return_value = mock_llm

        _setup_prompt_mock(mock_load)
        _setup_config_mock(mock_config)

        result = await technical_analysis_node(_base_state())

        ta = result["technical_analysis"]
        assert ta["pattern_summary"] == "Ascending triangle forming"


# ---------------------------------------------------------------------------
# Test technical_analysis_node — combine output
# ---------------------------------------------------------------------------


class TestNodeCombine:
    @patch(_PATCH_CONFIG)
    @patch(_PATCH_LOAD_PROMPT)
    @patch(_PATCH_LLM_CLIENT)
    @patch(_PATCH_GET_PRICES)
    @patch(_PATCH_GET_INDICATORS)
    async def test_output_has_all_required_fields(
        self, mock_indicators, mock_prices, mock_llm_cls, mock_load, mock_config,
    ) -> None:
        dt = datetime(2026, 3, 28, tzinfo=timezone.utc)
        mock_indicators.return_value = (_full_raw_indicators(), dt)
        mock_prices.return_value = _make_ohlcv_df(50)

        mock_llm = AsyncMock()
        mock_llm.call.return_value = "analysis text"
        mock_llm_cls.return_value = mock_llm

        _setup_prompt_mock(mock_load)
        _setup_config_mock(mock_config)

        result = await technical_analysis_node(_base_state())

        ta = result["technical_analysis"]
        assert ta is not None
        assert "indicator_summary" in ta
        assert "pattern_summary" in ta
        assert "signals" in ta
        assert "support_levels" in ta
        assert "resistance_levels" in ta
        assert "confidence" in ta
        assert "data_as_of" in ta
        assert "data_source" in ta

        signals = ta["signals"]
        assert signals["trend"] in ("uptrend", "downtrend", "sideways")
        assert signals["momentum"] in ("bullish", "bearish", "neutral")
        assert signals["volatility"] in ("high", "low", "normal")
        assert isinstance(signals["volume_confirmation"], bool)

        assert isinstance(ta["support_levels"], list)
        assert isinstance(ta["resistance_levels"], list)
        assert 0 <= ta["confidence"] <= 0.95


# ---------------------------------------------------------------------------
# Test insufficient data (AC #3)
# ---------------------------------------------------------------------------


class TestNodeInsufficientData:
    @patch(_PATCH_CONFIG)
    @patch(_PATCH_LLM_CLIENT)
    @patch(_PATCH_GET_PRICES)
    @patch(_PATCH_GET_INDICATORS)
    async def test_returns_graceful_message(
        self, mock_indicators, mock_prices, mock_llm_cls, mock_config,
    ) -> None:
        mock_indicators.return_value = ({}, None)
        mock_prices.return_value = _make_ohlcv_df(10)  # < 30 rows

        mock_llm_cls.return_value = AsyncMock()
        _setup_config_mock(mock_config)

        result = await technical_analysis_node(_base_state())

        assert result["technical_analysis"] is None
        assert "Không đủ dữ liệu" in result["error"]
        assert "technical_analysis" in result["failed_agents"]


# ---------------------------------------------------------------------------
# Test no indicators in DB — on-the-fly recalculation
# ---------------------------------------------------------------------------


class TestNodeNoIndicators:
    @patch(_PATCH_CALC_INDICATORS)
    @patch(_PATCH_CONFIG)
    @patch(_PATCH_LOAD_PROMPT)
    @patch(_PATCH_LLM_CLIENT)
    @patch(_PATCH_GET_PRICES)
    @patch(_PATCH_GET_INDICATORS)
    async def test_recalculates_on_the_fly(
        self, mock_indicators, mock_prices, mock_llm_cls, mock_load, mock_config,
        mock_calc,
    ) -> None:
        mock_indicators.return_value = ({}, None)  # No indicators in DB
        mock_prices.return_value = _make_ohlcv_df(50)

        # Simulate calculate_indicators returning records
        mock_record = MagicMock()
        mock_record.indicator_name = "RSI_14"
        mock_record.indicator_value = Decimal("55.0")
        mock_record.data_as_of = datetime(2026, 3, 28, tzinfo=timezone.utc)
        mock_calc.return_value = [mock_record]

        mock_llm = AsyncMock()
        mock_llm.call.return_value = "analysis text"
        mock_llm_cls.return_value = mock_llm

        _setup_prompt_mock(mock_load)
        _setup_config_mock(mock_config)

        result = await technical_analysis_node(_base_state())

        mock_calc.assert_called_once()
        assert result["technical_analysis"] is not None


# ---------------------------------------------------------------------------
# Test LLM failure Phase 1 — Phase 2 still runs
# ---------------------------------------------------------------------------


class TestNodePhase1Failure:
    @patch(_PATCH_CONFIG)
    @patch(_PATCH_LOAD_PROMPT)
    @patch(_PATCH_LLM_CLIENT)
    @patch(_PATCH_GET_PRICES)
    @patch(_PATCH_GET_INDICATORS)
    async def test_phase2_still_runs(
        self, mock_indicators, mock_prices, mock_llm_cls, mock_load, mock_config,
    ) -> None:
        from shared.llm.client import LLMCallError

        dt = datetime(2026, 3, 28, tzinfo=timezone.utc)
        mock_indicators.return_value = (_full_raw_indicators(), dt)
        mock_prices.return_value = _make_ohlcv_df(50)

        mock_llm = AsyncMock()
        # Phase 1 fails, Phase 2 succeeds
        mock_llm.call.side_effect = [
            LLMCallError("Rate limit"),
            "Pattern detected: breakout",
        ]
        mock_llm_cls.return_value = mock_llm

        _setup_prompt_mock(mock_load)
        _setup_config_mock(mock_config)

        result = await technical_analysis_node(_base_state())

        ta = result["technical_analysis"]
        assert ta is not None
        assert ta["indicator_summary"] is None
        assert ta["pattern_summary"] == "Pattern detected: breakout"


# ---------------------------------------------------------------------------
# Test LLM failure Phase 2 — indicator analysis preserved
# ---------------------------------------------------------------------------


class TestNodePhase2Failure:
    @patch(_PATCH_CONFIG)
    @patch(_PATCH_LOAD_PROMPT)
    @patch(_PATCH_LLM_CLIENT)
    @patch(_PATCH_GET_PRICES)
    @patch(_PATCH_GET_INDICATORS)
    async def test_indicator_analysis_preserved(
        self, mock_indicators, mock_prices, mock_llm_cls, mock_load, mock_config,
    ) -> None:
        from shared.llm.client import LLMCallError

        dt = datetime(2026, 3, 28, tzinfo=timezone.utc)
        mock_indicators.return_value = (_full_raw_indicators(), dt)
        mock_prices.return_value = _make_ohlcv_df(50)

        mock_llm = AsyncMock()
        # Phase 1 succeeds, Phase 2 fails
        mock_llm.call.side_effect = [
            "RSI neutral",
            LLMCallError("Timeout"),
        ]
        mock_llm_cls.return_value = mock_llm

        _setup_prompt_mock(mock_load)
        _setup_config_mock(mock_config)

        result = await technical_analysis_node(_base_state())

        ta = result["technical_analysis"]
        assert ta is not None
        assert ta["indicator_summary"] == "RSI neutral"
        assert ta["pattern_summary"] is None


# ---------------------------------------------------------------------------
# Test both phases fail
# ---------------------------------------------------------------------------


class TestNodeBothPhasesFail:
    @patch(_PATCH_CONFIG)
    @patch(_PATCH_LOAD_PROMPT)
    @patch(_PATCH_LLM_CLIENT)
    @patch(_PATCH_GET_PRICES)
    @patch(_PATCH_GET_INDICATORS)
    async def test_returns_none_and_failed_agents(
        self, mock_indicators, mock_prices, mock_llm_cls, mock_load, mock_config,
    ) -> None:
        from shared.llm.client import LLMCallError

        dt = datetime(2026, 3, 28, tzinfo=timezone.utc)
        mock_indicators.return_value = (_full_raw_indicators(), dt)
        mock_prices.return_value = _make_ohlcv_df(50)

        mock_llm = AsyncMock()
        mock_llm.call.side_effect = LLMCallError("All retries failed")
        mock_llm_cls.return_value = mock_llm

        _setup_prompt_mock(mock_load)
        _setup_config_mock(mock_config)

        result = await technical_analysis_node(_base_state())

        assert result["technical_analysis"] is None
        assert "technical_analysis" in result["failed_agents"]


# ---------------------------------------------------------------------------
# Test logging events
# ---------------------------------------------------------------------------


class TestNodeLogging:
    @patch(_PATCH_CONFIG)
    @patch(_PATCH_LOAD_PROMPT)
    @patch(_PATCH_LLM_CLIENT)
    @patch(_PATCH_GET_PRICES)
    @patch(_PATCH_GET_INDICATORS)
    @patch("services.app.agents.technical_analysis.node.logger")
    async def test_logs_key_events(
        self, mock_logger, mock_indicators, mock_prices, mock_llm_cls, mock_load,
        mock_config,
    ) -> None:
        dt = datetime(2026, 3, 28, tzinfo=timezone.utc)
        mock_indicators.return_value = (_full_raw_indicators(), dt)
        mock_prices.return_value = _make_ohlcv_df(50)

        mock_llm = AsyncMock()
        mock_llm.call.return_value = "analysis"
        mock_llm_cls.return_value = mock_llm

        _setup_prompt_mock(mock_load)
        _setup_config_mock(mock_config)

        await technical_analysis_node(_base_state())

        log_events = [call.args[0] for call in mock_logger.info.call_args_list]
        assert "agent_started" in log_events
        assert "indicator_retrieval_started" in log_events
        assert "indicator_retrieval_completed" in log_events
        assert "ohlcv_retrieval_started" in log_events
        assert "ohlcv_retrieval_completed" in log_events
        assert "agent_completed" in log_events
