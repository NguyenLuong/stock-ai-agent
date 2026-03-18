"""Tests for technical_analysis/indicators.yaml and pattern_recognition.yaml prompt rendering."""

from __future__ import annotations

from pathlib import Path

import pytest

from shared.llm.prompt_loader import PromptLoader, PromptRenderError, reset_prompt_loader


@pytest.fixture(autouse=True)
def _reset_loader():
    reset_prompt_loader()
    yield
    reset_prompt_loader()


@pytest.fixture()
def loader() -> PromptLoader:
    """PromptLoader pointing at real config/ directory."""
    config_dir = Path(__file__).resolve().parents[3] / "config"
    return PromptLoader(config_dir)


# ---------------------------------------------------------------------------
# Full indicator data fixtures
# ---------------------------------------------------------------------------

def _full_variables() -> dict:
    """Return a complete set of all 10 indicator variables."""
    return {
        "ticker": "HPG",
        "analysis_date": "2026-03-17",
        "rsi": {"value": 45.2, "period": 14},
        "macd": {"macd": 1.23, "signal": 0.98, "histogram": 0.25},
        "bollinger": {"upper": 30000, "middle": 28500, "lower": 27000, "current_price": 28800},
        "sma": {"sma_20": 28600, "sma_50": 27800, "sma_200": 26500},
        "volume_analysis": {"avg_volume": 15000000, "current_volume": 18000000, "ratio": 1.2},
        "atr": {"value": 850.5, "period": 14},
        "donchian": {"upper": 30500, "middle": 28750, "lower": 27000, "current_price": 28800},
        "relative_strength": {"value": 1.15},
    }


def _null_new_variables() -> dict:
    """Return variables with new indicators set to None (backward compat)."""
    return {
        "ticker": "HPG",
        "analysis_date": "2026-03-17",
        "rsi": {"value": 45.2, "period": 14},
        "macd": {"macd": 1.23, "signal": 0.98, "histogram": 0.25},
        "bollinger": {"upper": 30000, "middle": 28500, "lower": 27000, "current_price": 28800},
        "sma": {"sma_20": 28600, "sma_50": 27800, "sma_200": 26500},
        "volume_analysis": {"avg_volume": 15000000, "current_volume": 18000000, "ratio": 1.2},
        "atr": None,
        "donchian": None,
        "relative_strength": None,
    }


def _all_null_variables() -> dict:
    """Return variables with ALL indicators set to None."""
    return {
        "ticker": "HPG",
        "analysis_date": "2026-03-17",
        "rsi": None,
        "macd": None,
        "bollinger": None,
        "sma": None,
        "volume_analysis": None,
        "atr": None,
        "donchian": None,
        "relative_strength": None,
    }


# ===========================================================================
# indicators.yaml — Full render
# ===========================================================================

class TestIndicatorsFullRender:
    """AC1: Template bao gồm 10 sections cho tất cả indicators."""

    def test_render_all_10_indicators(self, loader: PromptLoader):
        result = loader.load("technical_analysis/indicators", **_full_variables())

        assert result.name == "technical_analysis_indicators"
        assert result.version == "2.0"
        assert result.model_key == "triage"

        text = result.text
        # Verify all 10 indicator sections present with values
        assert "RSI (Relative Strength Index)" in text
        assert "45.2" in text
        assert "MACD" in text
        assert "1.23" in text
        assert "Bollinger Bands" in text
        assert "30000" in text
        assert "Đường trung bình (SMA)" in text
        assert "28600" in text
        assert "Khối lượng" in text
        assert "15000000" in text
        assert "ATR (Average True Range)" in text
        assert "850.5" in text
        assert "Donchian Channel" in text
        assert "30500" in text
        assert "Relative Strength vs VN-Index" in text
        assert "1.15" in text

    def test_analysis_framework_5_dimensions(self, loader: PromptLoader):
        """AC2: Prompt yêu cầu phân tích 5 chiều."""
        result = loader.load("technical_analysis/indicators", **_full_variables())
        text = result.text

        assert "Xu hướng (Trend)" in text
        assert "Momentum" in text
        assert "Biến động (Volatility)" in text
        assert "Sức mạnh tương đối (Relative Strength)" in text
        assert "Volume Confirmation" in text

    def test_ma_cross_analysis(self, loader: PromptLoader):
        """AC2: Xu hướng — MA cross."""
        result = loader.load("technical_analysis/indicators", **_full_variables())
        text = result.text
        # sma_20 (28600) > sma_50 (27800) → Golden cross
        assert "Golden cross" in text

    def test_donchian_breakout_in_channel(self, loader: PromptLoader):
        """Donchian: price between upper and lower."""
        result = loader.load("technical_analysis/indicators", **_full_variables())
        text = result.text
        assert "Trong kênh Donchian" in text

    def test_donchian_breakout_bullish(self, loader: PromptLoader):
        """Donchian: price at upper → breakout bullish."""
        variables = _full_variables()
        variables["donchian"]["current_price"] = 30500
        result = loader.load("technical_analysis/indicators", **variables)
        assert "Breakout bullish" in result.text

    def test_donchian_breakdown_bearish(self, loader: PromptLoader):
        """Donchian: price at lower → breakdown bearish."""
        variables = _full_variables()
        variables["donchian"]["current_price"] = 27000
        result = loader.load("technical_analysis/indicators", **variables)
        assert "Breakdown bearish" in result.text

    def test_rsi_overbought(self, loader: PromptLoader):
        variables = _full_variables()
        variables["rsi"]["value"] = 75
        result = loader.load("technical_analysis/indicators", **variables)
        assert "Quá mua" in result.text

    def test_rsi_oversold(self, loader: PromptLoader):
        variables = _full_variables()
        variables["rsi"]["value"] = 25
        result = loader.load("technical_analysis/indicators", **variables)
        assert "Quá bán" in result.text

    def test_relative_strength_outperform(self, loader: PromptLoader):
        result = loader.load("technical_analysis/indicators", **_full_variables())
        assert "OUTPERFORM" in result.text

    def test_relative_strength_underperform(self, loader: PromptLoader):
        variables = _full_variables()
        variables["relative_strength"]["value"] = 0.85
        result = loader.load("technical_analysis/indicators", **variables)
        assert "UNDERPERFORM" in result.text

    def test_volume_high(self, loader: PromptLoader):
        variables = _full_variables()
        variables["volume_analysis"]["ratio"] = 2.0
        result = loader.load("technical_analysis/indicators", **variables)
        assert "Volume cao" in result.text

    def test_volume_low(self, loader: PromptLoader):
        variables = _full_variables()
        variables["volume_analysis"]["ratio"] = 0.3
        result = loader.load("technical_analysis/indicators", **variables)
        assert "Volume thấp" in result.text

    def test_macd_bullish_cross(self, loader: PromptLoader):
        """H1 fix: MACD line > Signal → Bullish cross."""
        variables = _full_variables()
        variables["macd"] = {"macd": 1.23, "signal": 0.98, "histogram": 0.25}
        result = loader.load("technical_analysis/indicators", **variables)
        assert "Bullish cross" in result.text

    def test_macd_bearish_cross(self, loader: PromptLoader):
        """H1 fix: MACD line < Signal → Bearish cross."""
        variables = _full_variables()
        variables["macd"] = {"macd": 0.50, "signal": 0.98, "histogram": -0.48}
        result = loader.load("technical_analysis/indicators", **variables)
        assert "Bearish cross" in result.text

    def test_bollinger_overextended_bullish(self, loader: PromptLoader):
        """M4: price > upper → overextended bullish."""
        variables = _full_variables()
        variables["bollinger"]["current_price"] = 31000
        result = loader.load("technical_analysis/indicators", **variables)
        assert "Overextended bullish" in result.text

    def test_bollinger_overextended_bearish(self, loader: PromptLoader):
        """M4: price < lower → overextended bearish."""
        variables = _full_variables()
        variables["bollinger"]["current_price"] = 26000
        result = loader.load("technical_analysis/indicators", **variables)
        assert "Overextended bearish" in result.text

    def test_relative_strength_equal_one(self, loader: PromptLoader):
        """L1: RS == 1.0 → NGANG BẰNG."""
        variables = _full_variables()
        variables["relative_strength"]["value"] = 1.0
        result = loader.load("technical_analysis/indicators", **variables)
        assert "NGANG BẰNG" in result.text

    def test_disclaimer_present(self, loader: PromptLoader):
        result = loader.load("technical_analysis/indicators", **_full_variables())
        assert "không phải khuyến nghị đầu tư" in result.text
        assert "KHÔNG tự tạo số liệu" in result.text


# ===========================================================================
# indicators.yaml — Null handling (AC3)
# ===========================================================================

class TestIndicatorsNullHandling:
    """AC3: Sections hiển thị 'Không có dữ liệu' khi indicator = None."""

    def test_new_indicators_null(self, loader: PromptLoader):
        """New indicators (ATR, Donchian, RS) = None → 'Không có dữ liệu'."""
        result = loader.load("technical_analysis/indicators", **_null_new_variables())
        text = result.text

        assert "Không có dữ liệu ATR" in text
        assert "Không có dữ liệu Donchian Channel" in text
        assert "Không có dữ liệu Relative Strength" in text

        # Old indicators should still render values
        assert "45.2" in text
        assert "1.23" in text

    def test_all_indicators_null(self, loader: PromptLoader):
        """All indicators = None → all sections show 'Không có dữ liệu'."""
        result = loader.load("technical_analysis/indicators", **_all_null_variables())
        text = result.text

        assert "Không có dữ liệu RSI" in text
        assert "Không có dữ liệu MACD" in text
        assert "Không có dữ liệu Bollinger Bands" in text
        assert "Không có dữ liệu SMA" in text
        assert "Không có dữ liệu khối lượng" in text
        assert "Không có dữ liệu ATR" in text
        assert "Không có dữ liệu Donchian Channel" in text
        assert "Không có dữ liệu Relative Strength" in text

    def test_no_render_error_with_all_variables(self, loader: PromptLoader):
        """No PromptRenderError when all variables provided (including new ones)."""
        result = loader.load("technical_analysis/indicators", **_full_variables())
        assert result.text  # non-empty

    def test_no_render_error_with_null_variables(self, loader: PromptLoader):
        """No PromptRenderError when indicators are None."""
        result = loader.load("technical_analysis/indicators", **_all_null_variables())
        assert result.text  # non-empty

    def test_sma_with_bollinger_null_donchian_available(self, loader: PromptLoader):
        """M2: sma has data, bollinger=None, donchian available → uses donchian.current_price."""
        variables = _full_variables()
        variables["bollinger"] = None
        result = loader.load("technical_analysis/indicators", **variables)
        text = result.text
        assert "Không có dữ liệu Bollinger Bands" in text
        # SMA section should still compare price using donchian.current_price
        assert "vs MA20" in text
        assert "28800" in text  # donchian.current_price

    def test_sma_with_bollinger_and_donchian_both_null(self, loader: PromptLoader):
        """M2: sma has data, both bollinger and donchian = None → fallback message."""
        variables = _full_variables()
        variables["bollinger"] = None
        variables["donchian"] = None
        result = loader.load("technical_analysis/indicators", **variables)
        text = result.text
        assert "Không có giá hiện tại để so sánh với MA" in text

    def test_render_error_missing_required_variable(self, loader: PromptLoader):
        """PromptRenderError when required variable (ticker) is not passed."""
        with pytest.raises(PromptRenderError):
            loader.load("technical_analysis/indicators")


# ===========================================================================
# pattern_recognition.yaml — indicators_summary
# ===========================================================================

class TestPatternRecognitionIndicatorsSummary:
    """AC2: pattern_recognition.yaml renders with optional indicators_summary."""

    def _pattern_variables(self, include_summary: bool = False) -> dict:
        variables = {
            "ticker": "HPG",
            "analysis_date": "2026-03-17",
            "ohlcv": [
                {"date": "2026-03-14", "open": 28500, "high": 29000, "low": 28200, "close": 28800, "volume": 18000000},
                {"date": "2026-03-13", "open": 28000, "high": 28600, "low": 27800, "close": 28500, "volume": 16000000},
            ],
            "support_levels": [27000, 26500],
            "resistance_levels": [30000, 31000],
            "trend": "uptrend",
        }
        if include_summary:
            variables["indicators_summary"] = "RSI: 45.2 (Trung tính), MACD: Bullish, RS: 1.15 (Outperform)"
        else:
            variables["indicators_summary"] = None
        return variables

    def test_render_with_indicators_summary(self, loader: PromptLoader):
        result = loader.load("technical_analysis/pattern_recognition", **self._pattern_variables(include_summary=True))
        text = result.text

        assert "Tóm tắt chỉ báo kỹ thuật" in text
        assert "RSI: 45.2 (Trung tính)" in text
        assert result.version == "1.1"

    def test_render_without_indicators_summary(self, loader: PromptLoader):
        result = loader.load("technical_analysis/pattern_recognition", **self._pattern_variables(include_summary=False))
        text = result.text

        assert "Tóm tắt chỉ báo kỹ thuật" not in text
        assert result.version == "1.1"

    def test_render_success_no_error(self, loader: PromptLoader):
        """No PromptRenderError when indicators_summary is provided."""
        result = loader.load("technical_analysis/pattern_recognition", **self._pattern_variables(include_summary=True))
        assert result.text
        assert result.model_key == "triage"
