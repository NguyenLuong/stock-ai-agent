"""Unit tests for indicator_calculator — calculation logic, data sufficiency, RS calc."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import patch

import pandas as pd
import pytest

from services.crawler.market_data.indicator_calculator import (
    MIN_PERIODS,
    IndicatorRecord,
    _calculate_relative_strength,
    calculate_indicators,
)


def _make_ohlcv_df(n: int = 250) -> pd.DataFrame:
    """Create a dummy OHLCV DataFrame with n rows."""
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "time": dates,
        "open": [100.0 + i * 0.1 for i in range(n)],
        "high": [105.0 + i * 0.1 for i in range(n)],
        "low": [95.0 + i * 0.1 for i in range(n)],
        "close": [102.0 + i * 0.1 for i in range(n)],
        "volume": [1000000.0 + i * 1000 for i in range(n)],
    })


def _make_vnindex_df(n: int = 250) -> pd.DataFrame:
    """Create a dummy VN-Index DataFrame."""
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "time": dates,
        "open": [1200.0 + i * 0.5 for i in range(n)],
        "high": [1210.0 + i * 0.5 for i in range(n)],
        "low": [1190.0 + i * 0.5 for i in range(n)],
        "close": [1205.0 + i * 0.5 for i in range(n)],
        "volume": [500000000.0 for _ in range(n)],
    })


class TestCalculateIndicators:
    """Tests for calculate_indicators function."""

    def test_returns_all_19_indicators_with_sufficient_data(self) -> None:
        """With 250 rows + VN-Index data, all 19 indicator values are calculated."""
        df = _make_ohlcv_df(250)
        df_vnindex = _make_vnindex_df(250)

        records = calculate_indicators("HPG", df, df_vnindex)

        names = {r.indicator_name for r in records}
        expected = {
            "SMA_20", "SMA_50", "SMA_200",
            "RSI_14",
            "MACD_LINE", "MACD_SIGNAL", "MACD_HISTOGRAM",
            "VOLUME_AVG_20", "VOLUME_CURRENT", "VOLUME_RATIO",
            "ATR_14",
            "BB_UPPER", "BB_MIDDLE", "BB_LOWER",
            "DC_UPPER", "DC_LOWER", "DC_MIDDLE",
            "RS_VNINDEX",
        }
        assert expected.issubset(names), f"Missing: {expected - names}"

    def test_returns_indicator_records(self) -> None:
        """Each result is an IndicatorRecord with correct types."""
        df = _make_ohlcv_df(50)
        records = calculate_indicators("HPG", df)

        assert len(records) > 0
        for rec in records:
            assert isinstance(rec, IndicatorRecord)
            assert isinstance(rec.indicator_name, str)
            assert isinstance(rec.indicator_value, Decimal)
            assert rec.data_as_of is not None

    def test_data_as_of_is_last_date(self) -> None:
        """data_as_of matches the last row's time."""
        df = _make_ohlcv_df(50)
        records = calculate_indicators("HPG", df)

        last_date = df["time"].iloc[-1]
        for rec in records:
            assert rec.data_as_of == last_date.to_pydatetime()

    def test_empty_dataframe_returns_empty(self) -> None:
        """Empty input returns empty list."""
        df = pd.DataFrame()
        records = calculate_indicators("HPG", df)
        assert records == []

    def test_insufficient_data_skips_sma200(self) -> None:
        """With only 50 rows, SMA_200 is not calculated."""
        df = _make_ohlcv_df(50)
        records = calculate_indicators("HPG", df)

        names = {r.indicator_name for r in records}
        assert "SMA_200" not in names
        assert "SMA_20" in names
        assert "SMA_50" in names

    def test_insufficient_data_for_all_with_10_rows(self) -> None:
        """With only 10 rows, no SMA/RSI/MACD indicators are possible."""
        df = _make_ohlcv_df(10)
        records = calculate_indicators("HPG", df)

        names = {r.indicator_name for r in records}
        assert "SMA_20" not in names
        assert "SMA_50" not in names
        assert "RSI_14" not in names
        assert "MACD_LINE" not in names

    @patch("services.crawler.market_data.indicator_calculator.logger")
    def test_logs_warning_for_insufficient_data(self, mock_logger) -> None:
        """Warning logged when data insufficient for an indicator."""
        df = _make_ohlcv_df(10)
        calculate_indicators("HPG", df)

        # Should have multiple warnings for insufficient data
        assert mock_logger.warning.call_count > 0

    def test_no_vnindex_skips_rs(self) -> None:
        """Without VN-Index data, RS_VNINDEX is not calculated."""
        df = _make_ohlcv_df(50)
        records = calculate_indicators("HPG", df, df_vnindex=None)

        names = {r.indicator_name for r in records}
        assert "RS_VNINDEX" not in names

    def test_vnindex_empty_df_skips_rs(self) -> None:
        """With empty VN-Index DataFrame, RS_VNINDEX is not calculated."""
        df = _make_ohlcv_df(50)
        df_vnindex = _make_vnindex_df(5)  # Too few rows

        records = calculate_indicators("HPG", df, df_vnindex)

        names = {r.indicator_name for r in records}
        assert "RS_VNINDEX" not in names

    def test_rsi_in_valid_range(self) -> None:
        """RSI value should be between 0 and 100."""
        df = _make_ohlcv_df(50)
        records = calculate_indicators("HPG", df)

        rsi_records = [r for r in records if r.indicator_name == "RSI_14"]
        assert len(rsi_records) == 1
        rsi_val = float(rsi_records[0].indicator_value)
        assert 0 <= rsi_val <= 100

    def test_volume_ratio_calculated(self) -> None:
        """Volume ratio = current / avg_20."""
        df = _make_ohlcv_df(50)
        records = calculate_indicators("HPG", df)

        vol_map = {r.indicator_name: float(r.indicator_value) for r in records
                   if r.indicator_name.startswith("VOLUME_")}
        assert "VOLUME_AVG_20" in vol_map
        assert "VOLUME_CURRENT" in vol_map
        assert "VOLUME_RATIO" in vol_map
        expected_ratio = vol_map["VOLUME_CURRENT"] / vol_map["VOLUME_AVG_20"]
        assert abs(vol_map["VOLUME_RATIO"] - expected_ratio) < 0.01


class TestCalculateRelativeStrength:
    """Tests for _calculate_relative_strength helper."""

    def test_rs_calculation(self) -> None:
        """RS = (ticker_return) / (vnindex_return)."""
        df_ticker = _make_ohlcv_df(50)
        df_vnindex = _make_vnindex_df(50)

        rs = _calculate_relative_strength(df_ticker, df_vnindex, periods=20)
        assert rs is not None
        assert rs > 0

    def test_insufficient_ticker_data(self) -> None:
        """Returns None if ticker data < periods + 1."""
        df_ticker = _make_ohlcv_df(10)
        df_vnindex = _make_vnindex_df(50)

        rs = _calculate_relative_strength(df_ticker, df_vnindex, periods=20)
        assert rs is None

    def test_insufficient_vnindex_data(self) -> None:
        """Returns None if VN-Index data < periods + 1."""
        df_ticker = _make_ohlcv_df(50)
        df_vnindex = _make_vnindex_df(10)

        rs = _calculate_relative_strength(df_ticker, df_vnindex, periods=20)
        assert rs is None

    def test_zero_price_returns_none(self) -> None:
        """Returns None if past price is zero."""
        df_ticker = _make_ohlcv_df(50)
        df_vnindex = _make_vnindex_df(50)
        # Set past price to zero
        df_ticker.iloc[-21, df_ticker.columns.get_loc("close")] = 0.0

        rs = _calculate_relative_strength(df_ticker, df_vnindex, periods=20)
        assert rs is None


class TestMinPeriods:
    """Tests for MIN_PERIODS configuration."""

    def test_all_indicators_have_min_periods(self) -> None:
        """All expected indicators have min_periods defined."""
        expected_keys = {
            "SMA_20", "SMA_50", "SMA_200", "RSI_14", "MACD",
            "VOLUME", "ATR_14", "BOLLINGER", "DONCHIAN", "RS_VNINDEX",
        }
        assert set(MIN_PERIODS.keys()) == expected_keys

    def test_sma200_requires_200(self) -> None:
        assert MIN_PERIODS["SMA_200"] == 200

    def test_rsi_requires_15(self) -> None:
        assert MIN_PERIODS["RSI_14"] == 15

    def test_macd_requires_35(self) -> None:
        assert MIN_PERIODS["MACD"] == 35
