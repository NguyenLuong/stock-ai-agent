"""Unit tests for indicator_manager — orchestration, trading day, error handling."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from services.crawler.market_data.indicator_calculator import IndicatorRecord
from services.crawler.market_data.indicator_manager import (
    IndicatorCalculationResult,
    run_indicator_calculation,
)


def _make_ohlcv_df(n: int = 50) -> pd.DataFrame:
    """Create a dummy OHLCV DataFrame."""
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "time": dates,
        "open": [100.0 + i * 0.1 for i in range(n)],
        "high": [105.0 + i * 0.1 for i in range(n)],
        "low": [95.0 + i * 0.1 for i in range(n)],
        "close": [102.0 + i * 0.1 for i in range(n)],
        "volume": [1000000.0 + i * 1000 for i in range(n)],
    })


def _make_indicator_records() -> list[IndicatorRecord]:
    """Create sample indicator records."""
    data_as_of = datetime(2026, 3, 14, tzinfo=timezone.utc)
    return [
        IndicatorRecord("SMA_20", Decimal("102.5"), data_as_of),
        IndicatorRecord("RSI_14", Decimal("55.3"), data_as_of),
    ]


class TestRunIndicatorCalculation:
    """Tests for run_indicator_calculation orchestrator."""

    @patch("services.crawler.market_data.indicator_manager.load_ticker_config")
    @patch("services.crawler.market_data.indicator_manager.now_utc")
    async def test_non_trading_day_skips(self, mock_now, mock_load_config) -> None:
        """Non-trading day skips all calculation."""
        # Saturday
        mock_now.return_value = datetime(2026, 3, 14, 10, 0, tzinfo=timezone.utc)

        from services.crawler.market_data.ticker_config import TickerConfig
        mock_load_config.return_value = TickerConfig(
            tickers=["HPG", "VNM"], total_count=2, enabled_groups=1, holidays=[],
        )

        result = await run_indicator_calculation()

        assert isinstance(result, IndicatorCalculationResult)
        assert result.skipped_reason == "non_trading_day"
        assert result.success_count == 0

    @patch("services.crawler.market_data.indicator_manager.load_ticker_config")
    @patch("services.crawler.market_data.indicator_manager.now_utc")
    async def test_empty_tickers_returns_early(self, mock_now, mock_load_config) -> None:
        """No tickers configured returns empty result."""
        mock_now.return_value = datetime(2026, 3, 16, 10, 0, tzinfo=timezone.utc)  # Monday

        from services.crawler.market_data.ticker_config import TickerConfig
        mock_load_config.return_value = TickerConfig(
            tickers=[], total_count=0, enabled_groups=0, holidays=[],
        )

        result = await run_indicator_calculation()

        assert result.total_tickers == 0
        assert result.skipped_reason is None

    @patch("services.crawler.market_data.indicator_manager.save_technical_indicators", new_callable=AsyncMock)
    @patch("services.crawler.market_data.indicator_manager.calculate_indicators")
    @patch("services.crawler.market_data.indicator_manager.get_stock_prices_df", new_callable=AsyncMock)
    @patch("services.crawler.market_data.indicator_manager.load_ticker_config")
    @patch("services.crawler.market_data.indicator_manager.now_utc")
    async def test_successful_calculation(
        self, mock_now, mock_load_config, mock_get_df, mock_calc, mock_save,
    ) -> None:
        """Successful calculation for all tickers."""
        mock_now.return_value = datetime(2026, 3, 16, 10, 0, tzinfo=timezone.utc)

        from services.crawler.market_data.ticker_config import TickerConfig
        mock_load_config.return_value = TickerConfig(
            tickers=["HPG"], total_count=1, enabled_groups=1, holidays=[],
        )

        df = _make_ohlcv_df()
        mock_get_df.return_value = df
        mock_calc.return_value = _make_indicator_records()
        mock_save.return_value = 2

        result = await run_indicator_calculation()

        assert result.success_count == 1
        assert result.failed_count == 0
        assert result.rows_inserted == 2
        assert result.indicators_calculated == 2
        mock_save.assert_awaited_once()

    @patch("services.crawler.market_data.indicator_manager.save_technical_indicators", new_callable=AsyncMock)
    @patch("services.crawler.market_data.indicator_manager.calculate_indicators")
    @patch("services.crawler.market_data.indicator_manager.get_stock_prices_df", new_callable=AsyncMock)
    @patch("services.crawler.market_data.indicator_manager.load_ticker_config")
    @patch("services.crawler.market_data.indicator_manager.now_utc")
    async def test_ticker_error_continues(
        self, mock_now, mock_load_config, mock_get_df, mock_calc, mock_save,
    ) -> None:
        """Error on one ticker doesn't crash — continues with next."""
        mock_now.return_value = datetime(2026, 3, 16, 10, 0, tzinfo=timezone.utc)

        from services.crawler.market_data.ticker_config import TickerConfig
        mock_load_config.return_value = TickerConfig(
            tickers=["BAD", "HPG"], total_count=2, enabled_groups=1, holidays=[],
        )

        df = _make_ohlcv_df()
        # First call (VNINDEX) returns empty, second (BAD) errors, third (HPG) succeeds
        mock_get_df.side_effect = [
            pd.DataFrame(),  # VN-Index
            Exception("DB error"),  # BAD
            df,  # HPG
        ]
        mock_calc.return_value = _make_indicator_records()
        mock_save.return_value = 2

        result = await run_indicator_calculation()

        assert result.success_count == 1
        assert result.failed_count == 1
        assert len(result.errors) == 1
        assert result.errors[0]["ticker"] == "BAD"

    @patch("services.crawler.market_data.indicator_manager.get_stock_prices_df", new_callable=AsyncMock)
    @patch("services.crawler.market_data.indicator_manager.load_ticker_config")
    @patch("services.crawler.market_data.indicator_manager.now_utc")
    async def test_empty_ohlcv_skips_ticker(
        self, mock_now, mock_load_config, mock_get_df,
    ) -> None:
        """Ticker with no OHLCV data is skipped (no error)."""
        mock_now.return_value = datetime(2026, 3, 16, 10, 0, tzinfo=timezone.utc)

        from services.crawler.market_data.ticker_config import TickerConfig
        mock_load_config.return_value = TickerConfig(
            tickers=["HPG"], total_count=1, enabled_groups=1, holidays=[],
        )

        # VN-Index empty, HPG empty
        mock_get_df.return_value = pd.DataFrame()

        result = await run_indicator_calculation()

        assert result.failed_count == 0
        assert result.rows_inserted == 0

    @patch("services.crawler.market_data.indicator_manager.save_technical_indicators", new_callable=AsyncMock)
    @patch("services.crawler.market_data.indicator_manager.calculate_indicators")
    @patch("services.crawler.market_data.indicator_manager.get_stock_prices_df", new_callable=AsyncMock)
    @patch("services.crawler.market_data.indicator_manager.load_ticker_config")
    @patch("services.crawler.market_data.indicator_manager.now_utc")
    @patch("services.crawler.market_data.indicator_manager.asyncio.sleep", new_callable=AsyncMock)
    async def test_rate_limiting_between_tickers(
        self, mock_sleep, mock_now, mock_load_config, mock_get_df, mock_calc, mock_save,
    ) -> None:
        """asyncio.sleep(0.1) called between tickers."""
        mock_now.return_value = datetime(2026, 3, 16, 10, 0, tzinfo=timezone.utc)

        from services.crawler.market_data.ticker_config import TickerConfig
        mock_load_config.return_value = TickerConfig(
            tickers=["HPG", "VNM"], total_count=2, enabled_groups=1, holidays=[],
        )

        df = _make_ohlcv_df()
        mock_get_df.return_value = df
        mock_calc.return_value = _make_indicator_records()
        mock_save.return_value = 2

        await run_indicator_calculation()

        mock_sleep.assert_any_call(0.1)

    @patch("services.crawler.market_data.indicator_manager.save_technical_indicators", new_callable=AsyncMock)
    @patch("services.crawler.market_data.indicator_manager.calculate_indicators")
    @patch("services.crawler.market_data.indicator_manager.get_stock_prices_df", new_callable=AsyncMock)
    @patch("services.crawler.market_data.indicator_manager.load_ticker_config")
    @patch("services.crawler.market_data.indicator_manager.now_utc")
    async def test_vnindex_fetched_once(
        self, mock_now, mock_load_config, mock_get_df, mock_calc, mock_save,
    ) -> None:
        """VN-Index data is fetched once and passed to all tickers."""
        mock_now.return_value = datetime(2026, 3, 16, 10, 0, tzinfo=timezone.utc)

        from services.crawler.market_data.ticker_config import TickerConfig
        mock_load_config.return_value = TickerConfig(
            tickers=["HPG", "VNM"], total_count=2, enabled_groups=1, holidays=[],
        )

        vnindex_df = _make_ohlcv_df(100)
        ticker_df = _make_ohlcv_df(50)
        # First call is VNINDEX, then HPG, then VNM
        mock_get_df.side_effect = [vnindex_df, ticker_df, ticker_df]
        mock_calc.return_value = _make_indicator_records()
        mock_save.return_value = 2

        await run_indicator_calculation()

        # calculate_indicators called with vnindex df for each ticker
        assert mock_calc.call_count == 2
        for call in mock_calc.call_args_list:
            assert call[1].get("df_vnindex") is not None or call[0][2] is not None

    def test_result_dataclass_fields(self) -> None:
        """IndicatorCalculationResult has all required fields."""
        result = IndicatorCalculationResult(
            total_tickers=10,
            success_count=8,
            failed_count=2,
            rows_inserted=150,
            indicators_calculated=100,
            duration_seconds=5.0,
            skipped_reason=None,
        )
        assert hasattr(result, "total_tickers")
        assert hasattr(result, "success_count")
        assert hasattr(result, "failed_count")
        assert hasattr(result, "rows_inserted")
        assert hasattr(result, "indicators_calculated")
        assert hasattr(result, "duration_seconds")
        assert hasattr(result, "skipped_reason")
        assert hasattr(result, "errors")
        assert result.errors == []
