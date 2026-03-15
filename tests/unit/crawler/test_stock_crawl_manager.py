"""Unit tests for stock_crawl_manager — orchestration, initial vs incremental, error handling, rate limiting."""

from __future__ import annotations

from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from services.crawler.market_data.stock_crawl_manager import (
    INITIAL_THRESHOLD,
    VN_FIXED_HOLIDAYS,
    StockCrawlResult,
    is_trading_day,
    run_stock_crawl,
)


class TestIsTradingDay:
    """Tests for is_trading_day helper."""

    def test_weekday_is_trading_day(self) -> None:
        """Monday-Friday are trading days (no holidays)."""
        # 2026-03-16 is Monday
        assert is_trading_day(date(2026, 3, 16), holidays=[]) is True

    def test_saturday_not_trading_day(self) -> None:
        """Saturday is not a trading day."""
        # 2026-03-14 is Saturday
        assert is_trading_day(date(2026, 3, 14), holidays=[]) is False

    def test_sunday_not_trading_day(self) -> None:
        """Sunday is not a trading day."""
        # 2026-03-15 is Sunday
        assert is_trading_day(date(2026, 3, 15), holidays=[]) is False

    def test_fixed_holiday_not_trading_day(self) -> None:
        """Fixed holidays (1/1, 30/4, 1/5, 2/9) are not trading days."""
        # 2026-01-01 is Thursday (New Year)
        assert is_trading_day(date(2026, 1, 1), holidays=[]) is False
        # 2026-04-30
        assert is_trading_day(date(2026, 4, 30), holidays=[]) is False
        # 2026-05-01
        assert is_trading_day(date(2026, 5, 1), holidays=[]) is False
        # 2026-09-02
        assert is_trading_day(date(2026, 9, 2), holidays=[]) is False

    def test_variable_holiday_not_trading_day(self) -> None:
        """Variable holidays from config are not trading days."""
        holidays = [date(2026, 2, 16), date(2026, 2, 17)]
        assert is_trading_day(date(2026, 2, 16), holidays=holidays) is False
        assert is_trading_day(date(2026, 2, 18), holidays=holidays) is True

    def test_fixed_holidays_match_spec(self) -> None:
        """VN_FIXED_HOLIDAYS contains expected entries."""
        assert (1, 1) in VN_FIXED_HOLIDAYS
        assert (4, 30) in VN_FIXED_HOLIDAYS
        assert (5, 1) in VN_FIXED_HOLIDAYS
        assert (9, 2) in VN_FIXED_HOLIDAYS

    @patch("services.crawler.market_data.stock_crawl_manager.logger")
    def test_warns_when_year_missing_from_holidays(self, mock_logger) -> None:
        """Warning logged when current year has no holidays in config."""
        holidays_2026 = [date(2026, 2, 16)]
        # 2027 date with only 2026 holidays → should warn
        is_trading_day(date(2027, 3, 10), holidays=holidays_2026)
        mock_logger.warning.assert_called_once_with(
            "no_holiday_config_for_year",
            year=2027,
            component="stock_crawler",
        )


class TestRunStockCrawl:
    """Tests for run_stock_crawl orchestrator."""

    @patch("services.crawler.market_data.stock_crawl_manager.load_ticker_config")
    @patch("services.crawler.market_data.stock_crawl_manager.count_stock_prices_batch")
    @patch("services.crawler.market_data.stock_crawl_manager.VnstockClient")
    @patch("services.crawler.market_data.stock_crawl_manager.save_stock_prices")
    @patch("services.crawler.market_data.stock_crawl_manager.now_utc")
    async def test_initial_crawl_for_new_ticker(
        self,
        mock_now,
        mock_save,
        mock_client_cls,
        mock_count_batch,
        mock_load_config,
    ) -> None:
        """Ticker with < INITIAL_THRESHOLD rows triggers initial crawl (1Y)."""
        mock_now.return_value = datetime(2026, 3, 16, 10, 0, tzinfo=timezone.utc)  # Monday

        from services.crawler.market_data.ticker_config import TickerConfig
        mock_load_config.return_value = TickerConfig(
            tickers=["VNM"], total_count=1, enabled_groups=1, holidays=[],
        )
        mock_count_batch.return_value = {"VNM": 5}  # < 30

        mock_client = MagicMock()
        df = pd.DataFrame({
            "time": [datetime(2026, 1, 1)],
            "open": [100.0], "high": [105.0], "low": [99.0],
            "close": [103.0], "volume": [1000000],
        })
        df.attrs["data_source"] = "vnstock"
        mock_client.get_stock_history.return_value = df
        mock_client_cls.return_value = mock_client

        mock_save.return_value = 1

        result = await run_stock_crawl()

        assert isinstance(result, StockCrawlResult)
        assert result.initial_count == 1
        assert result.incremental_count == 0
        assert result.success_count == 1
        mock_client.get_stock_history.assert_called_once_with("VNM", length="1Y")

    @patch("services.crawler.market_data.stock_crawl_manager.load_ticker_config")
    @patch("services.crawler.market_data.stock_crawl_manager.count_stock_prices_batch")
    @patch("services.crawler.market_data.stock_crawl_manager.VnstockClient")
    @patch("services.crawler.market_data.stock_crawl_manager.save_stock_prices")
    @patch("services.crawler.market_data.stock_crawl_manager.now_utc")
    async def test_incremental_crawl_for_existing_ticker(
        self,
        mock_now,
        mock_save,
        mock_client_cls,
        mock_count_batch,
        mock_load_config,
    ) -> None:
        """Ticker with >= INITIAL_THRESHOLD rows triggers incremental crawl (1b)."""
        mock_now.return_value = datetime(2026, 3, 16, 10, 0, tzinfo=timezone.utc)

        from services.crawler.market_data.ticker_config import TickerConfig
        mock_load_config.return_value = TickerConfig(
            tickers=["VNM"], total_count=1, enabled_groups=1, holidays=[],
        )
        mock_count_batch.return_value = {"VNM": 250}

        mock_client = MagicMock()
        df = pd.DataFrame({
            "time": [datetime(2026, 3, 16)],
            "open": [100.0], "high": [105.0], "low": [99.0],
            "close": [103.0], "volume": [500000],
        })
        df.attrs["data_source"] = "vnstock"
        mock_client.get_stock_history.return_value = df
        mock_client_cls.return_value = mock_client

        mock_save.return_value = 1

        result = await run_stock_crawl()

        assert result.incremental_count == 1
        assert result.initial_count == 0
        mock_client.get_stock_history.assert_called_once_with("VNM", length="1b")

    @patch("services.crawler.market_data.stock_crawl_manager.load_ticker_config")
    @patch("services.crawler.market_data.stock_crawl_manager.count_stock_prices_batch")
    @patch("services.crawler.market_data.stock_crawl_manager.VnstockClient")
    @patch("services.crawler.market_data.stock_crawl_manager.save_stock_prices")
    @patch("services.crawler.market_data.stock_crawl_manager.now_utc")
    async def test_non_trading_day_skips_incremental_but_runs_initial(
        self,
        mock_now,
        mock_save,
        mock_client_cls,
        mock_count_batch,
        mock_load_config,
    ) -> None:
        """Non-trading day: skip incremental, but still run initial for new tickers."""
        # Saturday
        mock_now.return_value = datetime(2026, 3, 14, 10, 0, tzinfo=timezone.utc)

        from services.crawler.market_data.ticker_config import TickerConfig
        mock_load_config.return_value = TickerConfig(
            tickers=["VNM", "HPG"], total_count=2, enabled_groups=1, holidays=[],
        )
        # VNM is new (needs initial), HPG has data (would be incremental)
        mock_count_batch.return_value = {"VNM": 0, "HPG": 250}

        mock_client = MagicMock()
        df = pd.DataFrame({
            "time": [datetime(2026, 1, 1)],
            "open": [100.0], "high": [105.0], "low": [99.0],
            "close": [103.0], "volume": [1000000],
        })
        df.attrs["data_source"] = "vnstock"
        mock_client.get_stock_history.return_value = df
        mock_client_cls.return_value = mock_client
        mock_save.return_value = 1

        result = await run_stock_crawl()

        assert result.initial_count == 1  # VNM initial still runs
        assert result.skipped_count == 1  # HPG skipped (incremental on non-trading day)
        assert result.skipped_reason == "non_trading_day"

    @patch("services.crawler.market_data.stock_crawl_manager.load_ticker_config")
    @patch("services.crawler.market_data.stock_crawl_manager.count_stock_prices_batch")
    @patch("services.crawler.market_data.stock_crawl_manager.VnstockClient")
    @patch("services.crawler.market_data.stock_crawl_manager.save_stock_prices")
    @patch("services.crawler.market_data.stock_crawl_manager.now_utc")
    async def test_non_trading_day_all_existing_skips_entirely(
        self,
        mock_now,
        mock_save,
        mock_client_cls,
        mock_count_batch,
        mock_load_config,
    ) -> None:
        """Non-trading day with all existing tickers → skip entirely."""
        mock_now.return_value = datetime(2026, 3, 14, 10, 0, tzinfo=timezone.utc)

        from services.crawler.market_data.ticker_config import TickerConfig
        mock_load_config.return_value = TickerConfig(
            tickers=["VNM"], total_count=1, enabled_groups=1, holidays=[],
        )
        mock_count_batch.return_value = {"VNM": 250}

        result = await run_stock_crawl()

        assert result.skipped_reason == "non_trading_day"
        assert result.success_count == 0
        mock_client_cls.assert_not_called()

    @patch("services.crawler.market_data.stock_crawl_manager.load_ticker_config")
    @patch("services.crawler.market_data.stock_crawl_manager.count_stock_prices_batch")
    @patch("services.crawler.market_data.stock_crawl_manager.VnstockClient")
    @patch("services.crawler.market_data.stock_crawl_manager.save_stock_prices")
    @patch("services.crawler.market_data.stock_crawl_manager.now_utc")
    async def test_ticker_error_continues_next(
        self,
        mock_now,
        mock_save,
        mock_client_cls,
        mock_count_batch,
        mock_load_config,
    ) -> None:
        """Error on one ticker does not crash — continues with next."""
        mock_now.return_value = datetime(2026, 3, 16, 10, 0, tzinfo=timezone.utc)

        from services.crawler.market_data.ticker_config import TickerConfig
        mock_load_config.return_value = TickerConfig(
            tickers=["BAD", "VNM"], total_count=2, enabled_groups=1, holidays=[],
        )
        mock_count_batch.return_value = {"BAD": 0, "VNM": 0}

        mock_client = MagicMock()
        # First call fails, second succeeds
        df = pd.DataFrame({
            "time": [datetime(2026, 1, 1)],
            "open": [100.0], "high": [105.0], "low": [99.0],
            "close": [103.0], "volume": [1000000],
        })
        df.attrs["data_source"] = "vnstock"
        mock_client.get_stock_history.side_effect = [
            Exception("API error for BAD"),
            df,
        ]
        mock_client_cls.return_value = mock_client
        mock_save.return_value = 1

        result = await run_stock_crawl()

        assert result.success_count == 1
        assert result.failed_count == 1
        assert len(result.errors) == 1
        assert result.errors[0]["ticker"] == "BAD"

    @patch("services.crawler.market_data.stock_crawl_manager.load_ticker_config")
    @patch("services.crawler.market_data.stock_crawl_manager.count_stock_prices_batch")
    @patch("services.crawler.market_data.stock_crawl_manager.VnstockClient")
    @patch("services.crawler.market_data.stock_crawl_manager.save_stock_prices")
    @patch("services.crawler.market_data.stock_crawl_manager.now_utc")
    @patch("services.crawler.market_data.stock_crawl_manager.asyncio.sleep", new_callable=AsyncMock)
    async def test_rate_limiting_sleep_between_tickers(
        self,
        mock_sleep,
        mock_now,
        mock_save,
        mock_client_cls,
        mock_count_batch,
        mock_load_config,
    ) -> None:
        """asyncio.sleep(1.0) is called between ticker fetches."""
        mock_now.return_value = datetime(2026, 3, 16, 10, 0, tzinfo=timezone.utc)

        from services.crawler.market_data.ticker_config import TickerConfig
        mock_load_config.return_value = TickerConfig(
            tickers=["VNM", "HPG"], total_count=2, enabled_groups=1, holidays=[],
        )
        mock_count_batch.return_value = {"VNM": 0, "HPG": 0}

        mock_client = MagicMock()
        df = pd.DataFrame({
            "time": [datetime(2026, 1, 1)],
            "open": [100.0], "high": [105.0], "low": [99.0],
            "close": [103.0], "volume": [1000000],
        })
        df.attrs["data_source"] = "vnstock"
        mock_client.get_stock_history.return_value = df
        mock_client_cls.return_value = mock_client
        mock_save.return_value = 1

        await run_stock_crawl()

        # Sleep called between tickers (after first ticker, before second)
        assert mock_sleep.call_count >= 1
        mock_sleep.assert_any_call(1.0)

    @patch("services.crawler.market_data.stock_crawl_manager.load_ticker_config")
    @patch("services.crawler.market_data.stock_crawl_manager.count_stock_prices_batch")
    @patch("services.crawler.market_data.stock_crawl_manager.now_utc")
    async def test_result_dataclass_fields(
        self,
        mock_now,
        mock_count_batch,
        mock_load_config,
    ) -> None:
        """StockCrawlResult has all required fields."""
        mock_now.return_value = datetime(2026, 3, 14, 10, 0, tzinfo=timezone.utc)  # Saturday

        from services.crawler.market_data.ticker_config import TickerConfig
        mock_load_config.return_value = TickerConfig(
            tickers=[], total_count=0, enabled_groups=0, holidays=[],
        )
        mock_count_batch.return_value = {}

        result = await run_stock_crawl()

        assert hasattr(result, "total_tickers")
        assert hasattr(result, "success_count")
        assert hasattr(result, "failed_count")
        assert hasattr(result, "initial_count")
        assert hasattr(result, "incremental_count")
        assert hasattr(result, "skipped_count")
        assert hasattr(result, "rows_inserted")
        assert hasattr(result, "duration_seconds")
        assert hasattr(result, "skipped_reason")
        assert hasattr(result, "errors")
