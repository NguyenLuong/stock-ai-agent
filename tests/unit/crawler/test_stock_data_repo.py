"""Unit tests for stock_data_repo."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from services.crawler.market_data.stock_data_repo import (
    _extract_ratios_from_df,
    _period_to_datetime,
    _to_aware_datetime,
    save_financial_ratios,
    save_stock_prices,
)


class TestSaveStockPrices:
    """Tests for save_stock_prices."""

    @pytest.mark.asyncio
    @patch("services.crawler.market_data.stock_data_repo.get_async_session")
    async def test_maps_dataframe_to_market_data_orm(
        self, mock_get_session: MagicMock
    ) -> None:
        """Test DataFrame columns are correctly mapped to MarketData ORM fields."""
        mock_session = AsyncMock()
        mock_session.add = MagicMock()  # add() is sync, avoid coroutine warning
        # Batch query returns no existing dates
        mock_existing = MagicMock()
        mock_existing.all.return_value = []
        mock_session.execute.return_value = mock_existing

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.return_value = mock_session
        mock_get_session.return_value = mock_ctx

        df = pd.DataFrame(
            {
                "time": [datetime(2026, 1, 1)],
                "open": [25.5],
                "high": [26.0],
                "low": [25.0],
                "close": [25.8],
                "volume": [1000000],
            }
        )

        inserted = await save_stock_prices("HPG", df, "vnstock")

        assert inserted == 1
        mock_session.add.assert_called_once()
        record = mock_session.add.call_args[0][0]
        assert record.ticker_symbol == "HPG"
        assert record.data_type == "stock_price"
        assert record.open_price == Decimal("25.5")
        assert record.high_price == Decimal("26.0")
        assert record.low_price == Decimal("25.0")
        assert record.close_price == Decimal("25.8")
        assert record.volume == 1000000
        assert record.data_source == "vnstock"
        mock_session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("services.crawler.market_data.stock_data_repo.get_async_session")
    async def test_upsert_skips_existing_records(
        self, mock_get_session: MagicMock
    ) -> None:
        """Test upsert logic skips records that already exist."""
        mock_session = AsyncMock()
        mock_session.add = MagicMock()
        # Batch query returns the date as already existing
        existing_date = datetime(2026, 1, 1, tzinfo=timezone.utc)
        mock_existing = MagicMock()
        mock_existing.all.return_value = [(existing_date,)]
        mock_session.execute.return_value = mock_existing

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.return_value = mock_session
        mock_get_session.return_value = mock_ctx

        df = pd.DataFrame(
            {
                "time": [datetime(2026, 1, 1)],
                "open": [25.5],
                "high": [26.0],
                "low": [25.0],
                "close": [25.8],
                "volume": [1000000],
            }
        )

        inserted = await save_stock_prices("HPG", df, "vnstock")

        assert inserted == 0
        mock_session.add.assert_not_called()

    @pytest.mark.asyncio
    @patch("services.crawler.market_data.stock_data_repo.get_async_session")
    async def test_empty_dataframe_returns_zero(
        self, mock_get_session: MagicMock
    ) -> None:
        """Test empty DataFrame returns 0 without DB access."""
        inserted = await save_stock_prices("HPG", pd.DataFrame(), "vnstock")
        assert inserted == 0
        mock_get_session.assert_not_called()


class TestSaveFinancialRatios:
    """Tests for save_financial_ratios."""

    @pytest.mark.asyncio
    @patch("services.crawler.market_data.stock_data_repo.get_async_session")
    async def test_saves_ratio_records(self, mock_get_session: MagicMock) -> None:
        """Test financial ratios are saved correctly."""
        mock_session = AsyncMock()
        mock_session.add = MagicMock()  # add() is sync
        mock_existing = MagicMock()
        mock_existing.all.return_value = []
        mock_session.execute.return_value = mock_existing

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.return_value = mock_session
        mock_get_session.return_value = mock_ctx

        df = pd.DataFrame(
            [
                {"item": "P/E", "2025-Q4": 12.5},
                {"item": "P/B", "2025-Q4": 1.8},
                {"item": "ROE", "2025-Q4": 15.3},
                {"item": "EPS", "2025-Q4": 3500.0},
            ]
        )

        inserted = await save_financial_ratios("HPG", df, "vnstock")

        assert inserted == 1  # One period = one record
        mock_session.add.assert_called_once()
        record = mock_session.add.call_args[0][0]
        assert record.ticker_symbol == "HPG"
        assert record.data_type == "financial_ratio"
        assert record.pe_ratio == Decimal("12.5")
        assert record.pb_ratio == Decimal("1.8")
        assert record.roe == Decimal("15.3")
        assert record.eps == Decimal("3500.0")

    @pytest.mark.asyncio
    @patch("services.crawler.market_data.stock_data_repo.get_async_session")
    async def test_upsert_skips_existing_ratios(
        self, mock_get_session: MagicMock
    ) -> None:
        """Test upsert logic skips ratio records that already exist."""
        mock_session = AsyncMock()
        mock_session.add = MagicMock()
        # Batch query returns Q4 date as existing
        existing_date = datetime(2025, 12, 1, tzinfo=timezone.utc)
        mock_existing = MagicMock()
        mock_existing.all.return_value = [(existing_date,)]
        mock_session.execute.return_value = mock_existing

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.return_value = mock_session
        mock_get_session.return_value = mock_ctx

        df = pd.DataFrame(
            [
                {"item": "P/E", "2025-Q4": 12.5},
                {"item": "P/B", "2025-Q4": 1.8},
                {"item": "ROE", "2025-Q4": 15.3},
                {"item": "EPS", "2025-Q4": 3500.0},
            ]
        )

        inserted = await save_financial_ratios("HPG", df, "vnstock")

        assert inserted == 0
        mock_session.add.assert_not_called()


class TestExtractRatiosFromDf:
    """Tests for _extract_ratios_from_df helper."""

    def test_extracts_ratios_correctly(self) -> None:
        """Test ratio extraction from KBS format DataFrame."""
        df = pd.DataFrame(
            [
                {"item": "P/E", "2025-Q4": 12.5},
                {"item": "P/B", "2025-Q4": 1.8},
                {"item": "ROE", "2025-Q4": 15.3},
                {"item": "EPS", "2025-Q4": 3500.0},
            ]
        )
        result = _extract_ratios_from_df(df)
        assert "2025-Q4" in result
        assert result["2025-Q4"]["pe_ratio"] == Decimal("12.5")
        assert result["2025-Q4"]["pb_ratio"] == Decimal("1.8")
        assert result["2025-Q4"]["roe"] == Decimal("15.3")
        assert result["2025-Q4"]["eps"] == Decimal("3500.0")

    def test_returns_empty_dict_without_item_column(self) -> None:
        """Test returns empty dict when 'item' column is missing."""
        df = pd.DataFrame({"something": [1, 2, 3]})
        assert _extract_ratios_from_df(df) == {}


class TestPeriodToDatetime:
    """Tests for _period_to_datetime helper."""

    def test_quarter_format(self) -> None:
        """Test '2025-Q4' → 2025-12-01."""
        result = _period_to_datetime("2025-Q4")
        assert result == datetime(2025, 12, 1, tzinfo=timezone.utc)

    def test_year_format(self) -> None:
        """Test '2024' → 2024-12-31."""
        result = _period_to_datetime("2024")
        assert result == datetime(2024, 12, 31, tzinfo=timezone.utc)


class TestToAwareDatetime:
    """Tests for _to_aware_datetime helper."""

    def test_naive_datetime_gets_utc(self) -> None:
        """Test naive datetime is made UTC-aware."""
        dt = datetime(2026, 1, 1)
        result = _to_aware_datetime(dt)
        assert result.tzinfo == timezone.utc

    def test_aware_datetime_unchanged(self) -> None:
        """Test aware datetime passes through unchanged."""
        dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
        result = _to_aware_datetime(dt)
        assert result == dt

    def test_pandas_timestamp(self) -> None:
        """Test pandas Timestamp is converted correctly."""
        ts = pd.Timestamp("2026-01-01")
        result = _to_aware_datetime(ts)
        assert result.tzinfo == timezone.utc
