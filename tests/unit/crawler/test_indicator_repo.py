"""Unit tests for indicator_repo — save/upsert, batch query, get_stock_prices_df."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from services.crawler.market_data.indicator_repo import (
    get_latest_indicator_date,
    get_stock_prices_df,
    save_technical_indicators,
)


def _make_indicators() -> list[dict]:
    """Create sample indicator dicts."""
    return [
        {"indicator_name": "SMA_20", "indicator_value": Decimal("102.5")},
        {"indicator_name": "RSI_14", "indicator_value": Decimal("55.3")},
        {"indicator_name": "MACD_LINE", "indicator_value": Decimal("1.23")},
    ]


class TestSaveTechnicalIndicators:
    """Tests for save_technical_indicators function."""

    @patch("services.crawler.market_data.indicator_repo.get_async_session")
    async def test_saves_new_indicators(self, mock_get_session) -> None:
        """New indicators are inserted into DB."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.all.return_value = []  # No existing records
        mock_session.execute.return_value = mock_result

        mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

        data_as_of = datetime(2026, 3, 14, tzinfo=timezone.utc)
        indicators = _make_indicators()

        inserted = await save_technical_indicators("HPG", indicators, data_as_of)

        assert inserted == 3
        assert mock_session.add.call_count == 3
        mock_session.commit.assert_awaited_once()

    @patch("services.crawler.market_data.indicator_repo.get_async_session")
    async def test_skips_existing_indicators(self, mock_get_session) -> None:
        """Existing indicators (by indicator_name) are not re-inserted."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        # SMA_20 already exists
        mock_result.all.return_value = [("SMA_20",)]
        mock_session.execute.return_value = mock_result

        mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

        data_as_of = datetime(2026, 3, 14, tzinfo=timezone.utc)
        indicators = _make_indicators()

        inserted = await save_technical_indicators("HPG", indicators, data_as_of)

        assert inserted == 2  # Only RSI_14 and MACD_LINE
        assert mock_session.add.call_count == 2

    @patch("services.crawler.market_data.indicator_repo.get_async_session")
    async def test_empty_indicators_returns_zero(self, mock_get_session) -> None:
        """Empty indicator list returns 0 without DB call."""
        data_as_of = datetime(2026, 3, 14, tzinfo=timezone.utc)

        inserted = await save_technical_indicators("HPG", [], data_as_of)

        assert inserted == 0
        mock_get_session.assert_not_called()

    @patch("services.crawler.market_data.indicator_repo.get_async_session")
    async def test_none_values_filtered_out(self, mock_get_session) -> None:
        """Indicators with None values are not saved."""
        indicators = [
            {"indicator_name": "SMA_20", "indicator_value": Decimal("102.5")},
            {"indicator_name": "RS_VNINDEX", "indicator_value": None},
        ]
        data_as_of = datetime(2026, 3, 14, tzinfo=timezone.utc)

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.all.return_value = []
        mock_session.execute.return_value = mock_result

        mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

        inserted = await save_technical_indicators("HPG", indicators, data_as_of)

        assert inserted == 1  # Only SMA_20

    @patch("services.crawler.market_data.indicator_repo.get_async_session")
    async def test_all_none_values_returns_zero(self, mock_get_session) -> None:
        """All-None indicators returns 0 without DB call."""
        indicators = [
            {"indicator_name": "RS_VNINDEX", "indicator_value": None},
        ]
        data_as_of = datetime(2026, 3, 14, tzinfo=timezone.utc)

        inserted = await save_technical_indicators("HPG", indicators, data_as_of)

        assert inserted == 0

    @patch("services.crawler.market_data.indicator_repo.get_async_session")
    async def test_naive_datetime_gets_utc(self, mock_get_session) -> None:
        """Naive datetime is converted to UTC-aware."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.all.return_value = []
        mock_session.execute.return_value = mock_result

        mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

        data_as_of = datetime(2026, 3, 14)  # Naive
        indicators = [{"indicator_name": "SMA_20", "indicator_value": Decimal("100")}]

        await save_technical_indicators("HPG", indicators, data_as_of)

        # Verify commit was called (function didn't crash on naive datetime)
        mock_session.commit.assert_awaited_once()


class TestGetLatestIndicatorDate:
    """Tests for get_latest_indicator_date function."""

    @patch("services.crawler.market_data.indicator_repo.get_async_session")
    async def test_returns_date_when_exists(self, mock_get_session) -> None:
        """Returns latest data_as_of for existing indicators."""
        expected_date = datetime(2026, 3, 14, tzinfo=timezone.utc)

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.first.return_value = (expected_date,)
        mock_session.execute.return_value = mock_result

        mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await get_latest_indicator_date("HPG")

        assert result == expected_date

    @patch("services.crawler.market_data.indicator_repo.get_async_session")
    async def test_returns_none_when_no_data(self, mock_get_session) -> None:
        """Returns None when no indicators exist."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.first.return_value = None
        mock_session.execute.return_value = mock_result

        mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await get_latest_indicator_date("HPG")

        assert result is None


class TestGetStockPricesDf:
    """Tests for get_stock_prices_df function."""

    @patch("services.crawler.market_data.indicator_repo.get_async_session")
    async def test_returns_dataframe_with_correct_columns(self, mock_get_session) -> None:
        """Returns DataFrame with expected columns."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.all.return_value = [
            (datetime(2026, 3, 14, tzinfo=timezone.utc), Decimal("100"), Decimal("105"), Decimal("99"), Decimal("103"), 1000000),
        ]
        mock_session.execute.return_value = mock_result

        mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

        df = await get_stock_prices_df("HPG")

        assert list(df.columns) == ["time", "open", "high", "low", "close", "volume"]
        assert len(df) == 1
        assert df["close"].dtype == float
        assert df["volume"].dtype == float

    @patch("services.crawler.market_data.indicator_repo.get_async_session")
    async def test_returns_empty_df_when_no_data(self, mock_get_session) -> None:
        """Returns empty DataFrame when no rows found."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.all.return_value = []
        mock_session.execute.return_value = mock_result

        mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

        df = await get_stock_prices_df("HPG")

        assert df.empty
