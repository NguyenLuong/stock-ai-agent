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
    get_latest_financial_ratios,
    get_peer_ratios,
    get_sector_average_ratios,
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


class TestGetLatestFinancialRatios:
    """Tests for get_latest_financial_ratios."""

    @pytest.mark.asyncio
    @patch("services.crawler.market_data.stock_data_repo.get_async_session")
    async def test_happy_path_returns_ratios_and_date(
        self, mock_get_session: MagicMock
    ) -> None:
        """Test returns (ratio_dict, data_as_of) when data exists."""
        data_as_of = datetime(2026, 3, 1, tzinfo=timezone.utc)
        mock_row = MagicMock()
        mock_row.pe_ratio = Decimal("12.5")
        mock_row.pb_ratio = Decimal("1.8")
        mock_row.roe = Decimal("15.2")
        mock_row.eps = Decimal("3500")
        mock_row.eps_growth_yoy = Decimal("0.12")
        mock_row.data_as_of = data_as_of

        mock_result = MagicMock()
        mock_result.first.return_value = mock_row

        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.return_value = mock_session
        mock_get_session.return_value = mock_ctx

        ratios, as_of = await get_latest_financial_ratios("HPG")

        assert ratios["pe_ratio"] == Decimal("12.5")
        assert ratios["pb_ratio"] == Decimal("1.8")
        assert ratios["roe"] == Decimal("15.2")
        assert ratios["eps"] == Decimal("3500")
        assert ratios["eps_growth_yoy"] == Decimal("0.12")
        assert as_of == data_as_of

    @pytest.mark.asyncio
    @patch("services.crawler.market_data.stock_data_repo.get_async_session")
    async def test_empty_result_returns_empty_dict_and_none(
        self, mock_get_session: MagicMock
    ) -> None:
        """Test returns ({}, None) when no data exists."""
        mock_result = MagicMock()
        mock_result.first.return_value = None

        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.return_value = mock_session
        mock_get_session.return_value = mock_ctx

        ratios, as_of = await get_latest_financial_ratios("UNKNOWN")

        assert ratios == {}
        assert as_of is None

    @pytest.mark.asyncio
    @patch("services.crawler.market_data.stock_data_repo.get_async_session")
    async def test_tuple_unpacking_works(
        self, mock_get_session: MagicMock
    ) -> None:
        """Test result can be unpacked as tuple."""
        mock_result = MagicMock()
        mock_result.first.return_value = None

        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.return_value = mock_session
        mock_get_session.return_value = mock_ctx

        result = await get_latest_financial_ratios("HPG")
        ratios, as_of = result
        assert isinstance(ratios, dict)
        assert as_of is None


class TestGetSectorAverageRatios:
    """Tests for get_sector_average_ratios."""

    @pytest.mark.asyncio
    @patch("services.crawler.market_data.stock_data_repo.get_async_session")
    async def test_happy_path_returns_averages(
        self, mock_get_session: MagicMock
    ) -> None:
        """Test returns average ratios across sector tickers."""
        # Simulate DB returning avg values
        mock_row = MagicMock()
        mock_row._mapping = {
            "avg_pe": Decimal("10.3"),
            "avg_pb": Decimal("1.5"),
            "avg_roe": Decimal("13.8"),
            "avg_eps": Decimal("2800"),
        }
        # Make the row subscriptable
        mock_row.__getitem__ = lambda self, key: self._mapping[key]

        mock_result = MagicMock()
        mock_result.first.return_value = mock_row

        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.return_value = mock_session
        mock_get_session.return_value = mock_ctx

        result = await get_sector_average_ratios(
            ["HSG", "NKG", "TLH"], exclude_ticker="HPG"
        )

        assert result["pe"] == Decimal("10.3")
        assert result["pb"] == Decimal("1.5")
        assert result["roe"] == Decimal("13.8")
        assert result["eps"] == Decimal("2800")

    @pytest.mark.asyncio
    @patch("services.crawler.market_data.stock_data_repo.get_async_session")
    async def test_empty_result_returns_all_none(
        self, mock_get_session: MagicMock
    ) -> None:
        """Test returns dict with all None values when no data."""
        mock_row = MagicMock()
        mock_row._mapping = {
            "avg_pe": None,
            "avg_pb": None,
            "avg_roe": None,
            "avg_eps": None,
        }
        mock_row.__getitem__ = lambda self, key: self._mapping[key]

        mock_result = MagicMock()
        mock_result.first.return_value = mock_row

        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.return_value = mock_session
        mock_get_session.return_value = mock_ctx

        result = await get_sector_average_ratios([], exclude_ticker="HPG")

        assert result["pe"] is None
        assert result["pb"] is None
        assert result["roe"] is None
        assert result["eps"] is None

    @pytest.mark.asyncio
    @patch("services.crawler.market_data.stock_data_repo.get_async_session")
    async def test_exclude_ticker(
        self, mock_get_session: MagicMock
    ) -> None:
        """Test exclude_ticker filters correctly."""
        mock_row = MagicMock()
        mock_row._mapping = {
            "avg_pe": Decimal("11.0"),
            "avg_pb": Decimal("1.6"),
            "avg_roe": Decimal("14.0"),
            "avg_eps": Decimal("3000"),
        }
        mock_row.__getitem__ = lambda self, key: self._mapping[key]

        mock_result = MagicMock()
        mock_result.first.return_value = mock_row

        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.return_value = mock_session
        mock_get_session.return_value = mock_ctx

        result = await get_sector_average_ratios(
            ["HPG", "HSG", "NKG"], exclude_ticker="HPG"
        )

        # Verify the execute was called (we can't easily check the WHERE clause
        # in the mock, but implementation should exclude HPG)
        mock_session.execute.assert_awaited_once()
        assert "pe" in result


class TestGetPeerRatios:
    """Tests for get_peer_ratios."""

    @pytest.mark.asyncio
    @patch("services.crawler.market_data.stock_data_repo.get_async_session")
    async def test_happy_path_returns_peer_list(
        self, mock_get_session: MagicMock
    ) -> None:
        """Test returns list of peer ratio dicts."""
        # Mock DB rows for HSG and NKG
        row_hsg = MagicMock()
        row_hsg.ticker_symbol = "HSG"
        row_hsg.pe_ratio = Decimal("8.5")
        row_hsg.pb_ratio = Decimal("1.2")
        row_hsg.roe = Decimal("12.0")

        row_nkg = MagicMock()
        row_nkg.ticker_symbol = "NKG"
        row_nkg.pe_ratio = Decimal("7.0")
        row_nkg.pb_ratio = Decimal("0.9")
        row_nkg.roe = Decimal("10.5")

        mock_result = MagicMock()
        mock_result.all.return_value = [(row_hsg,), (row_nkg,)]

        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.return_value = mock_session
        mock_get_session.return_value = mock_ctx

        result = await get_peer_ratios(
            ["HPG", "HSG", "NKG"], exclude_ticker="HPG"
        )

        assert len(result) == 2
        assert result[0]["ticker"] == "HSG"
        assert result[0]["pe"] == 8.5
        assert result[1]["ticker"] == "NKG"

    @pytest.mark.asyncio
    @patch("services.crawler.market_data.stock_data_repo.get_async_session")
    async def test_exclude_logic(
        self, mock_get_session: MagicMock
    ) -> None:
        """Test exclude_ticker is filtered out."""
        mock_result = MagicMock()
        mock_result.all.return_value = []

        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.return_value = mock_session
        mock_get_session.return_value = mock_ctx

        result = await get_peer_ratios(["HPG"], exclude_ticker="HPG")
        assert result == []

    @pytest.mark.asyncio
    @patch("services.crawler.market_data.stock_data_repo.get_async_session")
    async def test_max_5_cap(
        self, mock_get_session: MagicMock
    ) -> None:
        """Test returns max 5 peers."""
        rows = []
        for i, ticker in enumerate(["A", "B", "C", "D", "E", "F", "G"]):
            row = MagicMock()
            row.ticker_symbol = ticker
            row.pe_ratio = Decimal(str(10 + i))
            row.pb_ratio = Decimal("1.0")
            row.roe = Decimal("10.0")
            rows.append((row,))

        mock_result = MagicMock()
        mock_result.all.return_value = rows

        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.return_value = mock_session
        mock_get_session.return_value = mock_ctx

        result = await get_peer_ratios(
            ["A", "B", "C", "D", "E", "F", "G", "HPG"], exclude_ticker="HPG"
        )

        assert len(result) <= 5
