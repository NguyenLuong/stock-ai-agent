"""Tests for macro data repository — upsert and dedup logic."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from shared.models.market_data import MarketDataCreate
from services.crawler.macro.macro_data_repo import save_macro_indicators


class TestSaveMacroIndicators:
    """Tests for save_macro_indicators."""

    @patch("services.crawler.macro.macro_data_repo.get_async_session")
    async def test_inserts_new_indicators(self, mock_get_session: MagicMock) -> None:
        mock_session = AsyncMock()
        mock_session.add = MagicMock()
        # No existing records
        mock_existing = MagicMock()
        mock_existing.all.return_value = []
        mock_session.execute.return_value = mock_existing

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.return_value = mock_session
        mock_get_session.return_value = mock_ctx

        indicators = [
            MarketDataCreate(
                data_type="macro_indicator",
                indicator_name="vn_index_close",
                indicator_value=Decimal("1255.5"),
                data_as_of=datetime(2026, 3, 10, tzinfo=timezone.utc),
                data_source="vnstock",
            ),
            MarketDataCreate(
                data_type="macro_indicator",
                indicator_name="usd_vnd_rate",
                indicator_value=Decimal("25850.0"),
                data_as_of=datetime(2026, 3, 10, tzinfo=timezone.utc),
                data_source="vnstock",
            ),
        ]

        inserted = await save_macro_indicators(indicators)

        assert inserted == 2
        assert mock_session.add.call_count == 2
        mock_session.commit.assert_awaited_once()

    @patch("services.crawler.macro.macro_data_repo.get_async_session")
    async def test_skips_duplicates_by_indicator_and_date(self, mock_get_session: MagicMock) -> None:
        mock_session = AsyncMock()
        mock_session.add = MagicMock()
        # One existing record
        mock_existing = MagicMock()
        mock_existing.all.return_value = [
            ("vn_index_close", date(2026, 3, 10)),
        ]
        mock_session.execute.return_value = mock_existing

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.return_value = mock_session
        mock_get_session.return_value = mock_ctx

        indicators = [
            MarketDataCreate(
                data_type="macro_indicator",
                indicator_name="vn_index_close",
                indicator_value=Decimal("1255.5"),
                data_as_of=datetime(2026, 3, 10, tzinfo=timezone.utc),
                data_source="vnstock",
            ),
            MarketDataCreate(
                data_type="macro_indicator",
                indicator_name="usd_vnd_rate",
                indicator_value=Decimal("25850.0"),
                data_as_of=datetime(2026, 3, 10, tzinfo=timezone.utc),
                data_source="vnstock",
            ),
        ]

        inserted = await save_macro_indicators(indicators)

        # Only usd_vnd_rate should be inserted (vn_index_close already exists)
        assert inserted == 1
        assert mock_session.add.call_count == 1

    @patch("services.crawler.macro.macro_data_repo.get_async_session")
    async def test_empty_list_returns_zero(self, mock_get_session: MagicMock) -> None:
        inserted = await save_macro_indicators([])
        assert inserted == 0
        mock_get_session.assert_not_called()

    @patch("services.crawler.macro.macro_data_repo.get_async_session")
    async def test_maps_fields_to_orm(self, mock_get_session: MagicMock) -> None:
        mock_session = AsyncMock()
        mock_session.add = MagicMock()
        mock_existing = MagicMock()
        mock_existing.all.return_value = []
        mock_session.execute.return_value = mock_existing

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.return_value = mock_session
        mock_get_session.return_value = mock_ctx

        indicators = [
            MarketDataCreate(
                data_type="macro_indicator",
                indicator_name="sbv_interest_rate",
                indicator_value=Decimal("4.5"),
                data_as_of=datetime(2026, 3, 10, tzinfo=timezone.utc),
                data_source="sbv.gov.vn",
            ),
        ]

        await save_macro_indicators(indicators)

        record = mock_session.add.call_args[0][0]
        assert record.data_type == "macro_indicator"
        assert record.indicator_name == "sbv_interest_rate"
        assert record.indicator_value == Decimal("4.5")
        assert record.data_source == "sbv.gov.vn"
        assert record.ticker_symbol is None
