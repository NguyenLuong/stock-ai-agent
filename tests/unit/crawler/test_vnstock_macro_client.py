"""Tests for VnstockMacroClient — VN-Index, exchange rate, foreign flow."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from services.crawler.macro.vnstock_macro_client import VnstockMacroClient


class TestGetVnIndex:
    """Tests for VN-Index fetch via vnstock."""

    def setup_method(self) -> None:
        VnstockMacroClient.reset()

    def test_returns_close_and_volume_on_success(self) -> None:
        client = VnstockMacroClient()

        with patch.object(client, "_fetch_vn_index", return_value=(1255.5, 500_000_000.0, datetime(2026, 3, 10, tzinfo=timezone.utc))):
            results = client.get_vn_index()

        assert len(results) == 2
        close_result = results[0]
        volume_result = results[1]

        assert close_result.indicator_name == "vn_index_close"
        assert close_result.indicator_value == 1255.5
        assert close_result.data_source == "vnstock"
        assert close_result.success is True

        assert volume_result.indicator_name == "vn_index_volume"
        assert volume_result.indicator_value == 500_000_000.0
        assert volume_result.data_source == "vnstock"
        assert volume_result.success is True

    def test_fallback_to_mock_on_api_error(self) -> None:
        client = VnstockMacroClient()

        with patch.object(client, "_fetch_vn_index", side_effect=ConnectionError("unavailable")):
            results = client.get_vn_index()

        assert len(results) == 2
        assert results[0].data_source == "mock"
        assert results[1].data_source == "mock"
        assert results[0].success is True

    def test_singleton_pattern(self) -> None:
        VnstockMacroClient.reset()
        a = VnstockMacroClient.get_instance()
        b = VnstockMacroClient.get_instance()
        assert a is b

    def test_reset_clears_singleton(self) -> None:
        a = VnstockMacroClient.get_instance()
        VnstockMacroClient.reset()
        b = VnstockMacroClient.get_instance()
        assert a is not b


class TestGetExchangeRate:
    """Tests for USD/VND exchange rate fetch."""

    def setup_method(self) -> None:
        VnstockMacroClient.reset()

    def test_returns_rate_on_success(self) -> None:
        client = VnstockMacroClient()

        with patch.object(client, "_fetch_exchange_rate", return_value=(25_850.0, datetime(2026, 3, 10, tzinfo=timezone.utc))):
            result = client.get_exchange_rate()

        assert result.indicator_name == "usd_vnd_rate"
        assert result.indicator_value == 25_850.0
        assert result.data_source == "vnstock"
        assert result.success is True

    def test_fallback_to_mock_on_error(self) -> None:
        client = VnstockMacroClient()

        with patch.object(client, "_fetch_exchange_rate", side_effect=TimeoutError("timeout")):
            result = client.get_exchange_rate()

        assert result.data_source == "mock"
        assert result.success is True
        assert result.indicator_name == "usd_vnd_rate"


class TestGetForeignFlow:
    """Tests for foreign net flow (always mock)."""

    def test_returns_mock_data(self) -> None:
        client = VnstockMacroClient()
        result = client.get_foreign_flow()

        assert result.indicator_name == "foreign_net_flow"
        assert result.data_source == "mock"
        assert result.success is True


class TestAsyncWrappers:
    """Tests for async wrappers."""

    def setup_method(self) -> None:
        VnstockMacroClient.reset()

    async def test_aget_vn_index(self) -> None:
        client = VnstockMacroClient()

        with patch.object(client, "_fetch_vn_index", return_value=(1255.5, 500_000_000.0, datetime(2026, 3, 10, tzinfo=timezone.utc))):
            results = await client.aget_vn_index()

        assert len(results) == 2
        assert results[0].indicator_name == "vn_index_close"

    async def test_aget_exchange_rate(self) -> None:
        client = VnstockMacroClient()

        with patch.object(client, "_fetch_exchange_rate", return_value=(25_850.0, datetime(2026, 3, 10, tzinfo=timezone.utc))):
            result = await client.aget_exchange_rate()

        assert result.indicator_name == "usd_vnd_rate"

    async def test_aget_foreign_flow(self) -> None:
        client = VnstockMacroClient()
        result = await client.aget_foreign_flow()

        assert result.indicator_name == "foreign_net_flow"
        assert result.data_source == "mock"
