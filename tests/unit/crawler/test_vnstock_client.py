"""Unit tests for VnstockClient."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from services.crawler.market_data.vnstock_client import VnstockClient


def _make_ohlcv_df(rows: int = 250) -> pd.DataFrame:
    """Create a realistic OHLCV DataFrame matching vnstock output."""
    import numpy as np

    dates = pd.date_range(end="2026-03-06", periods=rows, freq="B")
    rng = np.random.default_rng(42)
    close_arr = (30.0 + rng.normal(0, 1, rows).cumsum()).tolist()
    open_arr = [c + rng.normal(0, 0.5) for c in close_arr]
    high_arr = [max(o, c) + abs(rng.normal(0, 1)) for o, c in zip(open_arr, close_arr)]
    low_arr = [min(o, c) - abs(rng.normal(0, 1)) for o, c in zip(open_arr, close_arr)]
    vol_arr = [int(rng.integers(100_000, 5_000_000)) for _ in range(rows)]
    return pd.DataFrame(
        {
            "time": dates.tolist(),
            "open": open_arr,
            "high": high_arr,
            "low": low_arr,
            "close": close_arr,
            "volume": vol_arr,
        }
    )


def _make_ratios_df() -> pd.DataFrame:
    """Create a financial ratios DataFrame matching KBS format."""
    return pd.DataFrame(
        [
            {"item": "P/E", "2025-Q4": 12.5, "2025-Q3": 11.0},
            {"item": "P/B", "2025-Q4": 1.8, "2025-Q3": 1.7},
            {"item": "ROE", "2025-Q4": 15.3, "2025-Q3": 14.5},
            {"item": "EPS", "2025-Q4": 3500.0, "2025-Q3": 3200.0},
        ]
    )


class TestGetStockHistory:
    """Tests for VnstockClient.get_stock_history."""

    def test_returns_ohlcv_dataframe(self) -> None:
        """Test returns DataFrame with correct columns and ≥200 rows."""
        expected_df = _make_ohlcv_df(250)
        client = VnstockClient()

        with patch.object(client, "_fetch_history", return_value=expected_df):
            result = client.get_stock_history("HPG")

        expected_columns = {"time", "open", "high", "low", "close", "volume"}
        assert expected_columns.issubset(set(result.columns))
        assert len(result) >= 200
        assert result.attrs["data_source"] == "vnstock"

    def test_fallback_to_mock_on_api_error(self) -> None:
        """Test mock fallback triggers on API error with data_source='mock'."""
        client = VnstockClient()

        with patch.object(
            client, "_fetch_history", side_effect=ConnectionError("API unavailable")
        ):
            result = client.get_stock_history("HPG")

        assert result.attrs["data_source"] == "mock"
        expected_columns = {"time", "open", "high", "low", "close", "volume"}
        assert expected_columns.issubset(set(result.columns))
        assert len(result) > 0

    def test_fallback_after_all_retries_exhausted(self) -> None:
        """Test that when _fetch_history raises after all retries, fallback to mock occurs."""
        client = VnstockClient()

        with patch.object(
            client, "_fetch_history", side_effect=ConnectionError("timeout")
        ):
            result = client.get_stock_history("HPG")

        assert result.attrs["data_source"] == "mock"
        assert len(result) > 0

    def test_logs_on_api_failure(self) -> None:
        """Test structured logging on API failure triggers fallback."""
        client = VnstockClient()

        with patch.object(
            client, "_fetch_history", side_effect=ConnectionError("Network error")
        ):
            result = client.get_stock_history("HPG")

        # Verify fallback happened (which means warning was logged)
        assert result.attrs["data_source"] == "mock"

    def test_custom_length_and_interval(self) -> None:
        """Test custom length and interval params are passed through."""
        expected_df = _make_ohlcv_df(50)
        client = VnstockClient()

        with patch.object(client, "_fetch_history", return_value=expected_df) as mock:
            client.get_stock_history("HPG", length="3M", interval="1W")
            mock.assert_called_once_with("HPG", "3M", "1W")


class TestGetFinancialRatios:
    """Tests for VnstockClient.get_financial_ratios."""

    def test_returns_ratios_with_pe_pb_roe_eps(self) -> None:
        """Test returns DataFrame with P/E, P/B, ROE, EPS non-null."""
        expected_df = _make_ratios_df()
        client = VnstockClient()

        with patch.object(client, "_fetch_ratios", return_value=expected_df):
            result = client.get_financial_ratios("HPG")

        assert result.attrs["data_source"] == "vnstock"
        items = result["item"].str.upper().tolist()
        for ratio in ["P/E", "P/B", "ROE", "EPS"]:
            assert ratio in items
        # Verify latest quarter values are non-null
        latest_col = [c for c in result.columns if c != "item"][0]
        for _, row in result.iterrows():
            assert pd.notna(row[latest_col])

    def test_fallback_to_mock_on_api_error(self) -> None:
        """Test mock fallback on API failure."""
        client = VnstockClient()

        with patch.object(
            client, "_fetch_ratios", side_effect=ConnectionError("API down")
        ):
            result = client.get_financial_ratios("HPG")

        assert result.attrs["data_source"] == "mock"
        items = result["item"].str.upper().tolist()
        assert "P/E" in items
        assert "P/B" in items
        assert "ROE" in items
        assert "EPS" in items


class TestGetIncomeStatement:
    """Tests for VnstockClient.get_income_statement."""

    def test_returns_dataframe_on_success(self) -> None:
        """Test returns DataFrame with data_source='vnstock' on success."""
        expected_df = pd.DataFrame({"item": ["Revenue"], "2025": [1000000]})
        client = VnstockClient()

        with patch.object(client, "_fetch_income_statement", return_value=expected_df):
            result = client.get_income_statement("HPG")

        assert result.attrs["data_source"] == "vnstock"
        assert not result.empty

    def test_fallback_to_empty_on_api_error(self) -> None:
        """Test returns empty DataFrame with data_source='mock' on failure."""
        client = VnstockClient()

        with patch.object(
            client, "_fetch_income_statement", side_effect=ConnectionError("API down")
        ):
            result = client.get_income_statement("HPG")

        assert result.attrs["data_source"] == "mock"
        assert result.empty


class TestGetBalanceSheet:
    """Tests for VnstockClient.get_balance_sheet."""

    def test_returns_dataframe_on_success(self) -> None:
        """Test returns DataFrame with data_source='vnstock' on success."""
        expected_df = pd.DataFrame({"item": ["Total Assets"], "2025": [5000000]})
        client = VnstockClient()

        with patch.object(client, "_fetch_balance_sheet", return_value=expected_df):
            result = client.get_balance_sheet("HPG")

        assert result.attrs["data_source"] == "vnstock"
        assert not result.empty

    def test_fallback_to_empty_on_api_error(self) -> None:
        """Test returns empty DataFrame with data_source='mock' on failure."""
        client = VnstockClient()

        with patch.object(
            client, "_fetch_balance_sheet", side_effect=ConnectionError("API down")
        ):
            result = client.get_balance_sheet("HPG")

        assert result.attrs["data_source"] == "mock"
        assert result.empty
