"""Unit tests for mock data generators."""

from __future__ import annotations

from services.crawler.market_data.mock_data import (
    generate_mock_financial_ratios,
    generate_mock_stock_price,
)


class TestGenerateMockStockPrice:
    """Tests for generate_mock_stock_price."""

    def test_has_correct_columns(self) -> None:
        """Test mock DataFrame has OHLCV columns matching vnstock output."""
        df = generate_mock_stock_price("HPG")
        expected = {"time", "open", "high", "low", "close", "volume"}
        assert expected == set(df.columns)

    def test_has_realistic_values(self) -> None:
        """Test mock data has positive, realistic price and volume values."""
        df = generate_mock_stock_price("HPG")
        assert len(df) == 250  # Default rows
        assert (df["open"] > 0).all()
        assert (df["close"] > 0).all()
        assert (df["high"] >= df["low"]).all()
        assert (df["volume"] > 0).all()

    def test_has_mock_data_source(self) -> None:
        """Test mock DataFrame has data_source='mock' in attrs."""
        df = generate_mock_stock_price("VNM")
        assert df.attrs["data_source"] == "mock"

    def test_different_tickers_produce_different_data(self) -> None:
        """Test different tickers seed different random data."""
        df1 = generate_mock_stock_price("HPG")
        df2 = generate_mock_stock_price("VNM")
        assert not df1["close"].equals(df2["close"])


class TestGenerateMockFinancialRatios:
    """Tests for generate_mock_financial_ratios."""

    def test_has_correct_columns(self) -> None:
        """Test mock ratios DataFrame has item column and period columns."""
        df = generate_mock_financial_ratios("HPG")
        assert "item" in df.columns
        # Should have period columns beyond 'item'
        period_cols = [c for c in df.columns if c != "item"]
        assert len(period_cols) > 0

    def test_has_pe_pb_roe_eps(self) -> None:
        """Test all required ratio items are present."""
        df = generate_mock_financial_ratios("HPG")
        items = df["item"].tolist()
        assert "P/E" in items
        assert "P/B" in items
        assert "ROE" in items
        assert "EPS" in items

    def test_has_mock_data_source(self) -> None:
        """Test mock DataFrame has data_source='mock' in attrs."""
        df = generate_mock_financial_ratios("HPG")
        assert df.attrs["data_source"] == "mock"
