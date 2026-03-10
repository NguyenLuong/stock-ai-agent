"""Mock data generators for fallback when vnstock API is unavailable.

All mock DataFrames mirror the column structure of real vnstock responses
and include data_source="mock" in attrs metadata.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd


def generate_mock_stock_price(ticker: str, rows: int = 250) -> pd.DataFrame:
    """Generate realistic mock OHLCV stock price data.

    Returns DataFrame with columns [time, open, high, low, close, volume]
    matching vnstock Quote.history() output structure.
    """
    end_date = datetime.now(tz=timezone.utc)
    dates = [end_date - timedelta(days=i) for i in range(rows)]
    dates.reverse()

    rng = np.random.default_rng(seed=hash(ticker) % 2**32)

    # Start with a realistic base price
    base_price = 25.0 + rng.random() * 50.0
    prices = [base_price]
    for _ in range(1, rows):
        change = rng.normal(0, 0.02)  # ~2% daily volatility
        prices.append(max(1.0, prices[-1] * (1 + change)))

    close_prices = np.array(prices)
    # Generate OHLV from close
    daily_range = close_prices * rng.uniform(0.01, 0.04, size=rows)
    open_prices = close_prices + rng.normal(0, 1, size=rows) * daily_range * 0.3
    high_prices = np.maximum(open_prices, close_prices) + abs(
        rng.normal(0, 1, size=rows)
    ) * daily_range * 0.5
    low_prices = np.minimum(open_prices, close_prices) - abs(
        rng.normal(0, 1, size=rows)
    ) * daily_range * 0.5

    volumes = rng.integers(100_000, 10_000_000, size=rows)

    df = pd.DataFrame(
        {
            "time": dates,
            "open": np.round(open_prices, 2),
            "high": np.round(high_prices, 2),
            "low": np.round(low_prices, 2),
            "close": np.round(close_prices, 2),
            "volume": volumes,
        }
    )
    df.attrs["data_source"] = "mock"
    return df


def generate_mock_financial_ratios(ticker: str) -> pd.DataFrame:
    """Generate realistic mock financial ratios data.

    Returns DataFrame matching vnstock Finance.ratio() output structure
    with P/E, P/B, ROE, EPS fields.
    """
    rng = np.random.default_rng(seed=hash(ticker) % 2**32)

    # Generate ratio data in KBS format: rows = ratio items, columns = periods
    periods = ["2025-Q4", "2025-Q3", "2025-Q2", "2025-Q1"]

    items = {
        "P/E": [round(rng.uniform(5, 25), 2) for _ in periods],
        "P/B": [round(rng.uniform(0.5, 5.0), 2) for _ in periods],
        "ROE": [round(rng.uniform(5, 30), 2) for _ in periods],
        "EPS": [round(rng.uniform(1000, 8000), 0) for _ in periods],
        "EPS Growth YoY": [round(rng.uniform(-20, 40), 2) for _ in periods],
    }

    rows = []
    for item_name, values in items.items():
        row: dict = {"item": item_name}
        for period, val in zip(periods, values):
            row[period] = val
        rows.append(row)

    df = pd.DataFrame(rows)
    df.attrs["data_source"] = "mock"
    return df
