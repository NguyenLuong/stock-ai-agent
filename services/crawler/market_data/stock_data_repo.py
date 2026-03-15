"""Data persistence for stock market data.

Saves OHLCV prices and financial ratios to the market_data table
using shared async DB sessions and the MarketData ORM model.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pandas as pd
from sqlalchemy import func, select

from shared.db.client import get_async_session
from shared.db.orm.market_data import MarketData
from shared.logging import get_logger
from shared.models.market_data import MarketDataCreate

logger = get_logger("stock_data_repo")


async def save_stock_prices(
    ticker: str,
    df: pd.DataFrame,
    data_source: str,
) -> int:
    """Save OHLCV price rows to market_data table.

    Maps DataFrame columns: time→data_as_of, open→open_price, high→high_price,
    low→low_price, close→close_price, volume→volume.

    Implements upsert: skips if (ticker_symbol, data_type, data_as_of) already exists.

    Returns number of rows inserted.
    """
    if df.empty:
        return 0

    # Build all candidate records with Pydantic validation
    candidates: list[tuple[datetime, MarketDataCreate]] = []
    for _, row in df.iterrows():
        data_as_of = _to_aware_datetime(row["time"])
        validated = MarketDataCreate(
            ticker_symbol=ticker,
            data_type="stock_price",
            open_price=Decimal(str(row["open"])),
            high_price=Decimal(str(row["high"])),
            low_price=Decimal(str(row["low"])),
            close_price=Decimal(str(row["close"])),
            volume=int(row["volume"]),
            data_as_of=data_as_of,
            data_source=data_source,
        )
        candidates.append((data_as_of, validated))

    inserted = 0
    async with get_async_session() as session:
        # Batch-fetch existing records to avoid N+1 queries
        all_dates = [c[0] for c in candidates]
        existing_result = await session.execute(
            select(MarketData.data_as_of).where(
                MarketData.ticker_symbol == ticker,
                MarketData.data_type == "stock_price",
                MarketData.data_as_of.in_(all_dates),
            )
        )
        existing_dates = {row[0] for row in existing_result.all()}

        for data_as_of, validated in candidates:
            if data_as_of in existing_dates:
                continue
            record = MarketData(**validated.model_dump())
            session.add(record)
            inserted += 1

        await session.commit()

    logger.info(
        "stock_prices_saved",
        ticker=ticker,
        inserted=inserted,
        total_rows=len(df),
        data_source=data_source,
        component="stock_data_repo",
    )
    return inserted


async def count_stock_prices_batch(tickers: list[str]) -> dict[str, int]:
    """Batch count stock_price rows for multiple tickers."""
    if not tickers:
        return {}
    async with get_async_session() as session:
        result = await session.execute(
            select(
                MarketData.ticker_symbol,
                func.count(MarketData.id),
            )
            .where(MarketData.ticker_symbol.in_(tickers))
            .where(MarketData.data_type == "stock_price")
            .group_by(MarketData.ticker_symbol)
        )
        counts = {row[0]: row[1] for row in result.all()}
        return {t: counts.get(t, 0) for t in tickers}


async def save_financial_ratios(
    ticker: str,
    df: pd.DataFrame,
    data_source: str,
) -> int:
    """Save financial ratio rows to market_data table.

    Extracts P/E, P/B, ROE, EPS from ratio DataFrame.
    Implements upsert: skips if (ticker_symbol, data_type, data_as_of) already exists.

    Returns number of rows inserted.
    """
    if df.empty:
        return 0

    # Parse ratio data — KBS format: rows=items, columns=periods
    ratio_map = _extract_ratios_from_df(df)
    if not ratio_map:
        return 0

    # Build candidates with Pydantic validation
    candidates: list[tuple[datetime, MarketDataCreate]] = []
    for period_str, ratios in ratio_map.items():
        data_as_of = _period_to_datetime(period_str)
        validated = MarketDataCreate(
            ticker_symbol=ticker,
            data_type="financial_ratio",
            pe_ratio=ratios.get("pe_ratio"),
            pb_ratio=ratios.get("pb_ratio"),
            roe=ratios.get("roe"),
            eps=ratios.get("eps"),
            eps_growth_yoy=ratios.get("eps_growth_yoy"),
            data_as_of=data_as_of,
            data_source=data_source,
        )
        candidates.append((data_as_of, validated))

    inserted = 0
    async with get_async_session() as session:
        # Batch-fetch existing records to avoid N+1 queries
        all_dates = [c[0] for c in candidates]
        existing_result = await session.execute(
            select(MarketData.data_as_of).where(
                MarketData.ticker_symbol == ticker,
                MarketData.data_type == "financial_ratio",
                MarketData.data_as_of.in_(all_dates),
            )
        )
        existing_dates = {row[0] for row in existing_result.all()}

        for data_as_of, validated in candidates:
            if data_as_of in existing_dates:
                continue
            record = MarketData(**validated.model_dump())
            session.add(record)
            inserted += 1

        await session.commit()

    logger.info(
        "financial_ratios_saved",
        ticker=ticker,
        inserted=inserted,
        data_source=data_source,
        component="stock_data_repo",
    )
    return inserted


def _extract_ratios_from_df(
    df: pd.DataFrame,
) -> dict[str, dict[str, Decimal | None]]:
    """Extract ratio values from KBS-format DataFrame.

    KBS format: rows have 'item' column with ratio names,
    period columns like '2025-Q4', '2024', etc.
    """
    if "item" not in df.columns:
        return {}

    # Get period columns (exclude 'item' and 'item_id')
    period_cols = [c for c in df.columns if c not in ("item", "item_id")]

    # Build ratio name mapping (case-insensitive)
    ratio_key_map = {
        "p/e": "pe_ratio",
        "pe": "pe_ratio",
        "p/b": "pb_ratio",
        "pb": "pb_ratio",
        "roe": "roe",
        "eps": "eps",
        "eps growth yoy": "eps_growth_yoy",
    }

    result: dict[str, dict[str, Decimal | None]] = {}

    for period in period_cols:
        ratios: dict[str, Decimal | None] = {}
        for _, row in df.iterrows():
            item_name = str(row["item"]).strip().lower()
            orm_field = ratio_key_map.get(item_name)
            if orm_field is not None:
                val = row.get(period)
                if pd.notna(val):
                    ratios[orm_field] = Decimal(str(val))
        if ratios:
            result[period] = ratios

    return result


def _period_to_datetime(period_str: str) -> datetime:
    """Convert period string like '2025-Q4' or '2024' to datetime."""
    if "-Q" in period_str:
        year, quarter = period_str.split("-Q")
        month = int(quarter) * 3
        return datetime(int(year), month, 1, tzinfo=timezone.utc)
    # Year-only format
    return datetime(int(period_str), 12, 31, tzinfo=timezone.utc)


def _to_aware_datetime(value: object) -> datetime:
    """Convert a value to timezone-aware datetime."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    if isinstance(value, pd.Timestamp):
        dt = value.to_pydatetime()
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt
    # Fallback: try parsing string
    dt = pd.Timestamp(value).to_pydatetime()
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt
