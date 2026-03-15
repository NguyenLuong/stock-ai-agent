"""Data persistence for technical indicators.

Saves calculated indicator values to the market_data table
using shared async DB sessions and the MarketData ORM model.
Upsert composite key: (ticker_symbol, data_type="technical_indicator", indicator_name, data_as_of).
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pandas as pd
from sqlalchemy import select

from shared.db.client import get_async_session
from shared.db.orm.market_data import MarketData
from shared.logging import get_logger
from shared.models.market_data import MarketDataCreate

logger = get_logger("indicator_repo")


async def save_technical_indicators(
    ticker: str,
    indicators: list[dict],
    data_as_of: datetime,
    data_source: str = "calculated",
) -> int:
    """Save technical indicator records to market_data table.

    Each indicator dict has: indicator_name (str), indicator_value (Decimal|None).
    Implements upsert: skips if (ticker_symbol, data_type, indicator_name, data_as_of) exists.

    Returns number of rows inserted.
    """
    if not indicators:
        return 0

    # Ensure timezone-aware
    if data_as_of.tzinfo is None:
        data_as_of = data_as_of.replace(tzinfo=timezone.utc)

    # Filter out None values
    valid_indicators = [
        ind for ind in indicators if ind.get("indicator_value") is not None
    ]
    if not valid_indicators:
        return 0

    inserted = 0
    async with get_async_session() as session:
        # Batch-fetch existing records for this ticker+date
        indicator_names = [ind["indicator_name"] for ind in valid_indicators]
        existing_result = await session.execute(
            select(MarketData.indicator_name).where(
                MarketData.ticker_symbol == ticker,
                MarketData.data_type == "technical_indicator",
                MarketData.data_as_of == data_as_of,
                MarketData.indicator_name.in_(indicator_names),
            )
        )
        existing_names = {row[0] for row in existing_result.all()}

        for ind in valid_indicators:
            if ind["indicator_name"] in existing_names:
                continue

            validated = MarketDataCreate(
                ticker_symbol=ticker,
                data_type="technical_indicator",
                indicator_name=ind["indicator_name"],
                indicator_value=Decimal(str(ind["indicator_value"])),
                data_as_of=data_as_of,
                data_source=data_source,
            )
            record = MarketData(**validated.model_dump())
            session.add(record)
            inserted += 1

        await session.commit()

    logger.info(
        "technical_indicators_saved",
        ticker=ticker,
        inserted=inserted,
        total_indicators=len(valid_indicators),
        data_as_of=data_as_of.isoformat(),
        component="technical_indicators",
    )
    return inserted


async def get_latest_indicator_date(ticker: str) -> datetime | None:
    """Get the most recent data_as_of for technical indicators of a ticker.

    Returns None if no indicators exist for this ticker.
    """
    async with get_async_session() as session:
        result = await session.execute(
            select(MarketData.data_as_of)
            .where(
                MarketData.ticker_symbol == ticker,
                MarketData.data_type == "technical_indicator",
            )
            .order_by(MarketData.data_as_of.desc())
            .limit(1)
        )
        row = result.first()
        return row[0] if row else None


async def get_stock_prices_df(ticker: str, limit: int = 300) -> pd.DataFrame:
    """Fetch OHLCV data from DB as pandas DataFrame for indicator calculation.

    Returns DataFrame with columns: [time, open, high, low, close, volume]
    sorted by data_as_of ASC (oldest first -- required for pandas-ta).
    limit=300 provides enough data for MA200 + buffer.
    """
    async with get_async_session() as session:
        result = await session.execute(
            select(
                MarketData.data_as_of,
                MarketData.open_price,
                MarketData.high_price,
                MarketData.low_price,
                MarketData.close_price,
                MarketData.volume,
            )
            .where(MarketData.ticker_symbol == ticker)
            .where(MarketData.data_type == "stock_price")
            .order_by(MarketData.data_as_of.asc())
            .limit(limit)
        )
        rows = result.all()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df["volume"] = df["volume"].astype(float)
    return df
