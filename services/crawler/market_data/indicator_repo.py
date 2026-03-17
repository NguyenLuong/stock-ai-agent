"""Data persistence for technical indicators.

Saves calculated indicator values to the market_data table
using shared async DB sessions and the MarketData ORM model.
Upsert composite key: (ticker_symbol, data_type="technical_indicator", indicator_name, data_as_of).
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pandas as pd
from sqlalchemy import select, desc

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
    Implements upsert: updates existing records if value changed, inserts new ones.
    Composite key: (ticker_symbol, data_type, indicator_name, data_as_of).

    Returns number of rows inserted + updated.
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
    updated = 0
    async with get_async_session() as session:
        # Batch-fetch existing records for this ticker+date
        indicator_names = [ind["indicator_name"] for ind in valid_indicators]
        existing_result = await session.execute(
            select(MarketData).where(
                MarketData.ticker_symbol == ticker,
                MarketData.data_type == "technical_indicator",
                MarketData.data_as_of == data_as_of,
                MarketData.indicator_name.in_(indicator_names),
            )
        )
        existing_map = {
            row.indicator_name: row for row in existing_result.scalars().all()
        }

        for ind in valid_indicators:
            new_value = Decimal(str(ind["indicator_value"]))

            if ind["indicator_name"] in existing_map:
                existing_record = existing_map[ind["indicator_name"]]
                if existing_record.indicator_value != new_value:
                    existing_record.indicator_value = new_value
                    existing_record.data_source = data_source
                    updated += 1
                continue

            validated = MarketDataCreate(
                ticker_symbol=ticker,
                data_type="technical_indicator",
                indicator_name=ind["indicator_name"],
                indicator_value=new_value,
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
        updated=updated,
        total_indicators=len(valid_indicators),
        data_as_of=data_as_of.isoformat(),
        component="technical_indicators",
    )
    return inserted + updated



async def get_stock_prices_df(ticker: str, limit: int = 300) -> pd.DataFrame:
    """Fetch most recent OHLCV data from DB as pandas DataFrame for indicator calculation.

    Returns DataFrame with columns: [time, open, high, low, close, volume]
    sorted by data_as_of ASC (oldest first -- required for pandas-ta).
    limit=300 fetches the 300 most recent rows, enough for MA200 + buffer.
    """
    # Subquery: get the N most recent rows (DESC), then sort ASC for pandas-ta
    async with get_async_session() as session:
        subquery = (
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
            .order_by(desc(MarketData.data_as_of))
            .limit(limit)
        ).subquery()

        result = await session.execute(
            select(subquery).order_by(subquery.c.data_as_of.asc())
        )
        rows = result.all()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df["volume"] = df["volume"].astype(float)
    return df
