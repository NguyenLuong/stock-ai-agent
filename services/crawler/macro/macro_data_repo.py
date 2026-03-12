"""Data persistence for macro indicators.

Saves macro indicator values to the market_data table with data_type="macro_indicator".
Deduplicates by (indicator_name, date(data_as_of)) — one value per indicator per day.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import func, select

from shared.db.client import get_async_session
from shared.db.orm.market_data import MarketData
from shared.logging import get_logger
from shared.models.market_data import MarketDataCreate

logger = get_logger("macro_data_repo")


async def save_macro_indicators(indicators: list[MarketDataCreate]) -> int:
    """Save macro indicators to market_data table with upsert (skip existing).

    Deduplicates by (indicator_name, date(data_as_of), data_type="macro_indicator").

    Returns count of newly inserted rows.
    """
    if not indicators:
        return 0

    async with get_async_session() as session:
        # Batch-fetch existing records to avoid N+1 queries
        indicator_dates = [
            (ind.indicator_name, ind.data_as_of.date()) for ind in indicators
        ]
        indicator_names = list({name for name, _ in indicator_dates})

        existing_result = await session.execute(
            select(
                MarketData.indicator_name,
                func.date(MarketData.data_as_of),
            ).where(
                MarketData.data_type == "macro_indicator",
                MarketData.indicator_name.in_(indicator_names),
            )
        )
        existing_set = {(row[0], row[1]) for row in existing_result.all()}

        inserted = 0
        for indicator in indicators:
            key = (indicator.indicator_name, indicator.data_as_of.date())
            if key in existing_set:
                continue
            record = MarketData(**indicator.model_dump())
            session.add(record)
            inserted += 1

        await session.commit()

    logger.info(
        "macro_indicators_saved",
        inserted=inserted,
        total=len(indicators),
        component="macro_data_repo",
    )
    return inserted


async def get_last_macro_fetch_time() -> datetime | None:
    """Return the most recent data_as_of timestamp for any macro indicator.

    Used by AC3 to log last successful fetch when all sources are unavailable.
    """
    async with get_async_session() as session:
        result = await session.execute(
            select(func.max(MarketData.data_as_of)).where(
                MarketData.data_type == "macro_indicator",
            )
        )
        return result.scalar_one_or_none()
