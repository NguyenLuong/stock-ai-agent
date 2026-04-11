"""Recommendation persistence — save morning briefing and alert outputs."""

from __future__ import annotations

import uuid

from shared.db import get_async_session
from shared.db.orm.recommendation import Recommendation
from shared.logging import get_logger
from shared.models.recommendation import RecommendationCreate

logger = get_logger("recommendation_repo")


async def save_recommendation(rec: RecommendationCreate) -> uuid.UUID:
    """Persist a recommendation to the recommendations table.

    Returns the generated UUID for the new record.
    """
    async with get_async_session() as session:
        obj = Recommendation(**rec.model_dump())
        session.add(obj)
        await session.commit()
        await session.refresh(obj)

        logger.info(
            "recommendation_saved",
            component="recommendation_repo",
            type=rec.type,
            ticker_symbol=rec.ticker_symbol,
            id=str(obj.id),
        )
        return obj.id
