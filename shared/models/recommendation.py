"""Recommendation model — final AI investment recommendations."""
import uuid
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel


class RecommendationBase(BaseModel):
    type: str
    ticker_symbol: str | None = None
    content: str
    bull_case: str | None = None
    bear_case: str | None = None
    confidence_score: Decimal | None = None
    risk_level: str | None = None
    agents_used: list[str] | None = None
    agents_failed: list[str] | None = None
    data_sources: dict | None = None


class RecommendationCreate(RecommendationBase):
    pass


class Recommendation(RecommendationBase):
    id: uuid.UUID
    created_at: datetime

    model_config = {"from_attributes": True}
