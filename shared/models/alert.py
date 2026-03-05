"""Alert model — real-time event alerts for Telegram delivery."""
import uuid
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel


class AlertBase(BaseModel):
    ticker_symbol: str
    event_type: str
    severity: str
    detected_at: datetime
    raw_data: dict
    analysis: str | None = None
    confidence_score: Decimal | None = None
    telegram_sent: bool = False
    telegram_sent_at: datetime | None = None


class AlertCreate(AlertBase):
    pass


class Alert(AlertBase):
    id: uuid.UUID
    created_at: datetime

    model_config = {"from_attributes": True}
