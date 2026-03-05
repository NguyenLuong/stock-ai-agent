"""MarketData model — OHLCV price data, macro indicators, financial ratios."""
import uuid
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel


class MarketDataBase(BaseModel):
    ticker_symbol: str | None = None
    data_type: str
    # Stock price fields
    open_price: Decimal | None = None
    high_price: Decimal | None = None
    low_price: Decimal | None = None
    close_price: Decimal | None = None
    volume: int | None = None
    # Macro indicator fields
    indicator_name: str | None = None
    indicator_value: Decimal | None = None
    # Financial ratio fields
    pe_ratio: Decimal | None = None
    pb_ratio: Decimal | None = None
    roe: Decimal | None = None
    eps: Decimal | None = None
    eps_growth_yoy: Decimal | None = None
    # Metadata
    data_as_of: datetime
    data_source: str


class MarketDataCreate(MarketDataBase):
    pass


class MarketData(MarketDataBase):
    id: uuid.UUID
    created_at: datetime

    model_config = {"from_attributes": True}
