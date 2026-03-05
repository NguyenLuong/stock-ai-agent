"""SQLAlchemy ORM model for the market_data table."""
import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import BigInteger, DateTime, Index, Numeric, String, func, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from shared.db.base import Base


class MarketData(Base):
    __tablename__ = "market_data"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    ticker_symbol: Mapped[str | None] = mapped_column(String(20), nullable=True)
    data_type: Mapped[str] = mapped_column(String(50), nullable=False)
    # Stock price fields
    open_price: Mapped[Decimal | None] = mapped_column(Numeric(15, 2), nullable=True)
    high_price: Mapped[Decimal | None] = mapped_column(Numeric(15, 2), nullable=True)
    low_price: Mapped[Decimal | None] = mapped_column(Numeric(15, 2), nullable=True)
    close_price: Mapped[Decimal | None] = mapped_column(Numeric(15, 2), nullable=True)
    volume: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    # Macro indicator fields
    indicator_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    indicator_value: Mapped[Decimal | None] = mapped_column(
        Numeric(20, 6), nullable=True
    )
    # Financial ratio fields
    pe_ratio: Mapped[Decimal | None] = mapped_column(Numeric(10, 4), nullable=True)
    pb_ratio: Mapped[Decimal | None] = mapped_column(Numeric(10, 4), nullable=True)
    roe: Mapped[Decimal | None] = mapped_column(Numeric(10, 4), nullable=True)
    eps: Mapped[Decimal | None] = mapped_column(Numeric(15, 2), nullable=True)
    eps_growth_yoy: Mapped[Decimal | None] = mapped_column(
        Numeric(10, 4), nullable=True
    )
    # Metadata
    data_as_of: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    data_source: Mapped[str] = mapped_column(String(50), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("idx_market_data_ticker", "ticker_symbol"),
        Index("idx_market_data_type", "data_type"),
        Index("idx_market_data_as_of", text("data_as_of DESC")),
    )
