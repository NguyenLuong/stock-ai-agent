"""Pydantic models for macro data crawler results."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from shared.models.market_data import MarketDataCreate


@dataclass
class MacroDataResult:
    """Result of fetching a single macro indicator."""

    indicator_name: str
    indicator_value: float | None
    data_as_of: datetime
    data_source: str  # "vnstock", "sbv.gov.vn", or "mock"
    success: bool
    error: str | None = None

    def to_market_data_create(self) -> MarketDataCreate:
        """Convert to MarketDataCreate for DB persistence."""
        return MarketDataCreate(
            ticker_symbol=None,
            data_type="macro_indicator",
            indicator_name=self.indicator_name,
            indicator_value=Decimal(str(self.indicator_value)) if self.indicator_value is not None else None,
            data_as_of=self.data_as_of,
            data_source=self.data_source,
        )


@dataclass
class MacroCrawlResult:
    """Aggregate result of a macro crawl cycle."""

    results: list[MacroDataResult] = field(default_factory=list)
    saved_count: int = 0

    @property
    def succeeded(self) -> list[MacroDataResult]:
        return [r for r in self.results if r.success]

    @property
    def failed(self) -> list[MacroDataResult]:
        return [r for r in self.results if not r.success]

    @property
    def failed_indicators(self) -> list[str]:
        return [r.indicator_name for r in self.failed]
