"""Stock crawl orchestrator — coordinates stock price history crawl for all configured tickers.

Handles initial bulk load (1Y) for new tickers and daily incremental updates (1b)
for existing tickers. Includes trading day detection and rate limiting.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import date

from market_data.stock_data_repo import (
    count_stock_prices_batch,
    save_stock_prices,
)
from market_data.ticker_config import load_ticker_config
from market_data.vnstock_client import VnstockClient
from shared.logging import get_logger
from shared.utils.datetime_utils import now_utc

logger = get_logger("stock_crawl_manager")

INITIAL_THRESHOLD = 30

VN_FIXED_HOLIDAYS = [
    (1, 1),   # Tet Duong lich
    (4, 30),  # Ngay Giai phong
    (5, 1),   # Quoc te Lao dong
    (9, 2),   # Quoc khanh
]


@dataclass
class StockCrawlResult:
    """Summary of stock crawl execution."""

    total_tickers: int
    success_count: int
    failed_count: int
    initial_count: int
    incremental_count: int
    skipped_count: int
    rows_inserted: int
    duration_seconds: float
    skipped_reason: str | None
    errors: list[dict] = field(default_factory=list)


def is_trading_day(d: date, holidays: list[date] | None = None) -> bool:
    """Check if date is a VN stock market trading day.

    Returns False for weekends, VN fixed holidays, and variable holidays from config.
    """
    if d.weekday() >= 5:
        return False
    if (d.month, d.day) in VN_FIXED_HOLIDAYS:
        return False
    if holidays:
        holiday_years = {h.year for h in holidays}
        if d.year not in holiday_years:
            logger.warning(
                "no_holiday_config_for_year",
                year=d.year,
                component="stock_crawler",
            )
        if d in holidays:
            return False
    return True


async def run_stock_crawl() -> StockCrawlResult:
    """Orchestrate stock price crawl for all configured tickers.

    1. Load ticker config
    2. Check trading day
    3. For each ticker: initial (1Y) or incremental (1b) crawl
    4. Rate limit between requests
    """
    start = time.monotonic()

    config = load_ticker_config()
    tickers = config.tickers
    today = now_utc().date()
    trading_day = is_trading_day(today, holidays=config.holidays)

    if not trading_day:
        logger.info(
            "non_trading_day_detected",
            date=today.isoformat(),
            component="stock_crawler",
        )

    if not tickers:
        return StockCrawlResult(
            total_tickers=0,
            success_count=0,
            failed_count=0,
            initial_count=0,
            incremental_count=0,
            skipped_count=0,
            rows_inserted=0,
            duration_seconds=round(time.monotonic() - start, 2),
            skipped_reason="non_trading_day" if not trading_day else None,
            errors=[],
        )

    # Batch count existing rows
    row_counts = await count_stock_prices_batch(tickers)

    # Determine which tickers need initial vs incremental
    initial_tickers: list[str] = []
    incremental_tickers: list[str] = []
    for ticker in tickers:
        if row_counts.get(ticker, 0) < INITIAL_THRESHOLD:
            initial_tickers.append(ticker)
        else:
            incremental_tickers.append(ticker)

    # On non-trading day: skip incremental, only run initial for new tickers
    skipped_count = 0
    if not trading_day:
        skipped_count = len(incremental_tickers)
        incremental_tickers = []

    # If nothing to do
    if not initial_tickers and not incremental_tickers:
        return StockCrawlResult(
            total_tickers=len(tickers),
            success_count=0,
            failed_count=0,
            initial_count=0,
            incremental_count=0,
            skipped_count=skipped_count,
            rows_inserted=0,
            duration_seconds=round(time.monotonic() - start, 2),
            skipped_reason="non_trading_day" if not trading_day else None,
            errors=[],
        )

    client = VnstockClient()
    success_count = 0
    failed_count = 0
    initial_done = 0
    incremental_done = 0
    total_rows = 0
    errors: list[dict] = []

    # Process initial tickers
    for i, ticker in enumerate(initial_tickers):
        try:
            df = client.get_stock_history(ticker, length="1Y")
            data_source = df.attrs.get("data_source", "vnstock")
            inserted = await save_stock_prices(ticker, df, data_source)
            total_rows += inserted
            initial_done += 1
            success_count += 1
            logger.info(
                "stock_initial_crawl_done",
                ticker=ticker,
                rows=inserted,
                component="stock_crawler",
            )
        except Exception as exc:
            failed_count += 1
            errors.append({"ticker": ticker, "error": str(exc)})
            logger.warning(
                "stock_crawl_ticker_failed",
                ticker=ticker,
                error=str(exc),
                crawl_type="initial",
                component="stock_crawler",
            )
        if i < len(initial_tickers) - 1 or incremental_tickers:
            await asyncio.sleep(1.0)

    # Process incremental tickers
    for i, ticker in enumerate(incremental_tickers):
        try:
            df = client.get_stock_history(ticker, length="1b")
            data_source = df.attrs.get("data_source", "vnstock")
            inserted = await save_stock_prices(ticker, df, data_source)
            total_rows += inserted
            incremental_done += 1
            success_count += 1
            logger.info(
                "stock_incremental_crawl_done",
                ticker=ticker,
                rows=inserted,
                component="stock_crawler",
            )
        except Exception as exc:
            failed_count += 1
            errors.append({"ticker": ticker, "error": str(exc)})
            logger.warning(
                "stock_crawl_ticker_failed",
                ticker=ticker,
                error=str(exc),
                crawl_type="incremental",
                component="stock_crawler",
            )
        if i < len(incremental_tickers) - 1:
            await asyncio.sleep(1.0)

    duration = round(time.monotonic() - start, 2)

    logger.info(
        "stock_crawl_complete",
        total_tickers=len(tickers),
        success=success_count,
        failed=failed_count,
        initial=initial_done,
        incremental=incremental_done,
        skipped=skipped_count,
        rows_inserted=total_rows,
        duration_seconds=duration,
        component="stock_crawler",
    )

    return StockCrawlResult(
        total_tickers=len(tickers),
        success_count=success_count,
        failed_count=failed_count,
        initial_count=initial_done,
        incremental_count=incremental_done,
        skipped_count=skipped_count,
        rows_inserted=total_rows,
        duration_seconds=duration,
        skipped_reason="non_trading_day" if not trading_day else None,
        errors=errors,
    )
