"""Technical indicator calculation orchestrator.

Coordinates indicator calculation for all configured tickers:
1. Load ticker config
2. Check trading day
3. Fetch VN-Index data once
4. For each ticker: fetch OHLCV -> calculate -> save
5. Return structured result
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from services.crawler.market_data.indicator_calculator import calculate_indicators
from services.crawler.market_data.indicator_repo import (
    get_stock_prices_df,
    save_technical_indicators,
)
from services.crawler.market_data.stock_crawl_manager import is_trading_day
from services.crawler.market_data.ticker_config import load_ticker_config
from shared.logging import get_logger
from shared.utils.datetime_utils import now_utc

logger = get_logger("indicator_manager")

VNINDEX_TICKER = "VNINDEX"


@dataclass
class IndicatorCalculationResult:
    """Summary of indicator calculation execution."""

    total_tickers: int
    success_count: int
    failed_count: int
    rows_inserted: int
    indicators_calculated: int
    duration_seconds: float
    skipped_reason: str | None
    errors: list[dict] = field(default_factory=list)


async def run_indicator_calculation() -> IndicatorCalculationResult:
    """Orchestrate technical indicator calculation for all configured tickers.

    1. Load ticker config
    2. Check trading day — skip if non-trading day
    3. Fetch VN-Index OHLCV once for RS calculation
    4. For each ticker: fetch OHLCV -> calculate indicators -> save
    5. Rate limit with asyncio.sleep(0.1) between tickers
    """
    start = time.monotonic()

    config = load_ticker_config()
    tickers = config.tickers
    today = now_utc().date()

    if not is_trading_day(today, holidays=config.holidays):
        duration = round(time.monotonic() - start, 2)
        logger.info(
            "indicator_calculation_skipped",
            reason="non_trading_day",
            date=today.isoformat(),
            component="technical_indicators",
        )
        return IndicatorCalculationResult(
            total_tickers=len(tickers),
            success_count=0,
            failed_count=0,
            rows_inserted=0,
            indicators_calculated=0,
            duration_seconds=duration,
            skipped_reason="non_trading_day",
        )

    if not tickers:
        duration = round(time.monotonic() - start, 2)
        return IndicatorCalculationResult(
            total_tickers=0,
            success_count=0,
            failed_count=0,
            rows_inserted=0,
            indicators_calculated=0,
            duration_seconds=duration,
            skipped_reason=None,
        )

    # Fetch VN-Index data once for Relative Strength calculation
    df_vnindex = None
    try:
        df_vnindex = await get_stock_prices_df(VNINDEX_TICKER, limit=300)
        if df_vnindex.empty:
            df_vnindex = None
            logger.warning(
                "vnindex_data_not_found",
                message="VN-Index data not available in DB, RS calculation will be skipped",
                component="technical_indicators",
            )
    except Exception as exc:
        logger.warning(
            "vnindex_fetch_failed",
            error=str(exc),
            component="technical_indicators",
        )

    success_count = 0
    failed_count = 0
    total_rows = 0
    total_indicators = 0
    errors: list[dict] = []

    for i, ticker in enumerate(tickers):
        try:
            df_ohlcv = await get_stock_prices_df(ticker, limit=300)
            if df_ohlcv.empty:
                logger.warning(
                    "no_ohlcv_data",
                    ticker=ticker,
                    component="technical_indicators",
                )
                continue

            indicator_records = calculate_indicators(ticker, df_ohlcv, df_vnindex)
            if not indicator_records:
                logger.warning(
                    "no_indicators_calculated",
                    ticker=ticker,
                    component="technical_indicators",
                )
                success_count += 1
                continue

            # Convert IndicatorRecord to dict for save
            indicators_to_save = [
                {
                    "indicator_name": rec.indicator_name,
                    "indicator_value": rec.indicator_value,
                }
                for rec in indicator_records
            ]

            data_as_of = indicator_records[0].data_as_of
            inserted = await save_technical_indicators(
                ticker=ticker,
                indicators=indicators_to_save,
                data_as_of=data_as_of,
                data_source="calculated",
            )

            total_rows += inserted
            total_indicators += len(indicator_records)
            success_count += 1

            logger.info(
                "ticker_indicators_done",
                ticker=ticker,
                indicators=len(indicator_records),
                inserted=inserted,
                component="technical_indicators",
            )
        except Exception as exc:
            failed_count += 1
            errors.append({"ticker": ticker, "error": str(exc)})
            logger.warning(
                "ticker_indicator_failed",
                ticker=ticker,
                error=str(exc),
                component="technical_indicators",
            )

        if i < len(tickers) - 1:
            await asyncio.sleep(0.1)

    duration = round(time.monotonic() - start, 2)

    logger.info(
        "indicator_calculation_complete",
        total_tickers=len(tickers),
        success=success_count,
        failed=failed_count,
        rows_inserted=total_rows,
        indicators_calculated=total_indicators,
        duration_seconds=duration,
        component="technical_indicators",
    )

    return IndicatorCalculationResult(
        total_tickers=len(tickers),
        success_count=success_count,
        failed_count=failed_count,
        rows_inserted=total_rows,
        indicators_calculated=total_indicators,
        duration_seconds=duration,
        skipped_reason=None,
        errors=errors,
    )
