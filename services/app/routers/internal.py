"""Internal trigger endpoints for scheduled pipeline execution.

Called by the Prefect scheduler via HTTP POST. Not intended for external use.
Security: requires X-Trigger-Source: prefect-scheduler header.
"""

from __future__ import annotations

import time
from dataclasses import asdict

from fastapi import APIRouter, Header, HTTPException

from shared.logging import get_logger

logger = get_logger("internal_trigger")

router = APIRouter(prefix="/internal/trigger", tags=["internal"])


def _validate_trigger_source(x_trigger_source: str) -> None:
    """Raise 403 if the trigger source header is not from the scheduler."""
    if x_trigger_source != "prefect-scheduler":
        raise HTTPException(status_code=403, detail="Unauthorized trigger source")


@router.post("/crawl")
async def trigger_crawl(
    x_trigger_source: str = Header(default=""),
) -> dict:
    """Trigger full crawl cycle: news + macro data."""
    _validate_trigger_source(x_trigger_source)

    from services.crawler.news.crawl_manager import run_news_crawl
    from services.crawler.macro.macro_crawl_manager import run_macro_crawl

    start = time.monotonic()
    errors: list[str] = []
    news_result_dict: dict = {}
    macro_result_dict: dict = {}

    try:
        news_result = await run_news_crawl()
        news_result_dict = asdict(news_result)
    except Exception as exc:
        errors.append(f"news_crawl: {exc}")
        logger.error(
            "news_crawl_failed",
            component="internal_trigger",
            error=str(exc),
        )

    try:
        macro_result = await run_macro_crawl()
        macro_result_dict = {
            "saved_count": macro_result.saved_count,
            "succeeded": len(macro_result.succeeded),
            "failed": len(macro_result.failed),
            "failed_indicators": macro_result.failed_indicators,
        }
    except Exception as exc:
        errors.append(f"macro_crawl: {exc}")
        logger.error(
            "macro_crawl_failed",
            component="internal_trigger",
            error=str(exc),
        )

    duration = round(time.monotonic() - start, 2)

    logger.info(
        "crawl_trigger_completed",
        component="internal_trigger",
        duration_seconds=duration,
        errors=len(errors),
    )

    return {
        "status": "ok" if not errors else "partial",
        "duration_seconds": duration,
        "news_crawl": news_result_dict,
        "macro_crawl": macro_result_dict,
        "errors": errors,
    }


@router.post("/stock-crawl")
async def trigger_stock_crawl(
    x_trigger_source: str = Header(default=""),
) -> dict:
    """Trigger stock history crawl for all configured tickers."""
    _validate_trigger_source(x_trigger_source)

    from services.crawler.market_data.stock_crawl_manager import run_stock_crawl

    start = time.monotonic()

    try:
        result = await run_stock_crawl()
        duration = round(time.monotonic() - start, 2)
        logger.info(
            "stock_crawl_trigger_completed",
            component="internal_trigger",
            duration_seconds=duration,
            success_count=result.success_count,
        )
        return {
            "status": "ok",
            "duration_seconds": duration,
            "result": {
                "total_tickers": result.total_tickers,
                "success_count": result.success_count,
                "failed_count": result.failed_count,
                "initial_count": result.initial_count,
                "incremental_count": result.incremental_count,
                "rows_inserted": result.rows_inserted,
                "duration_seconds": result.duration_seconds,
                "skipped_reason": result.skipped_reason,
                "errors": result.errors,
            },
        }
    except Exception as exc:
        duration = round(time.monotonic() - start, 2)
        logger.error(
            "stock_crawl_trigger_failed",
            component="internal_trigger",
            duration_seconds=duration,
            error=str(exc),
        )
        return {
            "status": "failed",
            "duration_seconds": duration,
            "error": str(exc),
        }


@router.post("/technical-indicators")
async def trigger_technical_indicators(
    x_trigger_source: str = Header(default=""),
) -> dict:
    """Trigger technical indicator calculation for all configured tickers."""
    _validate_trigger_source(x_trigger_source)

    from services.crawler.market_data.indicator_manager import run_indicator_calculation

    start = time.monotonic()
    try:
        result = await run_indicator_calculation()
        duration = round(time.monotonic() - start, 2)
        logger.info(
            "indicator_calculation_trigger_completed",
            component="internal_trigger",
            duration_seconds=duration,
            success_count=result.success_count,
        )
        return {
            "status": "ok",
            "duration_seconds": duration,
            "result": {
                "total_tickers": result.total_tickers,
                "success_count": result.success_count,
                "failed_count": result.failed_count,
                "rows_inserted": result.rows_inserted,
                "indicators_calculated": result.indicators_calculated,
                "duration_seconds": result.duration_seconds,
                "skipped_reason": result.skipped_reason,
                "errors": result.errors,
            },
        }
    except Exception as exc:
        duration = round(time.monotonic() - start, 2)
        logger.error(
            "indicator_calculation_trigger_failed",
            component="internal_trigger",
            duration_seconds=duration,
            error=str(exc),
        )
        return {
            "status": "failed",
            "duration_seconds": duration,
            "error": str(exc),
        }


@router.post("/embedding")
async def trigger_embedding(
    x_trigger_source: str = Header(default=""),
) -> dict:
    """Trigger article embedding pipeline."""
    _validate_trigger_source(x_trigger_source)

    from services.crawler.embedding.embedding_pipeline import run_embedding_pipeline

    start = time.monotonic()

    try:
        result = await run_embedding_pipeline()
        duration = round(time.monotonic() - start, 2)
        logger.info(
            "embedding_trigger_completed",
            component="internal_trigger",
            duration_seconds=duration,
            embedded_count=result.embedded_count,
        )
        return {
            "status": "ok",
            "duration_seconds": duration,
            "result": result.model_dump(),
        }
    except Exception as exc:
        duration = round(time.monotonic() - start, 2)
        logger.error(
            "embedding_trigger_failed",
            component="internal_trigger",
            duration_seconds=duration,
            error=str(exc),
        )
        return {
            "status": "failed",
            "duration_seconds": duration,
            "error": str(exc),
        }


@router.post("/lifecycle")
async def trigger_lifecycle(
    x_trigger_source: str = Header(default=""),
) -> dict:
    """Trigger data lifecycle cleanup pipeline."""
    _validate_trigger_source(x_trigger_source)

    from services.crawler.lifecycle.lifecycle_pipeline import run_lifecycle_pipeline

    start = time.monotonic()

    try:
        result = await run_lifecycle_pipeline()
        duration = round(time.monotonic() - start, 2)
        logger.info(
            "lifecycle_trigger_completed",
            component="internal_trigger",
            duration_seconds=duration,
            summarized_count=result.summarized_count,
        )
        return {
            "status": "ok",
            "duration_seconds": duration,
            "result": result.model_dump(),
        }
    except Exception as exc:
        duration = round(time.monotonic() - start, 2)
        logger.error(
            "lifecycle_trigger_failed",
            component="internal_trigger",
            duration_seconds=duration,
            error=str(exc),
        )
        return {
            "status": "failed",
            "duration_seconds": duration,
            "error": str(exc),
        }
