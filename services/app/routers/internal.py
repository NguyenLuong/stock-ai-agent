"""Internal trigger endpoints for scheduled pipeline execution.

Called by the Prefect scheduler via HTTP POST. Not intended for external use.
Security: requires X-Trigger-Source: prefect-scheduler header.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict
from datetime import date
from pathlib import Path

import yaml
from fastapi import APIRouter, Header, HTTPException, Request

from shared.logging import get_logger
from shared.utils.datetime_utils import now_utc, to_vn_display

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
    """Trigger news crawl cycle (including macro news via category tagging)."""
    _validate_trigger_source(x_trigger_source)

    from news.crawl_manager import run_news_crawl

    start = time.monotonic()
    errors: list[str] = []
    news_result_dict: dict = {}

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
        "errors": errors,
    }


@router.post("/stock-crawl")
async def trigger_stock_crawl(
    x_trigger_source: str = Header(default=""),
) -> dict:
    """Trigger stock history crawl for all configured tickers."""
    _validate_trigger_source(x_trigger_source)

    from market_data.stock_crawl_manager import run_stock_crawl

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

    from market_data.indicator_manager import run_indicator_calculation

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

    from embedding.embedding_pipeline import run_embedding_pipeline

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

    from lifecycle.lifecycle_pipeline import run_lifecycle_pipeline

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


@router.post("/test-telegram")
async def trigger_test_telegram(request: Request) -> dict:
    """Send a test Telegram message to verify bot connectivity."""
    telegram_bot = getattr(request.app.state, "telegram_bot", None)
    if telegram_bot is None:
        return {"status": "failed", "error": "Telegram bot not initialized"}

    test_msg = (
        f"🤖 Stock AI Agent — Telegram Bot hoạt động bình thường. "
        f"[{to_vn_display(now_utc())}]"
    )
    message_ids = await telegram_bot.sender.send_message(test_msg)
    if message_ids:
        return {"status": "ok", "message_sent": True}
    logger.error("test_telegram_failed", component="internal_trigger", reason="message queued, not delivered")
    return {"status": "failed", "error": "Message failed to deliver (queued for retry)"}


# ---------------------------------------------------------------------------
# Morning Briefing helpers
# ---------------------------------------------------------------------------

VN_FIXED_HOLIDAYS = [(1, 1), (4, 30), (5, 1), (9, 2)]


def _is_trading_day(d: date, variable_holidays: list[date] | None = None) -> bool:
    """Check if *d* is a HOSE trading day."""
    if d.weekday() >= 5:
        return False
    if (d.month, d.day) in VN_FIXED_HOLIDAYS:
        return False
    if variable_holidays and d in variable_holidays:
        return False
    return True


def _load_ticker_config() -> dict:
    """Read stock_tickers.yaml once and return the raw dict."""
    config_path = (
        Path(os.environ.get("CONFIG_DIR", "/app/config")) / "crawlers" / "stock_tickers.yaml"
    )
    try:
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.warning(
            "ticker_config_not_found",
            component="internal_trigger",
            path=str(config_path),
        )
        return {}


def _load_variable_holidays(raw: dict | None = None) -> list[date]:
    """Load variable holidays from stock_tickers.yaml."""
    if raw is None:
        raw = _load_ticker_config()
    holidays: list[date] = []
    for _year, date_list in raw.get("holidays", {}).items():
        if isinstance(date_list, list):
            for d in date_list:
                try:
                    holidays.append(date.fromisoformat(str(d)))
                except ValueError:
                    pass
    return holidays


def _load_watchlist(raw: dict | None = None) -> list[str]:
    """Load enabled tickers from stock_tickers.yaml."""
    if raw is None:
        raw = _load_ticker_config()
    tickers: list[str] = []
    seen: set[str] = set()
    for _name, group_data in raw.get("groups", {}).items():
        if not group_data.get("enabled", False):
            continue
        for ticker in group_data.get("tickers", []):
            t = str(ticker).strip().upper()
            if t and t not in seen:
                seen.add(t)
                tickers.append(t)
    return tickers


@router.post("/morning-briefing")
async def trigger_morning_briefing(
    request: Request,
    x_trigger_source: str = Header(default=""),
) -> dict:
    """Daily morning briefing pipeline — triggered by Prefect scheduler."""
    _validate_trigger_source(x_trigger_source)

    start = time.monotonic()
    started_at = now_utc()

    # AC #4: Trading day check — load config once for both holiday and watchlist reads
    today = started_at.date()
    ticker_config = _load_ticker_config()
    variable_holidays = _load_variable_holidays(ticker_config)
    if not _is_trading_day(today, variable_holidays):
        logger.info(
            "morning_briefing_skipped",
            component="internal_trigger",
            reason="non_trading_day",
            date=today.isoformat(),
        )
        return {
            "status": "skipped",
            "reason": "non_trading_day",
            "date": today.isoformat(),
            "duration_seconds": round(time.monotonic() - start, 2),
        }

    # AC #2: Load watchlist (reuse already-loaded config — no second file read)
    watchlist = _load_watchlist(ticker_config)
    if not watchlist:
        logger.error("morning_briefing_no_tickers", component="internal_trigger")
        return {"status": "failed", "error": "No tickers in watchlist"}

    # Lazy imports to avoid startup circular imports
    import asyncio

    from services.app.agents.morning_briefing_graph import morning_briefing_graph
    from shared.db.repositories.recommendation_repo import save_recommendation
    from shared.models.recommendation import RecommendationCreate
    from telegram.formatters.briefing_formatter import format_morning_briefing

    # AC #2: Run Morning Briefing Intelligence Graph (1 invocation)
    try:
        state = await asyncio.wait_for(
            morning_briefing_graph.ainvoke({
                "analysis_date": today.isoformat(),
                "watchlist": watchlist,
            }),
            timeout=3600.0,
        )
    except Exception as exc:
        duration = round(time.monotonic() - start, 2)
        logger.error(
            "morning_briefing_graph_failed",
            component="internal_trigger",
            error=str(exc),
            duration_seconds=duration,
        )
        return {"status": "failed", "error": str(exc), "duration_seconds": duration}

    market_result = state.get("market_result") or {}
    if not market_result:
        duration = round(time.monotonic() - start, 2)
        logger.error(
            "morning_briefing_no_result",
            component="internal_trigger",
            duration_seconds=duration,
        )
        return {
            "status": "failed",
            "error": "morning_briefing_graph returned no result",
            "duration_seconds": duration,
        }

    # AC #2: Format and send Telegram (1 message)
    telegram_status = "skipped"
    telegram_bot = getattr(request.app.state, "telegram_bot", None)
    if telegram_bot:
        try:
            formatted = format_morning_briefing(market_result)
            message_ids = await telegram_bot.sender.send_message(formatted)
            telegram_status = "delivered" if message_ids else "queued"
        except Exception as exc:
            telegram_status = "failed"
            logger.error(
                "morning_briefing_telegram_failed",
                component="internal_trigger",
                error=str(exc),
            )

    failed_steps = state.get("failed_steps", [])

    # AC #3: Save 1 record to recommendations table
    try:
        rec = RecommendationCreate(
            type="morning_briefing",
            ticker_symbol="MARKET",
            content=market_result.get("market_summary", ""),
            confidence_score=None,
            risk_level=None,
            agents_used=["market_context", "technical_analysis", "fundamental_analysis"],
            agents_failed=failed_steps,
            data_sources={"top_picks": market_result.get("top_picks", [])},
        )
        await save_recommendation(rec)
    except Exception as exc:
        logger.error(
            "morning_briefing_save_failed",
            component="internal_trigger",
            error=str(exc),
        )

    duration = round(time.monotonic() - start, 2)

    # AC #3: Structured pipeline log
    logger.info(
        "morning_briefing_pipeline_completed",
        component="internal_trigger",
        started_at=started_at.isoformat(),
        duration_seconds=duration,
        affected_sectors=market_result.get("affected_sectors", []),
        top_picks_count=len(market_result.get("top_picks", [])),
        telegram_status=telegram_status,
    )

    overall_status = "ok" if not failed_steps and telegram_status != "failed" else "partial"
    return {
        "status": overall_status,
        "duration_seconds": duration,
        "affected_sectors": market_result.get("affected_sectors", []),
        "top_picks_count": len(market_result.get("top_picks", [])),
        "telegram": telegram_status,
        "errors": failed_steps,
    }
