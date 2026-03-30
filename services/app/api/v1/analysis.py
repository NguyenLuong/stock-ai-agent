"""Stock analysis endpoint — invokes the orchestrator graph."""

from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter
from pydantic import BaseModel

from shared.logging import get_logger

logger = get_logger("analysis_api")

router = APIRouter(prefix="/api/v1", tags=["analysis"])


class AnalyzeStockRequest(BaseModel):
    ticker: str
    analysis_type: str = "morning_briefing"
    watchlist: list[str] = []


@router.post("/analyze-stock")
async def analyze_stock(req: AnalyzeStockRequest) -> dict:
    """Run the full orchestrator graph for a given ticker.

    Returns structured JSON with status, duration, and analysis result.
    Timeout: 120 seconds (AC #1 requirement).
    """
    from datetime import date, timezone

    from services.app.agents.graph import orchestrator_graph

    start = time.monotonic()

    initial_state = {
        "ticker": req.ticker,
        "analysis_type": req.analysis_type,
        "analysis_date": date.today().isoformat(),
        "watchlist": req.watchlist or [req.ticker],
    }

    try:
        result = await asyncio.wait_for(
            orchestrator_graph.ainvoke(initial_state),
            timeout=120,
        )
    except asyncio.TimeoutError:
        duration = round(time.monotonic() - start, 2)
        logger.error(
            "analysis_timeout",
            component="analysis_api",
            ticker=req.ticker,
            duration_seconds=duration,
        )
        return {
            "status": "failed",
            "duration_seconds": duration,
            "error": "Analysis timed out (>120s)",
        }
    except Exception as exc:
        duration = round(time.monotonic() - start, 2)
        logger.error(
            "analysis_failed",
            component="analysis_api",
            ticker=req.ticker,
            error=str(exc),
        )
        return {
            "status": "failed",
            "duration_seconds": duration,
            "error": str(exc),
        }

    duration = round(time.monotonic() - start, 2)
    failed = result.get("failed_agents", [])
    error = result.get("error")

    if error and result.get("synthesis_result") is None:
        status = "failed"
    elif failed:
        status = "partial"
    else:
        status = "ok"

    logger.info(
        "analysis_completed",
        component="analysis_api",
        ticker=req.ticker,
        status=status,
        duration_seconds=duration,
    )

    return {
        "status": status,
        "duration_seconds": duration,
        "result": {
            "synthesis": result.get("synthesis_result"),
            "confidence_score": result.get("confidence_score"),
            "failed_agents": failed,
            "error": error,
        },
    }
