"""Prefect flows — scheduled data pipelines.

Three independent flows:
- news_crawl_flow:  crawl → embedding
- stock_flow:       stock-crawl → technical-indicators
- data_cleanup_flow: lifecycle
"""

import os
import time

from prefect import flow

from shared.logging import get_logger
from shared.utils.datetime_utils import now_utc
from tasks.http_trigger import trigger_pipeline

logger = get_logger("prefect_scheduler")

APP_URL = os.environ.get("APP_URL", "http://app:8000")

NEWS_CRAWL_STEPS = [
    ("crawl", "Full news crawl (all sources)"),
    ("embedding", "Article embedding pipeline"),
]

STOCK_STEPS = [
    ("stock-crawl", "Stock price history crawl"),
    ("technical-indicators", "Technical indicator calculation"),
]


async def _run_steps(flow_name: str, steps: list[tuple[str, str]]) -> dict:
    """Execute a sequence of pipeline steps, logging each independently.

    Failures in one step do not block subsequent steps.
    """
    start = time.monotonic()
    started_at = now_utc()
    results: dict[str, dict] = {}
    errors: list[str] = []

    logger.info(
        f"{flow_name}_started",
        component="prefect_scheduler",
        started_at=started_at.isoformat(),
    )

    for endpoint, description in steps:
        step_start = time.monotonic()
        try:
            result = await trigger_pipeline(
                app_url=APP_URL,
                endpoint=endpoint,
            )
            duration = round(time.monotonic() - step_start, 2)
            results[endpoint] = {
                "status": "success",
                "duration_seconds": duration,
                "result": result,
            }
            logger.info(
                "pipeline_step_completed",
                component="prefect_scheduler",
                flow=flow_name,
                step=endpoint,
                description=description,
                duration_seconds=duration,
            )
        except Exception as exc:
            duration = round(time.monotonic() - step_start, 2)
            results[endpoint] = {
                "status": "failed",
                "duration_seconds": duration,
                "error": str(exc),
            }
            errors.append(f"{endpoint}: {exc}")
            logger.error(
                "pipeline_step_failed",
                component="prefect_scheduler",
                flow=flow_name,
                step=endpoint,
                description=description,
                duration_seconds=duration,
                error=str(exc),
            )

    total_duration = round(time.monotonic() - start, 2)
    summary = {
        "started_at": started_at.isoformat(),
        "total_duration_seconds": total_duration,
        "steps": results,
        "total_steps": len(steps),
        "successful_steps": sum(
            1 for r in results.values() if r["status"] == "success"
        ),
        "failed_steps": len(errors),
        "errors": errors,
    }

    logger.info(
        f"{flow_name}_completed",
        component="prefect_scheduler",
        total_duration_seconds=total_duration,
        successful_steps=summary["successful_steps"],
        failed_steps=summary["failed_steps"],
    )

    return summary


@flow(name="news-crawl", log_prints=True)
async def news_crawl_flow() -> dict:
    """News ingestion pipeline: crawl all sources → embed articles."""
    return await _run_steps("news_crawl", NEWS_CRAWL_STEPS)


@flow(name="stock-pipeline", log_prints=True)
async def stock_pipeline_flow() -> dict:
    """Stock data pipeline: crawl prices → calculate technical indicators."""
    return await _run_steps("stock_pipeline", STOCK_STEPS)


@flow(name="data-cleanup", log_prints=True)
async def data_cleanup_flow() -> dict:
    """Trigger lifecycle cleanup independently."""
    start = time.monotonic()
    started_at = now_utc()

    logger.info(
        "data_cleanup_started",
        component="prefect_scheduler",
        started_at=started_at.isoformat(),
    )

    try:
        result = await trigger_pipeline(
            app_url=APP_URL,
            endpoint="lifecycle",
        )
        duration = round(time.monotonic() - start, 2)
        logger.info(
            "data_cleanup_completed",
            component="prefect_scheduler",
            duration_seconds=duration,
        )
        return {
            "status": "success",
            "duration_seconds": duration,
            "result": result,
        }
    except Exception as exc:
        duration = round(time.monotonic() - start, 2)
        logger.error(
            "data_cleanup_failed",
            component="prefect_scheduler",
            duration_seconds=duration,
            error=str(exc),
        )
        return {
            "status": "failed",
            "duration_seconds": duration,
            "error": str(exc),
        }
