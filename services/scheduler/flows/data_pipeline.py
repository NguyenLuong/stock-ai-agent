"""Prefect flow — automated data ingestion pipeline.

Orchestrates crawl → embedding → lifecycle via HTTP POST to the app service.
Each step is independent: failure in one does not block the others.
"""

import os
import time

from prefect import flow

from shared.logging import get_logger
from shared.utils.datetime_utils import now_utc
from services.scheduler.tasks.http_trigger import trigger_pipeline

logger = get_logger("prefect_scheduler")

APP_URL = os.environ.get("APP_URL", "http://app:8000")

PIPELINE_STEPS = [
    ("crawl", "Full news + macro + vnstock crawl"),
    ("embedding", "Article embedding pipeline"),
    ("lifecycle", "Data lifecycle cleanup"),
]


@flow(name="data-pipeline", log_prints=True)
async def data_pipeline_flow() -> dict:
    """Orchestrate full data ingestion pipeline.

    Sequence: crawl → embedding → lifecycle.
    Each step logs its result independently. Failures are logged but do not
    block subsequent steps.

    Returns:
        Summary dict with per-step results and overall stats.
    """
    start = time.monotonic()
    started_at = now_utc()
    results: dict[str, dict] = {}
    errors: list[str] = []

    logger.info(
        "data_pipeline_started",
        component="prefect_scheduler",
        started_at=started_at.isoformat(),
    )

    for endpoint, description in PIPELINE_STEPS:
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
        "total_steps": len(PIPELINE_STEPS),
        "successful_steps": sum(
            1 for r in results.values() if r["status"] == "success"
        ),
        "failed_steps": len(errors),
        "errors": errors,
    }

    logger.info(
        "data_pipeline_completed",
        component="prefect_scheduler",
        total_duration_seconds=total_duration,
        successful_steps=summary["successful_steps"],
        failed_steps=summary["failed_steps"],
    )

    return summary


@flow(name="data-cleanup", log_prints=True)
async def data_cleanup_flow() -> dict:
    """Trigger lifecycle cleanup independently (runs at 2:00 AM)."""
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
