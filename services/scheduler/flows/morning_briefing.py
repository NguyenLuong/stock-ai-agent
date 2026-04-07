"""Prefect flow — daily morning briefing pipeline."""

import os
import time

from prefect import flow

from shared.logging import get_logger
from shared.utils.datetime_utils import now_utc
from tasks.http_trigger import trigger_pipeline

logger = get_logger("prefect_scheduler")

APP_URL = os.environ.get("APP_URL", "http://app:8000")


@flow(name="morning-briefing", log_prints=True)
async def morning_briefing_flow() -> dict:
    """Daily morning briefing pipeline: Prefect -> app -> orchestrator -> Telegram."""
    start = time.monotonic()
    started_at = now_utc()

    logger.info(
        "morning_briefing_flow_started",
        component="prefect_scheduler",
        started_at=started_at.isoformat(),
    )

    result = await trigger_pipeline(
        app_url=APP_URL,
        endpoint="morning-briefing",
    )

    duration = round(time.monotonic() - start, 2)
    logger.info(
        "morning_briefing_flow_completed",
        component="prefect_scheduler",
        duration_seconds=duration,
        result=result,
    )
    return result
