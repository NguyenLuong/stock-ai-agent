"""Reusable Prefect task for triggering pipelines via HTTP POST."""

import httpx
from prefect import task

from shared.logging import get_logger

logger = get_logger("prefect_scheduler")


@task(retries=2, retry_delay_seconds=30, timeout_seconds=1800)
async def trigger_pipeline(
    app_url: str,
    endpoint: str,
    timeout_seconds: float = 1800.0,
) -> dict:
    """POST to app internal endpoint and return result.

    Args:
        app_url: Base URL of the app service (e.g. http://app:8000).
        endpoint: Pipeline endpoint name (e.g. "crawl", "embedding").
        timeout_seconds: HTTP request timeout.

    Returns:
        JSON response from the app service.
    """
    url = f"{app_url}/internal/trigger/{endpoint}"
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        response = await client.post(
            url,
            headers={"X-Trigger-Source": "prefect-scheduler"},
        )
        response.raise_for_status()
        result = response.json()
        logger.info(
            "pipeline_triggered",
            component="prefect_scheduler",
            endpoint=endpoint,
            status=response.status_code,
            result=result,
        )
        return result
