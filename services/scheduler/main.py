"""Scheduler service entry point — Prefect serve() with cron schedules."""

import asyncio
import signal

from prefect import serve

from config_loader import load_schedules
from flows.data_pipeline import data_cleanup_flow, news_crawl_flow, stock_pipeline_flow
from shared.logging import configure_logging, get_logger

logger = get_logger("prefect_scheduler")

FLOW_REGISTRY = {
    "news_crawl": news_crawl_flow,
    "stock_pipeline": stock_pipeline_flow,
    "data_cleanup": data_cleanup_flow,
}

# Map schedule YAML flow names to internal flow keys.
SCHEDULE_TO_FLOW = {
    "news_crawl": "news_crawl",
    "stock_pipeline": "stock_pipeline",
    "data_cleanup": "data_cleanup",
}


def _build_deployments(
    schedules: list,
) -> list:
    """Build Prefect deployment objects from schedule entries.

    Each schedule entry creates its own deployment so that different cron
    schedules for the same flow all fire independently.
    """
    deployments = []

    for entry in schedules:
        flow_key = SCHEDULE_TO_FLOW.get(entry.name)
        if flow_key is None:
            logger.info(
                "schedule_skipped",
                component="prefect_scheduler",
                flow=entry.name,
                reason="no matching flow (may be epic 4+ scope)",
            )
            continue

        flow_fn = FLOW_REGISTRY.get(flow_key)
        if flow_fn is None:
            continue

        deployment = flow_fn.to_deployment(
            name=entry.name.replace("_", "-"),
            cron=entry.cron,
        )
        deployments.append(deployment)

        logger.info(
            "deployment_registered",
            component="prefect_scheduler",
            flow=flow_key,
            deployment=entry.name,
            cron=entry.cron,
        )

    return deployments


def _handle_signal(signum: int, _frame: object) -> None:
    """Handle SIGTERM/SIGINT for graceful shutdown via KeyboardInterrupt."""
    sig_name = signal.Signals(signum).name
    logger.info(
        "shutdown_signal_received",
        component="prefect_scheduler",
        signal=sig_name,
    )
    raise KeyboardInterrupt


def main() -> None:
    """Load schedules, register deployments, and start Prefect serve()."""
    configure_logging()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    logger.info("scheduler_starting", component="prefect_scheduler")

    schedules = load_schedules()
    if not schedules:
        logger.warning(
            "no_schedules_found",
            component="prefect_scheduler",
        )
        return

    deployments = _build_deployments(schedules)
    if not deployments:
        logger.warning(
            "no_deployments_registered",
            component="prefect_scheduler",
        )
        return

    logger.info(
        "scheduler_ready",
        component="prefect_scheduler",
        deployment_count=len(deployments),
    )

    asyncio.run(serve(*deployments))


if __name__ == "__main__":
    main()
