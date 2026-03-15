"""Scheduler service entry point — Prefect serve() with cron schedules."""

import asyncio
import signal

from prefect import serve

from services.scheduler.config_loader import load_schedules
from services.scheduler.flows.data_pipeline import data_cleanup_flow, data_pipeline_flow
from shared.logging import configure_logging, get_logger

logger = get_logger("prefect_scheduler")

FLOW_REGISTRY = {
    "data_pipeline": data_pipeline_flow,
    "data_cleanup": data_cleanup_flow,
}

# Map schedule YAML flow names to internal flow keys.
# Individual crawler entries in the YAML are grouped into data_pipeline.
SCHEDULE_TO_FLOW = {
    "crawler_vietstock": "data_pipeline",
    "crawler_cafef": "data_pipeline",
    "crawler_vneconomy": "data_pipeline",
    "stock_crawl": "data_pipeline",
    "data_cleanup": "data_cleanup",
}


def _build_deployments(
    schedules: list,
) -> list:
    """Build Prefect deployment objects from schedule entries.

    The YAML config has individual entries per crawler source, but our
    architecture uses a single data_pipeline flow for all crawlers.
    We pick the earliest cron from the crawler group and ignore duplicates.
    data_cleanup maps 1:1 to data_cleanup_flow.
    """
    seen_flows: dict[str, str] = {}
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

        if flow_key in seen_flows:
            continue

        flow_fn = FLOW_REGISTRY.get(flow_key)
        if flow_fn is None:
            continue

        seen_flows[flow_key] = entry.cron
        deployment = flow_fn.to_deployment(
            name=flow_key.replace("_", "-"),
            cron=entry.cron,
        )
        deployments.append(deployment)

        logger.info(
            "deployment_registered",
            component="prefect_scheduler",
            flow=flow_key,
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
