"""Scheduler service entry point."""
# TODO: Story 2.6 — replace with Prefect worker start
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Scheduler service starting (stub — Story 2.6)")
    while True:
        time.sleep(60)
