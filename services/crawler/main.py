"""Crawler service entry point."""
# TODO: Story 2.x — implement crawler orchestration
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Crawler service starting (stub — Story 2.x)")
    while True:
        time.sleep(60)
