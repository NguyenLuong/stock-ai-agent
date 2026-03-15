"""Ticker config loader — reads stock_tickers.yaml and produces deduplicated ticker list."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml

from shared.logging import get_logger

logger = get_logger("ticker_config")

_TICKER_PATTERN = re.compile(r"^[A-Z]{3,4}$")

DEFAULT_CONFIG_PATH = Path("config/crawlers/stock_tickers.yaml")


@dataclass
class TickerConfig:
    """Result of loading and merging ticker config."""

    tickers: list[str]
    total_count: int
    enabled_groups: int
    holidays: list[date] = field(default_factory=list)


def load_ticker_config(
    config_path: Path | None = None,
) -> TickerConfig:
    """Load stock_tickers.yaml, merge enabled groups, deduplicate, and validate.

    Args:
        config_path: Path to the YAML config file. Defaults to config/crawlers/stock_tickers.yaml.

    Returns:
        TickerConfig with deduplicated, validated ticker list.

    Raises:
        FileNotFoundError: If config file does not exist.
    """
    path = config_path or DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Ticker config not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    groups = raw.get("groups", {})
    seen: set[str] = set()
    tickers: list[str] = []
    enabled_count = 0

    for group_name, group_data in groups.items():
        if not group_data.get("enabled", False):
            continue
        enabled_count += 1
        for ticker in group_data.get("tickers", []):
            ticker_str = str(ticker).strip().upper()
            if ticker_str in seen:
                continue
            if not _TICKER_PATTERN.match(ticker_str):
                logger.warning(
                    "invalid_ticker_format",
                    ticker=ticker_str,
                    group=group_name,
                    component="ticker_config",
                )
                continue
            seen.add(ticker_str)
            tickers.append(ticker_str)

    # Parse holidays
    holidays_raw = raw.get("holidays", {})
    holidays: list[date] = []
    for _year, date_list in holidays_raw.items():
        if not isinstance(date_list, list):
            continue
        for d in date_list:
            try:
                holidays.append(date.fromisoformat(str(d)))
            except ValueError:
                logger.warning(
                    "invalid_holiday_date",
                    date=str(d),
                    component="ticker_config",
                )

    logger.info(
        "ticker_config_loaded",
        total_tickers=len(tickers),
        enabled_groups=enabled_count,
        total_groups=len(groups),
        holidays_count=len(holidays),
        component="ticker_config",
    )

    return TickerConfig(
        tickers=tickers,
        total_count=len(tickers),
        enabled_groups=enabled_count,
        holidays=holidays,
    )
