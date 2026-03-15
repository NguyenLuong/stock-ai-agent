"""Load and validate scheduler schedule configuration from YAML."""

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from shared.logging import get_logger

logger = get_logger("prefect_scheduler")

DEFAULT_CONFIG_PATH = Path(
    os.environ.get("CONFIG_DIR", "/app/config")
) / "scheduler" / "schedules.yaml"


@dataclass(frozen=True)
class ScheduleEntry:
    """A single flow schedule definition."""

    name: str
    cron: str
    description: str
    enabled: bool


_CRON_FIELD_RANGES = {
    0: (0, 59, "minute"),
    1: (0, 23, "hour"),
    2: (1, 31, "day of month"),
    3: (1, 12, "month"),
    4: (0, 7, "day of week"),
}


def _validate_cron(cron: str, flow_name: str) -> None:
    """Validate a cron expression has 5 fields with valid values.

    Raises ValueError on invalid field count or out-of-range numeric values.
    """
    parts = cron.strip().split()
    if len(parts) != 5:
        raise ValueError(
            f"Invalid cron expression for flow '{flow_name}': "
            f"expected 5 fields, got {len(parts)} in '{cron}'"
        )

    for idx, part in enumerate(parts):
        lo, hi, field_name = _CRON_FIELD_RANGES[idx]
        for segment in part.split(","):
            token = segment.split("/")[0].split("-")[0]
            if token == "*":
                continue
            try:
                val = int(token)
            except ValueError:
                raise ValueError(
                    f"Invalid cron {field_name} for flow '{flow_name}': "
                    f"'{token}' is not a valid value in '{cron}'"
                )
            if val < lo or val > hi:
                raise ValueError(
                    f"Invalid cron {field_name} for flow '{flow_name}': "
                    f"{val} is out of range ({lo}-{hi}) in '{cron}'"
                )


def load_schedules(config_path: Path | None = None) -> list[ScheduleEntry]:
    """Load schedule entries from YAML config file.

    Args:
        config_path: Path to schedules.yaml. Defaults to CONFIG_DIR/scheduler/schedules.yaml.

    Returns:
        List of enabled ScheduleEntry objects.

    Raises:
        FileNotFoundError: If config file does not exist.
        ValueError: If cron expression is invalid.
    """
    path = config_path or DEFAULT_CONFIG_PATH

    if not path.exists():
        raise FileNotFoundError(f"Schedule config not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    flows = data.get("flows", {})
    if not flows:
        logger.warning(
            "no_flows_configured",
            component="prefect_scheduler",
            config_path=str(path),
        )
        return []

    entries: list[ScheduleEntry] = []
    for name, config in flows.items():
        cron = config.get("cron", "")
        enabled = config.get("enabled", True)
        description = config.get("description", "")

        _validate_cron(cron, name)

        entry = ScheduleEntry(
            name=name,
            cron=cron,
            description=description,
            enabled=enabled,
        )

        if enabled:
            entries.append(entry)
            logger.info(
                "schedule_loaded",
                component="prefect_scheduler",
                flow=name,
                cron=cron,
                description=description,
            )
        else:
            logger.info(
                "schedule_disabled",
                component="prefect_scheduler",
                flow=name,
            )

    return entries
