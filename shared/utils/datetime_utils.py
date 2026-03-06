"""Date and time utility functions for Vietnam timezone handling.

WARNING: NEVER use datetime.now() — always use now_utc() from this module.
"""

from __future__ import annotations

import zoneinfo
from datetime import datetime, timezone

VN_TZ = zoneinfo.ZoneInfo("Asia/Ho_Chi_Minh")


def now_utc() -> datetime:
    """Return current time as timezone-aware UTC datetime. Never naive."""
    return datetime.now(timezone.utc)


def to_vn_display(dt: datetime) -> str:
    """Format UTC datetime to Vietnam timezone display string.

    Example: "07:30 sáng 05/03/2026"
    """
    vn_time = dt.astimezone(VN_TZ)
    hour = vn_time.hour
    if hour < 12:
        period = "sáng"
    elif hour < 18:
        period = "chiều"
    else:
        period = "tối"
    return vn_time.strftime(f"%H:%M {period} %d/%m/%Y")


def is_stale(dt: datetime, max_age_hours: float = 4.0) -> bool:
    """Return True if dt is older than max_age_hours ago."""
    age = now_utc() - dt.astimezone(timezone.utc)
    return age.total_seconds() > max_age_hours * 3600


def format_iso_utc(dt: datetime) -> str:
    """Format datetime as ISO 8601 UTC string ending in Z."""
    utc_dt = dt.astimezone(timezone.utc)
    return utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
