"""Tests for shared.utils.datetime_utils — timezone-aware helpers."""

from datetime import datetime, timedelta, timezone

from shared.utils.datetime_utils import format_iso_utc, is_stale, now_utc, to_vn_display


class TestNowUtc:
    def test_returns_timezone_aware(self):
        result = now_utc()
        assert result.tzinfo is not None
        assert result.tzinfo == timezone.utc

    def test_returns_datetime_type(self):
        result = now_utc()
        assert isinstance(result, datetime)


class TestIsStale:
    def test_stale_when_older_than_max_age(self):
        old = now_utc() - timedelta(hours=5)
        assert is_stale(old, max_age_hours=4.0) is True

    def test_not_stale_when_within_max_age(self):
        recent = now_utc() - timedelta(hours=1)
        assert is_stale(recent, max_age_hours=4.0) is False

    def test_default_max_age_is_4_hours(self):
        old = now_utc() - timedelta(hours=5)
        assert is_stale(old) is True


class TestFormatIsoUtc:
    def test_ends_with_z(self):
        dt = datetime(2026, 3, 5, 7, 30, 0, tzinfo=timezone.utc)
        result = format_iso_utc(dt)
        assert result.endswith("Z")
        assert result == "2026-03-05T07:30:00Z"

    def test_converts_non_utc_to_utc(self):
        from zoneinfo import ZoneInfo
        vn = ZoneInfo("Asia/Ho_Chi_Minh")
        dt = datetime(2026, 3, 5, 14, 30, 0, tzinfo=vn)  # VN +7
        result = format_iso_utc(dt)
        assert result == "2026-03-05T07:30:00Z"


class TestToVnDisplay:
    def test_format_morning(self):
        dt = datetime(2026, 3, 5, 0, 30, 0, tzinfo=timezone.utc)  # 7:30 VN
        result = to_vn_display(dt)
        assert "07:30" in result
        assert "sáng" in result
        assert "05/03/2026" in result

    def test_format_afternoon(self):
        dt = datetime(2026, 3, 5, 6, 0, 0, tzinfo=timezone.utc)  # 13:00 VN
        result = to_vn_display(dt)
        assert "13:00" in result
        assert "chiều" in result

    def test_format_evening(self):
        dt = datetime(2026, 3, 5, 12, 0, 0, tzinfo=timezone.utc)  # 19:00 VN
        result = to_vn_display(dt)
        assert "19:00" in result
        assert "tối" in result
