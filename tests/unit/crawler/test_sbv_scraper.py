"""Tests for SBV interest rate scraper."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx

from services.crawler.macro.sbv_scraper import SBVScraper, _extract_rate_number


# Sample HTML with interest rate in table
_HTML_TABLE = """
<html><body>
<table>
  <tr><th>Loại lãi suất</th><th>Mức (%/năm)</th></tr>
  <tr><td>Lãi suất tái cấp vốn</td><td>4.50%</td></tr>
  <tr><td>Lãi suất chiết khấu</td><td>3.00%</td></tr>
</table>
</body></html>
"""

# Sample HTML with interest rate in text
_HTML_TEXT = """
<html><body>
<p>Ngân hàng Nhà nước quyết định giữ nguyên lãi suất tái cấp vốn ở mức 4.5%/năm.</p>
</body></html>
"""

# Sample HTML without interest rate
_HTML_NO_RATE = """
<html><body>
<p>Tin tức tài chính ngày hôm nay.</p>
</body></html>
"""


class TestSBVScraper:
    """Tests for SBVScraper.fetch_interest_rate."""

    async def test_success_from_table(self) -> None:
        mock_response = MagicMock()
        mock_response.text = _HTML_TABLE
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = mock_response

        scraper = SBVScraper(client=mock_client, base_url="https://www.sbv.gov.vn")
        result = await scraper.fetch_interest_rate()

        assert result.success is True
        assert result.indicator_name == "sbv_interest_rate"
        assert result.indicator_value == 4.5
        assert result.data_source == "sbv.gov.vn"

    async def test_success_from_text(self) -> None:
        mock_response = MagicMock()
        mock_response.text = _HTML_TEXT
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = mock_response

        scraper = SBVScraper(client=mock_client)
        result = await scraper.fetch_interest_rate()

        assert result.success is True
        assert result.indicator_value == 4.5

    async def test_parse_failure_returns_not_success(self) -> None:
        mock_response = MagicMock()
        mock_response.text = _HTML_NO_RATE
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = mock_response

        scraper = SBVScraper(client=mock_client)
        result = await scraper.fetch_interest_rate()

        assert result.success is False
        assert result.indicator_value is None
        assert "parse" in result.error.lower()

    async def test_http_error_returns_failure(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.side_effect = httpx.ConnectError("Connection refused")

        scraper = SBVScraper(client=mock_client)
        result = await scraper.fetch_interest_rate()

        assert result.success is False
        assert result.error is not None

    async def test_robots_blocked_returns_failure(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_robots = AsyncMock()
        mock_robots.can_fetch.return_value = False

        scraper = SBVScraper(client=mock_client, robots_checker=mock_robots)
        result = await scraper.fetch_interest_rate()

        assert result.success is False
        assert "robots" in result.error.lower()
        mock_client.get.assert_not_called()

    async def test_timeout_returns_failure(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.side_effect = httpx.TimeoutException("Timeout")

        scraper = SBVScraper(client=mock_client)
        result = await scraper.fetch_interest_rate()

        assert result.success is False


class TestExtractRateNumber:
    """Tests for rate number extraction helper."""

    def test_extracts_percentage(self) -> None:
        assert _extract_rate_number("4.5%") == 4.5

    def test_extracts_percentage_with_comma(self) -> None:
        assert _extract_rate_number("4,5%") == 4.5

    def test_extracts_rate_per_year(self) -> None:
        assert _extract_rate_number("4.5%/năm") == 4.5

    def test_rejects_bare_number_without_percent(self) -> None:
        assert _extract_rate_number("4.50") is None

    def test_rejects_out_of_range(self) -> None:
        assert _extract_rate_number("0%") is None
        assert _extract_rate_number("99.9%") is None

    def test_no_number_returns_none(self) -> None:
        assert _extract_rate_number("no rate here") is None
