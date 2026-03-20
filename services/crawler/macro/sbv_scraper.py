"""SBV interest rate scraper — fetches policy rate from sbv.gov.vn.

Scrapes SBV refinancing rate (lai suat tai cap von) from the State Bank of Vietnam.
Uses httpx AsyncClient with RateLimitedTransport for rate limiting.
Falls back gracefully on parsing failures with multiple selector strategies.
"""

from __future__ import annotations

import re

import httpx
from bs4 import BeautifulSoup

from macro.models import MacroDataResult
from middleware.robots_checker import RobotsChecker
from shared.logging import get_logger
from shared.utils.datetime_utils import now_utc

logger = get_logger("sbv_scraper")

# Keywords for finding refinancing rate on SBV pages
_RATE_KEYWORDS = [
    "lãi suất tái cấp vốn",
    "lai suat tai cap von",
    "refinancing rate",
    "tái cấp vốn",
]


class SBVScraper:
    """Scrapes SBV policy interest rate from sbv.gov.vn."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        robots_checker: RobotsChecker | None = None,
        base_url: str = "https://www.sbv.gov.vn",
    ) -> None:
        self._client = client
        self._robots_checker = robots_checker
        self._base_url = base_url
        self._logger = logger

    async def fetch_interest_rate(self) -> MacroDataResult:
        """Fetch current SBV refinancing rate."""
        url = self._base_url
        try:
            # Check robots.txt if checker available
            if self._robots_checker is not None:
                allowed = await self._robots_checker.can_fetch(url)
                if not allowed:
                    return MacroDataResult(
                        indicator_name="sbv_interest_rate",
                        indicator_value=None,
                        data_as_of=now_utc(),
                        data_source="sbv.gov.vn",
                        success=False,
                        error="Blocked by robots.txt",
                    )

            response = await self._client.get(url)
            response.raise_for_status()

            rate = self._parse_interest_rate(response.text)
            if rate is not None:
                self._logger.info(
                    "sbv_interest_rate_fetched",
                    rate=rate,
                    component="sbv_scraper",
                )
                return MacroDataResult(
                    indicator_name="sbv_interest_rate",
                    indicator_value=rate,
                    data_as_of=now_utc(),
                    data_source="sbv.gov.vn",
                    success=True,
                )

            # Parsing failed — return failure
            self._logger.warning(
                "sbv_parse_failed",
                component="sbv_scraper",
                url=url,
            )
            return MacroDataResult(
                indicator_name="sbv_interest_rate",
                indicator_value=None,
                data_as_of=now_utc(),
                data_source="sbv.gov.vn",
                success=False,
                error="Could not parse interest rate from SBV page",
            )

        except (httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException) as exc:
            self._logger.warning(
                "sbv_fetch_failed",
                error=str(exc),
                component="sbv_scraper",
            )
            return MacroDataResult(
                indicator_name="sbv_interest_rate",
                indicator_value=None,
                data_as_of=now_utc(),
                data_source="sbv.gov.vn",
                success=False,
                error=str(exc),
            )
        except (ValueError, AttributeError, TypeError, KeyError) as exc:
            self._logger.warning(
                "sbv_parse_error",
                error=str(exc),
                error_type=type(exc).__name__,
                component="sbv_scraper",
            )
            return MacroDataResult(
                indicator_name="sbv_interest_rate",
                indicator_value=None,
                data_as_of=now_utc(),
                data_source="sbv.gov.vn",
                success=False,
                error=str(exc),
            )

    def _parse_interest_rate(self, html: str) -> float | None:
        """Parse refinancing rate from SBV HTML.

        Strategy:
        1. Try specific CSS selectors for interest rate tables
        2. Fallback: search for keywords in text and extract nearby numbers
        3. If all parsing fails, return None
        """
        soup = BeautifulSoup(html, "html.parser")

        # Strategy 1: Look for table cells containing rate keywords
        rate = self._extract_from_tables(soup)
        if rate is not None:
            return rate

        # Strategy 2: Search text for keywords and extract nearby percentages
        rate = self._extract_from_text(soup)
        if rate is not None:
            return rate

        return None

    @staticmethod
    def _extract_from_tables(soup: BeautifulSoup) -> float | None:
        """Extract rate from HTML tables containing rate keywords."""
        for table in soup.find_all("table"):
            rows = table.find_all("tr")
            for row in rows:
                cells = row.find_all(["td", "th"])
                row_text = " ".join(c.get_text(strip=True).lower() for c in cells)
                if any(kw in row_text for kw in _RATE_KEYWORDS):
                    # Look for a number (percentage) in the row
                    for cell in cells:
                        text = cell.get_text(strip=True)
                        rate = _extract_rate_number(text)
                        if rate is not None:
                            return rate
        return None

    @staticmethod
    def _extract_from_text(soup: BeautifulSoup) -> float | None:
        """Search full page text for rate keywords and nearby percentages."""
        full_text = soup.get_text(separator=" ")
        full_text_lower = full_text.lower()

        for keyword in _RATE_KEYWORDS:
            idx = full_text_lower.find(keyword)
            if idx == -1:
                continue
            # Search in a window around the keyword
            window = full_text[max(0, idx - 50) : idx + len(keyword) + 100]
            rate = _extract_rate_number(window)
            if rate is not None:
                return rate

        return None


def _extract_rate_number(text: str) -> float | None:
    """Extract a percentage-like number from text.

    Matches patterns like: 4.5%, 4,5%, 4.50, 6%/năm
    Returns the float value or None.
    """
    # Match numbers that look like interest rates (0-100 range, with optional %)
    # Ordered from most specific to least specific
    patterns = [
        r"(\d{1,2}[.,]\d{1,2})\s*%\s*/\s*năm",  # 4.5%/năm (most specific)
        r"(\d{1,2}[.,]\d{1,2})\s*%",  # 4.5% or 4,5%
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value_str = match.group(1).replace(",", ".")
            try:
                value = float(value_str)
                # Reasonable range for interest rates: 0-50%
                if 0 < value <= 50:
                    return value
            except ValueError:
                continue
    return None
