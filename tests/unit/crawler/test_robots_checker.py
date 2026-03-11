"""Tests for robots.txt compliance checker."""

import time
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from services.crawler.middleware.robots_checker import RobotsChecker

ROBOTS_TXT = """\
User-agent: *
Disallow: /private/
Disallow: /admin/
"""


class TestRobotsChecker:
    """Test robots.txt parsing, caching, and URL filtering."""

    @pytest.fixture
    def mock_client(self):
        """Return an AsyncMock httpx client that serves robots.txt."""
        client = AsyncMock(spec=httpx.AsyncClient)
        request = httpx.Request("GET", "https://example.com/robots.txt")
        response = httpx.Response(200, text=ROBOTS_TXT, request=request)
        client.get = AsyncMock(return_value=response)
        return client

    @pytest.fixture
    def checker(self, mock_client):
        return RobotsChecker(client=mock_client, user_agent="StockAIAgent/1.0")

    async def test_allowed_url(self, checker):
        result = await checker.can_fetch("https://example.com/news/article1")
        assert result is True

    async def test_disallowed_url_global(self, checker):
        result = await checker.can_fetch("https://example.com/private/data")
        assert result is False

    async def test_disallowed_url_admin(self, checker):
        result = await checker.can_fetch("https://example.com/admin/panel")
        assert result is False

    async def test_root_url_allowed(self, checker):
        result = await checker.can_fetch("https://example.com/")
        assert result is True

    async def test_robots_txt_fetched_once_per_domain(self, checker, mock_client):
        """robots.txt should be cached per domain."""
        await checker.can_fetch("https://example.com/page1")
        await checker.can_fetch("https://example.com/page2")
        await checker.can_fetch("https://example.com/page3")
        # Only one GET to robots.txt
        assert mock_client.get.call_count == 1
        mock_client.get.assert_called_once_with("https://example.com/robots.txt")

    async def test_different_domains_fetch_separately(self, checker, mock_client):
        """Each domain should get its own robots.txt fetch."""
        await checker.can_fetch("https://domain-a.com/page")
        await checker.can_fetch("https://domain-b.com/page")
        assert mock_client.get.call_count == 2

    async def test_robots_txt_fetch_failure_allows_all(self, mock_client):
        """If robots.txt can't be fetched, allow all URLs (graceful degradation)."""
        mock_client.get = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "Not Found",
                request=httpx.Request("GET", "https://example.com/robots.txt"),
                response=httpx.Response(404),
            )
        )
        checker = RobotsChecker(client=mock_client, user_agent="StockAIAgent/1.0")
        result = await checker.can_fetch("https://example.com/private/data")
        assert result is True

    async def test_robots_txt_network_error_allows_all(self, mock_client):
        """Network errors should not block crawling."""
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("timeout"))
        checker = RobotsChecker(client=mock_client, user_agent="StockAIAgent/1.0")
        result = await checker.can_fetch("https://example.com/any/page")
        assert result is True

    async def test_cache_ttl_expires(self, checker, mock_client):
        """After TTL expires, robots.txt should be re-fetched."""
        checker._cache_ttl_seconds = 0  # Expire immediately
        await checker.can_fetch("https://example.com/page1")
        # Force cache expiry
        for domain in checker._cache:
            checker._cache[domain] = (checker._cache[domain][0], 0)
        await checker.can_fetch("https://example.com/page2")
        assert mock_client.get.call_count == 2
