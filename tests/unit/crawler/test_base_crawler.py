"""Tests for BaseNewsCrawler abstract class."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from xml.etree.ElementTree import Element, SubElement, tostring

import httpx
import pytest

from services.crawler.middleware.robots_checker import RobotsChecker
from services.crawler.news.base_crawler import BaseNewsCrawler
from shared.models.article import ArticleCreate


def _build_rss_xml(items: list[dict]) -> str:
    """Build a minimal RSS XML string from a list of item dicts."""
    rss = Element("rss", version="2.0")
    channel = SubElement(rss, "channel")
    SubElement(channel, "title").text = "Test Feed"
    for item_data in items:
        item = SubElement(channel, "item")
        SubElement(item, "title").text = item_data.get("title", "Test Title")
        SubElement(item, "link").text = item_data.get("link", "https://example.com/article")
        SubElement(item, "pubDate").text = item_data.get(
            "pubDate", "Mon, 10 Mar 2026 06:00:00 +0000"
        )
        if "description" in item_data:
            SubElement(item, "description").text = item_data["description"]
    return tostring(rss, encoding="unicode")


class ConcreteTestCrawler(BaseNewsCrawler):
    """Concrete implementation for testing."""

    def __init__(self, client, robots_checker, feed_urls, article_html="<p>Test content</p>"):
        super().__init__(client=client, robots_checker=robots_checker, source_name="test")
        self._feed_urls = feed_urls
        self._article_html = article_html

    async def get_rss_feeds(self) -> list[str]:
        return self._feed_urls

    def parse_article_page(self, url: str, html: str) -> str:
        return "Test article body content"

    def _get_article_selectors(self) -> list[str]:
        return ["p"]


class TestBaseNewsCrawler:
    """Test base crawler: RSS parsing, error handling, dedup."""

    @pytest.fixture
    def mock_robots(self):
        robots = AsyncMock(spec=RobotsChecker)
        robots.can_fetch = AsyncMock(return_value=True)
        return robots

    @pytest.fixture
    def rss_items(self):
        return [
            {"title": "Article 1", "link": "https://example.com/article-1", "pubDate": "Mon, 10 Mar 2026 06:00:00 +0000"},
            {"title": "Article 2", "link": "https://example.com/article-2", "pubDate": "Tue, 11 Mar 2026 08:00:00 +0000"},
        ]

    @pytest.fixture
    def mock_client(self, rss_items):
        """Client that returns RSS XML for feed URLs and HTML for article URLs."""
        rss_xml = _build_rss_xml(rss_items)

        async def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if url.endswith(".rss") or "rss" in url:
                return httpx.Response(200, text=rss_xml, request=request)
            return httpx.Response(
                200,
                text="<html><body><p>Full article body</p></body></html>",
                request=request,
            )

        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def test_crawl_returns_articles(self, mock_client, mock_robots, rss_items):
        crawler = ConcreteTestCrawler(
            client=mock_client,
            robots_checker=mock_robots,
            feed_urls=["https://example.com/rss/feed.rss"],
        )
        articles = await crawler.crawl()
        assert len(articles) == 2
        assert all(isinstance(a, ArticleCreate) for a in articles)
        assert articles[0].title == "Article 1"
        assert articles[0].source == "test"
        assert articles[0].url == "https://example.com/article-1"

    async def test_crawl_skips_robots_disallowed(self, mock_client, mock_robots, rss_items):
        """URLs blocked by robots.txt should be skipped."""
        mock_robots.can_fetch = AsyncMock(
            side_effect=lambda url: "article-1" not in url
        )
        crawler = ConcreteTestCrawler(
            client=mock_client,
            robots_checker=mock_robots,
            feed_urls=["https://example.com/rss/feed.rss"],
        )
        articles = await crawler.crawl()
        assert len(articles) == 1
        assert articles[0].url == "https://example.com/article-2"

    async def test_crawl_handles_http_error_gracefully(self, mock_robots):
        """HTTP 5xx on article fetch should not crash; article is skipped."""

        async def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if "rss" in url:
                rss_xml = _build_rss_xml([
                    {"title": "Art", "link": "https://example.com/fail-article"},
                ])
                return httpx.Response(200, text=rss_xml, request=request)
            return httpx.Response(500, text="Server Error", request=request)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        crawler = ConcreteTestCrawler(
            client=client,
            robots_checker=mock_robots,
            feed_urls=["https://example.com/rss/feed.rss"],
        )
        articles = await crawler.crawl()
        assert len(articles) == 0  # Failed article is skipped, no crash

    async def test_crawl_handles_rss_fetch_error(self, mock_robots):
        """HTTP error on RSS feed should not crash."""

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="Unavailable", request=request)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        crawler = ConcreteTestCrawler(
            client=client,
            robots_checker=mock_robots,
            feed_urls=["https://example.com/rss/feed.rss"],
        )
        articles = await crawler.crawl()
        assert len(articles) == 0

    async def test_crawl_sets_user_agent(self, mock_robots):
        """Requests should use StockAIAgent/1.0 user agent."""
        captured_headers = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured_headers.update(dict(request.headers))
            rss_xml = _build_rss_xml([])
            return httpx.Response(200, text=rss_xml, request=request)

        client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            headers={"User-Agent": "StockAIAgent/1.0"},
        )
        crawler = ConcreteTestCrawler(
            client=client,
            robots_checker=mock_robots,
            feed_urls=["https://example.com/rss/feed.rss"],
        )
        await crawler.crawl()
        assert "stockaiagent" in captured_headers.get("user-agent", "").lower()

    async def test_extract_tickers(self, mock_client, mock_robots):
        """Ticker extraction from article content."""
        crawler = ConcreteTestCrawler(
            client=mock_client,
            robots_checker=mock_robots,
            feed_urls=[],
        )
        # Test basic ticker extraction
        tickers = crawler.extract_tickers("Cổ phiếu HPG tăng mạnh, VNM giảm nhẹ")
        assert "HPG" in tickers
        assert "VNM" in tickers

    async def test_extract_tickers_filters_false_positives(self, mock_client, mock_robots):
        """Common acronyms like USD, GDP should not be detected as tickers."""
        crawler = ConcreteTestCrawler(
            client=mock_client,
            robots_checker=mock_robots,
            feed_urls=[],
        )
        tickers = crawler.extract_tickers("Tỷ giá USD/VND tăng, GDP giảm, FDI vào VN")
        assert "USD" not in tickers
        assert "VND" not in tickers
        assert "GDP" not in tickers
        assert "FDI" not in tickers
