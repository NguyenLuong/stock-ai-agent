"""Tests for Vietstock RSS crawler."""

from unittest.mock import AsyncMock
from xml.etree.ElementTree import Element, SubElement, tostring

import httpx
import pytest

from services.crawler.middleware.robots_checker import RobotsChecker
from services.crawler.news.vietstock_crawler import VietstockCrawler


def _rss_xml(items):
    rss = Element("rss", version="2.0")
    channel = SubElement(rss, "channel")
    SubElement(channel, "title").text = "Vietstock"
    for it in items:
        item = SubElement(channel, "item")
        SubElement(item, "title").text = it["title"]
        SubElement(item, "link").text = it["link"]
        SubElement(item, "pubDate").text = it.get("pubDate", "Mon, 10 Mar 2026 06:00:00 +0000")
    return tostring(rss, encoding="unicode")


ARTICLE_HTML = """
<html><body>
<div class="content-detail">
  <p>HPG tăng 5% trong phiên sáng nay.</p>
  <p>Khối lượng giao dịch đạt 10 triệu cổ phiếu.</p>
</div>
</body></html>
"""

ARTICLE_HTML_FALLBACK = """
<html><body>
<div class="unknown-class">
  <p>Paragraph 1</p>
  <p>Paragraph 2</p>
  <p>Paragraph 3</p>
</div>
</body></html>
"""


class TestVietstockCrawler:

    @pytest.fixture
    def mock_robots(self):
        robots = AsyncMock(spec=RobotsChecker)
        robots.can_fetch = AsyncMock(return_value=True)
        return robots

    @pytest.fixture
    def mock_client(self):
        items = [{"title": "HPG tăng mạnh", "link": "https://vietstock.vn/article-1"}]

        async def handler(request: httpx.Request) -> httpx.Response:
            if "rss" in str(request.url):
                return httpx.Response(200, text=_rss_xml(items), request=request)
            return httpx.Response(200, text=ARTICLE_HTML, request=request)

        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def test_crawl_parses_rss_and_extracts_content(self, mock_client, mock_robots):
        crawler = VietstockCrawler(
            client=mock_client,
            robots_checker=mock_robots,
            rss_feeds=["https://vietstock.vn/rss/tai-chinh.rss"],
        )
        articles = await crawler.crawl()
        assert len(articles) == 1
        assert articles[0].source == "vietstock"
        assert articles[0].title == "HPG tăng mạnh"
        assert "10 triệu" in articles[0].raw_content

    async def test_parse_article_specific_selector(self, mock_client, mock_robots):
        crawler = VietstockCrawler(
            client=mock_client, robots_checker=mock_robots, rss_feeds=[]
        )
        content = crawler.parse_article_page("https://vietstock.vn/art", ARTICLE_HTML)
        assert "HPG tăng 5%" in content

    async def test_parse_article_fallback(self, mock_client, mock_robots):
        crawler = VietstockCrawler(
            client=mock_client, robots_checker=mock_robots, rss_feeds=[]
        )
        content = crawler.parse_article_page("https://vietstock.vn/art", ARTICLE_HTML_FALLBACK)
        assert "Paragraph 1" in content

    async def test_ticker_extracted_from_content(self, mock_client, mock_robots):
        crawler = VietstockCrawler(
            client=mock_client,
            robots_checker=mock_robots,
            rss_feeds=["https://vietstock.vn/rss/tai-chinh.rss"],
        )
        articles = await crawler.crawl()
        assert articles[0].ticker_symbol == "HPG"
