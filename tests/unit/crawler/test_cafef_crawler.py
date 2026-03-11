"""Tests for CafeF RSS crawler."""

from unittest.mock import AsyncMock
from xml.etree.ElementTree import Element, SubElement, tostring

import httpx
import pytest

from services.crawler.middleware.robots_checker import RobotsChecker
from services.crawler.news.cafef_crawler import CafeFCrawler


def _rss_xml(items):
    rss = Element("rss", version="2.0")
    channel = SubElement(rss, "channel")
    SubElement(channel, "title").text = "CafeF"
    for it in items:
        item = SubElement(channel, "item")
        SubElement(item, "title").text = it["title"]
        SubElement(item, "link").text = it["link"]
        SubElement(item, "pubDate").text = it.get("pubDate", "Mon, 10 Mar 2026 06:00:00 +0000")
    return tostring(rss, encoding="unicode")


ARTICLE_HTML = """
<html><body>
<div class="detail-content">
  <p>VNM công bố kết quả kinh doanh quý 1.</p>
  <p>Doanh thu tăng 15% so với cùng kỳ.</p>
</div>
</body></html>
"""

ARTICLE_HTML_FALLBACK = """
<html><body>
<article>
  <p>Article tag content here</p>
</article>
</body></html>
"""


class TestCafeFCrawler:

    @pytest.fixture
    def mock_robots(self):
        robots = AsyncMock(spec=RobotsChecker)
        robots.can_fetch = AsyncMock(return_value=True)
        return robots

    @pytest.fixture
    def mock_client(self):
        items = [{"title": "VNM báo lãi lớn", "link": "https://cafef.vn/article-1"}]

        async def handler(request: httpx.Request) -> httpx.Response:
            if "rss" in str(request.url):
                return httpx.Response(200, text=_rss_xml(items), request=request)
            return httpx.Response(200, text=ARTICLE_HTML, request=request)

        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def test_crawl_parses_rss_and_extracts_content(self, mock_client, mock_robots):
        crawler = CafeFCrawler(
            client=mock_client,
            robots_checker=mock_robots,
            rss_feeds=["https://cafef.vn/rss/trang-chu.rss"],
        )
        articles = await crawler.crawl()
        assert len(articles) == 1
        assert articles[0].source == "cafef"
        assert articles[0].title == "VNM báo lãi lớn"
        assert "15%" in articles[0].raw_content

    async def test_parse_article_specific_selector(self, mock_client, mock_robots):
        crawler = CafeFCrawler(
            client=mock_client, robots_checker=mock_robots, rss_feeds=[]
        )
        content = crawler.parse_article_page("https://cafef.vn/art", ARTICLE_HTML)
        assert "VNM công bố" in content

    async def test_parse_article_fallback_to_article_tag(self, mock_client, mock_robots):
        crawler = CafeFCrawler(
            client=mock_client, robots_checker=mock_robots, rss_feeds=[]
        )
        content = crawler.parse_article_page("https://cafef.vn/art", ARTICLE_HTML_FALLBACK)
        assert "Article tag content" in content

    async def test_ticker_extracted(self, mock_client, mock_robots):
        crawler = CafeFCrawler(
            client=mock_client,
            robots_checker=mock_robots,
            rss_feeds=["https://cafef.vn/rss/trang-chu.rss"],
        )
        articles = await crawler.crawl()
        assert articles[0].ticker_symbol == "VNM"
