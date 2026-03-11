"""Base news crawler with RSS parsing, robots.txt compliance, and error handling."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from datetime import datetime
from email.utils import parsedate_to_datetime
from xml.etree.ElementTree import ParseError

import httpx
from bs4 import BeautifulSoup
from defusedxml.ElementTree import fromstring
from shared.logging import get_logger
from shared.models.article import ArticleCreate
from shared.utils.datetime_utils import now_utc
from shared.utils.text_utils import normalize_whitespace

from services.crawler.middleware.robots_checker import RobotsChecker

# Vietnamese stock tickers: 3 uppercase letters
_TICKER_PATTERN = re.compile(r"\b([A-Z]{3})\b")
_FALSE_POSITIVE_TICKERS = frozenset({
    "CEO", "USD", "VND", "GDP", "CPI", "FDI", "ODA", "WTO", "IMF", "FED",
    "ETF", "IPO", "ROE", "ROA", "EPS", "BPS",
    "SBV", "SSC", "HNX", "HSX", "VSD",
})


class BaseNewsCrawler(ABC):
    """Abstract base class for RSS-based news crawlers."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        robots_checker: RobotsChecker,
        source_name: str,
    ) -> None:
        self.client = client
        self.robots_checker = robots_checker
        self.source_name = source_name
        self.logger = get_logger(f"crawler.{source_name}")

    @abstractmethod
    async def get_rss_feeds(self) -> list[str]:
        """Return list of RSS feed URLs for this source."""

    @abstractmethod
    def parse_article_page(self, url: str, html: str) -> str:
        """Extract article body text from full HTML page."""

    async def crawl(self) -> list[ArticleCreate]:
        """Main crawl flow: fetch RSS -> check robots -> fetch articles -> parse."""
        articles: list[ArticleCreate] = []
        seen_urls: set[str] = set()

        for feed_url in await self.get_rss_feeds():
            feed_articles = await self._process_feed(feed_url, seen_urls)
            articles.extend(feed_articles)

        self.logger.info(
            "crawl_complete",
            component=f"crawler.{self.source_name}",
            articles_count=len(articles),
        )
        return articles

    async def _process_feed(
        self, feed_url: str, seen_urls: set[str]
    ) -> list[ArticleCreate]:
        """Fetch and process a single RSS feed."""
        articles: list[ArticleCreate] = []

        try:
            response = await self.client.get(feed_url)
            response.raise_for_status()
        except (httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException) as exc:
            self.logger.error(
                "rss_fetch_failed",
                component=f"crawler.{self.source_name}",
                feed_url=feed_url,
                error=str(exc),
            )
            return articles

        try:
            rss_items = self._parse_rss_xml(response.text)
        except ParseError as exc:
            self.logger.error(
                "rss_parse_failed",
                component=f"crawler.{self.source_name}",
                feed_url=feed_url,
                error=str(exc),
            )
            return articles

        for item in rss_items:
            url = item.get("link", "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            article = await self._fetch_article(item)
            if article:
                articles.append(article)

        return articles

    async def _fetch_article(self, item: dict) -> ArticleCreate | None:
        """Fetch full article content and create ArticleCreate."""
        url = item["link"]

        # Check robots.txt
        if not await self.robots_checker.can_fetch(url):
            return None

        try:
            response = await self.client.get(url)
            response.raise_for_status()
        except (httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException) as exc:
            self.logger.warning(
                "article_fetch_failed",
                component=f"crawler.{self.source_name}",
                url=url,
                error=str(exc),
            )
            return None

        raw_content = self.parse_article_page(url, response.text)
        if raw_content:
            raw_content = normalize_whitespace(raw_content)

        # Extract ticker from title + content
        text_for_tickers = f"{item.get('title', '')} {raw_content or ''}"
        tickers = self.extract_tickers(text_for_tickers)
        ticker_symbol = tickers[0] if tickers else None

        published_at = self._parse_pub_date(item.get("pubDate", ""))

        return ArticleCreate(
            source=self.source_name,
            title=item.get("title", ""),
            url=url,
            published_at=published_at,
            raw_content=raw_content or None,
            ticker_symbol=ticker_symbol,
        )

    @staticmethod
    def _parse_rss_xml(xml_text: str) -> list[dict]:
        """Parse RSS XML and return list of item dicts with title, link, pubDate."""
        root = fromstring(xml_text)
        items = []
        for item_elem in root.iter("item"):
            item: dict[str, str] = {}
            for field in ("title", "link", "pubDate", "description"):
                elem = item_elem.find(field)
                if elem is not None and elem.text:
                    item[field] = elem.text.strip()
            if "link" in item:
                items.append(item)
        return items

    @staticmethod
    def _parse_pub_date(date_str: str) -> datetime:
        """Parse RSS pubDate to timezone-aware datetime."""
        if not date_str:
            return now_utc()
        try:
            return parsedate_to_datetime(date_str)
        except (ValueError, TypeError):
            return now_utc()

    @staticmethod
    def _fallback_extract(soup: BeautifulSoup) -> str:
        """Find the div with the most <p> children as fallback."""
        best_div = None
        max_p_count = 0
        for div in soup.find_all("div"):
            p_count = len(div.find_all("p", recursive=False))
            if p_count > max_p_count:
                max_p_count = p_count
                best_div = div
        if best_div:
            return best_div.get_text(separator=" ", strip=True)
        return ""

    @staticmethod
    def extract_tickers(text: str) -> list[str]:
        """Extract Vietnamese stock tickers from text."""
        matches = _TICKER_PATTERN.findall(text)
        return [t for t in dict.fromkeys(matches) if t not in _FALSE_POSITIVE_TICKERS]
