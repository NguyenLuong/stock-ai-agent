"""Vietstock RSS news crawler."""

from __future__ import annotations

import httpx
from bs4 import BeautifulSoup

from middleware.robots_checker import RobotsChecker
from news.base_crawler import BaseNewsCrawler


class VietstockCrawler(BaseNewsCrawler):
    """Crawl financial news from vietstock.vn via RSS feeds."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        robots_checker: RobotsChecker,
        rss_feeds: list[str],
    ) -> None:
        super().__init__(client=client, robots_checker=robots_checker, source_name="vietstock")
        self._rss_feeds = rss_feeds

    async def get_rss_feeds(self) -> list[str]:
        return self._rss_feeds

    def parse_article_page(self, url: str, html: str) -> str:
        """Extract article body from Vietstock HTML."""
        soup = BeautifulSoup(html, "lxml")

        # Try specific selectors first
        for selector in self._get_article_selectors():
            content = soup.select_one(selector)
            if content:
                return content.get_text(separator=" ", strip=True)

        # Fallback to <article> tag
        article = soup.find("article")
        if article:
            return article.get_text(separator=" ", strip=True)

        # Fallback to largest div with most <p> children
        return self._fallback_extract(soup)

    def _get_article_selectors(self) -> list[str]:
        return [
            "div.content-detail",
            "div.article-content",
            "div#content-detail",
        ]
