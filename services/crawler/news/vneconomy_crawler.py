"""VnEconomy RSS news crawler."""

from __future__ import annotations

import httpx
from bs4 import BeautifulSoup

from middleware.robots_checker import RobotsChecker
from news.base_crawler import BaseNewsCrawler


class VnEconomyCrawler(BaseNewsCrawler):
    """Crawl financial news from vneconomy.vn via RSS feeds."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        robots_checker: RobotsChecker,
        rss_feeds: list[dict | str],
    ) -> None:
        super().__init__(client=client, robots_checker=robots_checker, source_name="vneconomy")
        self._rss_feeds = rss_feeds

    async def get_rss_feeds(self) -> list[dict]:
        return self._rss_feeds

    def parse_article_page(self, url: str, html: str) -> str:
        """Extract article body from VnEconomy HTML."""
        soup = BeautifulSoup(html, "lxml")

        for selector in self._get_article_selectors():
            content = soup.select_one(selector)
            if content:
                return content.get_text(separator=" ", strip=True)

        article = soup.find("article")
        if article:
            return article.get_text(separator=" ", strip=True)

        return self._fallback_extract(soup)

    def _get_article_selectors(self) -> list[str]:
        return [
            "div.detail__content",
            "div.article-body",
            "div.content-detail",
        ]
