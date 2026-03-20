"""Crawl orchestrator — runs all enabled news crawlers sequentially."""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx
from shared.llm.config_loader import get_sources
from shared.logging import get_logger

from middleware.rate_limiter import RateLimitedTransport
from middleware.robots_checker import RobotsChecker
from news.article_repo import save_articles
from news.cafef_crawler import CafeFCrawler
from news.vietstock_crawler import VietstockCrawler
from news.vneconomy_crawler import VnEconomyCrawler

logger = get_logger("crawler.crawl_manager")

def _get_crawler_map():
    return {
        "vietstock": VietstockCrawler,
        "cafef": CafeFCrawler,
        "vneconomy": VnEconomyCrawler,
    }

_USER_AGENT = "StockAIAgent/1.0"


@dataclass
class CrawlResult:
    """Summary of a crawl cycle."""

    total_articles: int = 0
    new_articles: int = 0
    sources_crawled: int = 0
    errors: int = 0
    source_details: dict[str, int] = field(default_factory=dict)


async def run_news_crawl() -> CrawlResult:
    """Run all enabled news crawlers sequentially, save articles, return summary."""
    config = get_sources()
    sources = config.get("sources", {})
    result = CrawlResult()

    enabled_sources = {
        name: cfg for name, cfg in sources.items()
        if cfg.get("enabled", False) and name in _get_crawler_map()
    }

    if not enabled_sources:
        logger.info("no_enabled_sources", component="crawl_manager")
        return result

    all_articles = []

    for source_name, source_cfg in enabled_sources.items():
        rate_rps = source_cfg.get("rate_limit_rps", 1)
        rss_feeds = source_cfg.get("rss_feeds", [])

        transport = RateLimitedTransport(rate_per_second=rate_rps)
        async with httpx.AsyncClient(
            transport=transport,
            headers={"User-Agent": _USER_AGENT},
            timeout=30.0,
        ) as client:
            robots_checker = RobotsChecker(client=client, user_agent=_USER_AGENT)
            crawler_cls = _get_crawler_map()[source_name]
            crawler = crawler_cls(
                client=client,
                robots_checker=robots_checker,
                rss_feeds=rss_feeds,
            )

            try:
                articles = await crawler.crawl()
                all_articles.extend(articles)
                result.sources_crawled += 1
                result.source_details[source_name] = len(articles)
                logger.info(
                    "source_crawled",
                    component="crawl_manager",
                    source=source_name,
                    articles=len(articles),
                )
            except Exception as exc:
                result.errors += 1
                logger.error(
                    "source_crawl_failed",
                    component="crawl_manager",
                    source=source_name,
                    error=str(exc),
                )

    result.total_articles = len(all_articles)

    if all_articles:
        result.new_articles = await save_articles(all_articles)

    logger.info(
        "crawl_cycle_complete",
        component="crawl_manager",
        total_articles=result.total_articles,
        new_articles=result.new_articles,
        sources_crawled=result.sources_crawled,
        errors=result.errors,
    )
    return result
