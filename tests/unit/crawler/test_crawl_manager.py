"""Tests for crawl orchestrator."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from shared.models.article import ArticleCreate
from services.crawler.news.crawl_manager import run_news_crawl, CrawlResult


def _make_article(url: str, source: str) -> ArticleCreate:
    return ArticleCreate(
        source=source,
        title=f"Article from {source}",
        url=url,
        published_at=datetime(2026, 3, 10, tzinfo=timezone.utc),
        raw_content="content",
    )


MOCK_SOURCES_CONFIG = {
    "sources": {
        "vietstock": {
            "base_url": "https://vietstock.vn",
            "rate_limit_rps": 1,
            "enabled": True,
            "rss_feeds": ["https://vietstock.vn/rss/tai-chinh.rss"],
        },
        "cafef": {
            "base_url": "https://cafef.vn",
            "rate_limit_rps": 1,
            "enabled": True,
            "rss_feeds": ["https://cafef.vn/rss/trang-chu.rss"],
        },
        "vneconomy": {
            "base_url": "https://vneconomy.vn",
            "rate_limit_rps": 1,
            "enabled": False,  # Disabled
            "rss_feeds": ["https://vneconomy.vn/rss/chung-khoan.rss"],
        },
    }
}


class TestCrawlManager:

    @patch("services.crawler.news.crawl_manager.get_sources")
    @patch("services.crawler.news.crawl_manager.save_articles")
    @patch("services.crawler.news.crawl_manager._get_crawler_map")
    async def test_runs_enabled_crawlers_only(
        self, mock_crawler_map, mock_save, mock_get_sources
    ):
        mock_get_sources.return_value = MOCK_SOURCES_CONFIG

        mock_vs = AsyncMock()
        mock_vs.crawl.return_value = [
            _make_article("https://vietstock.vn/a1", "vietstock"),
            _make_article("https://vietstock.vn/a2", "vietstock"),
        ]
        MockVietstock = MagicMock(return_value=mock_vs)

        mock_cf = AsyncMock()
        mock_cf.crawl.return_value = [
            _make_article("https://cafef.vn/a1", "cafef"),
        ]
        MockCafeF = MagicMock(return_value=mock_cf)

        MockVnEconomy = MagicMock()

        mock_crawler_map.return_value = {
            "vietstock": MockVietstock,
            "cafef": MockCafeF,
            "vneconomy": MockVnEconomy,
        }

        mock_save.return_value = 3

        result = await run_news_crawl()

        assert isinstance(result, CrawlResult)
        assert result.total_articles == 3
        assert result.sources_crawled == 2
        assert result.errors == 0
        # VnEconomy disabled — not instantiated
        MockVnEconomy.assert_not_called()

    @patch("services.crawler.news.crawl_manager.get_sources")
    @patch("services.crawler.news.crawl_manager.save_articles")
    @patch("services.crawler.news.crawl_manager._get_crawler_map")
    async def test_graceful_degradation_on_source_failure(
        self, mock_crawler_map, mock_save, mock_get_sources
    ):
        """If one source fails, continue with others."""
        config = {
            "sources": {
                "vietstock": {
                    "base_url": "https://vietstock.vn",
                    "rate_limit_rps": 1,
                    "enabled": True,
                    "rss_feeds": ["https://vietstock.vn/rss/feed.rss"],
                },
                "cafef": {
                    "base_url": "https://cafef.vn",
                    "rate_limit_rps": 1,
                    "enabled": True,
                    "rss_feeds": ["https://cafef.vn/rss/feed.rss"],
                },
            }
        }
        mock_get_sources.return_value = config

        mock_vs = AsyncMock()
        mock_vs.crawl.side_effect = Exception("Connection refused")
        MockVietstock = MagicMock(return_value=mock_vs)

        mock_cf = AsyncMock()
        mock_cf.crawl.return_value = [_make_article("https://cafef.vn/a1", "cafef")]
        MockCafeF = MagicMock(return_value=mock_cf)

        mock_crawler_map.return_value = {
            "vietstock": MockVietstock,
            "cafef": MockCafeF,
        }

        mock_save.return_value = 1

        result = await run_news_crawl()
        assert result.total_articles == 1
        assert result.errors == 1
        assert result.sources_crawled == 1

    @patch("services.crawler.news.crawl_manager.get_sources")
    @patch("services.crawler.news.crawl_manager.save_articles")
    @patch("services.crawler.news.crawl_manager._get_crawler_map")
    async def test_no_enabled_sources(self, mock_crawler_map, mock_save, mock_get_sources):
        mock_get_sources.return_value = {
            "sources": {
                "vietstock": {"enabled": False, "rss_feeds": []},
            }
        }
        mock_crawler_map.return_value = {"vietstock": MagicMock()}
        result = await run_news_crawl()
        assert result.total_articles == 0
        assert result.sources_crawled == 0
        mock_save.assert_not_called()
