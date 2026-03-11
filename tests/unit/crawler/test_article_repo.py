"""Tests for article persistence repository."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.models.article import ArticleCreate
from services.crawler.news.article_repo import save_articles


def _make_article(url: str, title: str = "Test", source: str = "test") -> ArticleCreate:
    return ArticleCreate(
        source=source,
        title=title,
        url=url,
        published_at=datetime(2026, 3, 10, tzinfo=timezone.utc),
        raw_content="content",
    )


class TestArticleRepo:

    @pytest.fixture
    def mock_session(self):
        session = AsyncMock()
        session.add = MagicMock()
        session.commit = AsyncMock()
        # Default: no existing URLs
        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = []
        session.execute = AsyncMock(return_value=result_mock)
        return session

    @patch("services.crawler.news.article_repo.get_async_session")
    async def test_save_new_articles(self, mock_get_session, mock_session):
        mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

        articles = [
            _make_article("https://example.com/a1"),
            _make_article("https://example.com/a2"),
        ]
        count = await save_articles(articles)
        assert count == 2
        assert mock_session.add.call_count == 2
        mock_session.commit.assert_awaited_once()

    @patch("services.crawler.news.article_repo.get_async_session")
    async def test_dedup_skips_existing_urls(self, mock_get_session, mock_session):
        """Articles with existing URLs should not be inserted."""
        # Simulate one existing URL in DB
        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = ["https://example.com/a1"]
        mock_session.execute = AsyncMock(return_value=result_mock)
        mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

        articles = [
            _make_article("https://example.com/a1"),
            _make_article("https://example.com/a2"),
        ]
        count = await save_articles(articles)
        assert count == 1
        assert mock_session.add.call_count == 1

    @patch("services.crawler.news.article_repo.get_async_session")
    async def test_save_empty_list(self, mock_get_session, mock_session):
        mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

        count = await save_articles([])
        assert count == 0

    @patch("services.crawler.news.article_repo.get_async_session")
    async def test_all_duplicates_returns_zero(self, mock_get_session, mock_session):
        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = [
            "https://example.com/a1",
            "https://example.com/a2",
        ]
        mock_session.execute = AsyncMock(return_value=result_mock)
        mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

        articles = [
            _make_article("https://example.com/a1"),
            _make_article("https://example.com/a2"),
        ]
        count = await save_articles(articles)
        assert count == 0
        assert mock_session.add.call_count == 0
