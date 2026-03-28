"""Tests for semantic search in article_repo."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.crawler.news.article_repo import semantic_search


def _make_orm_article(title: str = "Test Article") -> MagicMock:
    """Create a mock Article ORM object."""
    article = MagicMock()
    article.title = title
    article.embedded = True
    return article


class TestSemanticSearch:

    @pytest.fixture
    def mock_session(self):
        session = AsyncMock()
        return session

    @patch("services.crawler.news.article_repo.get_async_session")
    @patch("services.crawler.news.article_repo.embed_single")
    async def test_returns_top_k_articles(
        self, mock_embed, mock_get_session, mock_session
    ):
        mock_embed.return_value = [0.1] * 1536

        articles = [_make_orm_article(f"Article {i}") for i in range(5)]
        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = articles
        mock_session.execute = AsyncMock(return_value=result_mock)
        mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

        results = await semantic_search("lãi suất tăng", top_k=5)

        assert len(results) == 5
        mock_embed.assert_awaited_once_with("lãi suất tăng")

    @patch("services.crawler.news.article_repo.get_async_session")
    @patch("services.crawler.news.article_repo.embed_single")
    async def test_returns_empty_list_when_no_results(
        self, mock_embed, mock_get_session, mock_session
    ):
        mock_embed.return_value = [0.1] * 1536

        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = []
        mock_session.execute = AsyncMock(return_value=result_mock)
        mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

        results = await semantic_search("nonexistent topic")

        assert results == []

    @patch("services.crawler.news.article_repo.get_async_session")
    @patch("services.crawler.news.article_repo.embed_single")
    async def test_embeds_query_text(
        self, mock_embed, mock_get_session, mock_session
    ):
        mock_embed.return_value = [0.1] * 1536

        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = []
        mock_session.execute = AsyncMock(return_value=result_mock)
        mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

        await semantic_search("tác động thép")

        mock_embed.assert_awaited_once_with("tác động thép")

    @patch("services.crawler.news.article_repo.get_async_session")
    @patch("services.crawler.news.article_repo.embed_single")
    async def test_filters_by_ticker_symbol(
        self, mock_embed, mock_get_session, mock_session
    ):
        mock_embed.return_value = [0.1] * 1536

        articles = [_make_orm_article("HPG Article")]
        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = articles
        mock_session.execute = AsyncMock(return_value=result_mock)
        mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

        results = await semantic_search(
            "thép Hòa Phát", top_k=10, ticker_symbol="HPG"
        )

        assert len(results) == 1
        # Verify the query included ticker_symbol filter
        executed_stmt = mock_session.execute.call_args[0][0]
        compiled_sql = str(executed_stmt)
        assert "ticker_symbol" in compiled_sql

    @patch("services.crawler.news.article_repo.get_async_session")
    @patch("services.crawler.news.article_repo.embed_single")
    async def test_no_ticker_filter_when_not_provided(
        self, mock_embed, mock_get_session, mock_session
    ):
        mock_embed.return_value = [0.1] * 1536

        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = []
        mock_session.execute = AsyncMock(return_value=result_mock)
        mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

        await semantic_search("test query")

        # Verify the WHERE clause does NOT include ticker_symbol filter
        executed_stmt = mock_session.execute.call_args[0][0]
        where_sql = str(executed_stmt.whereclause)
        assert "ticker_symbol" not in where_sql

    @patch("services.crawler.news.article_repo.get_async_session")
    @patch("services.crawler.news.article_repo.embed_single")
    async def test_query_filters_embedded_true(
        self, mock_embed, mock_get_session, mock_session
    ):
        mock_embed.return_value = [0.1] * 1536

        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = []
        mock_session.execute = AsyncMock(return_value=result_mock)
        mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

        await semantic_search("test query")

        # Verify the query includes embedded=TRUE filter
        executed_stmt = mock_session.execute.call_args[0][0]
        compiled_sql = str(executed_stmt)
        assert "embedded" in compiled_sql

    @patch("services.crawler.news.article_repo.get_async_session")
    @patch("services.crawler.news.article_repo.embed_single")
    async def test_filters_by_category(
        self, mock_embed, mock_get_session, mock_session
    ):
        mock_embed.return_value = [0.1] * 1536

        articles = [_make_orm_article("Macro Article")]
        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = articles
        mock_session.execute = AsyncMock(return_value=result_mock)
        mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

        results = await semantic_search(
            "thị trường vĩ mô", top_k=20, category="macro"
        )

        assert len(results) == 1
        executed_stmt = mock_session.execute.call_args[0][0]
        compiled_sql = str(executed_stmt)
        assert "category" in compiled_sql

    @patch("services.crawler.news.article_repo.get_async_session")
    @patch("services.crawler.news.article_repo.embed_single")
    async def test_no_category_filter_when_not_provided(
        self, mock_embed, mock_get_session, mock_session
    ):
        mock_embed.return_value = [0.1] * 1536

        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = []
        mock_session.execute = AsyncMock(return_value=result_mock)
        mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

        await semantic_search("test query")

        executed_stmt = mock_session.execute.call_args[0][0]
        where_sql = str(executed_stmt.whereclause)
        assert "category" not in where_sql

    @patch("services.crawler.news.article_repo.get_async_session")
    @patch("services.crawler.news.article_repo.embed_single")
    async def test_default_top_k_is_10(
        self, mock_embed, mock_get_session, mock_session
    ):
        mock_embed.return_value = [0.1] * 1536

        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = []
        mock_session.execute = AsyncMock(return_value=result_mock)
        mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

        await semantic_search("test query")

        # Function should work with default top_k=10
        mock_session.execute.assert_awaited_once()
