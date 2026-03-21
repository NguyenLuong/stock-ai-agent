"""Tests for the article embedding pipeline."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from embedding.embedding_pipeline import (
    _prepare_text,
    run_embedding_pipeline,
)
from embedding.models import EmbeddingPipelineResult


def _make_orm_article(
    title: str = "Test Article",
    raw_content: str | None = "Some content",
    summary: str | None = None,
    embedded: bool = False,
) -> MagicMock:
    """Create a mock Article ORM object."""
    article = MagicMock()
    article.id = uuid.uuid4()
    article.title = title
    article.raw_content = raw_content
    article.summary = summary
    article.embedded = embedded
    article.embedding = None
    return article


class TestPrepareText:

    def test_uses_title_and_raw_content(self):
        article = _make_orm_article(title="Tiêu đề", raw_content="Nội dung bài viết")
        result = _prepare_text(article)
        assert result == "Tiêu đề\n\nNội dung bài viết"

    def test_falls_back_to_summary_when_no_raw_content(self):
        article = _make_orm_article(
            title="Tiêu đề", raw_content=None, summary="Tóm tắt"
        )
        result = _prepare_text(article)
        assert result == "Tiêu đề\n\nTóm tắt"

    def test_returns_none_when_no_content_and_no_summary(self):
        article = _make_orm_article(
            title="Tiêu đề", raw_content=None, summary=None
        )
        result = _prepare_text(article)
        assert result is None

    def test_prefers_raw_content_over_summary(self):
        article = _make_orm_article(
            title="Title", raw_content="Raw", summary="Summary"
        )
        result = _prepare_text(article)
        assert result == "Title\n\nRaw"


class TestRunEmbeddingPipeline:

    @pytest.fixture
    def mock_session(self):
        session = AsyncMock()
        session.commit = AsyncMock()
        return session

    @patch("embedding.embedding_pipeline.get_async_session")
    @patch("embedding.embedding_pipeline.embed_texts")
    async def test_embeds_unprocessed_articles(
        self, mock_embed, mock_get_session, mock_session
    ):
        articles = [
            _make_orm_article(title="A1", raw_content="Content 1"),
            _make_orm_article(title="A2", raw_content="Content 2"),
        ]
        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = articles
        mock_session.execute = AsyncMock(return_value=result_mock)
        mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

        mock_embed.return_value = [[0.1] * 1536, [0.2] * 1536]

        result = await run_embedding_pipeline(batch_size=100)

        assert isinstance(result, EmbeddingPipelineResult)
        assert result.total == 2
        assert result.embedded_count == 2
        assert result.failed_count == 0
        assert result.skipped_count == 0
        mock_embed.assert_awaited_once()
        mock_session.commit.assert_awaited_once()

        # Verify articles were updated
        for article in articles:
            assert article.embedded is True
            assert article.embedding is not None

    @patch("embedding.embedding_pipeline.get_async_session")
    async def test_handles_empty_batch(self, mock_get_session, mock_session):
        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = []
        mock_session.execute = AsyncMock(return_value=result_mock)
        mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await run_embedding_pipeline()

        assert result.total == 0
        assert result.embedded_count == 0
        assert result.skipped_count == 0

    @patch("embedding.embedding_pipeline.get_async_session")
    @patch("embedding.embedding_pipeline.embed_texts")
    async def test_skips_articles_without_content(
        self, mock_embed, mock_get_session, mock_session
    ):
        articles = [
            _make_orm_article(title="Has Content", raw_content="Content"),
            _make_orm_article(title="No Content", raw_content=None, summary=None),
        ]
        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = articles
        mock_session.execute = AsyncMock(return_value=result_mock)
        mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

        mock_embed.return_value = [[0.1] * 1536]

        result = await run_embedding_pipeline()

        assert result.total == 2
        assert result.embedded_count == 1
        assert result.skipped_count == 1
        # Only one text should be embedded
        mock_embed.assert_awaited_once_with(["Has Content\n\nContent"], batch_size=100)

    @patch("embedding.embedding_pipeline.get_async_session")
    @patch("embedding.embedding_pipeline.embed_texts")
    async def test_handles_embedding_error(
        self, mock_embed, mock_get_session, mock_session
    ):
        articles = [_make_orm_article(title="A1", raw_content="Content")]
        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = articles
        mock_session.execute = AsyncMock(return_value=result_mock)
        mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

        mock_embed.side_effect = Exception("API error")

        result = await run_embedding_pipeline()

        assert result.total == 1
        assert result.embedded_count == 0
        assert result.failed_count == 1
        mock_session.commit.assert_not_awaited()
        mock_session.rollback.assert_awaited_once()

    @patch("embedding.embedding_pipeline.get_async_session")
    @patch("embedding.embedding_pipeline.embed_texts")
    async def test_uses_summary_fallback(
        self, mock_embed, mock_get_session, mock_session
    ):
        articles = [
            _make_orm_article(title="Title", raw_content=None, summary="Summary text"),
        ]
        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = articles
        mock_session.execute = AsyncMock(return_value=result_mock)
        mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

        mock_embed.return_value = [[0.1] * 1536]

        result = await run_embedding_pipeline()

        assert result.embedded_count == 1
        mock_embed.assert_awaited_once_with(
            ["Title\n\nSummary text"], batch_size=100
        )

    @patch("embedding.embedding_pipeline.get_async_session")
    @patch("embedding.embedding_pipeline.embed_texts")
    async def test_pipeline_result_has_duration(
        self, mock_embed, mock_get_session, mock_session
    ):
        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = []
        mock_session.execute = AsyncMock(return_value=result_mock)
        mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await run_embedding_pipeline()

        assert result.duration_seconds >= 0
