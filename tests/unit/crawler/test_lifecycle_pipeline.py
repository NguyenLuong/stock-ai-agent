"""Tests for the data lifecycle pipeline."""

from __future__ import annotations

import uuid
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.llm.client import LLMCallError

from lifecycle.lifecycle_pipeline import (
    _build_summary_prompt,
    _estimate_tokens,
    run_lifecycle_pipeline,
)
from lifecycle.models import LifecyclePipelineResult


def _make_orm_article(
    title: str = "Test Article",
    raw_content: str | None = "Nội dung bài viết gốc",
    summary: str | None = None,
    embedded: bool = True,
    embedding: list[float] | None = None,
    published_at=None,
) -> MagicMock:
    """Create a mock Article ORM object."""
    article = MagicMock()
    article.id = uuid.uuid4()
    article.title = title
    article.raw_content = raw_content
    article.summary = summary
    article.embedded = embedded
    article.embedding = embedding or [0.1] * 1536
    article.published_at = published_at
    return article


def _setup_mock_session(mock_get_session, mock_session, articles):
    """Wire up mock async session with articles query result."""
    result_mock = MagicMock()
    result_mock.scalars.return_value.all.return_value = articles
    mock_session.execute = AsyncMock(return_value=result_mock)
    mock_session.commit = AsyncMock()
    mock_session.rollback = AsyncMock()
    mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
    mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)


class TestBuildSummaryPrompt:

    @patch("lifecycle.lifecycle_pipeline.load_prompt")
    def test_calls_prompt_loader_with_correct_args(self, mock_load):
        mock_load.return_value = MagicMock(text="rendered prompt")
        article = _make_orm_article(title="VN-Index tăng mạnh", raw_content="Chi tiết...")

        result = _build_summary_prompt(article)

        mock_load.assert_called_once_with(
            "lifecycle/summarize_article",
            title="VN-Index tăng mạnh",
            raw_content="Chi tiết...",
        )
        assert result == "rendered prompt"


class TestRunLifecyclePipeline:

    @pytest.fixture
    def mock_session(self):
        session = AsyncMock()
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        return session

    @patch("lifecycle.lifecycle_pipeline.load_prompt")
    @patch("lifecycle.lifecycle_pipeline.get_async_session")
    @patch("lifecycle.lifecycle_pipeline.call_llm")
    async def test_summarizes_old_articles_successfully(
        self, mock_llm, mock_get_session, mock_load_prompt, mock_session
    ):
        articles = [
            _make_orm_article(title="Article 1", raw_content="Content 1"),
            _make_orm_article(title="Article 2", raw_content="Content 2"),
        ]
        _setup_mock_session(mock_get_session, mock_session, articles)
        mock_load_prompt.return_value = MagicMock(text="prompt text")
        mock_llm.return_value = "Tóm tắt bài viết"

        result = await run_lifecycle_pipeline()

        assert isinstance(result, LifecyclePipelineResult)
        assert result.total == 2
        assert result.summarized_count == 2
        assert result.skipped_count == 0
        assert result.failed_count == 0
        assert mock_llm.await_count == 2
        mock_session.commit.assert_awaited_once()

        for article in articles:
            assert article.summary == "Tóm tắt bài viết"
            assert article.raw_content is None

    @patch("lifecycle.lifecycle_pipeline.load_prompt")
    @patch("lifecycle.lifecycle_pipeline.get_async_session")
    @patch("lifecycle.lifecycle_pipeline.call_llm")
    async def test_skips_articles_already_summarized(
        self, mock_llm, mock_get_session, mock_load_prompt, mock_session
    ):
        articles = [
            _make_orm_article(title="Already done", raw_content="Content", summary="Existing summary"),
        ]
        _setup_mock_session(mock_get_session, mock_session, articles)

        result = await run_lifecycle_pipeline()

        assert result.total == 1
        assert result.summarized_count == 0
        assert result.skipped_count == 1
        assert result.failed_count == 0
        mock_llm.assert_not_awaited()

    @patch("lifecycle.lifecycle_pipeline.get_async_session")
    async def test_handles_empty_batch(self, mock_get_session, mock_session):
        _setup_mock_session(mock_get_session, mock_session, [])

        result = await run_lifecycle_pipeline()

        assert result.total == 0
        assert result.summarized_count == 0
        assert result.skipped_count == 0
        assert result.failed_count == 0
        assert result.duration_seconds >= 0

    @patch("lifecycle.lifecycle_pipeline.load_prompt")
    @patch("lifecycle.lifecycle_pipeline.get_async_session")
    @patch("lifecycle.lifecycle_pipeline.call_llm")
    async def test_sets_raw_content_to_none_after_summarization(
        self, mock_llm, mock_get_session, mock_load_prompt, mock_session
    ):
        article = _make_orm_article(title="Test", raw_content="Full content here")
        _setup_mock_session(mock_get_session, mock_session, [article])
        mock_load_prompt.return_value = MagicMock(text="prompt")
        mock_llm.return_value = "Summary text"

        await run_lifecycle_pipeline()

        assert article.raw_content is None
        assert article.summary == "Summary text"

    @patch("lifecycle.lifecycle_pipeline.load_prompt")
    @patch("lifecycle.lifecycle_pipeline.get_async_session")
    @patch("lifecycle.lifecycle_pipeline.call_llm")
    async def test_retains_embedding_vector(
        self, mock_llm, mock_get_session, mock_load_prompt, mock_session
    ):
        original_embedding = [0.5] * 1536
        article = _make_orm_article(
            title="Test",
            raw_content="Content",
            embedded=True,
            embedding=original_embedding,
        )
        _setup_mock_session(mock_get_session, mock_session, [article])
        mock_load_prompt.return_value = MagicMock(text="prompt")
        mock_llm.return_value = "Summary"

        await run_lifecycle_pipeline()

        assert article.embedding == original_embedding
        assert article.embedded is True

    @patch("lifecycle.lifecycle_pipeline.load_prompt")
    @patch("lifecycle.lifecycle_pipeline.get_async_session")
    @patch("lifecycle.lifecycle_pipeline.call_llm")
    async def test_handles_llm_failure_gracefully(
        self, mock_llm, mock_get_session, mock_load_prompt, mock_session
    ):
        articles = [
            _make_orm_article(title="Good", raw_content="Content 1"),
            _make_orm_article(title="Bad", raw_content="Content 2"),
        ]
        _setup_mock_session(mock_get_session, mock_session, articles)
        mock_load_prompt.return_value = MagicMock(text="prompt")
        mock_llm.side_effect = [
            "Good summary",
            Exception("LLM API error"),
        ]

        result = await run_lifecycle_pipeline()

        assert result.total == 2
        assert result.summarized_count == 1
        assert result.failed_count == 1
        # First article should be summarized
        assert articles[0].summary == "Good summary"
        assert articles[0].raw_content is None
        # Second article should retain raw_content
        assert articles[1].raw_content == "Content 2"

    @patch("lifecycle.lifecycle_pipeline.load_prompt")
    @patch("lifecycle.lifecycle_pipeline.get_async_session")
    @patch("lifecycle.lifecycle_pipeline.call_llm")
    async def test_logs_pipeline_summary(
        self, mock_llm, mock_get_session, mock_load_prompt, mock_session
    ):
        articles = [
            _make_orm_article(title="A1", raw_content="C1"),
            _make_orm_article(title="A2", raw_content="C2", summary="Already done"),
        ]
        _setup_mock_session(mock_get_session, mock_session, articles)
        mock_load_prompt.return_value = MagicMock(text="prompt")
        mock_llm.return_value = "Summary"

        result = await run_lifecycle_pipeline()

        assert result.total == 2
        assert result.summarized_count == 1
        assert result.skipped_count == 1
        assert result.duration_seconds >= 0

    @patch("lifecycle.lifecycle_pipeline.load_prompt")
    @patch("lifecycle.lifecycle_pipeline.get_async_session")
    @patch("lifecycle.lifecycle_pipeline.call_llm")
    async def test_processes_in_batches(
        self, mock_llm, mock_get_session, mock_load_prompt, mock_session
    ):
        """Verify commit is called per batch, not per article."""
        articles = [
            _make_orm_article(title=f"Article {i}", raw_content=f"Content {i}")
            for i in range(3)
        ]
        _setup_mock_session(mock_get_session, mock_session, articles)
        mock_load_prompt.return_value = MagicMock(text="prompt")
        mock_llm.return_value = "Summary"

        # batch_size=2 means 2 batches: [0,1] and [2]
        result = await run_lifecycle_pipeline(batch_size=2)

        assert result.total == 3
        assert result.summarized_count == 3
        # 2 commits: one for batch [0,1], one for batch [2]
        assert mock_session.commit.await_count == 2

    @patch("lifecycle.lifecycle_pipeline.load_prompt")
    @patch("lifecycle.lifecycle_pipeline.get_async_session")
    @patch("lifecycle.lifecycle_pipeline.call_llm")
    async def test_uses_correct_llm_params(
        self, mock_llm, mock_get_session, mock_load_prompt, mock_session
    ):
        articles = [_make_orm_article(title="Test", raw_content="Content")]
        _setup_mock_session(mock_get_session, mock_session, articles)
        mock_load_prompt.return_value = MagicMock(text="rendered prompt")
        mock_llm.return_value = "Summary"

        await run_lifecycle_pipeline()

        mock_llm.assert_awaited_once_with(
            prompt="rendered prompt",
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=500,
            component="data_lifecycle",
        )

    @patch("lifecycle.lifecycle_pipeline.load_prompt")
    @patch("lifecycle.lifecycle_pipeline.get_async_session")
    @patch("lifecycle.lifecycle_pipeline.call_llm")
    async def test_strips_whitespace_from_summary(
        self, mock_llm, mock_get_session, mock_load_prompt, mock_session
    ):
        article = _make_orm_article(title="Test", raw_content="Content")
        _setup_mock_session(mock_get_session, mock_session, [article])
        mock_load_prompt.return_value = MagicMock(text="prompt")
        mock_llm.return_value = "  Summary with spaces  \n"

        await run_lifecycle_pipeline()

        assert article.summary == "Summary with spaces"

    @patch("lifecycle.lifecycle_pipeline.load_prompt")
    @patch("lifecycle.lifecycle_pipeline.get_async_session")
    @patch("lifecycle.lifecycle_pipeline.call_llm")
    async def test_tokens_used_is_estimated(
        self, mock_llm, mock_get_session, mock_load_prompt, mock_session
    ):
        """Verify tokens_used is non-zero after successful summarization."""
        article = _make_orm_article(title="Test", raw_content="Some content here")
        _setup_mock_session(mock_get_session, mock_session, [article])
        mock_load_prompt.return_value = MagicMock(text="rendered prompt for article")
        mock_llm.return_value = "Tóm tắt nội dung"

        result = await run_lifecycle_pipeline()

        assert result.tokens_used > 0
        expected = _estimate_tokens("rendered prompt for article", "Tóm tắt nội dung")
        assert result.tokens_used == expected

    @patch("lifecycle.lifecycle_pipeline.load_prompt")
    @patch("lifecycle.lifecycle_pipeline.get_async_session")
    @patch("lifecycle.lifecycle_pipeline.call_llm")
    async def test_handles_llm_call_error(
        self, mock_llm, mock_get_session, mock_load_prompt, mock_session
    ):
        """Verify LLMCallError is caught gracefully."""
        article = _make_orm_article(title="Test", raw_content="Content")
        _setup_mock_session(mock_get_session, mock_session, [article])
        mock_load_prompt.return_value = MagicMock(text="prompt")
        mock_llm.side_effect = LLMCallError("All retries exhausted")

        result = await run_lifecycle_pipeline()

        assert result.failed_count == 1
        assert result.summarized_count == 0
        assert article.raw_content == "Content"

    @patch("lifecycle.lifecycle_pipeline.load_prompt")
    @patch("lifecycle.lifecycle_pipeline.get_async_session")
    @patch("lifecycle.lifecycle_pipeline.call_llm")
    async def test_batch_commit_failure_triggers_rollback(
        self, mock_llm, mock_get_session, mock_load_prompt, mock_session
    ):
        """Verify session.rollback() is called when commit fails."""
        article = _make_orm_article(title="Test", raw_content="Content")
        _setup_mock_session(mock_get_session, mock_session, [article])
        mock_load_prompt.return_value = MagicMock(text="prompt")
        mock_llm.return_value = "Summary"
        mock_session.commit = AsyncMock(side_effect=Exception("DB commit failed"))

        await run_lifecycle_pipeline()

        mock_session.rollback.assert_awaited_once()


class TestEstimateTokens:

    def test_estimates_from_prompt_and_summary(self):
        prompt = "a" * 300  # 300 chars
        summary = "b" * 150  # 150 chars
        result = _estimate_tokens(prompt, summary)
        assert result == (300 + 150) // 3  # 150

    def test_empty_strings_return_zero(self):
        assert _estimate_tokens("", "") == 0
