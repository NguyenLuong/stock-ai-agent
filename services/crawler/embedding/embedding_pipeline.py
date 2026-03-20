"""Embedding pipeline for processing unembedded articles."""

from __future__ import annotations

from sqlalchemy import select

from shared.db import get_async_session
from shared.db.orm.article import Article
from shared.llm.embedder import embed_texts
from shared.logging import get_logger
from shared.utils.datetime_utils import now_utc

from embedding.models import EmbeddingPipelineResult

logger = get_logger("embedding_pipeline")


async def run_embedding_pipeline(batch_size: int = 100) -> EmbeddingPipelineResult:
    """Embed all unprocessed articles.

    Queries articles where embedded=FALSE, prepares text from title + content,
    embeds via shared embedder in batches, and updates articles with vectors.
    """
    start = now_utc()

    async with get_async_session() as session:
        result = await session.execute(
            select(Article).where(Article.embedded.is_(False))
        )
        articles = list(result.scalars().all())

        if not articles:
            logger.info(
                "no_unembedded_articles",
                component="embedding_pipeline",
            )
            duration = (now_utc() - start).total_seconds()
            return EmbeddingPipelineResult(
                total=0,
                embedded_count=0,
                failed_count=0,
                skipped_count=0,
                duration_seconds=round(duration, 2),
            )

        texts: list[str] = []
        valid_articles: list[Article] = []
        skipped = 0

        for article in articles:
            text = _prepare_text(article)
            if text:
                texts.append(text)
                valid_articles.append(article)
            else:
                skipped += 1
                logger.warning(
                    "article_no_content",
                    component="embedding_pipeline",
                    article_id=str(article.id),
                    title=article.title,
                )

        embedded_count = 0
        failed_count = 0

        if valid_articles:
            try:
                embeddings = await embed_texts(texts, batch_size=batch_size)

                for article, embedding in zip(valid_articles, embeddings):
                    article.embedding = embedding
                    article.embedded = True
                    embedded_count += 1

                await session.commit()
            except Exception:
                await session.rollback()
                failed_count = len(valid_articles)
                embedded_count = 0
                logger.exception(
                    "embedding_pipeline_error",
                    component="embedding_pipeline",
                    failed_count=failed_count,
                )

    duration = (now_utc() - start).total_seconds()

    logger.info(
        "embedding_pipeline_complete",
        component="embedding_pipeline",
        total=len(articles),
        embedded=embedded_count,
        skipped=skipped,
        failed=failed_count,
        duration_seconds=round(duration, 2),
    )

    return EmbeddingPipelineResult(
        total=len(articles),
        embedded_count=embedded_count,
        failed_count=failed_count,
        skipped_count=skipped,
        duration_seconds=round(duration, 2),
    )


def _prepare_text(article: Article) -> str | None:
    """Prepare article text for embedding. Returns None if no content."""
    content = article.raw_content or article.summary
    if not content:
        return None
    return f"{article.title}\n\n{content}"
