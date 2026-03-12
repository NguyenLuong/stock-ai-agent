"""Article persistence with batch URL deduplication and semantic search."""

from __future__ import annotations

from sqlalchemy import select

from shared.db import get_async_session
from shared.db.orm.article import Article
from shared.llm.embedder import embed_single
from shared.logging import get_logger
from shared.models.article import ArticleCreate

logger = get_logger("crawler.article_repo")


async def save_articles(articles: list[ArticleCreate]) -> int:
    """Batch insert articles, skipping duplicates by URL.

    Returns count of newly inserted articles.
    """
    if not articles:
        return 0

    async with get_async_session() as session:
        # Batch query existing URLs
        urls = [a.url for a in articles]
        result = await session.execute(
            select(Article.url).where(Article.url.in_(urls))
        )
        existing_urls = set(result.scalars().all())

        # Filter out duplicates
        new_articles = [a for a in articles if a.url not in existing_urls]

        if not new_articles:
            logger.info(
                "no_new_articles",
                component="article_repo",
                total=len(articles),
                duplicates=len(articles),
            )
            return 0

        # Bulk insert
        for article in new_articles:
            session.add(Article(**article.model_dump()))

        await session.commit()

        logger.info(
            "articles_saved",
            component="article_repo",
            inserted=len(new_articles),
            duplicates=len(existing_urls),
        )
        return len(new_articles)


async def semantic_search(
    query: str,
    top_k: int = 10,
    ticker_symbol: str | None = None,
) -> list[Article]:
    """Semantic search over embedded articles using pgvector cosine distance."""
    query_vector = await embed_single(query)

    async with get_async_session() as session:
        stmt = select(Article).where(Article.embedded.is_(True))
        if ticker_symbol:
            stmt = stmt.where(Article.ticker_symbol == ticker_symbol)
        stmt = stmt.order_by(
            Article.embedding.cosine_distance(query_vector)
        ).limit(top_k)

        result = await session.execute(stmt)
        return list(result.scalars().all())
