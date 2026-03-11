"""Article persistence with batch URL deduplication."""

from __future__ import annotations

from sqlalchemy import select
from shared.db import get_async_session
from shared.db.orm.article import Article
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
