"""Lifecycle pipeline for summarizing old articles and freeing raw_content storage."""

from __future__ import annotations

from datetime import timedelta

from sqlalchemy import select, and_

from shared.db import get_async_session
from shared.db.orm.article import Article
from shared.llm.client import LLMCallError, call_llm
from shared.llm.prompt_loader import load_prompt
from shared.logging import get_logger
from shared.utils.datetime_utils import now_utc

from services.crawler.lifecycle.models import LifecyclePipelineResult

logger = get_logger("data_lifecycle")

MAX_ARTICLE_AGE_DAYS = 30
SUMMARIZATION_BATCH_SIZE = 50
MAX_ARTICLES_PER_RUN = 1000

# Rough token estimation: Vietnamese text averages ~1 token per 3 chars
_CHARS_PER_TOKEN_ESTIMATE = 3


def _estimate_tokens(prompt: str, summary: str) -> int:
    """Estimate token usage from prompt + summary character lengths."""
    return (len(prompt) + len(summary)) // _CHARS_PER_TOKEN_ESTIMATE


async def run_lifecycle_pipeline(
    max_age_days: int = MAX_ARTICLE_AGE_DAYS,
    batch_size: int = SUMMARIZATION_BATCH_SIZE,
) -> LifecyclePipelineResult:
    """Summarize old articles and free raw_content storage.

    Queries articles older than max_age_days with raw_content still present,
    generates a Vietnamese summary via LLM, then clears raw_content.
    Embedding vectors are retained for continued semantic search.
    """
    now = now_utc()
    start = now
    cutoff_date = now - timedelta(days=max_age_days)

    async with get_async_session() as session:
        # 1. Lấy các bài viết cũ hơn cutoff_date mà vẫn còn raw_content
        #    Giới hạn MAX_ARTICLES_PER_RUN để tránh OOM khi có quá nhiều bài tồn đọng
        result = await session.execute(
            select(Article)
            .where(
                and_(
                    Article.published_at < cutoff_date,
                    Article.raw_content.isnot(None),
                )
            )
            .limit(MAX_ARTICLES_PER_RUN)
        )
        articles = list(result.scalars().all())

        if not articles:
            logger.info(
                "no_articles_for_lifecycle",
                component="data_lifecycle",
                cutoff_date=cutoff_date.isoformat(),
            )
            duration = (now_utc() - start).total_seconds()
            return LifecyclePipelineResult(
                total=0,
                summarized_count=0,
                skipped_count=0,
                failed_count=0,
                tokens_used=0,
                duration_seconds=round(duration, 2),
            )

        summarized = 0
        failed = 0
        skipped = 0
        total_tokens = 0

        # 2. Xử lý theo batch — commit sau mỗi batch (không phải mỗi article)
        #    để cân bằng giữa hiệu năng và an toàn dữ liệu
        for i in range(0, len(articles), batch_size):
            batch = articles[i : i + batch_size]
            for article in batch:
                try:
                    # Bỏ qua bài đã có summary (phòng trường hợp chạy lại)
                    if article.summary:
                        skipped += 1
                        continue

                    # Tạo summary bằng LLM (gpt-4o-mini, temp=0.3 cho tóm tắt chính xác)
                    prompt = _build_summary_prompt(article)
                    summary = await call_llm(
                        prompt=prompt,
                        model="gpt-4o-mini",
                        temperature=0.3,
                        max_tokens=500,
                        component="data_lifecycle",
                    )

                    # Lưu summary, xóa raw_content để giải phóng dung lượng
                    # QUAN TRỌNG: Giữ nguyên embedding + embedded — KHÔNG chạm vào
                    article.summary = summary.strip()
                    article.raw_content = None
                    summarized += 1
                    total_tokens += _estimate_tokens(prompt, summary)

                except (LLMCallError, Exception) as e:
                    # Ghi log lỗi và tiếp tục — không crash pipeline vì 1 bài lỗi
                    logger.error(
                        "summarization_failed",
                        component="data_lifecycle",
                        error=str(e),
                        article_id=str(article.id),
                    )
                    failed += 1

            # Commit sau mỗi batch, rollback nếu commit thất bại
            try:
                await session.commit()
            except Exception as e:
                await session.rollback()
                logger.error(
                    "batch_commit_failed",
                    component="data_lifecycle",
                    error=str(e),
                    batch_offset=i,
                )

    duration = (now_utc() - start).total_seconds()

    # 3. Ghi log tổng kết pipeline (AC2: structured log với count + tokens)
    logger.info(
        "lifecycle_pipeline_complete",
        component="data_lifecycle",
        total=len(articles),
        summarized=summarized,
        skipped=skipped,
        failed=failed,
        tokens_used=total_tokens,
        duration_seconds=round(duration, 2),
    )

    return LifecyclePipelineResult(
        total=len(articles),
        summarized_count=summarized,
        skipped_count=skipped,
        failed_count=failed,
        tokens_used=total_tokens,
        duration_seconds=round(duration, 2),
    )


def _build_summary_prompt(article: Article) -> str:
    """Tạo prompt tóm tắt cho bài viết bằng Jinja2 template.

    Template nằm tại config/prompts/lifecycle/summarize_article.yaml,
    yêu cầu LLM tóm tắt tiếng Việt 200-400 ký tự, giữ số liệu + mã cổ phiếu.
    """
    rendered = load_prompt(
        "lifecycle/summarize_article",
        title=article.title,
        raw_content=article.raw_content,
    )
    return rendered.text
