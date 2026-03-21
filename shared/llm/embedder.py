"""Text embedding client using OpenAI embeddings API."""

from __future__ import annotations

from openai import (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from shared.llm.client import _get_client
from shared.logging import get_logger

logger = get_logger("embedder")

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
# text-embedding-3-small limit: 8192 tokens.
# Vietnamese text averages ~1.5-2.5 chars/token (mixed Vietnamese + numbers/URLs).
# Conservative limit: ~8000 tokens * 2 chars/token = 16000 chars.
MAX_INPUT_CHARS = 16000


def _truncate(text: str, max_chars: int = MAX_INPUT_CHARS) -> str:
    """Truncate text to stay within embedding model token limit."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


@retry(
    wait=wait_exponential(min=1, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(
        (RateLimitError, APITimeoutError, APIConnectionError)
    ),
    reraise=True,
)
async def _embed_batch(texts: list[str]) -> list[list[float]]:
    client = _get_client()
    truncated = [_truncate(t) for t in texts]
    response = await client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=truncated,
    )
    return [item.embedding for item in response.data]


async def embed_texts(
    texts: list[str], batch_size: int = 100
) -> list[list[float]]:
    """Embed texts in batches to avoid rate limits.

    Returns a list of float vectors, each of dimension 1536.
    """
    all_embeddings: list[list[float]] = []
    total = len(texts)
    batch_num = 0

    for i in range(0, total, batch_size):
        batch_num += 1
        batch = texts[i : i + batch_size]
        logger.info(
            "embedding_batch",
            batch_num=batch_num,
            batch_size=len(batch),
            total_texts=total,
        )
        embeddings = await _embed_batch(batch)
        all_embeddings.extend(embeddings)

    return all_embeddings


async def embed_single(text: str) -> list[float]:
    """Embed a single text string. Convenience wrapper around embed_texts."""
    results = await embed_texts([text])
    return results[0]
