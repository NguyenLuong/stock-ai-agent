"""Text processing and cleaning utilities."""

from __future__ import annotations

import re

TELEGRAM_CHAR_LIMIT: int = 3800


def chunk_telegram(text: str) -> list[str]:
    """Split text into chunks of ≤3800 chars for Telegram.

    If >1 chunk, prefix each with (1/N), (2/N) etc.
    Prefers splitting at paragraph boundaries (\\n\\n).
    Each chunk including its prefix is guaranteed to be ≤3800 chars.
    """
    if len(text) <= TELEGRAM_CHAR_LIMIT:
        return [text]

    # Reserve space for "(N/N) " prefix (up to 10 chars covers 99+ parts).
    effective_limit = TELEGRAM_CHAR_LIMIT - 10

    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        # Account for \n\n separator between paragraphs in chunk
        separator_len = 2 if current else 0
        para_len = len(para)

        if current_len + separator_len + para_len <= effective_limit:
            current.append(para)
            current_len += separator_len + para_len
        else:
            # Flush current chunk if non-empty
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_len = 0

            # Handle single paragraph > effective_limit: force-split
            if para_len > effective_limit:
                while para:
                    chunks.append(para[:effective_limit])
                    para = para[effective_limit:]
            else:
                current = [para]
                current_len = para_len

    if current:
        chunks.append("\n\n".join(current))

    # Prefix with (1/N) etc.
    total = len(chunks)
    if total > 1:
        chunks = [f"({i + 1}/{total}) {chunk}" for i, chunk in enumerate(chunks)]

    return chunks


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/newlines to single, strip edges."""
    return re.sub(r"\s+", " ", text).strip()
