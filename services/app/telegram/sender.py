"""Telegram message sender with retry and queue support."""

from __future__ import annotations

import asyncio

from tenacity import RetryError, retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ._compat import Bot, TelegramError

from shared.logging import get_logger
from shared.utils.text_utils import chunk_telegram

logger = get_logger("telegram_bot")


class TelegramSender:
    """Send messages to a Telegram chat with chunking, retry, and queue."""

    def __init__(self, bot_token: str, chat_id: str) -> None:
        self._bot = Bot(token=bot_token)
        self._chat_id = chat_id
        self._queue: list[str] = []

    @retry(
        wait=wait_exponential(min=1, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(TelegramError),
    )
    async def _send_single(self, text: str, parse_mode: str | None = None) -> int:
        """Send one message, return message_id."""
        msg = await self._bot.send_message(chat_id=self._chat_id, text=text, parse_mode=parse_mode)
        return msg.message_id

    async def send_message(self, text: str, parse_mode: str | None = None) -> list[int]:
        """Chunk text and send each part. Queue on total failure."""
        chunks = chunk_telegram(text)
        total = len(chunks)
        message_ids: list[int] = []

        for i, chunk in enumerate(chunks):
            try:
                mid = await self._send_single(chunk, parse_mode=parse_mode)
                message_ids.append(mid)
                logger.info(
                    "telegram_message_sent",
                    component="telegram_bot",
                    chat_id=self._chat_id,
                    chunk_index=i,
                    total_chunks=total,
                )
            except (TelegramError, RetryError) as e:
                self._queue.append(chunk)
                logger.error(
                    "telegram_send_failed_queued",
                    component="telegram_bot",
                    error=str(e),
                    queue_size=len(self._queue),
                )

            if i < total - 1:
                await asyncio.sleep(1)

        return message_ids

    async def flush_queue(self) -> int:
        """Retry sending queued messages. Return count of successfully sent."""
        if not self._queue:
            return 0

        sent = 0
        remaining: list[str] = []

        for msg in self._queue:
            try:
                await self._send_single(msg)
                sent += 1
            except (TelegramError, RetryError) as e:
                remaining.append(msg)
                logger.error(
                    "telegram_flush_queue_failed",
                    component="telegram_bot",
                    error=str(e),
                    remaining_in_queue=len(remaining),
                )

        self._queue = remaining
        return sent
