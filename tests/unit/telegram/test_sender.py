"""Tests for TelegramSender — chunking, retry, and queue behaviour."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from services.app.telegram._compat import TelegramError
from services.app.telegram.sender import TelegramSender


@pytest.fixture
def mock_bot():
    bot = AsyncMock()
    bot.send_message.return_value = MagicMock(message_id=12345)
    return bot


@pytest.fixture
def sender(mock_bot):
    s = TelegramSender(bot_token="fake-token", chat_id="123")
    s._bot = mock_bot
    return s


class TestSendMessage:
    @pytest.mark.asyncio
    async def test_single_message_under_limit(self, sender, mock_bot):
        """Message <=3800 chars → 1 API call, return 1 message_id."""
        text = "Hello" * 100  # 500 chars
        ids = await sender.send_message(text)

        assert len(ids) == 1
        assert ids[0] == 12345
        mock_bot.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_long_message_chunked(self, sender, mock_bot):
        """Message >3800 chars → multiple chunks, N API calls."""
        text = ("A" * 1900 + "\n\n") * 3  # ~5700 chars with separators
        # Use itertools.cycle to avoid StopIteration regardless of chunk count
        import itertools
        mock_bot.send_message.side_effect = itertools.cycle([
            MagicMock(message_id=1),
            MagicMock(message_id=2),
            MagicMock(message_id=3),
        ])

        ids = await sender.send_message(text)

        assert len(ids) >= 2
        assert mock_bot.send_message.call_count >= 2

    @pytest.mark.asyncio
    async def test_retry_on_telegram_error_then_success(self, sender, mock_bot):
        """TelegramError on first calls → retry → success on 3rd."""
        mock_bot.send_message.side_effect = [
            TelegramError("fail1"),
            TelegramError("fail2"),
            MagicMock(message_id=999),
        ]

        ids = await sender.send_message("short msg")

        assert 999 in ids
        assert mock_bot.send_message.call_count == 3

    @pytest.mark.asyncio
    async def test_all_retries_fail_queued(self, sender, mock_bot):
        """TelegramError on all 3 attempts → message queued, no exception."""
        mock_bot.send_message.side_effect = TelegramError("always fail")

        ids = await sender.send_message("will fail")

        assert ids == []
        assert len(sender._queue) == 1

    @pytest.mark.asyncio
    async def test_flush_queue_success(self, sender, mock_bot):
        """flush_queue sends queued message → removed from queue."""
        sender._queue = ["queued msg"]
        mock_bot.send_message.return_value = MagicMock(message_id=55)

        sent = await sender.flush_queue()

        assert sent == 1
        assert len(sender._queue) == 0

    @pytest.mark.asyncio
    async def test_flush_queue_empty(self, sender):
        """flush_queue with empty queue → return 0."""
        sent = await sender.flush_queue()

        assert sent == 0

    @pytest.mark.asyncio
    async def test_flush_queue_failure_stays_in_queue(self, sender, mock_bot):
        """flush_queue failure → message stays in queue."""
        sender._queue = ["will fail"]
        mock_bot.send_message.side_effect = TelegramError("still failing")

        sent = await sender.flush_queue()

        assert sent == 0
        assert len(sender._queue) == 1

    @pytest.mark.asyncio
    async def test_send_message_passes_parse_mode(self, sender, mock_bot):
        """parse_mode is forwarded to bot.send_message."""
        ids = await sender.send_message("hello", parse_mode="Markdown")

        _, kwargs = mock_bot.send_message.call_args
        assert kwargs.get("parse_mode") == "Markdown"
        assert len(ids) == 1
