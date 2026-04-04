"""Telegram Bot with polling mode and sender access."""

from __future__ import annotations

import os

from shared.logging import get_logger

from ._compat import Application
from .sender import TelegramSender

logger = get_logger("telegram_bot")


class TelegramBot:
    """Manage Telegram Application lifecycle and expose a sender."""

    def __init__(self) -> None:
        bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        if not bot_token or not chat_id:
            raise ValueError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set")
        self._application = Application.builder().token(bot_token).build()
        self.sender = TelegramSender(bot_token=bot_token, chat_id=chat_id)

    async def start_polling(self) -> None:
        """Initialize and start polling (non-blocking)."""
        await self._application.initialize()
        await self._application.start()
        await self._application.updater.start_polling()

    async def stop(self) -> None:
        """Graceful shutdown."""
        await self._application.updater.stop()
        await self._application.stop()
        await self._application.shutdown()
