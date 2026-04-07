from contextlib import asynccontextmanager

from fastapi import FastAPI

from services.app.api.router import router
from shared.logging import get_logger

logger = get_logger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup — Telegram bot
    try:
        from services.app.telegram.bot import TelegramBot

        telegram_bot = TelegramBot()
        await telegram_bot.start_polling()
        app.state.telegram_bot = telegram_bot
        logger.info("telegram_bot_started", component="app")
    except ValueError as e:
        logger.warning("telegram_bot_skipped", component="app", reason=str(e))
        app.state.telegram_bot = None

    yield

    # Shutdown
    if getattr(app.state, "telegram_bot", None):
        await app.state.telegram_bot.stop()
        logger.info("telegram_bot_stopped", component="app")


app = FastAPI(title="Stock AI Agent", lifespan=lifespan)
app.include_router(router)
