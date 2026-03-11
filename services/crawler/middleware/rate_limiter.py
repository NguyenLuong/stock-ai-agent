"""Per-domain rate limiter wrapping httpx AsyncBaseTransport."""

from __future__ import annotations

import asyncio

import httpx
from pyrate_limiter import Duration, Limiter, Rate
from shared.logging import get_logger

logger = get_logger("crawler.middleware.rate_limiter")


class RateLimitedTransport(httpx.AsyncBaseTransport):
    """Wraps an httpx async transport with per-domain rate limiting."""

    def __init__(
        self,
        rate_per_second: int = 1,
        wrapped_transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._rate_per_second = rate_per_second
        self._transport = wrapped_transport or httpx.AsyncHTTPTransport()
        self._limiters: dict[str, Limiter] = {}

    def _get_limiter(self, domain: str) -> Limiter:
        if domain not in self._limiters:
            rate = Rate(self._rate_per_second, Duration.SECOND)
            self._limiters[domain] = Limiter(rate)
            logger.info(
                "rate_limiter_created",
                component="rate_limiter",
                domain=domain,
                rate_per_second=self._rate_per_second,
            )
        return self._limiters[domain]

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        domain = request.url.host
        limiter = self._get_limiter(domain)
        await asyncio.to_thread(limiter.try_acquire, domain)
        logger.debug(
            "request_permitted",
            component="rate_limiter",
            domain=domain,
            url=str(request.url),
        )
        return await self._transport.handle_async_request(request)
