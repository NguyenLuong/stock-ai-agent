"""Robots.txt compliance checker with per-domain caching."""

from __future__ import annotations

import time
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx
from shared.logging import get_logger

logger = get_logger("crawler.middleware.robots_checker")

_DEFAULT_CACHE_TTL = 86400  # 24 hours


class RobotsChecker:
    """Check robots.txt before every request, caching per domain."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        user_agent: str = "StockAIAgent/1.0",
        cache_ttl_seconds: int = _DEFAULT_CACHE_TTL,
    ) -> None:
        self._client = client
        self._user_agent = user_agent
        self._cache_ttl_seconds = cache_ttl_seconds
        # domain -> (RobotFileParser | None, fetched_at_monotonic)
        self._cache: dict[str, tuple[RobotFileParser | None, float]] = {}

    async def can_fetch(self, url: str) -> bool:
        """Return True if the URL is allowed by robots.txt for our user agent."""
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.hostname or ""
        origin = f"{parsed.scheme}://{domain}"

        parser = await self._get_parser(domain, origin)
        if parser is None:
            return True  # graceful degradation

        allowed = parser.can_fetch(self._user_agent, url)
        if not allowed:
            logger.info(
                "url_disallowed_by_robots",
                component="robots_checker",
                domain=domain,
                url=url,
            )
        return allowed

    async def _get_parser(self, domain: str, origin: str) -> RobotFileParser | None:
        """Get cached parser or fetch and parse robots.txt."""
        if domain in self._cache:
            parser, fetched_at = self._cache[domain]
            if (time.monotonic() - fetched_at) < self._cache_ttl_seconds:
                return parser

        return await self._fetch_and_cache(domain, origin)

    async def _fetch_and_cache(
        self, domain: str, origin: str
    ) -> RobotFileParser | None:
        robots_url = f"{origin}/robots.txt"
        try:
            response = await self._client.get(robots_url)
            response.raise_for_status()
            parser = RobotFileParser()
            parser.parse(response.text.splitlines())
            self._cache[domain] = (parser, time.monotonic())
            logger.debug(
                "robots_txt_fetched",
                component="robots_checker",
                domain=domain,
            )
            return parser
        except (httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException) as exc:
            logger.warning(
                "robots_txt_fetch_failed",
                component="robots_checker",
                domain=domain,
                error=str(exc),
            )
            # Cache a permissive parser so we don't retry every request
            self._cache[domain] = (None, time.monotonic())
            return None
