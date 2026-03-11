"""Tests for per-domain rate limiter middleware."""

import asyncio
import time

import httpx
import pytest

from services.crawler.middleware.rate_limiter import RateLimitedTransport


class TestRateLimitedTransport:
    """Test per-domain rate limiting with httpx transport wrapper."""

    @pytest.fixture
    def recorded_transport(self):
        """Transport that records request timestamps per domain."""
        timestamps: dict[str, list[float]] = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            domain = request.url.host
            timestamps.setdefault(domain, []).append(time.monotonic())
            return httpx.Response(200, text="ok")

        return httpx.MockTransport(handler), timestamps

    async def test_single_request_passes_through(self, recorded_transport):
        mock_transport, timestamps = recorded_transport
        transport = RateLimitedTransport(
            rate_per_second=1, wrapped_transport=mock_transport
        )
        async with httpx.AsyncClient(transport=transport) as client:
            resp = await client.get("https://example.com/page1")
        assert resp.status_code == 200
        assert len(timestamps["example.com"]) == 1

    async def test_rate_limits_same_domain(self, recorded_transport):
        """Multiple requests to same domain should be throttled ~1 req/s."""
        mock_transport, timestamps = recorded_transport
        transport = RateLimitedTransport(
            rate_per_second=1, wrapped_transport=mock_transport
        )
        async with httpx.AsyncClient(transport=transport) as client:
            for _ in range(3):
                await client.get("https://example.com/page")

        times = timestamps["example.com"]
        assert len(times) == 3
        # Between consecutive requests, at least ~0.8s should elapse
        for i in range(1, len(times)):
            elapsed = times[i] - times[i - 1]
            assert elapsed >= 0.8, f"Request {i} was too fast: {elapsed:.2f}s"

    async def test_different_domains_not_throttled(self, recorded_transport):
        """Requests to different domains should NOT be rate-limited against each other."""
        mock_transport, timestamps = recorded_transport
        transport = RateLimitedTransport(
            rate_per_second=1, wrapped_transport=mock_transport
        )
        start = time.monotonic()
        async with httpx.AsyncClient(transport=transport) as client:
            await client.get("https://domain-a.com/page")
            await client.get("https://domain-b.com/page")
            await client.get("https://domain-c.com/page")
        elapsed = time.monotonic() - start
        # 3 different domains, 1 request each → should complete quickly
        assert elapsed < 1.5, f"Different domains were throttled: {elapsed:.2f}s"
        assert len(timestamps["domain-a.com"]) == 1
        assert len(timestamps["domain-b.com"]) == 1
        assert len(timestamps["domain-c.com"]) == 1

    async def test_concurrent_requests_same_domain_throttled(self, recorded_transport):
        """Concurrent requests to same domain should still be serialized."""
        mock_transport, timestamps = recorded_transport
        transport = RateLimitedTransport(
            rate_per_second=1, wrapped_transport=mock_transport
        )
        async with httpx.AsyncClient(transport=transport) as client:
            tasks = [client.get("https://example.com/page") for _ in range(3)]
            await asyncio.gather(*tasks)

        times = timestamps["example.com"]
        assert len(times) == 3
        total_elapsed = times[-1] - times[0]
        # 3 requests at 1/s → at least ~1.6s total span
        assert total_elapsed >= 1.6, f"Concurrent requests not throttled: {total_elapsed:.2f}s"

    async def test_custom_rate(self, recorded_transport):
        """Custom rate_per_second=2 should allow 2 req/s."""
        mock_transport, timestamps = recorded_transport
        transport = RateLimitedTransport(
            rate_per_second=2, wrapped_transport=mock_transport
        )
        async with httpx.AsyncClient(transport=transport) as client:
            for _ in range(2):
                await client.get("https://example.com/page")

        times = timestamps["example.com"]
        assert len(times) == 2
        # 2 requests at 2/s → should complete within ~0.8s
        elapsed = times[-1] - times[0]
        assert elapsed < 0.8, f"Rate 2/s too slow: {elapsed:.2f}s"
