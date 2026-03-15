"""Tests for HTTP trigger task — mock HTTP responses."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch, MagicMock

import httpx
import pytest

from services.scheduler.tasks.http_trigger import trigger_pipeline


class TestTriggerPipeline:
    """Tests for the trigger_pipeline Prefect task."""

    @patch("services.scheduler.tasks.http_trigger.httpx.AsyncClient")
    async def test_successful_trigger(self, mock_client_cls):
        """POST to correct URL, return JSON response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok", "count": 5}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = await trigger_pipeline.fn(
            app_url="http://app:8000",
            endpoint="crawl",
        )

        assert result == {"status": "ok", "count": 5}
        mock_client.post.assert_called_once_with(
            "http://app:8000/internal/trigger/crawl",
            headers={"X-Trigger-Source": "prefect-scheduler"},
        )

    @patch("services.scheduler.tasks.http_trigger.httpx.AsyncClient")
    async def test_timeout_raises(self, mock_client_cls):
        """Timeout during HTTP POST should propagate exception."""
        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.ReadTimeout("timed out")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        with pytest.raises(httpx.ReadTimeout):
            await trigger_pipeline.fn(
                app_url="http://app:8000",
                endpoint="crawl",
            )

    @patch("services.scheduler.tasks.http_trigger.httpx.AsyncClient")
    async def test_server_error_raises(self, mock_client_cls):
        """500 error should raise via raise_for_status."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error",
            request=MagicMock(),
            response=mock_response,
        )

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        with pytest.raises(httpx.HTTPStatusError):
            await trigger_pipeline.fn(
                app_url="http://app:8000",
                endpoint="crawl",
            )

    @patch("services.scheduler.tasks.http_trigger.httpx.AsyncClient")
    async def test_custom_timeout(self, mock_client_cls):
        """Custom timeout is passed to AsyncClient."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok"}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        await trigger_pipeline.fn(
            app_url="http://app:8000",
            endpoint="embedding",
            timeout_seconds=60.0,
        )

        mock_client_cls.assert_called_once_with(timeout=60.0)

    @patch("services.scheduler.tasks.http_trigger.httpx.AsyncClient")
    async def test_different_endpoints(self, mock_client_cls):
        """Each endpoint builds the correct URL."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok"}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        for endpoint in ("crawl", "embedding", "lifecycle"):
            mock_client.post.reset_mock()
            await trigger_pipeline.fn(
                app_url="http://app:8000",
                endpoint=endpoint,
            )
            mock_client.post.assert_called_once_with(
                f"http://app:8000/internal/trigger/{endpoint}",
                headers={"X-Trigger-Source": "prefect-scheduler"},
            )
