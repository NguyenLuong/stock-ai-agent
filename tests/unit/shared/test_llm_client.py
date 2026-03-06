"""Tests for shared.llm.client — OpenAI async client + tenacity retry."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import APIConnectionError, RateLimitError

from shared.llm.client import LLMCallError, LLMClient, reset_llm_client


@pytest.fixture(autouse=True)
def _reset_client():
    reset_llm_client()
    yield
    reset_llm_client()


def _make_response(content: str = "Hello", prompt_tokens: int = 10, completion_tokens: int = 20):
    """Create a mock OpenAI chat completion response."""
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens

    message = MagicMock()
    message.content = content

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


class TestLLMClientRetry:
    async def test_retry_succeeds_after_transient_failures(self):
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=[
                RateLimitError(
                    message="rate limit",
                    response=MagicMock(status_code=429),
                    body=None,
                ),
                RateLimitError(
                    message="rate limit",
                    response=MagicMock(status_code=429),
                    body=None,
                ),
                _make_response("success"),
            ]
        )

        client = LLMClient(client=mock_client)
        result = await client.call(prompt="test", component="test")

        assert result == "success"
        assert mock_client.chat.completions.create.call_count == 3

    async def test_all_failures_raises_llm_call_error(self):
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=APIConnectionError(request=MagicMock())
        )

        client = LLMClient(client=mock_client)
        with pytest.raises(LLMCallError) as exc_info:
            await client.call(prompt="test", component="test")

        assert "failed after 3 attempts" in str(exc_info.value)
        assert exc_info.value.last_exception is not None

    async def test_missing_api_key_raises_runtime_error(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        reset_llm_client()

        client = LLMClient()  # No injected client
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            await client.call(prompt="test", component="test")

    async def test_log_emission_on_success(self, capsys):
        import json
        import structlog
        from shared.logging.setup import configure_logging

        structlog.reset_defaults()
        configure_logging(env="production")

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_make_response("ok", prompt_tokens=15, completion_tokens=25)
        )

        client = LLMClient(client=mock_client)
        await client.call(prompt="test", model="gpt-4o-mini", component="test")

        output = capsys.readouterr().out.strip()
        data = json.loads(output)
        assert data["component"] == "test"
        assert data["context"]["model"] == "gpt-4o-mini"
        assert data["context"]["prompt_tokens"] == 15
        assert data["context"]["completion_tokens"] == 25

    async def test_retry_emits_warning_log(self, capsys):
        import json
        import structlog
        from openai import RateLimitError
        from shared.logging.setup import configure_logging

        structlog.reset_defaults()
        configure_logging(env="production")

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=[
                RateLimitError(
                    message="rate limit",
                    response=MagicMock(status_code=429),
                    body=None,
                ),
                _make_response("ok"),
            ]
        )

        client = LLMClient(client=mock_client)
        await client.call(prompt="test", model="gpt-4o-mini", component="test")

        lines = [l for l in capsys.readouterr().out.strip().splitlines() if l]
        # First line should be the retry warning
        retry_log = json.loads(lines[0])
        assert retry_log["level"] == "warning"
        assert retry_log["message"] == "llm_call_retry"
        assert retry_log["context"]["attempt"] == 1
        assert retry_log["component"] == "test"
