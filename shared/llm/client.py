"""OpenAI async client with tenacity retry wrapper."""

from __future__ import annotations

import os

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from shared.logging import get_logger

logger = get_logger("llm")


def _log_retry_attempt(retry_state: RetryCallState) -> None:
    """Log each retry attempt with component, model, attempt, error."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(
        "llm_call_retry",
        attempt=retry_state.attempt_number,
        error=str(exc),
        component=retry_state.kwargs.get("component", "unknown"),
        model=retry_state.kwargs.get("model", "unknown"),
    )


class LLMCallError(Exception):
    """Raised when all LLM retry attempts fail."""

    def __init__(self, message: str, last_exception: Exception | None = None) -> None:
        super().__init__(message)
        self.last_exception = last_exception


_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable is not set. "
                "Please set it before using the LLM client."
            )
        _client = AsyncOpenAI(api_key=api_key)
    return _client


def reset_llm_client() -> None:
    """Reset the singleton client — useful for testing."""
    global _client
    _client = None


class LLMClient:
    """Async OpenAI client with automatic retry on transient failures."""

    def __init__(self, client: AsyncOpenAI | None = None) -> None:
        self._client = client

    def _get_client(self) -> AsyncOpenAI:
        if self._client is not None:
            return self._client
        return _get_client()

    @retry(
        wait=wait_exponential(min=1, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(
            (RateLimitError, APITimeoutError, APIConnectionError)
        ),
        reraise=True,
        before_sleep=_log_retry_attempt,
    )
    async def _call_openai(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        component: str,
    ) -> str:
        client = self._get_client()
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content or ""
        usage = response.usage
        logger.info(
            "llm_call_success",
            component=component,
            model=model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )
        return content

    async def call(
        self,
        prompt: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        component: str = "default",
    ) -> str:
        """Call OpenAI with retry. Raises LLMCallError on total failure."""
        try:
            return await self._call_openai(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                component=component,
            )
        except (RateLimitError, APITimeoutError, APIConnectionError) as exc:
            raise LLMCallError(
                f"LLM call failed after 3 attempts: {exc}", last_exception=exc
            ) from exc


async def call_llm(
    prompt: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: int = 2000,
    component: str = "default",
) -> str:
    """Module-level convenience function for LLM calls."""
    client = LLMClient()
    return await client.call(
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        component=component,
    )
