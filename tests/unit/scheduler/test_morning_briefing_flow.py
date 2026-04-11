"""Tests for the Prefect morning_briefing_flow."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
@patch("flows.morning_briefing.trigger_pipeline", new_callable=AsyncMock)
async def test_flow_calls_trigger_pipeline_with_correct_endpoint(mock_trigger):
    """morning_briefing_flow calls trigger_pipeline with endpoint='morning-briefing'."""
    mock_trigger.return_value = {"status": "ok", "duration_seconds": 5.0}

    from flows.morning_briefing import morning_briefing_flow

    result = await morning_briefing_flow.fn()

    mock_trigger.assert_called_once()
    call_kwargs = mock_trigger.call_args
    assert call_kwargs.kwargs.get("endpoint") == "morning-briefing" or (
        len(call_kwargs.args) >= 2 and call_kwargs.args[1] == "morning-briefing"
    )


@pytest.mark.asyncio
@patch("flows.morning_briefing.trigger_pipeline", new_callable=AsyncMock)
async def test_flow_returns_trigger_result(mock_trigger):
    """Flow returns the result dict from trigger_pipeline."""
    expected = {"status": "ok", "duration_seconds": 12.3, "top_picks_count": 3}
    mock_trigger.return_value = expected

    from flows.morning_briefing import morning_briefing_flow

    result = await morning_briefing_flow.fn()

    assert result == expected


@pytest.mark.asyncio
@patch("flows.morning_briefing.trigger_pipeline", new_callable=AsyncMock)
async def test_flow_propagates_exception(mock_trigger):
    """Flow propagates exceptions from trigger_pipeline (Prefect handles retries)."""
    mock_trigger.side_effect = Exception("Connection refused")

    from flows.morning_briefing import morning_briefing_flow

    with pytest.raises(Exception, match="Connection refused"):
        await morning_briefing_flow.fn()


@pytest.mark.asyncio
@patch("flows.morning_briefing.trigger_pipeline", new_callable=AsyncMock)
async def test_flow_does_not_suppress_exceptions_allowing_prefect_retry(mock_trigger):
    """Flow must not catch trigger_pipeline exceptions.

    Prefect's retry logic (retries=2 on trigger_pipeline task) depends on the
    exception propagating unmodified.  This test verifies: after a transient
    failure the flow would succeed on a subsequent Prefect-managed invocation.
    """
    success_result = {"status": "ok", "duration_seconds": 8.1}
    # First call fails (simulates what Prefect would retry); second succeeds.
    mock_trigger.side_effect = [Exception("Transient error"), success_result]

    from flows.morning_briefing import morning_briefing_flow

    # First invocation (Prefect would catch and retry at the task level)
    with pytest.raises(Exception, match="Transient error"):
        await morning_briefing_flow.fn()

    # Second invocation (Prefect retry) succeeds and returns the result
    mock_trigger.side_effect = None
    mock_trigger.return_value = success_result
    result = await morning_briefing_flow.fn()
    assert result == success_result
