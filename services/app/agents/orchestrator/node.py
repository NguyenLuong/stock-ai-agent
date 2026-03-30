"""Orchestrator node — dispatch 3 analysis agents in parallel and collect results."""

from __future__ import annotations

import asyncio
import time

from importlib import import_module

from services.app.agents.state import OrchestratorState
from shared.logging import get_logger

logger = get_logger("orchestrator")

# (name, module_path, function_name) — resolved at call time so patches work
_AGENT_SPECS: list[tuple[str, str, str]] = [
    ("market_context", "services.app.agents.market_context.node", "market_context_node"),
    ("technical_analysis", "services.app.agents.technical_analysis.node", "technical_analysis_node"),
    ("fundamental_analysis", "services.app.agents.fundamental_analysis.node", "fundamental_analysis_node"),
]


async def dispatch_and_collect(state: OrchestratorState) -> dict:
    """Fan-out: call 3 agent nodes concurrently, fan-in: merge results.

    Returns a dict containing only the keys that changed (LangGraph pattern).
    """
    start = time.monotonic()

    logger.info(
        "dispatch_start",
        component="orchestrator",
        ticker=state.get("ticker"),
        analysis_type=state.get("analysis_type"),
        agent_count=len(_AGENT_SPECS),
    )

    # Resolve functions at call time (enables mocking)
    agent_funcs = [
        (name, getattr(import_module(mod_path), fn_name))
        for name, mod_path, fn_name in _AGENT_SPECS
    ]

    results = await asyncio.gather(
        *(fn(state) for _, fn in agent_funcs),
        return_exceptions=True,
    )

    failed_agents: list[str] = list(state.get("failed_agents", []))
    merged: dict = {}

    for (name, _), result in zip(agent_funcs, results):
        if isinstance(result, Exception):
            logger.error(
                "agent_exception",
                component="orchestrator",
                agent=name,
                error=str(result),
            )
            failed_agents.append(name)
        else:
            merged.update(result)
            logger.info(
                "agent_completed",
                component="orchestrator",
                agent=name,
            )

    duration = round(time.monotonic() - start, 2)
    logger.info(
        "dispatch_completed",
        component="orchestrator",
        duration_seconds=duration,
        failed_count=len(failed_agents),
        success_count=len(_AGENT_SPECS) - len(failed_agents),
    )

    merged["failed_agents"] = failed_agents
    return merged
