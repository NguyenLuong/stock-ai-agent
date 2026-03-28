"""LangGraph agent state definitions shared across all analysis agents."""

from __future__ import annotations

from typing import TypedDict


class MarketContextState(TypedDict, total=False):
    """State for the Market Context Agent LangGraph node.

    Input fields are set by the orchestrator; processing and output fields
    are populated by the agent node itself.
    """

    # --- Input fields (set by orchestrator) ---
    ticker: str
    analysis_type: str  # "morning_briefing" | "alert" | "deep_analysis"
    analysis_date: str  # ISO date string, e.g. "2026-03-25"

    # --- Processing fields (intermediate data) ---
    macro_articles: list[dict]
    stock_articles: list[dict]

    # --- Output fields ---
    market_summary: dict | None
    # market_summary structure:
    # {
    #     "macro_summary": str | None,
    #     "stock_summary": str | None,
    #     "affected_sectors": list[str],
    #     "confidence": float,
    #     "data_as_of": str,       # ISO timestamp
    #     "sources": list[str],    # Unique sources
    # }

    failed_agents: list[str]
    error: str
