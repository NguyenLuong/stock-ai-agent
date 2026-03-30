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


class TechnicalAnalysisState(TypedDict, total=False):
    """State for the Technical Analysis Agent LangGraph node.

    Input fields are set by the orchestrator; processing and output fields
    are populated by the agent node itself.
    """

    # --- Input fields (set by orchestrator) ---
    ticker: str
    analysis_type: str  # "morning_briefing" | "alert" | "deep_analysis"
    analysis_date: str  # ISO date string, e.g. "2026-03-28"

    # --- Processing fields (intermediate data) ---
    indicator_values: dict
    ohlcv_data: list[dict]

    # --- Output fields ---
    technical_analysis: dict | None
    # technical_analysis structure:
    # {
    #     "indicator_summary": str | None,     # Phase 1 output — LLM interpretation
    #     "pattern_summary": str | None,       # Phase 2 output — LLM pattern recognition
    #     "signals": {
    #         "trend": str,                    # "uptrend" | "downtrend" | "sideways"
    #         "momentum": str,                 # "bullish" | "bearish" | "neutral"
    #         "volatility": str,               # "high" | "low" | "normal"
    #         "volume_confirmation": bool,
    #     },
    #     "support_levels": list[float],
    #     "resistance_levels": list[float],
    #     "confidence": float,
    #     "data_as_of": str,                   # ISO timestamp
    #     "data_source": str,                  # "calculated" | "mock"
    # }

    failed_agents: list[str]
    error: str


class FundamentalAnalysisState(TypedDict, total=False):
    """State for the Fundamental Analysis Agent LangGraph node.

    Input fields are set by the orchestrator; output fields
    are populated by the agent node itself.
    """

    # --- Input fields (set by orchestrator) ---
    ticker: str
    analysis_type: str  # "morning_briefing" | "alert" | "deep_analysis"
    analysis_date: str  # ISO date string, e.g. "2026-03-29"

    # --- Output fields ---
    fundamental_analysis: dict | None
    # fundamental_analysis structure:
    # {
    #     "bctc_summary": str | None,          # Phase 1 output — LLM BCTC analysis
    #     "ratio_comparison": str | None,       # Phase 2 output — LLM ratio comparison
    #     "company_ratios": {                   # Raw ratio values from DB
    #         "pe": float | None,
    #         "pb": float | None,
    #         "roe": float | None,
    #         "eps": float | None,
    #         "eps_growth_yoy": float | None,
    #     },
    #     "sector_ratios": {                    # Sector averages
    #         "pe": float | None,
    #         "pb": float | None,
    #         "roe": float | None,
    #         "eps": float | None,
    #     },
    #     "sector_name": str,                   # Vietnamese sector name
    #     "signals": {
    #         "valuation": str,                 # "undervalued" | "overvalued" | "fair"
    #         "profitability": str,             # "strong" | "weak" | "average"
    #         "financial_health": str,          # "healthy" | "risky" | "neutral"
    #         "growth": str,                    # "growing" | "declining" | "stable"
    #     },
    #     "confidence": float,                  # 0.0-0.95
    #     "data_as_of": str,                    # ISO timestamp
    #     "data_source": str,                   # "vnstock" | "mock"
    # }

    failed_agents: list[str]
    error: str


class OrchestratorState(TypedDict, total=False):
    """State for the Orchestrator LangGraph graph.

    Superset of input fields needed by all 3 analysis agents plus
    orchestrator-specific tracking and output fields.
    """

    # --- Input fields (set by caller / endpoint) ---
    ticker: str
    analysis_type: str  # "morning_briefing" | "alert" | "deep_analysis"
    analysis_date: str  # ISO date string, e.g. "2026-03-29"
    watchlist: list[str]  # Danh mục theo dõi, used in synthesis prompt

    # --- Agent output fields (populated by dispatch_and_collect) ---
    market_summary: dict | None
    technical_analysis: dict | None
    fundamental_analysis: dict | None

    # --- Tracking fields ---
    failed_agents: list[str]

    # --- Orchestrator output fields ---
    synthesis_result: dict | None
    confidence_score: float | None
    error: str | None
