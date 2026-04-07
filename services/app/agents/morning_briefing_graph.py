"""Morning Briefing Intelligence Graph — sequential market-level pipeline.

Usage:
    state = await morning_briefing_graph.ainvoke({
        "analysis_date": "2026-04-01",
        "watchlist": ["HPG", "VNM", ...],  # full watchlist from stock_tickers.yaml
    })
    market_result = state["market_result"]  # -> format_morning_briefing(market_result)
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from services.app.agents.morning_briefing.nodes import (
    morning_market_context_node,
    sector_filter_node,
    technical_batch_node,
    fundamental_batch_node,
    morning_synthesis_node,
)
from services.app.agents.state import MorningBriefingState

graph_builder = StateGraph(MorningBriefingState)
graph_builder.add_node("market_context", morning_market_context_node)
graph_builder.add_node("sector_filter", sector_filter_node)
graph_builder.add_node("technical_batch", technical_batch_node)
graph_builder.add_node("fundamental_batch", fundamental_batch_node)
graph_builder.add_node("synthesis", morning_synthesis_node)

graph_builder.set_entry_point("market_context")
graph_builder.add_edge("market_context", "sector_filter")
graph_builder.add_edge("sector_filter", "technical_batch")
graph_builder.add_edge("technical_batch", "fundamental_batch")
graph_builder.add_edge("fundamental_batch", "synthesis")
graph_builder.add_edge("synthesis", END)

morning_briefing_graph = graph_builder.compile()
