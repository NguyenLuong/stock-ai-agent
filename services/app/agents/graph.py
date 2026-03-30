"""LangGraph orchestrator graph — fan-out / fan-in pipeline.

Usage:
    result = await orchestrator_graph.ainvoke({
        "ticker": "HPG",
        "analysis_type": "morning_briefing",
        "analysis_date": "2026-03-29",
        "watchlist": ["HPG", "VNM", "FPT"],
    })
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from services.app.agents.orchestrator.formatter import synthesize_node
from services.app.agents.orchestrator.node import dispatch_and_collect
from services.app.agents.state import OrchestratorState

graph_builder = StateGraph(OrchestratorState)
graph_builder.add_node("dispatch", dispatch_and_collect)
graph_builder.add_node("synthesize", synthesize_node)
graph_builder.set_entry_point("dispatch")
graph_builder.add_edge("dispatch", "synthesize")
graph_builder.add_edge("synthesize", END)

orchestrator_graph = graph_builder.compile()
