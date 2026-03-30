"""Synthesis node and agent output formatters for the orchestrator."""

from __future__ import annotations

from datetime import datetime, timezone

from services.app.agents.orchestrator.confidence import (
    calculate_confidence,
    confidence_display,
)
from services.app.agents.state import OrchestratorState
from shared.llm.client import LLMCallError, LLMClient
from shared.llm.config_loader import get_config_loader
from shared.llm.prompt_loader import load_prompt
from shared.logging import get_logger

logger = get_logger("orchestrator")

_DISCLAIMER = "Đây là tham khảo từ AI, không phải khuyến nghị mua/bán chính thức"

# Module-level singleton — avoids re-instantiation per call
_llm_client: LLMClient | None = None


def _get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


# ---------------------------------------------------------------------------
# Agent output → prompt string formatters
# ---------------------------------------------------------------------------


def format_agent_output_for_prompt(output_dict: dict | None, agent_type: str) -> str:
    """Convert an agent output dict to a formatted string for the synthesis prompt.

    Each agent type has its own format matching the fields it produces.
    Returns "⚠️ ... không khả dụng" when output is None.
    """
    if output_dict is None:
        return f"⚠️ {agent_type} không khả dụng"

    formatters = {
        "market_context": _format_market_context,
        "technical_analysis": _format_technical,
        "fundamental_analysis": _format_fundamental,
    }

    formatter = formatters.get(agent_type)
    if formatter is None:
        return str(output_dict)

    return formatter(output_dict)


def _format_market_context(d: dict) -> str:
    lines: list[str] = []
    if d.get("macro_summary"):
        lines.append(f"**Vĩ mô:** {d['macro_summary']}")
    if d.get("stock_summary"):
        lines.append(f"**Cổ phiếu:** {d['stock_summary']}")
    sectors = d.get("affected_sectors", [])
    if sectors:
        lines.append(f"**Ngành ảnh hưởng:** {', '.join(sectors)}")
    conf = d.get("confidence")
    if conf is not None:
        lines.append(f"**Confidence:** {conf:.2f}")
    dao = d.get("data_as_of")
    if dao:
        lines.append(f"**Data as of:** {dao}")
    sources = d.get("sources", [])
    if sources:
        lines.append(f"**Nguồn:** {', '.join(sources)}")
    return "\n".join(lines) if lines else str(d)


def _format_technical(d: dict) -> str:
    lines: list[str] = []
    if d.get("indicator_summary"):
        lines.append(f"**Chỉ báo:** {d['indicator_summary']}")
    if d.get("pattern_summary"):
        lines.append(f"**Mô hình giá:** {d['pattern_summary']}")
    signals = d.get("signals", {})
    if signals:
        sig_parts = [
            f"trend={signals.get('trend', 'N/A')}",
            f"momentum={signals.get('momentum', 'N/A')}",
            f"volatility={signals.get('volatility', 'N/A')}",
            f"vol_confirm={signals.get('volume_confirmation', 'N/A')}",
        ]
        lines.append(f"**Signals:** {', '.join(sig_parts)}")
    support = d.get("support_levels", [])
    resistance = d.get("resistance_levels", [])
    if support:
        lines.append(f"**Hỗ trợ:** {', '.join(str(s) for s in support)}")
    if resistance:
        lines.append(f"**Kháng cự:** {', '.join(str(r) for r in resistance)}")
    conf = d.get("confidence")
    if conf is not None:
        lines.append(f"**Confidence:** {conf:.2f}")
    dao = d.get("data_as_of")
    if dao:
        lines.append(f"**Data as of:** {dao}")
    return "\n".join(lines) if lines else str(d)


def _format_fundamental(d: dict) -> str:
    lines: list[str] = []
    if d.get("bctc_summary"):
        lines.append(f"**BCTC:** {d['bctc_summary']}")
    if d.get("ratio_comparison"):
        lines.append(f"**So sánh chỉ số:** {d['ratio_comparison']}")
    ratios = d.get("company_ratios", {})
    if ratios:
        ratio_parts = [f"{k}={v}" for k, v in ratios.items() if v is not None]
        if ratio_parts:
            lines.append(f"**Chỉ số DN:** {', '.join(ratio_parts)}")
    signals = d.get("signals", {})
    if signals:
        sig_parts = [
            f"valuation={signals.get('valuation', 'N/A')}",
            f"profitability={signals.get('profitability', 'N/A')}",
            f"health={signals.get('financial_health', 'N/A')}",
            f"growth={signals.get('growth', 'N/A')}",
        ]
        lines.append(f"**Signals:** {', '.join(sig_parts)}")
    conf = d.get("confidence")
    if conf is not None:
        lines.append(f"**Confidence:** {conf:.2f}")
    dao = d.get("data_as_of")
    if dao:
        lines.append(f"**Data as of:** {dao}")
    return "\n".join(lines) if lines else str(d)


# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------

_BULLISH_TECHNICAL = {"uptrend", "bullish"}
_BEARISH_TECHNICAL = {"downtrend", "bearish"}
_BULLISH_FUNDAMENTAL = {"undervalued", "growing"}
_BEARISH_FUNDAMENTAL = {"overvalued", "declining"}


def detect_conflicts(state: OrchestratorState) -> list[dict]:
    """Compare technical vs fundamental signals and return conflict dicts."""
    ta = state.get("technical_analysis")
    fa = state.get("fundamental_analysis")
    failed = set(state.get("failed_agents", []))

    if (
        "technical_analysis" in failed
        or "fundamental_analysis" in failed
        or not ta
        or not fa
    ):
        return []

    ta_signals = ta.get("signals", {})
    fa_signals = fa.get("signals", {})
    conflicts: list[dict] = []

    # Compare trend/momentum vs valuation
    ta_trend = ta_signals.get("trend", "")
    ta_momentum = ta_signals.get("momentum", "")
    fa_valuation = fa_signals.get("valuation", "")

    ta_is_bullish = ta_trend in _BULLISH_TECHNICAL or ta_momentum in _BULLISH_TECHNICAL
    ta_is_bearish = ta_trend in _BEARISH_TECHNICAL or ta_momentum in _BEARISH_TECHNICAL
    fa_is_bullish = fa_valuation in _BULLISH_FUNDAMENTAL
    fa_is_bearish = fa_valuation in _BEARISH_FUNDAMENTAL

    if ta_is_bullish and fa_is_bearish:
        conflicts.append({
            "topic": "Trend vs Valuation",
            "agent_a": "Technical Analysis",
            "agent_b": "Fundamental Analysis",
            "agent_a_signal": f"trend={ta_trend}, momentum={ta_momentum}",
            "agent_b_signal": f"valuation={fa_valuation}",
        })
    elif ta_is_bearish and fa_is_bullish:
        conflicts.append({
            "topic": "Trend vs Valuation",
            "agent_a": "Technical Analysis",
            "agent_b": "Fundamental Analysis",
            "agent_a_signal": f"trend={ta_trend}, momentum={ta_momentum}",
            "agent_b_signal": f"valuation={fa_valuation}",
        })

    # Compare momentum vs growth
    fa_growth = fa_signals.get("growth", "")
    momentum_bullish = ta_momentum in _BULLISH_TECHNICAL
    momentum_bearish = ta_momentum in _BEARISH_TECHNICAL
    growth_bullish = fa_growth in _BULLISH_FUNDAMENTAL
    growth_bearish = fa_growth in _BEARISH_FUNDAMENTAL

    if momentum_bullish and growth_bearish:
        conflicts.append({
            "topic": "Momentum vs Growth",
            "agent_a": "Technical Analysis",
            "agent_b": "Fundamental Analysis",
            "agent_a_signal": f"momentum={ta_momentum}",
            "agent_b_signal": f"growth={fa_growth}",
        })
    elif momentum_bearish and growth_bullish:
        conflicts.append({
            "topic": "Momentum vs Growth",
            "agent_a": "Technical Analysis",
            "agent_b": "Fundamental Analysis",
            "agent_a_signal": f"momentum={ta_momentum}",
            "agent_b_signal": f"growth={fa_growth}",
        })

    return conflicts


# ---------------------------------------------------------------------------
# Synthesis node
# ---------------------------------------------------------------------------


async def synthesize_node(state: OrchestratorState) -> dict:
    """Synthesize agent results into a unified recommendation.

    Returns dict with synthesis_result, confidence_score, and error keys.
    """
    failed = set(state.get("failed_agents", []))

    # AC #4: All 3 agents failed → structured error, NO LLM call
    if len(failed) == len(("market_context", "technical_analysis", "fundamental_analysis")):
        logger.critical(
            "all_agents_failed",
            component="orchestrator",
            ticker=state.get("ticker"),
        )
        return {
            "synthesis_result": None,
            "confidence_score": 0.0,
            "error": "Tất cả 3 agents đều thất bại — không thể tổng hợp phân tích",
        }

    ticker = state.get("ticker", "")
    analysis_date = state.get("analysis_date", "")
    watchlist = state.get("watchlist", [])

    # Format agent outputs for prompt
    market_str = format_agent_output_for_prompt(state.get("market_summary"), "market_context")
    technical_str = format_agent_output_for_prompt(state.get("technical_analysis"), "technical_analysis")
    fundamental_str = format_agent_output_for_prompt(state.get("fundamental_analysis"), "fundamental_analysis")

    # Calculate confidence score (rule-based)
    confidence = calculate_confidence(state)

    config = get_config_loader()
    llm = _get_llm_client()

    # --- Main synthesis ---
    try:
        rendered = load_prompt(
            "orchestrator/synthesis",
            ticker=ticker,
            analysis_date=analysis_date,
            market_context_result=market_str,
            technical_result=technical_str,
            fundamental_result=fundamental_str,
            watchlist=watchlist,
        )
        synthesis_text = await llm.call(
            prompt=rendered.text,
            model=config.get_model(rendered.model_key),
            temperature=config.get_temperature("creative"),
            component="orchestrator",
        )
    except LLMCallError as e:
        logger.error("synthesis_llm_failed", component="orchestrator", error=str(e))
        return {
            "synthesis_result": None,
            "confidence_score": confidence,
            "error": f"Synthesis LLM call failed: {e}",
        }

    # --- Conflict resolution ---
    conflicts = detect_conflicts(state)
    conflict_resolution_text: str | None = None

    if conflicts:
        logger.info(
            "conflicts_detected",
            component="orchestrator",
            conflict_count=len(conflicts),
        )
        try:
            cr_rendered = load_prompt(
                "orchestrator/conflict_resolution",
                ticker=ticker,
                analysis_date=analysis_date,
                conflicts=conflicts,
                market_context_result=market_str,
                technical_result=technical_str,
                fundamental_result=fundamental_str,
            )
            conflict_resolution_text = await llm.call(
                prompt=cr_rendered.text,
                model=config.get_model(cr_rendered.model_key),
                temperature=config.get_temperature("creative"),
                component="orchestrator",
            )
        except LLMCallError as e:
            logger.warning(
                "conflict_resolution_failed",
                component="orchestrator",
                error=str(e),
            )
            conflict_resolution_text = None

    # --- Data integrity: attach source timestamps ---
    data_sources: list[dict] = []
    for key, agent_name in [
        ("market_summary", "market_context"),
        ("technical_analysis", "technical_analysis"),
        ("fundamental_analysis", "fundamental_analysis"),
    ]:
        output = state.get(key)
        if output and agent_name not in failed:
            data_sources.append({
                "agent": agent_name,
                "data_as_of": output.get("data_as_of"),
                "source": output.get("data_source") or output.get("sources"),
            })

    # --- Staleness warnings ---
    stale_warnings: list[str] = []
    now = datetime.now(timezone.utc)
    for ds in data_sources:
        dao = ds.get("data_as_of")
        if not dao:
            continue
        try:
            ts = datetime.fromisoformat(dao)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            age_hours = (now - ts).total_seconds() / 3600
            if age_hours > 4:
                stale_warnings.append(
                    f"⚠️ Dữ liệu {ds['agent']} cập nhật lần cuối {dao}"
                )
        except (ValueError, TypeError):
            pass

    # --- Unavailable agent warnings ---
    unavailable_warnings: list[str] = []
    agent_display = {
        "market_context": "Market Context Analysis",
        "technical_analysis": "Technical Analysis",
        "fundamental_analysis": "Fundamental Analysis",
    }
    for agent_name in failed:
        display = agent_display.get(agent_name, agent_name)
        unavailable_warnings.append(f"⚠️ {display} không khả dụng")

    synthesis_result = {
        "synthesis": synthesis_text,
        "conflict_resolution": conflict_resolution_text,
        "conflicts": conflicts,
        "data_sources": data_sources,
        "stale_warnings": stale_warnings,
        "unavailable_warnings": unavailable_warnings,
        "confidence_display": confidence_display(confidence),
        "disclaimer": _DISCLAIMER,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    logger.info(
        "synthesis_completed",
        component="orchestrator",
        ticker=ticker,
        confidence=confidence,
        conflict_count=len(conflicts),
        failed_agents=list(failed),
    )

    return {
        "synthesis_result": synthesis_result,
        "confidence_score": confidence,
        "error": None,
    }
