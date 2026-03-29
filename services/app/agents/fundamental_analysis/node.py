"""Fundamental Analysis Agent node — 2-phase BCTC + ratio comparison analysis."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal

from services.app.agents.state import FundamentalAnalysisState
from services.crawler.market_data.stock_data_repo import (
    get_latest_financial_ratios,
    get_peer_ratios,
    get_sector_average_ratios,
    save_financial_ratios,
)
from services.crawler.market_data.ticker_config import get_sector_for_ticker
from services.crawler.market_data.vnstock_client import VnstockClient
from shared.llm.client import LLMCallError, LLMClient
from shared.llm.config_loader import get_config_loader
from shared.llm.prompt_loader import load_prompt
from shared.logging import get_logger

logger = get_logger("fundamental_analysis")

CONFIDENCE_BASE = 0.50


def _decimal_to_float(val: Decimal | None) -> float | None:
    """Convert Decimal to float, None stays None."""
    return float(val) if val is not None else None


def _build_ratio_for_prompt(
    ratios: dict[str, Decimal | None],
) -> dict[str, float | None]:
    """Convert raw DB ratios to prompt-friendly dict with short keys."""
    return {
        "pe": _decimal_to_float(ratios.get("pe_ratio")),
        "pb": _decimal_to_float(ratios.get("pb_ratio")),
        "roe": _decimal_to_float(ratios.get("roe")),
        "eps": _decimal_to_float(ratios.get("eps")),
        "eps_growth_yoy": _decimal_to_float(ratios.get("eps_growth_yoy")),
    }


def _determine_valuation(
    company_pe: float | None, sector_pe: float | None,
) -> str:
    """Rule-based valuation signal from P/E comparison."""
    if company_pe is None or sector_pe is None or sector_pe <= 0 or company_pe <= 0:
        return "fair"
    ratio = company_pe / sector_pe
    if ratio < 0.8:
        return "undervalued"
    if ratio > 1.2:
        return "overvalued"
    return "fair"


def _determine_profitability(roe: float | None) -> str:
    """Rule-based profitability signal from ROE."""
    if roe is None:
        return "average"
    if roe > 15:
        return "strong"
    if roe < 8:
        return "weak"
    return "average"


def _determine_growth(eps_growth: float | None) -> str:
    """Rule-based growth signal from EPS growth YoY."""
    if eps_growth is None:
        return "stable"
    if eps_growth > 0.10:
        return "growing"
    if eps_growth < -0.10:
        return "declining"
    return "stable"


def _calc_confidence(
    company_ratios: dict[str, float | None],
    sector_avg: dict[str, Decimal | None],
    phase1_ok: bool = True,
    phase2_ok: bool = True,
    data_as_of: datetime | None = None,
    peer_count: int = 0,
) -> float:
    """Rule-based confidence scoring. Base 0.50, max 0.95."""
    score = CONFIDENCE_BASE

    # Company ratio coverage
    core_fields = ["pe", "pb", "roe", "eps", "eps_growth_yoy"]
    available = sum(1 for f in core_fields if company_ratios.get(f) is not None)
    if available >= 5:
        score += 0.15
    elif available >= 3:
        score += 0.10
    elif available >= 1:
        score += 0.05

    # Sector average coverage (based on actual peer count with data)
    if peer_count >= 2:
        score += 0.10
    elif peer_count >= 1:
        score += 0.05

    # LLM phase success
    if phase1_ok and phase2_ok:
        score += 0.10
    elif phase1_ok or phase2_ok:
        score += 0.05

    # Data freshness (quarterly = <90 days)
    if data_as_of is not None:
        age = datetime.now(timezone.utc) - data_as_of
        if age.total_seconds() < 90 * 24 * 3600:
            score += 0.05

    return min(score, 0.95)


async def fundamental_analysis_node(state: FundamentalAnalysisState) -> dict:
    """LangGraph node: BCTC analysis → ratio comparison → combine.

    Never raises — returns graceful degradation on failure.
    """
    ticker = state.get("ticker", "")
    analysis_date = state.get(
        "analysis_date", datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    )

    logger.info(
        "agent_started",
        component="fundamental_analysis",
        ticker=ticker,
        analysis_date=analysis_date,
    )

    llm = LLMClient()
    config = get_config_loader()
    failed_agents: list[str] = list(state.get("failed_agents", []))

    # ── Phase 0: Data Retrieval ─────────────────────────────────────────
    logger.info("ratio_retrieval_started", component="fundamental_analysis")
    used_fallback = False
    try:
        company_ratios_raw, data_as_of = await get_latest_financial_ratios(ticker)
    except Exception as e:
        logger.error(
            "ratio_retrieval_failed", component="fundamental_analysis", error=str(e),
        )
        company_ratios_raw, data_as_of = {}, None

    logger.info(
        "ratio_retrieval_completed",
        component="fundamental_analysis",
        ratio_count=len(company_ratios_raw),
    )

    # Fallback: fetch on-the-fly via vnstock if no DB data
    if not company_ratios_raw:
        logger.warning(
            "no_ratios_in_db_fetching_vnstock",
            component="fundamental_analysis",
            ticker=ticker,
        )
        try:
            client = VnstockClient()
            df = await asyncio.to_thread(
                client.get_financial_ratios, ticker, "quarter",
            )
            fallback_source = getattr(df, "attrs", {}).get("data_source", "vnstock")
            await save_financial_ratios(ticker, df, data_source=fallback_source)
            company_ratios_raw, data_as_of = await get_latest_financial_ratios(ticker)
            used_fallback = True
        except Exception as e:
            logger.error(
                "vnstock_fallback_failed",
                component="fundamental_analysis",
                error=str(e),
            )

    # Check data sufficiency (AC #3)
    if not company_ratios_raw:
        logger.warning(
            "no_financial_data_available",
            component="fundamental_analysis",
            ticker=ticker,
        )
        failed_agents.append("fundamental_analysis")
        return {
            "fundamental_analysis": None,
            "error": f"Dữ liệu BCTC cho {ticker} không khả dụng — "
                     "không có trong DB và vnstock fallback cũng thất bại",
            "failed_agents": failed_agents,
        }

    # Sector info
    logger.info("sector_lookup_started", component="fundamental_analysis")
    sector_display_name, sector_tickers = await asyncio.to_thread(
        get_sector_for_ticker, ticker,
    )
    logger.info(
        "sector_lookup_completed",
        component="fundamental_analysis",
        sector=sector_display_name,
        peer_count=len(sector_tickers),
    )

    # Sector averages
    logger.info("sector_avg_retrieval_started", component="fundamental_analysis")
    try:
        sector_avg = await get_sector_average_ratios(
            sector_tickers, exclude_ticker=ticker,
        )
    except Exception as e:
        logger.warning(
            "sector_avg_retrieval_failed",
            component="fundamental_analysis",
            error=str(e),
        )
        sector_avg = {"pe": None, "pb": None, "roe": None, "eps": None}
    logger.info(
        "sector_avg_retrieval_completed", component="fundamental_analysis",
    )

    # Peer ratios
    try:
        peers = await get_peer_ratios(sector_tickers, exclude_ticker=ticker)
    except Exception as e:
        logger.warning(
            "peer_ratios_retrieval_failed",
            component="fundamental_analysis",
            error=str(e),
        )
        peers = []

    # Pre-compute prompt variables (H2/H3: outside LLM try blocks)
    company_prompt = _build_ratio_for_prompt(company_ratios_raw)

    # Sector avg for bctc_analysis prompt (needs pe, pb, roe keys)
    sector_avg_for_bctc: dict | None = None
    if any(v is not None for v in sector_avg.values()):
        sector_avg_for_bctc = {
            "pe": _decimal_to_float(sector_avg.get("pe")),
            "pb": _decimal_to_float(sector_avg.get("pb")),
            "roe": _decimal_to_float(sector_avg.get("roe")),
        }

    company_ratios_for_comparison = {
        "pe": company_prompt["pe"],
        "pb": company_prompt["pb"],
        "roe": company_prompt["roe"],
        "eps": company_prompt["eps"],
        "debt_to_equity": None,
        "current_ratio": None,
        "dividend_yield": None,
    }

    sector_ratios_for_comparison = {
        "pe": _decimal_to_float(sector_avg.get("pe")),
        "pb": _decimal_to_float(sector_avg.get("pb")),
        "roe": _decimal_to_float(sector_avg.get("roe")),
        "eps": _decimal_to_float(sector_avg.get("eps")),
        "debt_to_equity": None,
    }

    bctc_summary: str | None = None
    ratio_comparison: str | None = None
    phase1_ok = False
    phase2_ok = False

    # ── Phase 1: BCTC Analysis (LLM) ───────────────────────────────────
    try:
        logger.info("llm_call_started", component="fundamental_analysis", phase="bctc_analysis")
        bctc_prompt = load_prompt(
            "fundamental_analysis/bctc_analysis",
            ticker=ticker,
            analysis_date=analysis_date,
            pe_ratio=company_prompt["pe"],
            pb_ratio=company_prompt["pb"],
            roe=company_prompt["roe"],
            eps=company_prompt["eps"],
            revenue=None,
            net_profit=None,
            debt_to_equity=None,
            current_ratio=None,
            sector_avg=sector_avg_for_bctc,
        )
        model = config.get_model(bctc_prompt.model_key)
        temp = config.get_temperature()
        bctc_summary = await llm.call(
            prompt=bctc_prompt.text,
            model=model,
            temperature=temp,
            component="fundamental_analysis",
        )
        phase1_ok = True
        logger.info("llm_call_completed", component="fundamental_analysis", phase="bctc_analysis")
    except LLMCallError as e:
        logger.error(
            "llm_call_failed", component="fundamental_analysis", phase="bctc_analysis",
            error=str(e),
        )
        bctc_summary = None
    except Exception as e:
        logger.error(
            "phase_failed", component="fundamental_analysis", phase="bctc_analysis",
            error=str(e),
        )
        bctc_summary = None

    # ── Phase 2: Ratio Comparison (LLM) ─────────────────────────────────
    try:
        logger.info("llm_call_started", component="fundamental_analysis", phase="ratio_comparison")
        comparison_prompt = load_prompt(
            "fundamental_analysis/ratio_comparison",
            ticker=ticker,
            analysis_date=analysis_date,
            sector_name=sector_display_name,
            company_ratios=company_ratios_for_comparison,
            sector_ratios=sector_ratios_for_comparison,
            peers=peers,
        )
        ratio_comparison = await llm.call(
            prompt=comparison_prompt.text,
            model=model,
            temperature=temp,
            component="fundamental_analysis",
        )
        phase2_ok = True
        logger.info("llm_call_completed", component="fundamental_analysis", phase="ratio_comparison")
    except LLMCallError as e:
        logger.error(
            "llm_call_failed", component="fundamental_analysis", phase="ratio_comparison",
            error=str(e),
        )
        ratio_comparison = None
    except Exception as e:
        logger.error(
            "phase_failed", component="fundamental_analysis", phase="ratio_comparison",
            error=str(e),
        )
        ratio_comparison = None

    # ── Phase 3: Combine (code only, no LLM) ────────────────────────────
    if bctc_summary is None and ratio_comparison is None:
        logger.error("all_phases_failed", component="fundamental_analysis")
        if "fundamental_analysis" not in failed_agents:
            failed_agents.append("fundamental_analysis")
        return {
            "fundamental_analysis": None,
            "error": "Fundamental Analysis: both BCTC and ratio comparison phases failed",
            "failed_agents": failed_agents,
        }

    # Build signals dict (rule-based)
    signals = {
        "valuation": _determine_valuation(
            company_prompt["pe"],
            _decimal_to_float(sector_avg.get("pe")),
        ),
        "profitability": _determine_profitability(company_prompt["roe"]),
        "financial_health": "neutral",  # safe default — no D/E data available
        "growth": _determine_growth(company_prompt["eps_growth_yoy"]),
    }

    confidence = _calc_confidence(
        company_prompt, sector_avg,
        phase1_ok=phase1_ok, phase2_ok=phase2_ok,
        data_as_of=data_as_of,
        peer_count=len(peers),
    )

    effective_as_of = (
        data_as_of.isoformat() if data_as_of
        else datetime.now(timezone.utc).isoformat()
    )

    fundamental_analysis = {
        "bctc_summary": bctc_summary,
        "ratio_comparison": ratio_comparison,
        "company_ratios": company_prompt,
        "sector_ratios": {
            "pe": _decimal_to_float(sector_avg.get("pe")),
            "pb": _decimal_to_float(sector_avg.get("pb")),
            "roe": _decimal_to_float(sector_avg.get("roe")),
            "eps": _decimal_to_float(sector_avg.get("eps")),
        },
        "sector_name": sector_display_name,
        "signals": signals,
        "confidence": round(confidence, 2),
        "data_as_of": effective_as_of,
        "data_source": "vnstock_fallback" if used_fallback else "db",
    }

    logger.info(
        "agent_completed",
        component="fundamental_analysis",
        confidence=fundamental_analysis["confidence"],
        valuation=signals["valuation"],
        profitability=signals["profitability"],
        phase1_ok=phase1_ok,
        phase2_ok=phase2_ok,
    )

    return {"fundamental_analysis": fundamental_analysis}
