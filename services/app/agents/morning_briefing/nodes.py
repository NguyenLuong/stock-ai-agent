"""Morning Briefing pipeline nodes — 5-step sequential market analysis."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from pathlib import Path

import yaml

from services.app.agents.market_context.node import market_context_node
from services.app.agents.state import MorningBriefingState
from services.app.agents.technical_analysis.node import technical_analysis_node
from services.app.agents.fundamental_analysis.node import fundamental_analysis_node
from shared.logging import get_logger
from shared.utils.datetime_utils import now_utc

logger = get_logger("morning_briefing_graph")

# Max concurrent agent calls per batch step (avoids rate-limit / resource exhaustion).
_MAX_CONCURRENT_AGENT_CALLS = 10

# Maps stock_tickers.yaml group sector keys → keyword aliases found in
# market_context_node's affected_sectors output (Vietnamese names).
_SECTOR_KEYWORD_MAP: dict[str, list[str]] = {
    "banking": ["ngân hàng", "bank", "tài chính", "tín dụng", "lãi suất"],
    "steel": ["thép", "steel", "hsg", "hòa phát"],
    "oil_gas": ["dầu khí", "oil", "gas", "năng lượng", "energy", "xăng dầu"],
    "securities": ["chứng khoán", "securities", "môi giới"],
    "diversified": ["vn30", "blue-chip", "đa ngành"],
    # Sectors from market_context_node _SECTOR_ALIASES — no stock_tickers.yaml group yet,
    # but mapping exists so they won't be silently dropped when groups are added.
    "real_estate": ["bất động sản", "real estate", "bđs", "nhà đất"],
    "manufacturing": ["sản xuất", "manufacturing", "công nghiệp", "chế biến"],
    "technology": ["công nghệ", "technology", "tech", "phần mềm", "it"],
}

_BULLISH_KEYWORDS = ["tăng", "tích cực", "khả quan", "bullish", "phục hồi", "lạc quan"]
_BEARISH_KEYWORDS = ["giảm", "tiêu cực", "bearish", "rủi ro", "sụt", "bán tháo", "bi quan"]


def _normalize_sectors(raw_sectors: list[str]) -> list[str]:
    """Map Vietnamese sector names from market_context_node to stock_tickers.yaml keys."""
    matched: list[str] = []
    seen: set[str] = set()
    for raw in raw_sectors:
        raw_lower = raw.lower()
        for key, aliases in _SECTOR_KEYWORD_MAP.items():
            if key in seen:
                continue
            if any(alias in raw_lower for alias in aliases):
                matched.append(key)
                seen.add(key)
    return matched


def _infer_sentiment(text: str, confidence: float = 0.0) -> str:
    """Keyword-based sentiment from market summary text, with confidence tiebreaker.

    When keyword counts are tied, a high confidence (>0.6) from market_context_node
    suggests the LLM found a clear signal — default to "bullish" (market exists to go up).
    """
    lower = text.lower()
    bull = sum(1 for kw in _BULLISH_KEYWORDS if kw in lower)
    bear = sum(1 for kw in _BEARISH_KEYWORDS if kw in lower)
    if bull > bear:
        return "bullish"
    if bear > bull:
        return "bearish"
    # Tie or no keywords — use confidence as tiebreaker
    if bull == bear and bull > 0 and confidence > 0.6:
        return "bullish"
    return "neutral"


_EVENT_KEYWORDS = [
    "tăng", "giảm", "lãi suất", "lạm phát", "gdp", "fed", "sbv",
    "xuất khẩu", "nhập khẩu", "fdi", "trái phiếu", "tỷ giá",
    "dầu", "vàng", "cpi", "pmi", "sản lượng", "doanh thu",
    "lợi nhuận", "cổ tức", "ipo", "m&a", "phá sản",
]


def _extract_key_events(text: str, max_events: int = 5) -> list[str]:
    """Extract key event sentences from market summary text.

    Prioritises lines containing event-related keywords over generic text.
    """
    prioritised: list[str] = []
    fallback: list[str] = []

    for line in text.split("\n"):
        line = line.strip().lstrip("-•* ")
        if len(line) <= 20:
            continue
        snippet = line[:200]
        lower = snippet.lower()
        if any(kw in lower for kw in _EVENT_KEYWORDS):
            prioritised.append(snippet)
        else:
            fallback.append(snippet)

    events = prioritised[:max_events]
    remaining = max_events - len(events)
    if remaining > 0:
        events.extend(fallback[:remaining])
    return events


# ── Node 1: Market Context ──────────────────────────────────────────────


async def morning_market_context_node(state: MorningBriefingState) -> dict:
    """Step 1: Analyze market news, identify affected sectors."""
    failed_steps: list[str] = list(state.get("failed_steps", []))

    try:
        mc_result = await market_context_node({
            "ticker": "",
            "analysis_type": "morning_briefing",
            "analysis_date": state.get("analysis_date", ""),
        })
    except Exception as exc:
        logger.error(
            "morning_market_context_failed",
            component="morning_briefing_graph",
            error=str(exc),
        )
        failed_steps.append("market_context")
        return {
            "market_summary": None,
            "affected_sectors": [],
            "market_sentiment": "neutral",
            "key_events": [],
            "failed_steps": failed_steps,
        }

    market_summary = mc_result.get("market_summary")
    if market_summary is None:
        logger.warning(
            "morning_market_context_no_summary",
            component="morning_briefing_graph",
        )
        failed_steps.append("market_context")
        return {
            "market_summary": None,
            "affected_sectors": [],
            "market_sentiment": "neutral",
            "key_events": [],
            "failed_steps": failed_steps,
        }

    raw_sectors = market_summary.get("affected_sectors", [])
    affected_sectors = _normalize_sectors(raw_sectors)

    combined_text = (market_summary.get("macro_summary") or "") + " " + (
        market_summary.get("stock_summary") or ""
    )
    mc_confidence = market_summary.get("confidence", 0.0)
    market_sentiment = _infer_sentiment(combined_text, confidence=mc_confidence)
    key_events = _extract_key_events(combined_text)

    logger.info(
        "morning_market_context_completed",
        component="morning_briefing_graph",
        affected_sectors=affected_sectors,
        market_sentiment=market_sentiment,
    )

    return {
        "market_summary": market_summary,
        "affected_sectors": affected_sectors,
        "market_sentiment": market_sentiment,
        "key_events": key_events,
        "failed_steps": failed_steps,
    }


# ── Node 2: Sector Filter ───────────────────────────────────────────────


async def sector_filter_node(state: MorningBriefingState) -> dict:
    """Step 2: Filter tickers by affected sectors from stock_tickers.yaml."""
    affected_sectors = state.get("affected_sectors", [])
    watchlist = state.get("watchlist", [])

    config_path = (
        Path(os.environ.get("CONFIG_DIR", "/app/config"))
        / "crawlers"
        / "stock_tickers.yaml"
    )
    try:
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.error(
            "ticker_config_not_found",
            component="morning_briefing_graph",
            path=str(config_path),
        )
        return {"filtered_tickers": watchlist}

    filtered: list[str] = []
    seen: set[str] = set()

    if affected_sectors:
        for _group_name, group_data in raw.get("groups", {}).items():
            if not group_data.get("enabled", False):
                continue
            group_sector = group_data.get("sector", "")
            if group_sector not in affected_sectors:
                continue
            max_tickers = group_data.get("max_tickers")  # None = unlimited
            group_added = 0
            for ticker in group_data.get("tickers", []):
                if max_tickers is not None and group_added >= max_tickers:
                    break
                t = str(ticker).strip().upper()
                if t and t not in seen:
                    seen.add(t)
                    filtered.append(t)
                    group_added += 1

    # Early exit: pipeline aborted when filter yields no tickers
    if not filtered:
        abort_reason = (
            "no_sectors_identified" if not affected_sectors
            else "sectors_not_in_watchlist"
        )
        logger.warning(
            "morning_briefing_pipeline_aborted",
            component="morning_briefing_graph",
            reason=abort_reason,
            affected_sectors=affected_sectors,
        )
        return {
            "filtered_tickers": [],
            "pipeline_aborted": True,
            "abort_reason": abort_reason,
        }

    logger.info(
        "sector_filter_completed",
        component="morning_briefing_graph",
        filtered_count=len(filtered),
        total_watchlist=len(watchlist),
    )

    return {"filtered_tickers": filtered}


# ── Node 3: Technical Batch ─────────────────────────────────────────────


async def technical_batch_node(state: MorningBriefingState) -> dict:
    """Step 3: Technical analysis for each filtered ticker in parallel."""
    filtered_tickers = state.get("filtered_tickers", [])
    analysis_date = state.get("analysis_date", "")
    failed_steps: list[str] = list(state.get("failed_steps", []))

    sem = asyncio.Semaphore(_MAX_CONCURRENT_AGENT_CALLS)

    async def _run_ta(ticker: str) -> dict:
        async with sem:
            return await technical_analysis_node({
                "ticker": ticker,
                "analysis_type": "morning_briefing",
                "analysis_date": analysis_date,
            })

    results = await asyncio.gather(
        *[_run_ta(t) for t in filtered_tickers], return_exceptions=True,
    )

    technical_results: list[dict] = []
    notable_tickers: list[str] = []
    success_count = 0

    for t, r in zip(filtered_tickers, results):
        if isinstance(r, Exception):
            logger.error(
                "technical_analysis_ticker_failed",
                component="morning_briefing_graph",
                ticker=t,
                error=str(r),
            )
            technical_results.append({
                "ticker": t,
                "technical_analysis": None,
                "failed": True,
            })
            continue

        ta = r.get("technical_analysis")
        failed = ta is None
        technical_results.append({
            "ticker": t,
            "technical_analysis": ta,
            "failed": failed,
        })

        if not failed:
            success_count += 1
            signals = (ta or {}).get("signals", {})
            trend = signals.get("trend", "sideways")
            momentum = signals.get("momentum", "neutral")
            if trend != "sideways" or momentum != "neutral":
                notable_tickers.append(t)

    # Fallback: chỉ khi notable rỗng hoàn toàn → lấy 2 tickers đầu
    if not notable_tickers:
        notable_tickers = filtered_tickers[:2]

    if success_count == 0:
        failed_steps.append("technical_batch")

    logger.info(
        "technical_batch_completed",
        component="morning_briefing_graph",
        total=len(filtered_tickers),
        succeeded=success_count,
        notable=len(notable_tickers),
    )

    return {
        "technical_results": technical_results,
        "notable_tickers": notable_tickers,
        "failed_steps": failed_steps,
    }


# ── Node 4: Fundamental Batch ───────────────────────────────────────────


async def fundamental_batch_node(state: MorningBriefingState) -> dict:
    """Step 4: Fundamental analysis only for notable tickers in parallel."""
    notable_tickers = state.get("notable_tickers", [])
    analysis_date = state.get("analysis_date", "")
    failed_steps: list[str] = list(state.get("failed_steps", []))

    sem = asyncio.Semaphore(_MAX_CONCURRENT_AGENT_CALLS)

    async def _run_fa(ticker: str) -> dict:
        async with sem:
            return await fundamental_analysis_node({
                "ticker": ticker,
                "analysis_type": "morning_briefing",
                "analysis_date": analysis_date,
            })

    results = await asyncio.gather(
        *[_run_fa(t) for t in notable_tickers], return_exceptions=True,
    )

    fundamental_results: list[dict] = []
    success_count = 0

    for t, r in zip(notable_tickers, results):
        if isinstance(r, Exception):
            logger.error(
                "fundamental_analysis_ticker_failed",
                component="morning_briefing_graph",
                ticker=t,
                error=str(r),
            )
            fundamental_results.append({
                "ticker": t,
                "fundamental_analysis": None,
                "failed": True,
            })
            continue

        fa = r.get("fundamental_analysis")
        failed = fa is None
        fundamental_results.append({
            "ticker": t,
            "fundamental_analysis": fa,
            "failed": failed,
        })
        if not failed:
            success_count += 1

    if success_count == 0:
        failed_steps.append("fundamental_batch")

    logger.info(
        "fundamental_batch_completed",
        component="morning_briefing_graph",
        total=len(notable_tickers),
        succeeded=success_count,
    )

    return {
        "fundamental_results": fundamental_results,
        "failed_steps": failed_steps,
    }


# ── Node 5: Morning Synthesis ───────────────────────────────────────────


async def morning_synthesis_node(state: MorningBriefingState) -> dict:
    """Step 5: Compile all results into market_result dict."""
    # Early exit: pipeline aborted at sector filter — skip all downstream logic
    if state.get("pipeline_aborted"):
        market_result = {
            "pipeline_status": "aborted",
            "abort_reason": state.get("abort_reason", "unknown"),
            "market_sentiment": state.get("market_sentiment", "neutral"),
            "affected_sectors": state.get("affected_sectors", []),
            "key_events": state.get("key_events", []),
            "market_summary": (state.get("market_summary") or {}).get("macro_summary") or "",
            "top_picks": [],
            "stale_warnings": [],
            "unavailable_warnings": [],
            "disclaimer": "Thông tin chỉ mang tính chất tham khảo, không phải khuyến nghị đầu tư.",
            "generated_at": now_utc().isoformat(),
        }
        logger.info(
            "morning_synthesis_aborted",
            component="morning_briefing_graph",
            abort_reason=state.get("abort_reason"),
        )
        return {"market_result": market_result}

    market_summary = state.get("market_summary")
    market_sentiment = state.get("market_sentiment", "neutral")
    affected_sectors = state.get("affected_sectors", [])
    key_events = state.get("key_events", [])
    technical_results = state.get("technical_results", [])
    fundamental_results = state.get("fundamental_results", [])
    failed_steps = state.get("failed_steps", [])

    # Build fundamental lookup
    fundamental_map: dict[str, dict] = {}
    for item in fundamental_results:
        if not item.get("failed") and item.get("fundamental_analysis"):
            fundamental_map[item["ticker"]] = item["fundamental_analysis"]

    # Build top_picks: only tickers with BOTH technical + fundamental data
    top_picks: list[dict] = []
    for item in technical_results:
        if item.get("failed") or not item.get("technical_analysis"):
            continue
        ticker = item["ticker"]
        if ticker not in fundamental_map:
            continue
        ta = item["technical_analysis"]
        signals = ta.get("signals", {})
        top_picks.append({
            "ticker": ticker,
            "signal": signals.get("trend", "sideways"),
            "confidence": ta.get("confidence", 0.5),
            "summary": (ta.get("indicator_summary") or "")[:200],
        })

    # Sort by confidence descending, limit to 5
    top_picks.sort(key=lambda x: x["confidence"], reverse=True)
    top_picks = top_picks[:5]

    # Collect warnings
    stale_warnings: list[str] = []
    unavailable_warnings: list[str] = []
    stale_threshold_hours = 4
    current_time = now_utc()

    for item in technical_results:
        if item.get("failed"):
            unavailable_warnings.append(f"Technical analysis unavailable: {item['ticker']}")
        elif item.get("technical_analysis"):
            data_as_of = item["technical_analysis"].get("data_as_of", "")
            if data_as_of:
                try:
                    ts = datetime.fromisoformat(data_as_of)
                    if (current_time - ts).total_seconds() > stale_threshold_hours * 3600:
                        stale_warnings.append(f"Dữ liệu kỹ thuật {item['ticker']} đã cũ hơn {stale_threshold_hours}h")
                except (ValueError, TypeError):
                    pass

    for item in fundamental_results:
        if item.get("failed"):
            unavailable_warnings.append(f"Fundamental analysis unavailable: {item['ticker']}")
        elif item.get("fundamental_analysis"):
            data_as_of = item["fundamental_analysis"].get("data_as_of", "")
            if data_as_of:
                try:
                    ts = datetime.fromisoformat(data_as_of)
                    if (current_time - ts).total_seconds() > stale_threshold_hours * 3600:
                        stale_warnings.append(f"Dữ liệu cơ bản {item['ticker']} đã cũ hơn {stale_threshold_hours}h")
                except (ValueError, TypeError):
                    pass

    # Market summary text
    if top_picks:
        summary_text = (market_summary or {}).get("macro_summary") or ""
    else:
        summary_text = "Thị trường bình lặng hôm nay — không có tín hiệu kỹ thuật rõ ràng"

    generated_at = now_utc().isoformat()

    market_result: dict = {
        "market_sentiment": market_sentiment,
        "affected_sectors": affected_sectors,
        "key_events": key_events,
        "top_picks": top_picks,
        "market_summary": summary_text,
        "stale_warnings": stale_warnings,
        "unavailable_warnings": unavailable_warnings,
        "disclaimer": "Thông tin chỉ mang tính chất tham khảo, không phải khuyến nghị đầu tư.",
        "generated_at": generated_at,
    }

    logger.info(
        "morning_synthesis_completed",
        component="morning_briefing_graph",
        top_picks_count=len(top_picks),
        market_sentiment=market_sentiment,
    )

    return {"market_result": market_result}
