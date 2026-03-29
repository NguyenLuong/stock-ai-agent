"""Technical Analysis Agent node — 2-phase sequential indicator + pattern analysis."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal

import pandas as pd

from services.app.agents.state import TechnicalAnalysisState
from services.crawler.market_data.indicator_calculator import calculate_indicators
from services.crawler.market_data.indicator_repo import (
    get_latest_indicators,
    get_stock_prices_df,
)
from shared.llm.client import LLMCallError, LLMClient
from shared.llm.config_loader import get_config_loader
from shared.llm.prompt_loader import load_prompt
from shared.logging import get_logger

logger = get_logger("technical_analysis")

MIN_OHLCV_ROWS = 30

# Raw DB indicator_name → (group_key, field_key) for structured prompt dicts
INDICATOR_MAPPING: dict[str, tuple[str, str]] = {
    "RSI_14": ("rsi", "value"),
    "SMA_20": ("sma", "sma_20"),
    "SMA_50": ("sma", "sma_50"),
    "SMA_200": ("sma", "sma_200"),
    "MACD_LINE": ("macd", "macd"),
    "MACD_SIGNAL": ("macd", "signal"),
    "MACD_HISTOGRAM": ("macd", "histogram"),
    "BB_UPPER": ("bollinger", "upper"),
    "BB_MIDDLE": ("bollinger", "middle"),
    "BB_LOWER": ("bollinger", "lower"),
    "VOLUME_AVG_20": ("volume_analysis", "avg_volume"),
    "VOLUME_CURRENT": ("volume_analysis", "current_volume"),
    "VOLUME_RATIO": ("volume_analysis", "ratio"),
    "ATR_14": ("atr", "value"),
    "DC_UPPER": ("donchian", "upper"),
    "DC_MIDDLE": ("donchian", "middle"),
    "DC_LOWER": ("donchian", "lower"),
    "RS_VNINDEX": ("relative_strength", "value"),
}

# All 8 groups required by prompt template (Jinja2 StrictUndefined)
_ALL_GROUPS = (
    "rsi", "macd", "bollinger", "sma", "volume_analysis", "atr", "donchian",
    "relative_strength",
)


def _build_indicator_dicts(
    raw_indicators: dict[str, Decimal | None],
) -> dict:
    """Convert flat indicator map to structured dicts for prompt template.

    Always returns all 8 group keys (None for missing groups).
    """
    groups: dict[str, dict | None] = {g: None for g in _ALL_GROUPS}

    for ind_name, value in raw_indicators.items():
        mapping = INDICATOR_MAPPING.get(ind_name)
        if mapping is None:
            continue
        group_key, field_key = mapping
        if groups[group_key] is None:
            groups[group_key] = {}
        float_val = float(value) if value is not None else None
        groups[group_key][field_key] = float_val

    # Add fixed metadata
    if groups["rsi"] is not None:
        groups["rsi"]["period"] = 14
    if groups["atr"] is not None:
        groups["atr"]["period"] = 14

    return groups


def _determine_trend(sma: dict | None) -> str:
    """Determine trend from MA cross. Safe default: 'sideways'."""
    if sma is None:
        return "sideways"

    ma20 = sma.get("sma_20")
    ma50 = sma.get("sma_50")
    ma200 = sma.get("sma_200")

    if ma20 is None or ma50 is None or ma200 is None:
        return "sideways"

    if ma20 > ma50 > ma200:
        return "uptrend"
    if ma20 < ma50 < ma200:
        return "downtrend"
    return "sideways"


def _calculate_support_resistance(
    ohlcv_df: pd.DataFrame, window: int = 20,
) -> tuple[list[float], list[float]]:
    """Calculate support/resistance levels from OHLCV data.

    Returns (support_levels, resistance_levels) sorted ascending, max 3 each.
    """
    if len(ohlcv_df) < window:
        # Fallback: min/max of available data
        low_min = float(ohlcv_df["low"].min())
        high_max = float(ohlcv_df["high"].max())
        return [low_min], [high_max]

    # Rolling min/max
    rolling_low = ohlcv_df["low"].rolling(window=window, center=True).min()
    rolling_high = ohlcv_df["high"].rolling(window=window, center=True).max()

    # Find local minima (support) — value equals rolling min
    support_mask = ohlcv_df["low"] == rolling_low
    support_candidates = ohlcv_df.loc[support_mask, "low"].astype(float).tolist()

    # Find local maxima (resistance) — value equals rolling max
    resistance_mask = ohlcv_df["high"] == rolling_high
    resistance_candidates = ohlcv_df.loc[resistance_mask, "high"].astype(float).tolist()

    # Deduplicate: merge levels within ±2%
    support_levels = _deduplicate_levels(support_candidates)
    resistance_levels = _deduplicate_levels(resistance_candidates)

    # Fallback if no pivot points found
    if not support_levels:
        recent = ohlcv_df.tail(window)
        support_levels = [float(recent["low"].min())]
    if not resistance_levels:
        recent = ohlcv_df.tail(window)
        resistance_levels = [float(recent["high"].max())]

    # Get current price for proximity sorting
    current_price = float(ohlcv_df["close"].iloc[-1])

    # Top 3 closest to current price
    support_levels = sorted(support_levels, key=lambda x: abs(x - current_price))[:3]
    resistance_levels = sorted(resistance_levels, key=lambda x: abs(x - current_price))[:3]

    return sorted(support_levels), sorted(resistance_levels)


def _deduplicate_levels(levels: list[float], tolerance: float = 0.02) -> list[float]:
    """Merge levels within ±tolerance (2%) of each other."""
    if not levels:
        return []
    sorted_levels = sorted(levels)
    deduped = [sorted_levels[0]]
    for level in sorted_levels[1:]:
        if abs(deduped[-1]) < 1e-9 or abs(level - deduped[-1]) / abs(deduped[-1]) > tolerance:
            deduped.append(level)
    return deduped


def _build_ohlcv_for_prompt(
    ohlcv_df: pd.DataFrame, last_n: int = 30,
) -> list[dict]:
    """Convert last N rows of OHLCV DataFrame to list[dict] for prompt."""
    recent = ohlcv_df.tail(last_n).copy()
    recent = recent.rename(columns={"time": "date"})
    recent["date"] = recent["date"].astype(str)
    for col in ("open", "high", "low", "close", "volume"):
        recent[col] = recent[col].astype(float)
    return recent[["date", "open", "high", "low", "close", "volume"]].to_dict(orient="records")


def _calc_confidence(
    indicator_dicts: dict,
    ohlcv_count: int,
    phase1_ok: bool = True,
    phase2_ok: bool = True,
    data_as_of: datetime | None = None,
) -> float:
    """Rule-based confidence scoring.

    Base 0.50, max 0.95. See story Dev Notes for scoring table.
    """
    score = 0.50

    # OHLCV data depth
    if ohlcv_count >= 200:
        score += 0.15
    elif ohlcv_count >= 50:
        score += 0.10
    elif ohlcv_count >= 30:
        score += 0.05

    # Indicator availability — count using INDICATOR_MAPPING (authoritative list of 19 indicators)
    available_count = 0
    for ind_name, (group_key, field_key) in INDICATOR_MAPPING.items():
        group = indicator_dicts.get(group_key)
        if group is not None and group.get(field_key) is not None:
            available_count += 1

    if available_count >= 15:
        score += 0.10
    elif available_count >= 10:
        score += 0.05

    # LLM phase success
    if phase1_ok and phase2_ok:
        score += 0.10
    elif phase1_ok or phase2_ok:
        score += 0.05

    # Data freshness
    if data_as_of is not None:
        age = datetime.now(timezone.utc) - data_as_of
        if age.total_seconds() < 24 * 3600:
            score += 0.05

    return min(score, 0.95)


async def technical_analysis_node(state: TechnicalAnalysisState) -> dict:
    """LangGraph node: sequential indicator analysis → pattern recognition → combine.

    Never raises — returns graceful degradation on failure.
    """
    ticker = state.get("ticker", "")
    analysis_date = state.get(
        "analysis_date", datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    )

    logger.info(
        "agent_started",
        component="technical_analysis",
        ticker=ticker,
        analysis_date=analysis_date,
    )

    llm = LLMClient()
    config = get_config_loader()
    failed_agents: list[str] = list(state.get("failed_agents", []))

    # ── Phase 0: Data Retrieval ─────────────────────────────────────────
    try:
        logger.info("indicator_retrieval_started", component="technical_analysis")
        raw_indicators, data_as_of = await get_latest_indicators(ticker)
        logger.info(
            "indicator_retrieval_completed",
            component="technical_analysis",
            indicator_count=len(raw_indicators),
        )
    except Exception as e:
        logger.error(
            "indicator_retrieval_failed",
            component="technical_analysis",
            error=str(e),
        )
        raw_indicators, data_as_of = {}, None

    try:
        logger.info("ohlcv_retrieval_started", component="technical_analysis")
        ohlcv_df = await get_stock_prices_df(ticker, limit=300)
        logger.info(
            "ohlcv_retrieval_completed",
            component="technical_analysis",
            row_count=len(ohlcv_df),
        )
    except Exception as e:
        logger.error(
            "ohlcv_retrieval_failed", component="technical_analysis", error=str(e),
        )
        ohlcv_df = pd.DataFrame()

    # Check data sufficiency (AC #3)
    if len(ohlcv_df) < MIN_OHLCV_ROWS:
        logger.warning(
            "insufficient_data",
            component="technical_analysis",
            ohlcv_rows=len(ohlcv_df),
            min_required=MIN_OHLCV_ROWS,
        )
        failed_agents.append("technical_analysis")
        return {
            "technical_analysis": None,
            "error": f"Không đủ dữ liệu lịch sử để phân tích kỹ thuật "
                     f"(có {len(ohlcv_df)} phiên, cần tối thiểu {MIN_OHLCV_ROWS})",
            "failed_agents": failed_agents,
        }

    # Fallback: recalculate indicators on-the-fly if DB has none (M3: run sync in thread)
    if not raw_indicators:
        logger.warning(
            "no_indicators_in_db_recalculating",
            component="technical_analysis",
            ticker=ticker,
        )
        try:
            records = await asyncio.to_thread(
                calculate_indicators, ticker, ohlcv_df, None,
            )
            raw_indicators = {}
            latest_as_of: datetime | None = None
            for rec in records:
                if rec.indicator_value is not None:
                    raw_indicators[rec.indicator_name] = rec.indicator_value
                    if latest_as_of is None or rec.data_as_of > latest_as_of:
                        latest_as_of = rec.data_as_of
            data_as_of = latest_as_of
        except Exception as e:
            logger.error(
                "indicator_recalculation_failed",
                component="technical_analysis",
                error=str(e),
            )

    indicator_dicts = _build_indicator_dicts(raw_indicators)
    current_price = float(ohlcv_df["close"].iloc[-1])

    # Inject current_price into bollinger and donchian
    if indicator_dicts.get("bollinger") is not None:
        indicator_dicts["bollinger"]["current_price"] = current_price
    if indicator_dicts.get("donchian") is not None:
        indicator_dicts["donchian"]["current_price"] = current_price

    # Pre-compute data processing shared by Phase 2 and Phase 3 (H2/H3: outside try blocks)
    trend = _determine_trend(indicator_dicts.get("sma"))
    support_levels, resistance_levels = _calculate_support_resistance(ohlcv_df)
    ohlcv_for_prompt = _build_ohlcv_for_prompt(ohlcv_df, last_n=30)

    indicator_summary: str | None = None
    pattern_summary: str | None = None
    phase1_ok = False
    phase2_ok = False
    failed_agents_modified = False

    # ── Phase 1: Indicator Analysis (LLM) ───────────────────────────────
    try:
        logger.info("llm_call_started", component="technical_analysis", phase="indicators")
        indicators_prompt = load_prompt(
            "technical_analysis/indicators",
            ticker=ticker,
            analysis_date=analysis_date,
            rsi=indicator_dicts.get("rsi"),
            macd=indicator_dicts.get("macd"),
            bollinger=indicator_dicts.get("bollinger"),
            sma=indicator_dicts.get("sma"),
            volume_analysis=indicator_dicts.get("volume_analysis"),
            atr=indicator_dicts.get("atr"),
            donchian=indicator_dicts.get("donchian"),
            relative_strength=indicator_dicts.get("relative_strength"),
        )
        model = config.get_model(indicators_prompt.model_key)
        temp = config.get_temperature()
        indicator_summary = await llm.call(
            prompt=indicators_prompt.text,
            model=model,
            temperature=temp,
            component="technical_analysis",
        )
        phase1_ok = True
        logger.info("llm_call_completed", component="technical_analysis", phase="indicators")
    except LLMCallError as e:
        logger.error(
            "llm_call_failed", component="technical_analysis", phase="indicators",
            error=str(e),
        )
        indicator_summary = None
    except Exception as e:
        logger.error(
            "phase_failed", component="technical_analysis", phase="indicators",
            error=str(e),
        )
        indicator_summary = None

    # ── Phase 2: Pattern Recognition (LLM only — data already prepared) ─
    try:
        logger.info("llm_call_started", component="technical_analysis", phase="pattern_recognition")
        pattern_prompt = load_prompt(
            "technical_analysis/pattern_recognition",
            ticker=ticker,
            analysis_date=analysis_date,
            ohlcv=ohlcv_for_prompt,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            trend=trend,
            indicators_summary=indicator_summary,
        )
        pattern_summary = await llm.call(
            prompt=pattern_prompt.text,
            model=config.get_model(pattern_prompt.model_key),
            temperature=config.get_temperature(),
            component="technical_analysis",
        )
        phase2_ok = True
        logger.info("llm_call_completed", component="technical_analysis", phase="pattern_recognition")
    except LLMCallError as e:
        logger.error(
            "llm_call_failed", component="technical_analysis", phase="pattern_recognition",
            error=str(e),
        )
        pattern_summary = None
    except Exception as e:
        logger.error(
            "phase_failed", component="technical_analysis", phase="pattern_recognition",
            error=str(e),
        )
        pattern_summary = None

    # ── Phase 3: Combine (code only, no LLM) ────────────────────────────
    if indicator_summary is None and pattern_summary is None:
        logger.error("all_phases_failed", component="technical_analysis")
        if "technical_analysis" not in failed_agents:
            failed_agents.append("technical_analysis")
            failed_agents_modified = True
        return {
            "technical_analysis": None,
            "error": "Technical Analysis: both indicator and pattern phases failed",
            "failed_agents": failed_agents,
        }

    # Build signals dict (rule-based) — trend already computed above
    rsi_group = indicator_dicts.get("rsi")
    macd_group = indicator_dicts.get("macd")
    rsi_val = rsi_group.get("value") if rsi_group else None
    macd_hist = macd_group.get("histogram") if macd_group else None

    if rsi_val is not None and rsi_val > 70:
        momentum = "bearish"
    elif rsi_val is not None and rsi_val < 30:
        momentum = "bullish"
    elif macd_hist is not None and macd_hist > 0:
        momentum = "bullish"
    elif macd_hist is not None and macd_hist < 0:
        momentum = "bearish"
    else:
        momentum = "neutral"

    # Volatility from ATR + Bollinger width
    atr_group = indicator_dicts.get("atr")
    bb_group = indicator_dicts.get("bollinger")
    atr_val = atr_group.get("value") if atr_group else None
    bb_upper = bb_group.get("upper") if bb_group else None
    bb_lower = bb_group.get("lower") if bb_group else None

    if bb_upper is not None and bb_lower is not None and current_price > 0:
        bb_width = (bb_upper - bb_lower) / current_price
        if bb_width > 0.10:
            volatility = "high"
        elif bb_width < 0.04:
            volatility = "low"
        else:
            volatility = "normal"
    elif atr_val is not None and current_price > 0:
        atr_pct = atr_val / current_price
        if atr_pct > 0.04:
            volatility = "high"
        elif atr_pct < 0.015:
            volatility = "low"
        else:
            volatility = "normal"
    else:
        volatility = "normal"

    # Volume confirmation
    vol_group = indicator_dicts.get("volume_analysis")
    vol_ratio = vol_group.get("ratio") if vol_group else None
    volume_confirmation = vol_ratio is not None and vol_ratio > 1.5

    signals = {
        "trend": trend,
        "momentum": momentum,
        "volatility": volatility,
        "volume_confirmation": volume_confirmation,
    }

    confidence = _calc_confidence(
        indicator_dicts, len(ohlcv_df),
        phase1_ok=phase1_ok, phase2_ok=phase2_ok,
        data_as_of=data_as_of,
    )

    effective_as_of = data_as_of.isoformat() if data_as_of else datetime.now(timezone.utc).isoformat()

    technical_analysis = {
        "indicator_summary": indicator_summary,
        "pattern_summary": pattern_summary,
        "signals": signals,
        "support_levels": [float(s) for s in support_levels],
        "resistance_levels": [float(r) for r in resistance_levels],
        "confidence": round(confidence, 2),
        "data_as_of": effective_as_of,
        "data_source": "calculated",
    }

    logger.info(
        "agent_completed",
        component="technical_analysis",
        confidence=technical_analysis["confidence"],
        trend=trend,
        momentum=momentum,
        phase1_ok=phase1_ok,
        phase2_ok=phase2_ok,
    )

    result: dict = {"technical_analysis": technical_analysis}
    if failed_agents_modified:
        result["failed_agents"] = failed_agents
    return result
