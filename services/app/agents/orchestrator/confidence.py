"""Rule-based confidence scoring for orchestrator synthesis.

Score is 0.0–1.0, calculated from agent outputs — NO LLM involved.
Display: 🟢 ≥0.70 / 🟡 0.40–0.69 / 🔴 <0.40
"""

from __future__ import annotations

from datetime import datetime, timezone

from services.app.agents.state import OrchestratorState
from shared.logging import get_logger

logger = get_logger("orchestrator")

# Weight table (must sum to 1.0)
_W_COMPLETENESS = 0.25
_W_AGREEMENT = 0.25
_W_FRESHNESS = 0.20
_W_SIGNAL_STRENGTH = 0.15
_W_FUNDAMENTAL_HEALTH = 0.15

# Penalty per missing agent
_FAIL_PENALTY = 0.20

_ALL_AGENTS = ("market_context", "technical_analysis", "fundamental_analysis")


def calculate_confidence(state: OrchestratorState) -> float:
    """Return a rule-based confidence score in [0.0, 1.0]."""
    failed = set(state.get("failed_agents", []))
    available_count = len(_ALL_AGENTS) - len(failed)

    if available_count == 0:
        return 0.0

    # --- Factor 1: Data completeness (0–1) ---
    completeness = _calc_completeness(state, failed)

    # --- Factor 2: Agent agreement (0–1) ---
    agreement = _calc_agreement(state, failed)

    # --- Factor 3: Data freshness (0–1) ---
    freshness = _calc_freshness(state, failed)

    # --- Factor 4: Signal strength from technical (0–1) ---
    signal_strength = _read_confidence(state.get("technical_analysis"), "technical_analysis", failed)

    # --- Factor 5: Fundamental health (0–1) ---
    fundamental_health = _read_confidence(state.get("fundamental_analysis"), "fundamental_analysis", failed)

    # Redistribute weights when agents fail
    weights = _redistribute_weights(failed)

    raw_score = (
        weights["completeness"] * completeness
        + weights["agreement"] * agreement
        + weights["freshness"] * freshness
        + weights["signal_strength"] * signal_strength
        + weights["fundamental_health"] * fundamental_health
    )

    # Apply failure penalty
    penalty = len(failed) * _FAIL_PENALTY
    score = max(0.0, min(1.0, raw_score - penalty))

    logger.debug(
        "confidence_calculated",
        component="orchestrator",
        raw_score=round(raw_score, 3),
        penalty=round(penalty, 3),
        final_score=round(score, 3),
        failed_agents=list(failed),
        factors={
            "completeness": round(completeness, 3),
            "agreement": round(agreement, 3),
            "freshness": round(freshness, 3),
            "signal_strength": round(signal_strength, 3),
            "fundamental_health": round(fundamental_health, 3),
        },
    )

    return round(score, 3)


def confidence_display(score: float) -> str:
    """Return emoji indicator for confidence score."""
    if score >= 0.70:
        return f"🟢 {score * 100:.0f}%"
    if score >= 0.40:
        return f"🟡 {score * 100:.0f}%"
    return f"🔴 {score * 100:.0f}%"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _calc_completeness(state: OrchestratorState, failed: set[str]) -> float:
    """Score based on how many agents succeeded and key fields are present."""
    available = len(_ALL_AGENTS) - len(failed)
    base = available / len(_ALL_AGENTS)

    # Check key fields are not None
    field_checks = 0
    total_checks = 0

    ms = state.get("market_summary")
    if ms and "market_context" not in failed:
        total_checks += 2
        if ms.get("macro_summary") is not None:
            field_checks += 1
        if ms.get("stock_summary") is not None:
            field_checks += 1

    ta = state.get("technical_analysis")
    if ta and "technical_analysis" not in failed:
        total_checks += 2
        if ta.get("indicator_summary") is not None:
            field_checks += 1
        if ta.get("signals") is not None:
            field_checks += 1

    fa = state.get("fundamental_analysis")
    if fa and "fundamental_analysis" not in failed:
        total_checks += 2
        if fa.get("bctc_summary") is not None:
            field_checks += 1
        if fa.get("signals") is not None:
            field_checks += 1

    field_ratio = field_checks / total_checks if total_checks > 0 else 0.0
    return (base + field_ratio) / 2


def _calc_agreement(state: OrchestratorState, failed: set[str]) -> float:
    """Compare technical trend/momentum vs fundamental valuation/growth signals."""
    ta = state.get("technical_analysis")
    fa = state.get("fundamental_analysis")

    if (
        "technical_analysis" in failed
        or "fundamental_analysis" in failed
        or not ta
        or not fa
    ):
        return 0.5  # Neutral when we can't compare

    ta_signals = ta.get("signals", {})
    fa_signals = fa.get("signals", {})

    # Map signals to sentiment: bullish(1), neutral(0), bearish(-1)
    ta_sentiment = _signal_to_sentiment(ta_signals.get("trend"), ta_signals.get("momentum"))
    fa_sentiment = _fundamental_to_sentiment(fa_signals.get("valuation"), fa_signals.get("growth"))

    diff = abs(ta_sentiment - fa_sentiment)
    # diff 0 → 1.0, diff 1 → 0.5, diff 2 → 0.0
    return max(0.0, 1.0 - diff * 0.5)


def _signal_to_sentiment(trend: str | None, momentum: str | None) -> float:
    """Convert technical signals to a -1..1 sentiment score."""
    score = 0.0
    trend_map = {"uptrend": 1.0, "sideways": 0.0, "downtrend": -1.0}
    momentum_map = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}
    score += trend_map.get(trend or "", 0.0)
    score += momentum_map.get(momentum or "", 0.0)
    return max(-1.0, min(1.0, score / 2))


def _fundamental_to_sentiment(valuation: str | None, growth: str | None) -> float:
    """Convert fundamental signals to a -1..1 sentiment score."""
    score = 0.0
    val_map = {"undervalued": 1.0, "fair": 0.0, "overvalued": -1.0}
    growth_map = {"growing": 1.0, "stable": 0.0, "declining": -1.0}
    score += val_map.get(valuation or "", 0.0)
    score += growth_map.get(growth or "", 0.0)
    return max(-1.0, min(1.0, score / 2))


def _calc_freshness(state: OrchestratorState, failed: set[str]) -> float:
    """Score based on data_as_of age. <1h=1.0, 1-4h=0.7, >4h=0.4."""
    now = datetime.now(timezone.utc)
    scores: list[float] = []

    for key, agent_name in [
        ("market_summary", "market_context"),
        ("technical_analysis", "technical_analysis"),
        ("fundamental_analysis", "fundamental_analysis"),
    ]:
        if agent_name in failed:
            continue
        output = state.get(key)
        if not output:
            continue
        data_as_of = output.get("data_as_of")
        if not data_as_of:
            scores.append(0.4)
            continue
        try:
            ts = datetime.fromisoformat(data_as_of)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            age_hours = (now - ts).total_seconds() / 3600
            if age_hours < 1:
                scores.append(1.0)
            elif age_hours < 4:
                scores.append(0.7)
            else:
                scores.append(0.4)
        except (ValueError, TypeError):
            scores.append(0.4)

    return sum(scores) / len(scores) if scores else 0.4


def _read_confidence(
    output: dict | None,
    agent_name: str,
    failed: set[str],
) -> float:
    """Read confidence field from an agent output dict."""
    if agent_name in failed or not output:
        return 0.0
    return float(output.get("confidence", 0.0))


def _redistribute_weights(failed: set[str]) -> dict[str, float]:
    """Redistribute weights when agents fail, keeping total = 1.0."""
    weights = {
        "completeness": _W_COMPLETENESS,
        "agreement": _W_AGREEMENT,
        "freshness": _W_FRESHNESS,
        "signal_strength": _W_SIGNAL_STRENGTH,
        "fundamental_health": _W_FUNDAMENTAL_HEALTH,
    }

    if "technical_analysis" in failed and "fundamental_analysis" not in failed:
        # Redistribute signal_strength weight
        weights["fundamental_health"] += weights["signal_strength"]
        weights["signal_strength"] = 0.0
    elif "fundamental_analysis" in failed and "technical_analysis" not in failed:
        weights["signal_strength"] += weights["fundamental_health"]
        weights["fundamental_health"] = 0.0
    elif "technical_analysis" in failed and "fundamental_analysis" in failed:
        extra = weights["signal_strength"] + weights["fundamental_health"]
        weights["signal_strength"] = 0.0
        weights["fundamental_health"] = 0.0
        weights["completeness"] += extra / 3
        weights["agreement"] += extra / 3
        weights["freshness"] += extra / 3

    return weights
