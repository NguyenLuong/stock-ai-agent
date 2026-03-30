"""Data integrity guard — hallucination prevention and traceability checks.

All functions are pure, rule-based Python (no LLM calls).
State is read-only — never mutated.
"""

from __future__ import annotations

from services.app.agents.state import OrchestratorState
from shared.logging import get_logger

logger = get_logger("orchestrator")

_FORBIDDEN_PHRASES = [
    "chắc chắn",
    "đảm bảo",
    "nhất định",
    "guaranteed",
    "100%",
    "sẽ tăng mạnh",
    "sẽ giảm mạnh",
    "không thể sai",
    "không có rủi ro",
]


def validate_no_absolute_certainty(text: str) -> list[str]:
    """Scan text for forbidden absolute-certainty phrases.

    Returns list of violation strings (empty if clean).
    """
    if not text:
        return []
    lower = text.lower()
    violations: list[str] = []
    for phrase in _FORBIDDEN_PHRASES:
        if phrase.lower() in lower:
            violations.append(f"Phát hiện ngôn ngữ tuyệt đối: \"{phrase}\"")
    return violations


def compute_risk_assessment(state: OrchestratorState, confidence: float) -> dict:
    """Rule-based risk assessment. No LLM involved."""
    failed = state.get("failed_agents", [])
    ta = state.get("technical_analysis")
    volatility = ta.get("signals", {}).get("volatility", "") if ta else ""

    # Priority rules (high > medium > low)
    if confidence < 0.40 or volatility == "high" or len(failed) >= 2:
        level = "high"
    elif confidence < 0.70 or len(failed) == 1:
        level = "medium"
    else:
        level = "low"

    reasons: list[str] = []
    if confidence < 0.40:
        reasons.append(f"độ tin cậy thấp ({confidence:.0%})")
    elif 0.40 <= confidence < 0.70:
        reasons.append(f"độ tin cậy trung bình ({confidence:.0%})")
    if volatility == "high":
        reasons.append("biến động giá cao")
    if len(failed) >= 2:
        reasons.append(f"{len(failed)} agents không khả dụng")
    elif len(failed) == 1:
        reasons.append(f"1 agent không khả dụng ({failed[0]})")
    if not reasons:
        reasons.append(f"độ tin cậy tốt ({confidence:.0%}), không có dấu hiệu rủi ro cao")

    return {
        "level": level,
        "reasoning": "; ".join(reasons),
        "confidence_score": confidence,
        "failed_agents": list(failed),
    }


def compute_stop_loss(state: OrchestratorState) -> float | None:
    """Return nearest support level as stop-loss suggestion.

    support_levels is sorted ascending — [-1] is closest to current price.
    Returns None if data unavailable.
    """
    failed = set(state.get("failed_agents", []))
    if "technical_analysis" in failed:
        return None
    ta = state.get("technical_analysis")
    if not ta:
        return None
    support_levels = ta.get("support_levels", [])
    if not support_levels:
        return None
    return float(support_levels[-1])


def check_data_traceability(data_sources: list[dict]) -> list[str]:
    """Verify each data source entry has both 'source' and 'data_as_of'.

    Returns one warning per agent with missing fields, combining both issues.
    """
    warnings: list[str] = []
    for ds in data_sources:
        agent = ds.get("agent", "unknown")
        missing: list[str] = []
        if not ds.get("source"):
            missing.append("source")
        if not ds.get("data_as_of"):
            missing.append("data_as_of")
        if missing:
            warnings.append(f"⚠️ {agent}: thiếu {', '.join(missing)}")
    return warnings


def audit_null_fields(state: OrchestratorState) -> list[str]:
    """Return list of company_ratios field names that are None (unconfirmed).

    Used to verify synthesis text acknowledges missing data.
    """
    failed = set(state.get("failed_agents", []))
    if "fundamental_analysis" in failed:
        return []
    fa = state.get("fundamental_analysis")
    if not fa:
        return []
    ratios = fa.get("company_ratios", {})
    if not ratios:
        return []
    return [field for field, value in ratios.items() if value is None]
