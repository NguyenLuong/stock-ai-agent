"""Format synthesis_result into a Telegram-ready morning briefing string."""

from __future__ import annotations

import re

from shared.utils.datetime_utils import now_utc, to_vn_display

_RECOMMENDATION_KEYWORDS: dict[str, list[str]] = {
    "MUA": ["mua", "tăng trưởng", "khuyến nghị mua", "tích lũy"],
    "BÁN": ["bán", "giảm", "thoát hàng", "khuyến nghị bán"],
    "THEO DÕI": ["theo dõi", "trung lập", "chưa đủ dữ liệu", "cần thêm"],
}


def _parse_recommendation(synthesis_text: str) -> str:
    text_lower = synthesis_text.lower()
    for rec, keywords in _RECOMMENDATION_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return rec
    return "THEO DÕI"


def _parse_confidence_value(confidence_display: str) -> float:
    match = re.search(r"\((\d+)%\)", confidence_display)
    return int(match.group(1)) / 100.0 if match else 0.5


def _confidence_emoji(confidence: float) -> str:
    if confidence >= 0.70:
        return "🟢"
    elif confidence >= 0.40:
        return "🟡"
    return "🔴"


def format_morning_briefing(synthesis_result: dict, ticker: str) -> str:
    """Build a Telegram-formatted morning briefing from synthesis_result."""
    briefing_date = to_vn_display(now_utc())
    synthesis = synthesis_result["synthesis"]

    # Confidence section — confidence_display đã chứa emoji, không prepend thêm
    confidence_display = synthesis_result["confidence_display"]
    confidence_section = f"Độ tin cậy: {confidence_display}"

    # Risk section
    risk = synthesis_result["risk_assessment"]
    risk_section = f"⚖️ Rủi ro: {risk['level'].upper()} — {risk['reasoning']}"

    # Stop-loss section (optional)
    stop_loss = synthesis_result.get("stop_loss_suggestion")
    stop_loss_section = ""
    if stop_loss is not None:
        stop_loss_section = f"\n🛑 Stop-loss gợi ý: {stop_loss:,.0f}"

    # Alerts
    alerts: list[str] = (
        synthesis_result.get("stale_warnings", [])
        + synthesis_result.get("unavailable_warnings", [])
        + synthesis_result.get("integrity_violations", [])
    )
    alerts_section = ""
    if alerts:
        bullets = "\n".join(f"• {a}" for a in alerts)
        alerts_section = f"\n\n⚠️ *CẢNH BÁO*\n{bullets}"

    # Disclaimer
    disclaimer = synthesis_result.get("disclaimer", "")

    # Assemble
    parts = [
        f"📊 *MORNING BRIEFING — {briefing_date}*",
        f"*{ticker}*",
        "",
        "📋 *TÓM TẮT*",
        synthesis[:300].rsplit(" ", 1)[0] if len(synthesis) > 300 else synthesis,
        "",
        "🔍 *PHÂN TÍCH CHI TIẾT*",
        synthesis,
        "",
        confidence_section,
        risk_section,
    ]

    if stop_loss_section:
        parts.append(stop_loss_section.lstrip("\n"))

    if alerts_section:
        parts.append(alerts_section.lstrip("\n"))

    parts.append("")
    parts.append(f"_{disclaimer}_")

    return "\n".join(parts)
