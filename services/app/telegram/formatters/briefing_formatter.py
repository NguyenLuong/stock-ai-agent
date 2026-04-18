"""Format market_result into a Telegram-ready morning briefing string."""

from __future__ import annotations

from shared.utils.datetime_utils import now_utc, to_vn_display


def _signal_emoji(signal: str) -> str:
    if signal in ("uptrend", "bullish"):
        return "🟢"
    if signal in ("downtrend", "bearish"):
        return "🔴"
    return "🟡"


def _sentiment_emoji(sentiment: str) -> str:
    if sentiment == "bullish":
        return "📈"
    if sentiment == "bearish":
        return "📉"
    return "➡️"


def _escape_md(text: str) -> str:
    """Escape Telegram MarkdownV1 special characters in user-generated text."""
    for ch in ("*", "_", "`", "["):
        text = text.replace(ch, f"\\{ch}")
    return text


def format_morning_briefing(market_result: dict) -> str:
    """Build a Telegram-formatted morning briefing from market_result.

    market_result structure (from morning_briefing_graph):
        market_sentiment, affected_sectors, key_events, top_picks,
        market_summary, stale_warnings, unavailable_warnings,
        disclaimer, generated_at, pipeline_status, abort_reason.
    """
    pipeline_status = market_result.get("pipeline_status", "ok")
    abort_reason = market_result.get("abort_reason", "")

    if pipeline_status == "aborted":
        briefing_date = to_vn_display(now_utc())
        sentiment = market_result.get("market_sentiment", "neutral")
        key_events = market_result.get("key_events", [])
        summary = market_result.get("market_summary", "")
        disclaimer = market_result.get("disclaimer", "")

        reason_text = {
            "no_sectors_identified": "Market Context không xác định được ngành bị ảnh hưởng",
            "sectors_not_in_watchlist": "Các ngành bị ảnh hưởng không có trong danh mục theo dõi",
        }.get(abort_reason, "Không thể xác định tín hiệu thị trường")

        parts: list[str] = [
            f"📊 *MORNING BRIEFING — {briefing_date}*",
            f"{_sentiment_emoji(sentiment)} Tâm lý thị trường: *{sentiment.upper()}*",
            "",
            f"⚠️ *{reason_text}*",
            "Phân tích kỹ thuật và cơ bản không được thực thi hôm nay.",
        ]
        if summary:
            parts += ["", "📋 *TÓM TẮT VĨ MÔ*", _escape_md(summary[:500])]
        if key_events:
            parts += ["", "📰 *SỰ KIỆN CHÍNH*"]
            for event in key_events[:5]:
                parts.append(f"• {_escape_md(event)}")
        parts += ["", f"_{disclaimer}_"]
        return "\n".join(parts)

    briefing_date = to_vn_display(now_utc())
    sentiment = market_result.get("market_sentiment", "neutral")
    affected_sectors = market_result.get("affected_sectors", [])
    key_events = market_result.get("key_events", [])
    top_picks = market_result.get("top_picks", [])
    summary = market_result.get("market_summary", "")
    stale_warnings = market_result.get("stale_warnings", [])
    unavailable_warnings = market_result.get("unavailable_warnings", [])
    disclaimer = market_result.get("disclaimer", "")

    parts: list[str] = [
        f"📊 *MORNING BRIEFING — {briefing_date}*",
        f"{_sentiment_emoji(sentiment)} Tâm lý thị trường: *{sentiment.upper()}*",
    ]

    # Sectors
    if affected_sectors:
        sectors_str = ", ".join(affected_sectors)
        parts.append(f"🏭 Ngành ảnh hưởng: {sectors_str}")

    parts.append("")

    # Summary
    parts.append("📋 *TÓM TẮT*")
    if summary:
        parts.append(_escape_md(summary[:500]))
    parts.append("")

    # Key events
    if key_events:
        parts.append("📰 *SỰ KIỆN CHÍNH*")
        for event in key_events[:5]:
            parts.append(f"• {_escape_md(event)}")
        parts.append("")

    # Top picks
    if top_picks:
        parts.append("🎯 *TOP PICKS*")
        for pick in top_picks:
            emoji = _signal_emoji(pick.get("signal", "sideways"))
            conf = pick.get("confidence", 0)
            ticker = pick.get("ticker", "")
            pick_summary = _escape_md(pick.get("summary", "")[:100])
            parts.append(
                f"{emoji} *{ticker}* — {pick.get('signal', 'N/A')} "
                f"(conf: {conf:.0%})"
            )
            if pick_summary:
                parts.append(f"  {pick_summary}")
    else:
        parts.append("ℹ️ Không có tín hiệu nổi bật hôm nay")

    # Alerts
    alerts = stale_warnings + unavailable_warnings
    if alerts:
        parts.append("")
        parts.append("⚠️ *CẢNH BÁO*")
        for a in alerts:
            parts.append(f"• {a}")

    parts.append("")
    parts.append(f"_{disclaimer}_")

    return "\n".join(parts)
