"""Market Context Agent node — 3-phase sequential analysis of macro + stock news."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from services.crawler.news.article_repo import semantic_search
from shared.llm.client import LLMCallError, LLMClient
from shared.llm.config_loader import get_config_loader
from shared.llm.prompt_loader import load_prompt
from shared.logging import get_logger

from services.app.agents.state import MarketContextState

logger = get_logger("market_context_agent")

MAX_ARTICLE_AGE_HOURS = 12
MAX_MACRO_ARTICLES = 20
MAX_STOCK_ARTICLES = 15
# Approximate token budget per LLM call (chars ≈ tokens * 4)
MAX_INPUT_CHARS = 24_000  # ~6000 tokens
CONFIDENCE_BASE = 0.50


def _filter_recent(articles: list, hours: int = MAX_ARTICLE_AGE_HOURS) -> list:
    """Keep only articles published within the last *hours*.

    Accepts ORM Article objects (attribute access).
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    result = []
    for a in articles:
        pa = getattr(a, "published_at", None)
        if pa is not None and pa >= cutoff:
            result.append(a)
    return result


def _truncate_articles(
    articles: list,
    max_chars: int = MAX_INPUT_CHARS,
) -> tuple[list[dict], bool]:
    """Convert ORM articles to dicts, truncating raw_content if total exceeds budget.

    Returns (article_dicts, was_truncated).
    """
    dicts: list[dict] = []
    total_chars = 0
    truncated = False

    for a in articles:
        title = a.title or ""
        summary = a.summary or ""
        raw_content = a.raw_content or ""
        source = a.source or ""
        published_at = str(a.published_at) if a.published_at else ""

        entry_overhead = len(title) + len(summary) + len(source) + len(published_at) + 50
        remaining = max_chars - total_chars - entry_overhead

        if remaining <= 0:
            # Budget exhausted — still include article with title+summary (no raw_content)
            # so the LLM sees the headline context per AC #4.
            truncated = True
            dicts.append({
                "title": title,
                "source": source,
                "published_at": published_at,
                "raw_content": "",
                "summary": summary,
            })
            total_chars += entry_overhead
            continue

        if len(raw_content) > remaining:
            raw_content = raw_content[:remaining]
            truncated = True

        d = {
            "title": title,
            "source": source,
            "published_at": published_at,
            "raw_content": raw_content,
            "summary": summary,
        }
        dicts.append(d)
        total_chars += entry_overhead + len(raw_content)

    return dicts, truncated


def _calc_confidence(articles: list, phase_name: str) -> float:
    """Rule-based confidence score for a single phase."""
    if not articles:
        return 0.0

    score = CONFIDENCE_BASE
    count = len(articles)

    if count >= 5:
        score += 0.15
    elif count >= 3:
        score += 0.10
    elif count >= 1:
        score += 0.05

    sources = {a.get("source") if isinstance(a, dict) else getattr(a, "source", None) for a in articles}
    sources.discard(None)
    if len(sources) >= 2:
        score += 0.10

    # Recency bonus — check most recent article
    now = datetime.now(timezone.utc)
    most_recent = None
    for a in articles:
        pa = a.get("published_at") if isinstance(a, dict) else getattr(a, "published_at", None)
        if pa is None:
            continue
        if isinstance(pa, str):
            try:
                pa = datetime.fromisoformat(pa)
            except ValueError:
                continue
        if most_recent is None or pa > most_recent:
            most_recent = pa

    if most_recent is not None:
        age = now - most_recent
        if age < timedelta(hours=4):
            score += 0.10
        elif age < timedelta(hours=12):
            score += 0.05

    return min(score, 1.0)


async def market_context_node(state: MarketContextState) -> dict:
    """LangGraph node: sequential macro → stock → combine pipeline.

    Never raises — returns graceful degradation on failure.
    """
    ticker = state.get("ticker", "")
    analysis_date = state.get("analysis_date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))

    logger.info(
        "agent_started",
        component="market_context",
        ticker=ticker,
        analysis_date=analysis_date,
    )

    llm = LLMClient()
    config = get_config_loader()
    macro_summary: str | None = None
    stock_summary: str | None = None
    macro_article_dicts: list[dict] = []
    stock_article_dicts: list[dict] = []
    failed_agents: list[str] = list(state.get("failed_agents", []))

    # ── Phase 1: Macro Analysis ──────────────────────────────────────────
    try:
        logger.info("article_retrieval_started", component="market_context", phase="macro")
        raw_macro = await semantic_search(
            "thị trường vĩ mô kinh tế", top_k=MAX_MACRO_ARTICLES, category="macro",
        )
        recent_macro = _filter_recent(raw_macro)
        logger.info(
            "article_retrieval_completed",
            component="market_context",
            phase="macro",
            total=len(raw_macro),
            recent=len(recent_macro),
        )

        if recent_macro:
            macro_article_dicts, was_truncated = _truncate_articles(recent_macro)
            if was_truncated:
                logger.warning(
                    "articles_truncated",
                    component="market_context",
                    phase="macro",
                    article_count=len(macro_article_dicts),
                )

            macro_prompt = load_prompt(
                "market_context/macro_analysis",
                macro_articles=macro_article_dicts,
                analysis_date=analysis_date,
            )
            logger.info("llm_call_started", component="market_context", phase="macro")
            macro_summary = await llm.call(
                prompt=macro_prompt.text,
                model=config.get_model(macro_prompt.model_key),
                temperature=config.get_temperature(),
                component="market_context",
            )
            logger.info("llm_call_completed", component="market_context", phase="macro")
        else:
            macro_summary = "Không có tin tức vĩ mô mới trong 12 giờ qua"
    except LLMCallError as e:
        logger.error("market_context_llm_failed", component="market_context", phase="macro", error=str(e))
        macro_summary = None
        failed_agents.append("market_context_macro")
    except Exception as e:
        logger.error("market_context_failed", component="market_context", phase="macro", error=str(e))
        macro_summary = None
        failed_agents.append("market_context_macro")

    # ── Phase 2: Stock News Analysis ─────────────────────────────────────
    try:
        query = f"thị trường chứng khoán {ticker}" if ticker else "thị trường chứng khoán"
        logger.info("article_retrieval_started", component="market_context", phase="stock")
        raw_stock = await semantic_search(
            query, top_k=MAX_STOCK_ARTICLES, category="stock",
        )
        recent_stock = _filter_recent(raw_stock)
        logger.info(
            "article_retrieval_completed",
            component="market_context",
            phase="stock",
            total=len(raw_stock),
            recent=len(recent_stock),
        )

        if recent_stock:
            stock_article_dicts, was_truncated = _truncate_articles(recent_stock)
            if was_truncated:
                logger.warning(
                    "articles_truncated",
                    component="market_context",
                    phase="stock",
                    article_count=len(stock_article_dicts),
                )

            news_prompt = load_prompt(
                "market_context/news_analysis",
                articles=stock_article_dicts,
                query=query,
                analysis_date=analysis_date,
            )
            logger.info("llm_call_started", component="market_context", phase="stock")
            stock_summary = await llm.call(
                prompt=news_prompt.text,
                model=config.get_model(news_prompt.model_key),
                temperature=config.get_temperature(),
                component="market_context",
            )
            logger.info("llm_call_completed", component="market_context", phase="stock")
        else:
            stock_summary = "Không có tin tức chứng khoán mới trong 12 giờ qua"
    except LLMCallError as e:
        logger.error("market_context_llm_failed", component="market_context", phase="stock", error=str(e))
        stock_summary = None
        failed_agents.append("market_context_stock")
    except Exception as e:
        logger.error("market_context_failed", component="market_context", phase="stock", error=str(e))
        stock_summary = None
        failed_agents.append("market_context_stock")

    # ── Phase 3: Combine ─────────────────────────────────────────────────
    if macro_summary is None and stock_summary is None:
        logger.error("market_context_all_phases_failed", component="market_context")
        if "market_context" not in failed_agents:
            failed_agents.append("market_context")
        return {
            "market_summary": None,
            "error": "Market Context: both macro and stock phases failed",
            "failed_agents": failed_agents,
        }

    # Collect unique sources
    all_sources: set[str] = set()
    for d in macro_article_dicts + stock_article_dicts:
        if d.get("source"):
            all_sources.add(d["source"])

    # Collect affected sectors from LLM output text.
    # Keys = canonical sector names (matching macro_analysis.yaml prompt).
    # Values = alternative spellings / abbreviations that map to the same sector.
    _SECTOR_ALIASES: dict[str, list[str]] = {
        "ngân hàng": ["ngân hàng", "banking", "tín dụng", "lãi suất"],
        "bất động sản": ["bất động sản", "real estate", "bđs", "nhà đất"],
        "sản xuất": ["sản xuất", "manufacturing", "công nghiệp", "chế biến"],
        "công nghệ": ["công nghệ", "technology", "tech", "phần mềm", "it"],
        "năng lượng": ["năng lượng", "energy", "điện", "dầu khí", "xăng dầu"],
    }
    affected_sectors: list[str] = []
    combined_text = (macro_summary or "") + " " + (stock_summary or "")
    lower_text = combined_text.lower()
    for canonical, aliases in _SECTOR_ALIASES.items():
        if any(alias in lower_text for alias in aliases):
            affected_sectors.append(canonical)

    # Confidence: average of phases that had real articles.
    # When no articles were retrieved for a phase its confidence is 0 and excluded
    # from the average so that a "no data" response isn't confused with low-quality analysis.
    macro_conf = _calc_confidence(macro_article_dicts, "macro")
    stock_conf = _calc_confidence(stock_article_dicts, "stock")
    phase_scores = [s for s in [macro_conf, stock_conf] if s > 0]
    confidence = sum(phase_scores) / len(phase_scores) if phase_scores else 0.0

    data_as_of = datetime.now(timezone.utc).isoformat()

    market_summary = {
        "macro_summary": macro_summary,
        "stock_summary": stock_summary,
        "affected_sectors": affected_sectors,
        "confidence": round(confidence, 2),
        "data_as_of": data_as_of,
        "sources": sorted(all_sources),
    }

    logger.info(
        "agent_completed",
        component="market_context",
        confidence=market_summary["confidence"],
        sectors=len(affected_sectors),
        sources=len(all_sources),
    )

    result: dict = {"market_summary": market_summary}
    if failed_agents != list(state.get("failed_agents", [])):
        result["failed_agents"] = failed_agents
    return result
