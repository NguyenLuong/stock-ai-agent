"""Macro crawl orchestrator — coordinates all macro data sources.

Fetches macro indicators sequentially:
1. VN-Index (vnstock)
2. USD/VND exchange rate (vnstock)
3. Foreign net flow (mock — vnstock doesn't support market-wide)
4. SBV interest rate (sbv.gov.vn scrape)

Graceful degradation: if one source fails, continue with others.
"""

from __future__ import annotations

import httpx

from macro.macro_data_repo import get_last_macro_fetch_time, save_macro_indicators
from macro.models import MacroCrawlResult, MacroDataResult
from macro.sbv_scraper import SBVScraper
from macro.vnstock_macro_client import VnstockMacroClient
from middleware.rate_limiter import RateLimitedTransport
from middleware.robots_checker import RobotsChecker
from shared.llm.config_loader import get_sources
from shared.logging import get_logger

logger = get_logger("macro_crawl_manager")


async def run_macro_crawl() -> MacroCrawlResult:
    """Orchestrate macro data collection from all sources."""
    results: list[MacroDataResult] = []

    # Load macro config from sources.yaml
    config = get_sources()
    macro_config = config.get("macro", {})

    if not macro_config:
        logger.warning(
            "macro_config_missing",
            component="macro_crawl_manager",
            detail="No 'macro' section found in sources.yaml — crawl will do nothing",
        )

    # 1. Fetch VN-Index (vnstock) — produces 2 indicators: close + volume
    vn_config = macro_config.get("vn_index", {})
    if vn_config.get("enabled", False):
        try:
            vn_client = VnstockMacroClient.get_instance()
            vn_results = await vn_client.aget_vn_index()
            results.extend(vn_results)
        except Exception as exc:
            logger.warning(
                "vn_index_crawl_failed",
                error=str(exc),
                component="macro_crawl_manager",
            )

    # 2. Fetch USD/VND exchange rate (vnstock)
    fx_config = macro_config.get("exchange_rate", {})
    if fx_config.get("enabled", False):
        try:
            vn_client = VnstockMacroClient.get_instance()
            results.append(await vn_client.aget_exchange_rate())
        except Exception as exc:
            logger.warning(
                "exchange_rate_crawl_failed",
                error=str(exc),
                component="macro_crawl_manager",
            )

    # 3. Fetch foreign net flow (mock — vnstock doesn't support)
    flow_config = macro_config.get("foreign_flow", {})
    if flow_config.get("enabled", False):
        try:
            vn_client = VnstockMacroClient.get_instance()
            results.append(await vn_client.aget_foreign_flow())
        except Exception as exc:
            logger.warning(
                "foreign_flow_crawl_failed",
                error=str(exc),
                component="macro_crawl_manager",
            )

    # 4. Fetch SBV interest rate (sbv.gov.vn scrape)
    sbv_config = macro_config.get("sbv_interest_rate", {})
    if sbv_config.get("enabled", False):
        rate_rps = sbv_config.get("rate_limit_rps", 1)
        base_url = sbv_config.get("base_url", "https://www.sbv.gov.vn")
        try:
            transport = RateLimitedTransport(rate_per_second=rate_rps)
            async with httpx.AsyncClient(
                transport=transport,
                headers={"User-Agent": "StockAIAgent/1.0"},
                timeout=30.0,
            ) as client:
                robots_checker = RobotsChecker(client=client)
                sbv = SBVScraper(
                    client=client,
                    robots_checker=robots_checker,
                    base_url=base_url,
                )
                results.append(await sbv.fetch_interest_rate())
        except Exception as exc:
            logger.warning(
                "sbv_crawl_failed",
                error=str(exc),
                component="macro_crawl_manager",
            )

    # 5. Convert successful results to MarketDataCreate and save
    successful = [r for r in results if r.success]
    market_data_creates = [r.to_market_data_create() for r in successful]
    saved = await save_macro_indicators(market_data_creates)

    # 6. Log summary
    failed = [r for r in results if not r.success]

    # AC3: When all sources fail, log last successful fetch timestamp as warning
    if results and not successful:
        last_fetch_time = await get_last_macro_fetch_time()
        logger.warning(
            "all_macro_sources_failed",
            component="macro_crawl_manager",
            last_successful_fetch=last_fetch_time.isoformat() if last_fetch_time else "never",
            failed_indicators=[r.indicator_name for r in failed],
        )

    logger.info(
        "macro_crawl_complete",
        component="macro_crawl_manager",
        total=len(results),
        succeeded=len(successful),
        failed=len(failed),
        saved=saved,
        failed_indicators=[r.indicator_name for r in failed],
    )

    return MacroCrawlResult(
        results=results,
        saved_count=saved,
    )
