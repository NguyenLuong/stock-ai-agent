"""crawler.macro package — macro data crawlers for VN-Index, exchange rates, foreign flow, SBV rates."""

from services.crawler.macro.macro_crawl_manager import run_macro_crawl
from services.crawler.macro.macro_data_repo import save_macro_indicators
from services.crawler.macro.models import MacroCrawlResult, MacroDataResult
from services.crawler.macro.sbv_scraper import SBVScraper
from services.crawler.macro.vnstock_macro_client import VnstockMacroClient

__all__ = [
    "MacroCrawlResult",
    "MacroDataResult",
    "SBVScraper",
    "VnstockMacroClient",
    "run_macro_crawl",
    "save_macro_indicators",
]
