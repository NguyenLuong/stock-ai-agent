"""crawler.macro package — macro data crawlers for VN-Index, exchange rates, foreign flow, SBV rates."""

from macro.macro_crawl_manager import run_macro_crawl
from macro.macro_data_repo import save_macro_indicators
from macro.models import MacroCrawlResult, MacroDataResult
from macro.sbv_scraper import SBVScraper
from macro.vnstock_macro_client import VnstockMacroClient

__all__ = [
    "MacroCrawlResult",
    "MacroDataResult",
    "SBVScraper",
    "VnstockMacroClient",
    "run_macro_crawl",
    "save_macro_indicators",
]
