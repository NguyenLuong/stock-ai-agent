"""Verification script for vnstock macro data capabilities.

Tests vnstock library for:
- VN-Index data (OHLCV via Quote history)
- USD/VND exchange rate (via fx module)
- Foreign net flow data (via proprietary trade data)

Documents which indicators vnstock covers vs needs fallback sources.

Usage:
    python -m services.crawler.macro.verify_macro_sources

VERIFICATION RESULTS (vnstock >= 3.0):
======================================
1. VN-Index (VNINDEX):
   - Source: vnstock Quote(symbol="VNINDEX", source="VCI").history()
   - Returns: OHLCV DataFrame with [time, open, high, low, close, volume]
   - Status: SUPPORTED by vnstock
   - Indicators: vn_index_close, vn_index_volume

2. USD/VND Exchange Rate:
   - Source: vnstock Fx(symbol="USDVND", source="MSN")
   - Returns: DataFrame with exchange rate data
   - Status: SUPPORTED by vnstock (MSN source)
   - Indicators: usd_vnd_rate

3. Foreign Net Flow:
   - Source: vnstock does NOT have a direct foreign_flow() method for market-wide data
   - Fallback: Use VNINDEX quote data volume as proxy, or dedicated scraping
   - Status: NOT DIRECTLY SUPPORTED — use mock fallback
   - Note: vnstock has per-stock foreign trading data but not market-wide aggregation
   - Indicators: foreign_net_flow (mock fallback)

4. SBV Interest Rate:
   - Source: sbv.gov.vn HTML scraping (vnstock does not cover)
   - Status: NOT SUPPORTED by vnstock — requires HTML scraping
   - Indicators: sbv_interest_rate
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from shared.logging import get_logger

logger = get_logger("verify_macro_sources")


@dataclass
class VerificationResult:
    """Result of verifying a single macro data source."""

    indicator_name: str
    source: str
    supported: bool
    notes: str
    sample_value: float | None = None
    error: str | None = None


@dataclass
class VerificationReport:
    """Aggregate verification results."""

    results: list[VerificationResult] = field(default_factory=list)

    @property
    def supported_count(self) -> int:
        return sum(1 for r in self.results if r.supported)

    @property
    def fallback_count(self) -> int:
        return sum(1 for r in self.results if not r.supported)

    def summary(self) -> str:
        lines = ["=" * 60, "MACRO DATA SOURCE VERIFICATION REPORT", "=" * 60]
        for r in self.results:
            status = "SUPPORTED" if r.supported else "FALLBACK NEEDED"
            lines.append(f"\n{r.indicator_name}:")
            lines.append(f"  Source: {r.source}")
            lines.append(f"  Status: {status}")
            if r.sample_value is not None:
                lines.append(f"  Sample value: {r.sample_value}")
            if r.error:
                lines.append(f"  Error: {r.error}")
            lines.append(f"  Notes: {r.notes}")
        lines.append(f"\nTotal: {self.supported_count} supported, "
                      f"{self.fallback_count} need fallback")
        return "\n".join(lines)


def verify_vn_index() -> VerificationResult:
    """Test vnstock for VN-Index data (Quote history for VNINDEX)."""
    try:
        from vnstock import Quote

        quote = Quote(symbol="VNINDEX", source="VCI")
        df = quote.history(length="1M", interval="1D")

        if df.empty:
            return VerificationResult(
                indicator_name="vn_index_close",
                source="vnstock",
                supported=False,
                notes="vnstock returned empty DataFrame for VNINDEX",
            )

        latest = df.iloc[-1]
        return VerificationResult(
            indicator_name="vn_index_close",
            source="vnstock",
            supported=True,
            sample_value=float(latest["close"]),
            notes=(
                f"VNINDEX OHLCV via Quote(symbol='VNINDEX', source='VCI').history(). "
                f"Columns: {list(df.columns)}. Rows: {len(df)}"
            ),
        )
    except Exception as exc:
        return VerificationResult(
            indicator_name="vn_index_close",
            source="vnstock",
            supported=False,
            error=str(exc),
            notes="Failed to fetch VN-Index data via vnstock Quote",
        )


def verify_exchange_rate() -> VerificationResult:
    """Test vnstock for USD/VND exchange rate (fx module)."""
    try:
        from vnstock import Fx

        fx = Fx(symbol="USDVND", source="MSN")
        df = fx.history(length="1M", interval="1D")

        if df.empty:
            return VerificationResult(
                indicator_name="usd_vnd_rate",
                source="vnstock",
                supported=False,
                notes="vnstock Fx returned empty DataFrame for USDVND",
            )

        latest = df.iloc[-1]
        # Fx history columns may vary — try 'close' or 'rate'
        rate_col = "close" if "close" in df.columns else df.columns[-1]
        return VerificationResult(
            indicator_name="usd_vnd_rate",
            source="vnstock",
            supported=True,
            sample_value=float(latest[rate_col]),
            notes=(
                f"USDVND via Fx(symbol='USDVND', source='MSN').history(). "
                f"Columns: {list(df.columns)}. Rows: {len(df)}"
            ),
        )
    except Exception as exc:
        return VerificationResult(
            indicator_name="usd_vnd_rate",
            source="vnstock",
            supported=False,
            error=str(exc),
            notes="Failed to fetch USD/VND rate via vnstock Fx",
        )


def verify_foreign_flow() -> VerificationResult:
    """Test vnstock for foreign net flow data.

    vnstock does not have a market-wide foreign flow endpoint.
    Per-stock foreign trading exists but market-wide aggregation is not available.
    This indicator will use mock fallback.
    """
    # vnstock does not provide market-wide foreign net flow
    # Document this finding as the verification result
    return VerificationResult(
        indicator_name="foreign_net_flow",
        source="vnstock",
        supported=False,
        notes=(
            "vnstock does not provide market-wide foreign net flow aggregation. "
            "Per-stock foreign trading data may exist via stock().finance, "
            "but VNINDEX-level foreign_net_flow is not available. "
            "Will use mock fallback with data_source='mock'."
        ),
    )


def verify_sbv_interest_rate() -> VerificationResult:
    """Document that SBV interest rate requires HTML scraping."""
    return VerificationResult(
        indicator_name="sbv_interest_rate",
        source="sbv.gov.vn",
        supported=False,
        notes=(
            "SBV interest rate (lai suat tai cap von) is not available via vnstock. "
            "Requires HTML scraping from sbv.gov.vn. "
            "SBV updates rates infrequently (not daily). "
            "Implemented in sbv_scraper.py."
        ),
    )


def run_verification() -> VerificationReport:
    """Run all macro data source verifications."""
    report = VerificationReport()

    logger.info("verification_started", component="verify_macro_sources")

    report.results.append(verify_vn_index())
    report.results.append(verify_exchange_rate())
    report.results.append(verify_foreign_flow())
    report.results.append(verify_sbv_interest_rate())

    logger.info(
        "verification_complete",
        component="verify_macro_sources",
        supported=report.supported_count,
        fallback=report.fallback_count,
    )

    return report


async def arun_verification() -> VerificationReport:
    """Async wrapper for run_verification."""
    return await asyncio.to_thread(run_verification)


if __name__ == "__main__":
    report = run_verification()
    print(report.summary())
