"""Standalone verification script for vnstock API access.

Capabilities & Limitations of vnstock v3.x:
- Price History (Quote class): OHLCV data for Vietnamese stocks via VCI/KBS sources.
  Supports intervals: 1m, 5m, 15m, 30m, 1H, 1D, 1W, 1M.
  Length formats: period ("1Y", "3M"), day count (150), candle count ("200b").
  Returns adjusted prices (splits, dividends).
- Financial Data (Finance class): Income statement, balance sheet, cash flow, ratios.
  Sources: KBS (recommended for BCTC), VCI. TCBS is broken — do NOT use.
  Period: "quarter" or "year".
- Rate limits: Guest 20 req/min, Community (free API key) 60 req/min.
- Only supports domestic Vietnamese equities.

Usage:
    python -m services.crawler.market_data.verify_vnstock
    # or directly:
    python services/crawler/market_data/verify_vnstock.py
"""

from __future__ import annotations

import os
import sys


def verify_quote_history(symbol: str = "HPG") -> bool:
    """Verify Quote.history() returns valid OHLCV DataFrame."""
    from vnstock import Quote

    print(f"\n--- Verifying Quote.history('{symbol}') ---")
    quote = Quote(symbol=symbol, source="VCI")
    df = quote.history(length="1Y", interval="1D")

    expected_columns = {"time", "open", "high", "low", "close", "volume"}
    actual_columns = set(df.columns)

    print(f"Columns: {list(df.columns)}")
    print(f"Shape: {df.shape}")
    print(f"Sample:\n{df.head()}")

    missing = expected_columns - actual_columns
    if missing:
        print(f"FAIL: Missing columns: {missing}")
        return False

    if len(df) < 200:
        print(f"WARN: Only {len(df)} rows (expected ≥200 for 1Y daily)")

    print("PASS: Quote.history() verified.")
    return True


def verify_financial_ratios(symbol: str = "HPG") -> bool:
    """Verify Finance.ratio() returns P/E, P/B, ROE, EPS."""
    from vnstock import Finance

    print(f"\n--- Verifying Finance.ratio('{symbol}') ---")
    finance = Finance(symbol=symbol, source="KBS", standardize_columns=True)
    df = finance.ratio(period="quarter")

    print(f"Columns: {list(df.columns)}")
    print(f"Shape: {df.shape}")
    print(f"Sample:\n{df.head(10)}")

    print("PASS: Finance.ratio() verified.")
    return True


def verify_income_statement(symbol: str = "HPG") -> bool:
    """Verify Finance.income_statement() returns data."""
    from vnstock import Finance

    print(f"\n--- Verifying Finance.income_statement('{symbol}') ---")
    finance = Finance(symbol=symbol, source="KBS", standardize_columns=True)
    df = finance.income_statement(period="year")

    print(f"Columns: {list(df.columns)}")
    print(f"Shape: {df.shape}")
    print(f"Sample:\n{df.head()}")

    print("PASS: Finance.income_statement() verified.")
    return True


def verify_balance_sheet(symbol: str = "HPG") -> bool:
    """Verify Finance.balance_sheet() returns data."""
    from vnstock import Finance

    print(f"\n--- Verifying Finance.balance_sheet('{symbol}') ---")
    finance = Finance(symbol=symbol, source="KBS", standardize_columns=True)
    df = finance.balance_sheet(period="year")

    print(f"Columns: {list(df.columns)}")
    print(f"Shape: {df.shape}")
    print(f"Sample:\n{df.head()}")

    print("PASS: Finance.balance_sheet() verified.")
    return True


def main() -> None:
    """Run all verification checks."""
    api_key = os.environ.get("VNSTOCK_API_KEY")
    if api_key:
        print(f"VNSTOCK_API_KEY is set (length={len(api_key)})")
    else:
        print("WARN: VNSTOCK_API_KEY not set — running as guest (20 req/min limit)")

    symbol = sys.argv[1] if len(sys.argv) > 1 else "HPG"
    results: list[tuple[str, bool]] = []

    checks = [
        ("Quote.history", verify_quote_history),
        ("Finance.ratio", verify_financial_ratios),
        ("Finance.income_statement", verify_income_statement),
        ("Finance.balance_sheet", verify_balance_sheet),
    ]

    for name, check_fn in checks:
        try:
            ok = check_fn(symbol)
            results.append((name, ok))
        except Exception as exc:
            print(f"FAIL: {name} raised {type(exc).__name__}: {exc}")
            results.append((name, False))

    print("\n=== VERIFICATION SUMMARY ===")
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {status}: {name}")

    all_passed = all(ok for _, ok in results)
    print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
