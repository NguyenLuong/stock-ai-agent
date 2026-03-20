"""VnstockClient — wrapper around vnstock library for Vietnamese market data.

Provides stock price history, financial ratios, income statements, and balance sheets
with retry logic and mock data fallback on API failure.
"""

from __future__ import annotations

import asyncio

import pandas as pd
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from shared.logging import get_logger

# Transient network exceptions worth retrying
_RETRYABLE_EXCEPTIONS = (ConnectionError, TimeoutError, OSError)

logger = get_logger("vnstock_client")

# Default source for price data (VCI recommended)
DEFAULT_QUOTE_SOURCE = "VCI"
# Default source for financial data (KBS recommended for BCTC)
DEFAULT_FINANCE_SOURCE = "KBS"


def _log_retry_attempt(retry_state: RetryCallState) -> None:
    """Log each retry attempt."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(
        "vnstock_api_retry",
        attempt=retry_state.attempt_number,
        error=str(exc),
        component="vnstock_client",
    )


class VnstockAPIError(Exception):
    """Raised when vnstock API calls fail after all retries."""

    def __init__(self, message: str, last_exception: Exception | None = None) -> None:
        super().__init__(message)
        self.last_exception = last_exception


class VnstockClient:
    """Client for fetching Vietnamese stock market data via vnstock.

    Wraps the vnstock library with retry logic and mock data fallback.
    All public methods are synchronous (vnstock is sync); use asyncio.to_thread()
    when calling from async context.
    """

    def __init__(
        self,
        quote_source: str = DEFAULT_QUOTE_SOURCE,
        finance_source: str = DEFAULT_FINANCE_SOURCE,
    ) -> None:
        self._quote_source = quote_source
        self._finance_source = finance_source

    @retry(
        wait=wait_exponential(min=1, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
        reraise=True,
        before_sleep=_log_retry_attempt,
    )
    def _fetch_history(self, symbol: str, length: str, interval: str) -> pd.DataFrame:
        """Fetch price history with retry."""
        from vnstock import Quote

        quote = Quote(symbol=symbol, source=self._quote_source)
        return quote.history(length=length, interval=interval)

    @retry(
        wait=wait_exponential(min=1, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
        reraise=True,
        before_sleep=_log_retry_attempt,
    )
    def _fetch_ratios(self, symbol: str, period: str) -> pd.DataFrame:
        """Fetch financial ratios with retry."""
        from vnstock import Finance

        finance = Finance(
            symbol=symbol, source=self._finance_source, standardize_columns=True
        )
        return finance.ratio(period=period)

    @retry(
        wait=wait_exponential(min=1, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
        reraise=True,
        before_sleep=_log_retry_attempt,
    )
    def _fetch_income_statement(self, symbol: str, period: str) -> pd.DataFrame:
        """Fetch income statement with retry."""
        from vnstock import Finance

        finance = Finance(
            symbol=symbol, source=self._finance_source, standardize_columns=True
        )
        return finance.income_statement(period=period)

    @retry(
        wait=wait_exponential(min=1, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
        reraise=True,
        before_sleep=_log_retry_attempt,
    )
    def _fetch_balance_sheet(self, symbol: str, period: str) -> pd.DataFrame:
        """Fetch balance sheet with retry."""
        from vnstock import Finance

        finance = Finance(
            symbol=symbol, source=self._finance_source, standardize_columns=True
        )
        return finance.balance_sheet(period=period)

    def get_stock_history(
        self,
        ticker: str,
        length: str = "1Y",
        interval: str = "1D",
    ) -> pd.DataFrame:
        """Fetch OHLCV stock price history.

        Returns DataFrame with columns [time, open, high, low, close, volume].
        Falls back to mock data on API failure with data_source="mock" metadata.
        """
        try:
            df = self._fetch_history(ticker, length, interval)
            df.attrs["data_source"] = "vnstock"
            logger.info(
                "stock_history_fetched",
                ticker=ticker,
                rows=len(df),
                data_source="vnstock",
                component="vnstock_client",
            )
            return df
        except Exception as exc:
            logger.warning(
                "stock_history_fallback_to_mock",
                ticker=ticker,
                error=str(exc),
                component="vnstock_client",
            )
            from market_data.mock_data import (
                generate_mock_stock_price,
            )

            return generate_mock_stock_price(ticker)

    def get_financial_ratios(
        self,
        ticker: str,
        period: str = "quarter",
    ) -> pd.DataFrame:
        """Fetch financial ratios (P/E, P/B, ROE, EPS).

        Falls back to mock data on API failure.
        """
        try:
            df = self._fetch_ratios(ticker, period)
            df.attrs["data_source"] = "vnstock"
            logger.info(
                "financial_ratios_fetched",
                ticker=ticker,
                rows=len(df),
                data_source="vnstock",
                component="vnstock_client",
            )
            return df
        except Exception as exc:
            logger.warning(
                "financial_ratios_fallback_to_mock",
                ticker=ticker,
                error=str(exc),
                component="vnstock_client",
            )
            from market_data.mock_data import (
                generate_mock_financial_ratios,
            )

            return generate_mock_financial_ratios(ticker)

    def get_income_statement(
        self,
        ticker: str,
        period: str = "year",
    ) -> pd.DataFrame:
        """Fetch income statement data.

        Falls back to empty DataFrame with data_source="mock" on API failure.
        """
        try:
            df = self._fetch_income_statement(ticker, period)
            df.attrs["data_source"] = "vnstock"
            logger.info(
                "income_statement_fetched",
                ticker=ticker,
                rows=len(df),
                data_source="vnstock",
                component="vnstock_client",
            )
            return df
        except Exception as exc:
            logger.warning(
                "income_statement_fallback_to_mock",
                ticker=ticker,
                error=str(exc),
                component="vnstock_client",
            )
            df = pd.DataFrame()
            df.attrs["data_source"] = "mock"
            return df

    def get_balance_sheet(
        self,
        ticker: str,
        period: str = "year",
    ) -> pd.DataFrame:
        """Fetch balance sheet data.

        Falls back to empty DataFrame with data_source="mock" on API failure.
        """
        try:
            df = self._fetch_balance_sheet(ticker, period)
            df.attrs["data_source"] = "vnstock"
            logger.info(
                "balance_sheet_fetched",
                ticker=ticker,
                rows=len(df),
                data_source="vnstock",
                component="vnstock_client",
            )
            return df
        except Exception as exc:
            logger.warning(
                "balance_sheet_fallback_to_mock",
                ticker=ticker,
                error=str(exc),
                component="vnstock_client",
            )
            df = pd.DataFrame()
            df.attrs["data_source"] = "mock"
            return df

    async def aget_stock_history(
        self,
        ticker: str,
        length: str = "1Y",
        interval: str = "1D",
    ) -> pd.DataFrame:
        """Async wrapper for get_stock_history."""
        return await asyncio.to_thread(self.get_stock_history, ticker, length, interval)

    async def aget_financial_ratios(
        self,
        ticker: str,
        period: str = "quarter",
    ) -> pd.DataFrame:
        """Async wrapper for get_financial_ratios."""
        return await asyncio.to_thread(self.get_financial_ratios, ticker, period)

    async def aget_income_statement(
        self,
        ticker: str,
        period: str = "year",
    ) -> pd.DataFrame:
        """Async wrapper for get_income_statement."""
        return await asyncio.to_thread(self.get_income_statement, ticker, period)

    async def aget_balance_sheet(
        self,
        ticker: str,
        period: str = "year",
    ) -> pd.DataFrame:
        """Async wrapper for get_balance_sheet."""
        return await asyncio.to_thread(self.get_balance_sheet, ticker, period)
