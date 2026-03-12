"""VnstockMacroClient — fetches macro indicators via vnstock library.

Provides VN-Index close/volume, USD/VND exchange rate, and foreign net flow
with retry logic and mock data fallback on API failure.

Follows the same singleton + retry + mock pattern as vnstock_client.py.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from services.crawler.macro.models import MacroDataResult
from shared.logging import get_logger
from shared.utils.datetime_utils import now_utc

# Transient network exceptions worth retrying
_RETRYABLE_EXCEPTIONS = (ConnectionError, TimeoutError, OSError)

logger = get_logger("vnstock_macro_client")


def _log_retry_attempt(retry_state: RetryCallState) -> None:
    """Log each retry attempt."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(
        "vnstock_macro_retry",
        attempt=retry_state.attempt_number,
        error=str(exc),
        component="vnstock_macro_client",
    )


class VnstockMacroClient:
    """Fetches macro indicators via vnstock library.

    Wraps vnstock Quote and Fx with retry logic and mock fallback.
    Sync methods wrapped via asyncio.to_thread() for async contexts.
    """

    _instance: VnstockMacroClient | None = None

    @classmethod
    def get_instance(cls) -> VnstockMacroClient:
        """Return singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton — for test isolation."""
        cls._instance = None

    # ── VN-Index ──────────────────────────────────────────────────────────

    @retry(
        wait=wait_exponential(min=1, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
        reraise=True,
        before_sleep=_log_retry_attempt,
    )
    def _fetch_vn_index(self) -> tuple[float, float, datetime]:
        """Sync fetch VN-Index close + volume. Runs in thread via asyncio.to_thread()."""
        from vnstock import Quote

        quote = Quote(symbol="VNINDEX", source="VCI")
        df = quote.history(length="1M", interval="1D")

        if df.empty:
            raise ValueError("Empty DataFrame returned for VNINDEX")

        latest = df.iloc[-1]
        time_val = latest["time"]
        if isinstance(time_val, str):
            data_as_of = datetime.fromisoformat(time_val).replace(tzinfo=timezone.utc)
        else:
            data_as_of = time_val.to_pydatetime() if hasattr(time_val, "to_pydatetime") else now_utc()
            if data_as_of.tzinfo is None:
                data_as_of = data_as_of.replace(tzinfo=timezone.utc)

        return float(latest["close"]), float(latest["volume"]), data_as_of

    def get_vn_index(self) -> list[MacroDataResult]:
        """Fetch VN-Index close and volume. Returns two MacroDataResult items."""
        try:
            close, volume, data_as_of = self._fetch_vn_index()
            logger.info(
                "vn_index_fetched",
                close=close,
                volume=volume,
                data_source="vnstock",
                component="vnstock_macro_client",
            )
            return [
                MacroDataResult(
                    indicator_name="vn_index_close",
                    indicator_value=close,
                    data_as_of=data_as_of,
                    data_source="vnstock",
                    success=True,
                ),
                MacroDataResult(
                    indicator_name="vn_index_volume",
                    indicator_value=volume,
                    data_as_of=data_as_of,
                    data_source="vnstock",
                    success=True,
                ),
            ]
        except Exception as exc:
            logger.warning(
                "vn_index_fallback_to_mock",
                error=str(exc),
                component="vnstock_macro_client",
            )
            return self._mock_vn_index()

    async def aget_vn_index(self) -> list[MacroDataResult]:
        """Async wrapper for get_vn_index."""
        return await asyncio.to_thread(self.get_vn_index)

    @staticmethod
    def _mock_vn_index() -> list[MacroDataResult]:
        """Generate mock VN-Index data."""
        data_as_of = now_utc()
        return [
            MacroDataResult(
                indicator_name="vn_index_close",
                indicator_value=1250.0,
                data_as_of=data_as_of,
                data_source="mock",
                success=True,
            ),
            MacroDataResult(
                indicator_name="vn_index_volume",
                indicator_value=500_000_000.0,
                data_as_of=data_as_of,
                data_source="mock",
                success=True,
            ),
        ]

    # ── USD/VND Exchange Rate ─────────────────────────────────────────────

    @retry(
        wait=wait_exponential(min=1, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
        reraise=True,
        before_sleep=_log_retry_attempt,
    )
    def _fetch_exchange_rate(self) -> tuple[float, datetime]:
        """Sync fetch USD/VND rate via vnstock Fx."""
        from vnstock import Fx

        fx = Fx(symbol="USDVND", source="MSN")
        df = fx.history(length="1M", interval="1D")

        if df.empty:
            raise ValueError("Empty DataFrame returned for USDVND")

        latest = df.iloc[-1]
        rate_col = "close" if "close" in df.columns else df.columns[-1]
        time_col = "time" if "time" in df.columns else df.columns[0]

        time_val = latest[time_col]
        if isinstance(time_val, str):
            data_as_of = datetime.fromisoformat(time_val).replace(tzinfo=timezone.utc)
        else:
            data_as_of = time_val.to_pydatetime() if hasattr(time_val, "to_pydatetime") else now_utc()
            if data_as_of.tzinfo is None:
                data_as_of = data_as_of.replace(tzinfo=timezone.utc)

        return float(latest[rate_col]), data_as_of

    def get_exchange_rate(self) -> MacroDataResult:
        """Fetch USD/VND exchange rate."""
        try:
            rate, data_as_of = self._fetch_exchange_rate()
            logger.info(
                "exchange_rate_fetched",
                rate=rate,
                data_source="vnstock",
                component="vnstock_macro_client",
            )
            return MacroDataResult(
                indicator_name="usd_vnd_rate",
                indicator_value=rate,
                data_as_of=data_as_of,
                data_source="vnstock",
                success=True,
            )
        except Exception as exc:
            logger.warning(
                "exchange_rate_fallback_to_mock",
                error=str(exc),
                component="vnstock_macro_client",
            )
            return self._mock_exchange_rate()

    async def aget_exchange_rate(self) -> MacroDataResult:
        """Async wrapper for get_exchange_rate."""
        return await asyncio.to_thread(self.get_exchange_rate)

    @staticmethod
    def _mock_exchange_rate() -> MacroDataResult:
        """Generate mock USD/VND rate."""
        return MacroDataResult(
            indicator_name="usd_vnd_rate",
            indicator_value=25_850.0,
            data_as_of=now_utc(),
            data_source="mock",
            success=True,
        )

    # ── Foreign Net Flow ──────────────────────────────────────────────────

    def get_foreign_flow(self) -> MacroDataResult:
        """Fetch foreign net flow.

        vnstock does not support market-wide foreign flow aggregation.
        Always returns mock data with data_source="mock".
        """
        logger.info(
            "foreign_flow_mock_used",
            component="vnstock_macro_client",
            reason="vnstock does not support market-wide foreign flow",
        )
        return self._mock_foreign_flow()

    async def aget_foreign_flow(self) -> MacroDataResult:
        """Async wrapper for get_foreign_flow."""
        return await asyncio.to_thread(self.get_foreign_flow)

    @staticmethod
    def _mock_foreign_flow() -> MacroDataResult:
        """Generate mock foreign net flow data."""
        return MacroDataResult(
            indicator_name="foreign_net_flow",
            indicator_value=0.0,
            data_as_of=now_utc(),
            data_source="mock",
            success=True,
        )
