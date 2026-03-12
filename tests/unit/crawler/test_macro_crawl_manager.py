"""Tests for macro crawl orchestrator."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from services.crawler.macro.macro_crawl_manager import run_macro_crawl
from services.crawler.macro.models import MacroDataResult
from services.crawler.macro.vnstock_macro_client import VnstockMacroClient


def _make_result(name: str, value: float, source: str = "vnstock", success: bool = True) -> MacroDataResult:
    return MacroDataResult(
        indicator_name=name,
        indicator_value=value,
        data_as_of=datetime(2026, 3, 10, tzinfo=timezone.utc),
        data_source=source,
        success=success,
    )


_MOCK_CONFIG = {
    "sources": {},
    "macro": {
        "vn_index": {"source": "vnstock", "symbol": "VNINDEX", "enabled": True},
        "exchange_rate": {"source": "vnstock", "symbol": "USDVND", "enabled": True},
        "foreign_flow": {"source": "vnstock", "enabled": True},
        "sbv_interest_rate": {
            "source": "sbv.gov.vn",
            "base_url": "https://www.sbv.gov.vn",
            "rate_limit_rps": 1,
            "enabled": True,
        },
    },
}


class TestRunMacroCrawl:
    """Tests for run_macro_crawl orchestrator."""

    def setup_method(self) -> None:
        VnstockMacroClient.reset()

    @patch("services.crawler.macro.macro_crawl_manager.save_macro_indicators", new_callable=AsyncMock, return_value=5)
    @patch("services.crawler.macro.macro_crawl_manager.SBVScraper")
    @patch("services.crawler.macro.macro_crawl_manager.get_sources", return_value=_MOCK_CONFIG)
    async def test_all_sources_succeed(
        self,
        mock_get_sources: MagicMock,
        mock_sbv_cls: MagicMock,
        mock_save: AsyncMock,
    ) -> None:
        # Mock vnstock client
        mock_client = MagicMock()
        mock_client.aget_vn_index = AsyncMock(return_value=[
            _make_result("vn_index_close", 1255.5),
            _make_result("vn_index_volume", 500_000_000.0),
        ])
        mock_client.aget_exchange_rate = AsyncMock(return_value=_make_result("usd_vnd_rate", 25_850.0))
        mock_client.aget_foreign_flow = AsyncMock(return_value=_make_result("foreign_net_flow", 0.0, "mock"))

        with patch.object(VnstockMacroClient, "get_instance", return_value=mock_client):
            # Mock SBV scraper
            mock_sbv = AsyncMock()
            mock_sbv.fetch_interest_rate.return_value = _make_result("sbv_interest_rate", 4.5, "sbv.gov.vn")
            mock_sbv_cls.return_value = mock_sbv

            result = await run_macro_crawl()

        assert len(result.results) == 5
        assert len(result.succeeded) == 5
        assert len(result.failed) == 0
        assert result.saved_count == 5

    @patch("services.crawler.macro.macro_crawl_manager.save_macro_indicators", new_callable=AsyncMock, return_value=3)
    @patch("services.crawler.macro.macro_crawl_manager.SBVScraper")
    @patch("services.crawler.macro.macro_crawl_manager.get_sources", return_value=_MOCK_CONFIG)
    async def test_partial_failure_graceful_degradation(
        self,
        mock_get_sources: MagicMock,
        mock_sbv_cls: MagicMock,
        mock_save: AsyncMock,
    ) -> None:
        mock_client = MagicMock()
        mock_client.aget_vn_index = AsyncMock(return_value=[
            _make_result("vn_index_close", 1255.5),
            _make_result("vn_index_volume", 500_000_000.0),
        ])
        mock_client.aget_exchange_rate = AsyncMock(return_value=_make_result("usd_vnd_rate", 25_850.0))
        mock_client.aget_foreign_flow = AsyncMock(return_value=_make_result("foreign_net_flow", 0.0, "mock"))

        with patch.object(VnstockMacroClient, "get_instance", return_value=mock_client):
            # SBV fails
            mock_sbv = AsyncMock()
            mock_sbv.fetch_interest_rate.return_value = MacroDataResult(
                indicator_name="sbv_interest_rate",
                indicator_value=None,
                data_as_of=datetime(2026, 3, 10, tzinfo=timezone.utc),
                data_source="sbv.gov.vn",
                success=False,
                error="Connection refused",
            )
            mock_sbv_cls.return_value = mock_sbv

            result = await run_macro_crawl()

        assert len(result.succeeded) == 4
        assert len(result.failed) == 1
        assert result.failed_indicators == ["sbv_interest_rate"]

    @patch("services.crawler.macro.macro_crawl_manager.save_macro_indicators", new_callable=AsyncMock, return_value=0)
    @patch("services.crawler.macro.macro_crawl_manager.get_sources")
    async def test_all_disabled_returns_empty(
        self,
        mock_get_sources: MagicMock,
        mock_save: AsyncMock,
    ) -> None:
        mock_get_sources.return_value = {
            "macro": {
                "vn_index": {"enabled": False},
                "exchange_rate": {"enabled": False},
                "foreign_flow": {"enabled": False},
                "sbv_interest_rate": {"enabled": False},
            }
        }

        result = await run_macro_crawl()

        assert len(result.results) == 0
        assert result.saved_count == 0

    @patch("services.crawler.macro.macro_crawl_manager.save_macro_indicators", new_callable=AsyncMock, return_value=0)
    @patch("services.crawler.macro.macro_crawl_manager.get_sources", return_value=_MOCK_CONFIG)
    async def test_vnstock_exception_doesnt_crash(
        self,
        mock_get_sources: MagicMock,
        mock_save: AsyncMock,
    ) -> None:
        """If vnstock raises an unexpected exception, crawl continues with other sources."""
        mock_client = MagicMock()
        mock_client.aget_vn_index = AsyncMock(side_effect=RuntimeError("unexpected"))
        mock_client.aget_exchange_rate = AsyncMock(side_effect=RuntimeError("unexpected"))
        mock_client.aget_foreign_flow = AsyncMock(side_effect=RuntimeError("unexpected"))

        with patch.object(VnstockMacroClient, "get_instance", return_value=mock_client):
            with patch("services.crawler.macro.macro_crawl_manager.SBVScraper") as mock_sbv_cls:
                mock_sbv = AsyncMock()
                mock_sbv.fetch_interest_rate.return_value = _make_result("sbv_interest_rate", 4.5, "sbv.gov.vn")
                mock_sbv_cls.return_value = mock_sbv

                result = await run_macro_crawl()

        # Only SBV succeeded
        assert len(result.succeeded) == 1
        assert result.succeeded[0].indicator_name == "sbv_interest_rate"

    @patch("services.crawler.macro.macro_crawl_manager.get_last_macro_fetch_time", new_callable=AsyncMock)
    @patch("services.crawler.macro.macro_crawl_manager.save_macro_indicators", new_callable=AsyncMock, return_value=0)
    @patch("services.crawler.macro.macro_crawl_manager.SBVScraper")
    @patch("services.crawler.macro.macro_crawl_manager.get_sources", return_value=_MOCK_CONFIG)
    async def test_all_sources_fail_logs_last_fetch_timestamp(
        self,
        mock_get_sources: MagicMock,
        mock_sbv_cls: MagicMock,
        mock_save: AsyncMock,
        mock_get_last_fetch: AsyncMock,
    ) -> None:
        """AC3: When all sources fail, existing DB values are kept and last fetch timestamp is logged."""
        last_fetch = datetime(2026, 3, 9, 8, 0, 0, tzinfo=timezone.utc)
        mock_get_last_fetch.return_value = last_fetch

        # All vnstock sources fail via exception
        mock_client = MagicMock()
        mock_client.aget_vn_index = AsyncMock(side_effect=RuntimeError("down"))
        mock_client.aget_exchange_rate = AsyncMock(side_effect=RuntimeError("down"))
        mock_client.aget_foreign_flow = AsyncMock(side_effect=RuntimeError("down"))

        with patch.object(VnstockMacroClient, "get_instance", return_value=mock_client):
            # SBV also fails
            mock_sbv = AsyncMock()
            mock_sbv.fetch_interest_rate.return_value = MacroDataResult(
                indicator_name="sbv_interest_rate",
                indicator_value=None,
                data_as_of=datetime(2026, 3, 10, tzinfo=timezone.utc),
                data_source="sbv.gov.vn",
                success=False,
                error="Connection refused",
            )
            mock_sbv_cls.return_value = mock_sbv

            result = await run_macro_crawl()

        # All failed — no successful results
        assert len(result.succeeded) == 0
        assert len(result.failed) == 1  # only SBV returned a result (vnstock raised exceptions)
        assert result.saved_count == 0

        # Verify last fetch timestamp was queried (AC3: log last successful fetch)
        mock_get_last_fetch.assert_awaited_once()

        # DB was not written to — existing values preserved (AC3)
        mock_save.assert_awaited_once_with([])

    @patch("services.crawler.macro.macro_crawl_manager.get_last_macro_fetch_time", new_callable=AsyncMock)
    @patch("services.crawler.macro.macro_crawl_manager.save_macro_indicators", new_callable=AsyncMock, return_value=0)
    @patch("services.crawler.macro.macro_crawl_manager.SBVScraper")
    @patch("services.crawler.macro.macro_crawl_manager.get_sources", return_value=_MOCK_CONFIG)
    async def test_all_sources_fail_no_prior_data(
        self,
        mock_get_sources: MagicMock,
        mock_sbv_cls: MagicMock,
        mock_save: AsyncMock,
        mock_get_last_fetch: AsyncMock,
    ) -> None:
        """AC3: When all sources fail and no prior data exists, log 'never' as last fetch."""
        mock_get_last_fetch.return_value = None

        mock_client = MagicMock()
        mock_client.aget_vn_index = AsyncMock(side_effect=RuntimeError("down"))
        mock_client.aget_exchange_rate = AsyncMock(side_effect=RuntimeError("down"))
        mock_client.aget_foreign_flow = AsyncMock(side_effect=RuntimeError("down"))

        with patch.object(VnstockMacroClient, "get_instance", return_value=mock_client):
            mock_sbv = AsyncMock()
            mock_sbv.fetch_interest_rate.return_value = MacroDataResult(
                indicator_name="sbv_interest_rate",
                indicator_value=None,
                data_as_of=datetime(2026, 3, 10, tzinfo=timezone.utc),
                data_source="sbv.gov.vn",
                success=False,
                error="Timeout",
            )
            mock_sbv_cls.return_value = mock_sbv

            result = await run_macro_crawl()

        assert len(result.succeeded) == 0
        mock_get_last_fetch.assert_awaited_once()
