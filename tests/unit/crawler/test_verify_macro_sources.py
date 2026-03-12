"""Tests for macro data source verification script."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pandas as pd

from services.crawler.macro.verify_macro_sources import (
    VerificationReport,
    VerificationResult,
    run_verification,
    verify_exchange_rate,
    verify_foreign_flow,
    verify_sbv_interest_rate,
    verify_vn_index,
)


def _mock_vnstock_module() -> tuple[MagicMock, MagicMock]:
    """Create a mock vnstock module with Quote and Fx classes."""
    mock_module = ModuleType("vnstock")
    mock_quote_cls = MagicMock()
    mock_fx_cls = MagicMock()
    mock_module.Quote = mock_quote_cls  # type: ignore[attr-defined]
    mock_module.Fx = mock_fx_cls  # type: ignore[attr-defined]
    return mock_quote_cls, mock_fx_cls


class TestVerifyVnIndex:
    """Tests for VN-Index verification via vnstock."""

    def test_success_returns_supported(self) -> None:
        mock_quote_cls, _ = _mock_vnstock_module()
        mock_quote = MagicMock()
        mock_quote_cls.return_value = mock_quote
        df = pd.DataFrame({
            "time": ["2026-03-10"],
            "open": [1250.0],
            "high": [1260.0],
            "low": [1245.0],
            "close": [1255.5],
            "volume": [500_000_000],
        })
        mock_quote.history.return_value = df

        mock_mod = ModuleType("vnstock")
        mock_mod.Quote = mock_quote_cls  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"vnstock": mock_mod}):
            result = verify_vn_index()

        assert result.supported is True
        assert result.indicator_name == "vn_index_close"
        assert result.source == "vnstock"
        assert result.sample_value == 1255.5
        assert result.error is None

    def test_empty_df_returns_not_supported(self) -> None:
        mock_quote_cls = MagicMock()
        mock_quote = MagicMock()
        mock_quote_cls.return_value = mock_quote
        mock_quote.history.return_value = pd.DataFrame()

        mock_mod = ModuleType("vnstock")
        mock_mod.Quote = mock_quote_cls  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"vnstock": mock_mod}):
            result = verify_vn_index()

        assert result.supported is False
        assert "empty" in result.notes.lower()

    def test_import_error_returns_not_supported(self) -> None:
        # Remove vnstock from sys.modules so import fails
        with patch.dict(sys.modules, {"vnstock": None}):
            result = verify_vn_index()

        assert result.supported is False
        assert result.error is not None


class TestVerifyExchangeRate:
    """Tests for USD/VND exchange rate verification via vnstock."""

    def test_success_returns_supported(self) -> None:
        mock_fx_cls = MagicMock()
        mock_fx = MagicMock()
        mock_fx_cls.return_value = mock_fx
        df = pd.DataFrame({
            "time": ["2026-03-10"],
            "close": [25_850.0],
        })
        mock_fx.history.return_value = df

        mock_mod = ModuleType("vnstock")
        mock_mod.Fx = mock_fx_cls  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"vnstock": mock_mod}):
            result = verify_exchange_rate()

        assert result.supported is True
        assert result.indicator_name == "usd_vnd_rate"
        assert result.sample_value == 25_850.0

    def test_empty_df_returns_not_supported(self) -> None:
        mock_fx_cls = MagicMock()
        mock_fx = MagicMock()
        mock_fx_cls.return_value = mock_fx
        mock_fx.history.return_value = pd.DataFrame()

        mock_mod = ModuleType("vnstock")
        mock_mod.Fx = mock_fx_cls  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"vnstock": mock_mod}):
            result = verify_exchange_rate()

        assert result.supported is False

    def test_api_error_returns_not_supported(self) -> None:
        mock_fx_cls = MagicMock(side_effect=Exception("API error"))

        mock_mod = ModuleType("vnstock")
        mock_mod.Fx = mock_fx_cls  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"vnstock": mock_mod}):
            result = verify_exchange_rate()

        assert result.supported is False
        assert result.error == "API error"


class TestVerifyForeignFlow:
    """Tests for foreign net flow verification."""

    def test_returns_not_supported(self) -> None:
        result = verify_foreign_flow()

        assert result.supported is False
        assert result.indicator_name == "foreign_net_flow"
        assert "mock" in result.notes.lower()


class TestVerifySbvInterestRate:
    """Tests for SBV interest rate verification."""

    def test_returns_not_supported(self) -> None:
        result = verify_sbv_interest_rate()

        assert result.supported is False
        assert result.indicator_name == "sbv_interest_rate"
        assert "scraping" in result.notes.lower()


class TestVerificationReport:
    """Tests for VerificationReport dataclass."""

    def test_supported_count(self) -> None:
        report = VerificationReport(results=[
            VerificationResult("a", "vnstock", True, "ok"),
            VerificationResult("b", "vnstock", False, "no"),
            VerificationResult("c", "vnstock", True, "ok"),
        ])

        assert report.supported_count == 2
        assert report.fallback_count == 1

    def test_summary_contains_all_indicators(self) -> None:
        report = VerificationReport(results=[
            VerificationResult("vn_index_close", "vnstock", True, "ok", 1250.0),
            VerificationResult("foreign_net_flow", "vnstock", False, "mock"),
        ])

        summary = report.summary()
        assert "vn_index_close" in summary
        assert "foreign_net_flow" in summary
        assert "SUPPORTED" in summary
        assert "FALLBACK" in summary


class TestRunVerification:
    """Tests for the full verification run."""

    @patch("services.crawler.macro.verify_macro_sources.verify_vn_index")
    @patch("services.crawler.macro.verify_macro_sources.verify_exchange_rate")
    @patch("services.crawler.macro.verify_macro_sources.verify_foreign_flow")
    @patch("services.crawler.macro.verify_macro_sources.verify_sbv_interest_rate")
    def test_returns_report_with_all_indicators(
        self,
        mock_sbv: MagicMock,
        mock_flow: MagicMock,
        mock_fx: MagicMock,
        mock_vn: MagicMock,
    ) -> None:
        mock_vn.return_value = VerificationResult("vn_index_close", "vnstock", True, "ok")
        mock_fx.return_value = VerificationResult("usd_vnd_rate", "vnstock", True, "ok")
        mock_flow.return_value = VerificationResult("foreign_net_flow", "vnstock", False, "mock")
        mock_sbv.return_value = VerificationResult("sbv_interest_rate", "sbv.gov.vn", False, "scrape")

        report = run_verification()

        assert len(report.results) == 4
        assert report.supported_count == 2
        assert report.fallback_count == 2
