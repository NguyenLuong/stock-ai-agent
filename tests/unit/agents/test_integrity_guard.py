"""Tests for integrity_guard — hallucination prevention and data integrity checks."""

from __future__ import annotations

import pytest

from services.app.agents.orchestrator.integrity_guard import (
    audit_null_fields,
    check_data_traceability,
    compute_risk_assessment,
    compute_stop_loss,
    validate_no_absolute_certainty,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_state(**overrides) -> dict:
    s: dict = {
        "ticker": "HPG",
        "failed_agents": [],
        "technical_analysis": {
            "signals": {"volatility": "normal"},
            "support_levels": [27500.0, 28000.0],
            "confidence": 0.75,
        },
        "fundamental_analysis": {
            "company_ratios": {
                "pe": 8.5,
                "pb": 1.2,
                "roe": 18.0,
                "eps": 3500.0,
                "eps_growth_yoy": 15.0,
            },
        },
    }
    s.update(overrides)
    return s


# ===========================================================================
# validate_no_absolute_certainty
# ===========================================================================


class TestValidateNoAbsoluteCertainty:
    def test_clean_text_returns_empty(self):
        result = validate_no_absolute_certainty("HPG đang có xu hướng tăng nhẹ")
        assert result == []

    def test_single_violation(self):
        # "chắc chắn" triggers exactly 1 violation ("chắc" standalone removed from list)
        result = validate_no_absolute_certainty("Cổ phiếu này chắc chắn sẽ tăng")
        assert len(result) == 1
        assert any("chắc chắn" in v for v in result)

    def test_multiple_violations(self):
        text = "Đảm bảo lợi nhuận 100% và chắc chắn không có rủi ro"
        result = validate_no_absolute_certainty(text)
        # triggers: "đảm bảo", "100%", "chắc chắn", "không có rủi ro" → 4 violations
        assert len(result) == 4

    def test_100_percent_blocked(self):
        result = validate_no_absolute_certainty("Xác suất tăng 100%")
        assert any("100%" in v for v in result)

    def test_other_percent_not_blocked(self):
        result = validate_no_absolute_certainty("Độ tin cậy: 72%")
        assert result == []

    def test_case_insensitive_english(self):
        result = validate_no_absolute_certainty("This is Guaranteed profit")
        assert len(result) == 1

    def test_empty_text(self):
        result = validate_no_absolute_certainty("")
        assert result == []


# ===========================================================================
# compute_risk_assessment
# ===========================================================================


class TestComputeRiskAssessment:
    def test_low_risk(self):
        state = _base_state()
        result = compute_risk_assessment(state, confidence=0.80)
        assert result["level"] == "low"
        assert result["confidence_score"] == 0.80
        assert result["failed_agents"] == []

    def test_medium_risk_confidence(self):
        state = _base_state()
        result = compute_risk_assessment(state, confidence=0.55)
        assert result["level"] == "medium"
        assert "55%" in result["reasoning"]
        assert "trung bình" in result["reasoning"]

    def test_medium_risk_one_failure(self):
        state = _base_state(failed_agents=["market_context"])
        result = compute_risk_assessment(state, confidence=0.75)
        assert result["level"] == "medium"
        assert "market_context" in result["reasoning"]

    def test_high_risk_low_confidence(self):
        state = _base_state()
        result = compute_risk_assessment(state, confidence=0.35)
        assert result["level"] == "high"

    def test_high_risk_volatility(self):
        state = _base_state(
            technical_analysis={
                "signals": {"volatility": "high"},
                "support_levels": [28000.0],
                "confidence": 0.80,
            }
        )
        result = compute_risk_assessment(state, confidence=0.60)
        assert result["level"] == "high"

    def test_high_risk_two_failures(self):
        state = _base_state(
            failed_agents=["market_context", "technical_analysis"],
        )
        result = compute_risk_assessment(state, confidence=0.75)
        assert result["level"] == "high"

    def test_no_technical_analysis(self):
        state = _base_state(technical_analysis=None)
        result = compute_risk_assessment(state, confidence=0.80)
        assert result["level"] == "low"


# ===========================================================================
# compute_stop_loss
# ===========================================================================


class TestComputeStopLoss:
    def test_returns_highest_support(self):
        state = _base_state()
        result = compute_stop_loss(state)
        assert result == 28000.0

    def test_no_technical_analysis(self):
        state = _base_state(technical_analysis=None)
        assert compute_stop_loss(state) is None

    def test_empty_support_levels(self):
        state = _base_state(
            technical_analysis={
                "signals": {"volatility": "normal"},
                "support_levels": [],
                "confidence": 0.75,
            }
        )
        assert compute_stop_loss(state) is None

    def test_technical_in_failed_agents(self):
        state = _base_state(failed_agents=["technical_analysis"])
        assert compute_stop_loss(state) is None


# ===========================================================================
# check_data_traceability
# ===========================================================================


class TestCheckDataTraceability:
    def test_all_complete(self):
        data_sources = [
            {"agent": "market_context", "data_as_of": "2026-03-30T08:00:00Z", "source": "vietstock"},
            {"agent": "technical_analysis", "data_as_of": "2026-03-30T08:00:00Z", "source": "calculated"},
        ]
        assert check_data_traceability(data_sources) == []

    def test_missing_source(self):
        data_sources = [
            {"agent": "market_context", "data_as_of": "2026-03-30T08:00:00Z", "source": None},
        ]
        result = check_data_traceability(data_sources)
        assert len(result) == 1
        assert "market_context" in result[0]
        assert "source" in result[0]

    def test_missing_data_as_of(self):
        data_sources = [
            {"agent": "technical_analysis", "source": "calculated"},
        ]
        result = check_data_traceability(data_sources)
        assert len(result) == 1
        assert "technical_analysis" in result[0]
        assert "data_as_of" in result[0]

    def test_missing_both_combined(self):
        data_sources = [
            {"agent": "fundamental_analysis"},
        ]
        result = check_data_traceability(data_sources)
        assert len(result) == 1
        assert "source" in result[0]
        assert "data_as_of" in result[0]

    def test_empty_data_sources(self):
        assert check_data_traceability([]) == []


# ===========================================================================
# audit_null_fields
# ===========================================================================


class TestAuditNullFields:
    def test_all_present(self):
        state = _base_state()
        assert audit_null_fields(state) == []

    def test_one_null(self):
        state = _base_state()
        state["fundamental_analysis"]["company_ratios"]["pe"] = None
        result = audit_null_fields(state)
        assert result == ["pe"]

    def test_multiple_nulls(self):
        state = _base_state()
        state["fundamental_analysis"]["company_ratios"]["pe"] = None
        state["fundamental_analysis"]["company_ratios"]["eps_growth_yoy"] = None
        result = audit_null_fields(state)
        assert "pe" in result
        assert "eps_growth_yoy" in result

    def test_fundamental_none(self):
        state = _base_state(fundamental_analysis=None)
        assert audit_null_fields(state) == []

    def test_fundamental_in_failed_agents(self):
        state = _base_state(failed_agents=["fundamental_analysis"])
        assert audit_null_fields(state) == []

    def test_empty_company_ratios(self):
        state = _base_state()
        state["fundamental_analysis"]["company_ratios"] = {}
        assert audit_null_fields(state) == []


# ===========================================================================
# Integration: synthesize_node includes integrity guard fields
# ===========================================================================


class TestSynthesizeNodeIntegrity:
    """Verify synthesize_node output contains all integrity guard fields."""

    @pytest.fixture
    def _patch_llm(self):
        from unittest.mock import AsyncMock, MagicMock, patch

        with (
            patch("services.app.agents.orchestrator.formatter._get_llm_client") as mock_llm_cls,
            patch("services.app.agents.orchestrator.formatter.load_prompt") as mock_load,
            patch("services.app.agents.orchestrator.formatter.get_config_loader") as mock_config,
        ):
            cfg = MagicMock()
            cfg.get_model.return_value = "gpt-4o"
            cfg.get_temperature.return_value = 0.7
            mock_config.return_value = cfg

            rendered = MagicMock()
            rendered.text = "rendered prompt"
            rendered.model_key = "orchestrator_synthesis"
            mock_load.return_value = rendered

            llm = AsyncMock()
            llm.call.return_value = "KHUYẾN NGHỊ: THEO DÕI"
            mock_llm_cls.return_value = llm

            yield

    @pytest.mark.usefixtures("_patch_llm")
    async def test_synthesis_result_has_integrity_fields(self):
        from datetime import datetime, timezone

        from services.app.agents.orchestrator.formatter import synthesize_node

        state = {
            "ticker": "HPG",
            "analysis_type": "morning_briefing",
            "analysis_date": "2026-03-30",
            "watchlist": ["HPG"],
            "market_summary": {
                "macro_summary": "ok",
                "stock_summary": "ok",
                "affected_sectors": [],
                "confidence": 0.75,
                "data_as_of": datetime.now(timezone.utc).isoformat(),
                "sources": ["vietstock"],
            },
            "technical_analysis": {
                "indicator_summary": "RSI ok",
                "pattern_summary": "ok",
                "signals": {"trend": "uptrend", "momentum": "bullish", "volatility": "normal", "volume_confirmation": True},
                "support_levels": [27500.0, 28000.0],
                "resistance_levels": [30000.0],
                "confidence": 0.80,
                "data_as_of": datetime.now(timezone.utc).isoformat(),
                "data_source": "calculated",
            },
            "fundamental_analysis": {
                "bctc_summary": "ok",
                "ratio_comparison": "ok",
                "company_ratios": {"pe": 8.5, "pb": 1.2, "roe": 18.0, "eps": 3500.0, "eps_growth_yoy": 15.0},
                "sector_ratios": {"pe": 12.0, "pb": 1.5, "roe": 14.0, "eps": 2800.0},
                "sector_name": "Thép",
                "signals": {"valuation": "undervalued", "profitability": "strong", "financial_health": "healthy", "growth": "growing"},
                "confidence": 0.85,
                "data_as_of": datetime.now(timezone.utc).isoformat(),
                "data_source": "vnstock",
            },
            "failed_agents": [],
        }
        result = await synthesize_node(state)
        sr = result["synthesis_result"]

        assert "risk_assessment" in sr
        assert sr["risk_assessment"]["level"] in ("low", "medium", "high")
        assert "stop_loss_suggestion" in sr
        assert sr["stop_loss_suggestion"] == 28000.0
        assert "integrity_violations" in sr
        assert isinstance(sr["integrity_violations"], list)
        assert "traceability_warnings" in sr
        assert isinstance(sr["traceability_warnings"], list)
        assert "null_fields" in sr
        assert isinstance(sr["null_fields"], list)
