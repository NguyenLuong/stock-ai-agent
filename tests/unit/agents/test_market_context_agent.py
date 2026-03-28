"""Tests for Market Context Agent node."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.app.agents.market_context.node import (
    MAX_INPUT_CHARS,
    _calc_confidence,
    _filter_recent,
    _truncate_articles,
    market_context_node,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_article(
    title: str = "Test Article",
    source: str = "vietstock",
    raw_content: str = "Some content",
    summary: str = "Short summary",
    published_at: datetime | None = None,
    category: str = "macro",
) -> MagicMock:
    """Create a mock ORM Article."""
    art = MagicMock()
    art.title = title
    art.source = source
    art.raw_content = raw_content
    art.summary = summary
    art.published_at = published_at or datetime.now(timezone.utc)
    art.category = category
    art.embedded = True
    return art


def _base_state(**overrides) -> dict:
    """Return a minimal MarketContextState dict."""
    s: dict = {
        "ticker": "HPG",
        "analysis_type": "morning_briefing",
        "analysis_date": "2026-03-25",
    }
    s.update(overrides)
    return s


# Shared patch targets
_PATCH_SEARCH = "services.app.agents.market_context.node.semantic_search"
_PATCH_LLM_CLIENT = "services.app.agents.market_context.node.LLMClient"
_PATCH_LOAD_PROMPT = "services.app.agents.market_context.node.load_prompt"
_PATCH_CONFIG = "services.app.agents.market_context.node.get_config_loader"


def _setup_config_mock(mock_config_cls):
    cfg = MagicMock()
    cfg.get_model.return_value = "gpt-4o-mini"
    cfg.get_temperature.return_value = 0.3
    mock_config_cls.return_value = cfg
    return cfg


def _setup_prompt_mock(mock_load):
    rendered = MagicMock()
    rendered.text = "rendered prompt"
    rendered.model_key = "triage"
    mock_load.return_value = rendered
    return rendered


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------


class TestFilterRecent:
    def test_keeps_recent_articles(self):
        recent = _make_article(published_at=datetime.now(timezone.utc) - timedelta(hours=2))
        old = _make_article(published_at=datetime.now(timezone.utc) - timedelta(hours=24))
        result = _filter_recent([recent, old])
        assert len(result) == 1
        assert result[0] is recent

    def test_returns_empty_when_all_old(self):
        old = _make_article(published_at=datetime.now(timezone.utc) - timedelta(hours=24))
        assert _filter_recent([old]) == []

    def test_returns_all_when_all_recent(self):
        arts = [_make_article(published_at=datetime.now(timezone.utc) - timedelta(hours=i)) for i in range(5)]
        assert len(_filter_recent(arts)) == 5


class TestTruncateArticles:
    def test_no_truncation_for_small_articles(self):
        arts = [_make_article(raw_content="short") for _ in range(3)]
        dicts, truncated = _truncate_articles(arts)
        assert len(dicts) == 3
        assert not truncated

    def test_truncates_long_content(self):
        long_content = "x" * (MAX_INPUT_CHARS + 1000)
        arts = [_make_article(raw_content=long_content)]
        dicts, truncated = _truncate_articles(arts)
        assert truncated
        assert len(dicts[0]["raw_content"]) < len(long_content)

    def test_keeps_all_articles_but_truncates_content(self):
        # Each article ~half of budget → later ones get empty raw_content
        half_budget = MAX_INPUT_CHARS // 2
        arts = [_make_article(raw_content="y" * half_budget) for _ in range(4)]
        dicts, truncated = _truncate_articles(arts)
        assert truncated
        # All articles kept (title+summary preserved), but later ones have truncated/empty raw_content
        assert len(dicts) == 4
        assert dicts[-1]["raw_content"] == ""


class TestCalcConfidence:
    def test_zero_for_empty(self):
        assert _calc_confidence([], "macro") == 0.0

    def test_base_plus_count_bonus(self):
        arts = [{"source": "vietstock", "published_at": datetime.now(timezone.utc)} for _ in range(5)]
        score = _calc_confidence(arts, "macro")
        # base 0.50 + 0.15 (5+ articles) + 0.10 (recency < 4h) = 0.75
        # only 1 source so no multi-source bonus
        assert score == pytest.approx(0.75, abs=0.01)

    def test_multi_source_bonus(self):
        arts = [
            {"source": "vietstock", "published_at": datetime.now(timezone.utc)},
            {"source": "cafef", "published_at": datetime.now(timezone.utc)},
        ]
        score = _calc_confidence(arts, "macro")
        # base 0.50 + 0.05 (1-2 articles) + 0.10 (2 sources) + 0.10 (recency < 4h)
        assert score == pytest.approx(0.75, abs=0.01)

    def test_recency_mid_range(self):
        arts = [{"source": "vietstock", "published_at": datetime.now(timezone.utc) - timedelta(hours=6)}]
        score = _calc_confidence(arts, "stock")
        # base 0.50 + 0.05 (1 article) + 0.05 (4-12h)
        assert score == pytest.approx(0.60, abs=0.01)


# ---------------------------------------------------------------------------
# Integration-style tests for market_context_node
# ---------------------------------------------------------------------------


class TestMarketContextNodePhase1:
    """Test Phase 1 success: macro analysis."""

    @patch(_PATCH_CONFIG)
    @patch(_PATCH_LOAD_PROMPT)
    @patch(_PATCH_LLM_CLIENT)
    @patch(_PATCH_SEARCH)
    async def test_phase1_success_produces_macro_summary(
        self, mock_search, mock_llm_cls, mock_load, mock_config
    ):
        cfg = _setup_config_mock(mock_config)
        _setup_prompt_mock(mock_load)

        macro_articles = [_make_article(category="macro") for _ in range(3)]
        stock_articles = [_make_article(category="stock") for _ in range(2)]
        mock_search.side_effect = [macro_articles, stock_articles]

        llm_instance = AsyncMock()
        llm_instance.call = AsyncMock(side_effect=["Macro analysis result", "Stock analysis result"])
        mock_llm_cls.return_value = llm_instance

        result = await market_context_node(_base_state())

        assert result["market_summary"] is not None
        assert result["market_summary"]["macro_summary"] == "Macro analysis result"


class TestMarketContextNodePhase2:
    """Test Phase 2 success: stock news analysis."""

    @patch(_PATCH_CONFIG)
    @patch(_PATCH_LOAD_PROMPT)
    @patch(_PATCH_LLM_CLIENT)
    @patch(_PATCH_SEARCH)
    async def test_phase2_success_produces_stock_summary(
        self, mock_search, mock_llm_cls, mock_load, mock_config
    ):
        cfg = _setup_config_mock(mock_config)
        _setup_prompt_mock(mock_load)

        macro_articles = [_make_article(category="macro") for _ in range(3)]
        stock_articles = [_make_article(category="stock") for _ in range(2)]
        mock_search.side_effect = [macro_articles, stock_articles]

        llm_instance = AsyncMock()
        llm_instance.call = AsyncMock(side_effect=["Macro result", "Stock analysis result"])
        mock_llm_cls.return_value = llm_instance

        result = await market_context_node(_base_state())

        assert result["market_summary"]["stock_summary"] == "Stock analysis result"


class TestMarketContextNodePhase3:
    """Test Phase 3: combine produces complete market_summary."""

    @patch(_PATCH_CONFIG)
    @patch(_PATCH_LOAD_PROMPT)
    @patch(_PATCH_LLM_CLIENT)
    @patch(_PATCH_SEARCH)
    async def test_combine_has_all_fields(
        self, mock_search, mock_llm_cls, mock_load, mock_config
    ):
        _setup_config_mock(mock_config)
        _setup_prompt_mock(mock_load)

        macro_articles = [_make_article(source="vietstock", category="macro")]
        stock_articles = [_make_article(source="cafef", category="stock")]
        mock_search.side_effect = [macro_articles, stock_articles]

        llm_instance = AsyncMock()
        llm_instance.call = AsyncMock(side_effect=["Macro output", "Stock output"])
        mock_llm_cls.return_value = llm_instance

        result = await market_context_node(_base_state())
        ms = result["market_summary"]

        assert ms["macro_summary"] == "Macro output"
        assert ms["stock_summary"] == "Stock output"
        assert "data_as_of" in ms
        assert isinstance(ms["confidence"], float)
        assert isinstance(ms["sources"], list)
        assert isinstance(ms["affected_sectors"], list)


class TestMarketContextNodeEmptyMacro:
    """Test empty macro articles — Phase 2 should still run."""

    @patch(_PATCH_CONFIG)
    @patch(_PATCH_LOAD_PROMPT)
    @patch(_PATCH_LLM_CLIENT)
    @patch(_PATCH_SEARCH)
    async def test_empty_macro_still_runs_stock(
        self, mock_search, mock_llm_cls, mock_load, mock_config
    ):
        _setup_config_mock(mock_config)
        _setup_prompt_mock(mock_load)

        stock_articles = [_make_article(category="stock")]
        mock_search.side_effect = [[], stock_articles]  # empty macro

        llm_instance = AsyncMock()
        llm_instance.call = AsyncMock(return_value="Stock result")
        mock_llm_cls.return_value = llm_instance

        result = await market_context_node(_base_state())
        ms = result["market_summary"]

        assert "Không có tin tức vĩ mô" in ms["macro_summary"]
        assert ms["stock_summary"] == "Stock result"


class TestMarketContextNodeEmptyStock:
    """Test empty stock articles — macro_summary should be preserved."""

    @patch(_PATCH_CONFIG)
    @patch(_PATCH_LOAD_PROMPT)
    @patch(_PATCH_LLM_CLIENT)
    @patch(_PATCH_SEARCH)
    async def test_empty_stock_still_has_macro(
        self, mock_search, mock_llm_cls, mock_load, mock_config
    ):
        _setup_config_mock(mock_config)
        _setup_prompt_mock(mock_load)

        macro_articles = [_make_article(category="macro")]
        mock_search.side_effect = [macro_articles, []]  # empty stock

        llm_instance = AsyncMock()
        llm_instance.call = AsyncMock(return_value="Macro result")
        mock_llm_cls.return_value = llm_instance

        result = await market_context_node(_base_state())
        ms = result["market_summary"]

        assert ms["macro_summary"] == "Macro result"
        assert "Không có tin tức chứng khoán" in ms["stock_summary"]


class TestMarketContextNodeLLMFailurePhase1:
    """Test LLM failure in Phase 1 — Phase 2 should still run."""

    @patch(_PATCH_CONFIG)
    @patch(_PATCH_LOAD_PROMPT)
    @patch(_PATCH_LLM_CLIENT)
    @patch(_PATCH_SEARCH)
    async def test_llm_fail_macro_still_runs_stock(
        self, mock_search, mock_llm_cls, mock_load, mock_config
    ):
        from shared.llm.client import LLMCallError

        _setup_config_mock(mock_config)
        _setup_prompt_mock(mock_load)

        macro_articles = [_make_article(category="macro")]
        stock_articles = [_make_article(category="stock")]
        mock_search.side_effect = [macro_articles, stock_articles]

        llm_instance = AsyncMock()
        llm_instance.call = AsyncMock(
            side_effect=[LLMCallError("timeout"), "Stock result"]
        )
        mock_llm_cls.return_value = llm_instance

        result = await market_context_node(_base_state())
        ms = result["market_summary"]

        assert ms["macro_summary"] is None
        assert ms["stock_summary"] == "Stock result"
        assert "market_context_macro" in result.get("failed_agents", [])


class TestMarketContextNodeLLMFailurePhase2:
    """Test LLM failure in Phase 2 — macro_summary preserved."""

    @patch(_PATCH_CONFIG)
    @patch(_PATCH_LOAD_PROMPT)
    @patch(_PATCH_LLM_CLIENT)
    @patch(_PATCH_SEARCH)
    async def test_llm_fail_stock_preserves_macro(
        self, mock_search, mock_llm_cls, mock_load, mock_config
    ):
        from shared.llm.client import LLMCallError

        _setup_config_mock(mock_config)
        _setup_prompt_mock(mock_load)

        macro_articles = [_make_article(category="macro")]
        stock_articles = [_make_article(category="stock")]
        mock_search.side_effect = [macro_articles, stock_articles]

        llm_instance = AsyncMock()
        llm_instance.call = AsyncMock(
            side_effect=["Macro result", LLMCallError("timeout")]
        )
        mock_llm_cls.return_value = llm_instance

        result = await market_context_node(_base_state())
        ms = result["market_summary"]

        assert ms["macro_summary"] == "Macro result"
        assert ms["stock_summary"] is None
        assert "market_context_stock" in result.get("failed_agents", [])


class TestMarketContextNodeTruncation:
    """Test truncation with long articles — logs warning."""

    @patch(_PATCH_CONFIG)
    @patch(_PATCH_LOAD_PROMPT)
    @patch(_PATCH_LLM_CLIENT)
    @patch(_PATCH_SEARCH)
    async def test_truncation_occurs_and_still_succeeds(
        self, mock_search, mock_llm_cls, mock_load, mock_config
    ):
        _setup_config_mock(mock_config)
        _setup_prompt_mock(mock_load)

        long_article = _make_article(raw_content="x" * (MAX_INPUT_CHARS + 5000), category="macro")
        stock_articles = [_make_article(category="stock")]
        mock_search.side_effect = [[long_article], stock_articles]

        llm_instance = AsyncMock()
        llm_instance.call = AsyncMock(side_effect=["Macro truncated", "Stock result"])
        mock_llm_cls.return_value = llm_instance

        result = await market_context_node(_base_state())

        assert result["market_summary"] is not None
        assert result["market_summary"]["macro_summary"] == "Macro truncated"


class TestMarketContextNodeConfidenceScoring:
    """Test confidence is average of macro + stock scores."""

    @patch(_PATCH_CONFIG)
    @patch(_PATCH_LOAD_PROMPT)
    @patch(_PATCH_LLM_CLIENT)
    @patch(_PATCH_SEARCH)
    async def test_confidence_is_average(
        self, mock_search, mock_llm_cls, mock_load, mock_config
    ):
        _setup_config_mock(mock_config)
        _setup_prompt_mock(mock_load)

        # 5 macro articles from 2 sources → high confidence
        macro_articles = [
            _make_article(source="vietstock", category="macro") for _ in range(3)
        ] + [
            _make_article(source="cafef", category="macro") for _ in range(2)
        ]
        # 2 stock articles from 1 source → lower confidence
        stock_articles = [_make_article(source="vneconomy", category="stock") for _ in range(2)]
        mock_search.side_effect = [macro_articles, stock_articles]

        llm_instance = AsyncMock()
        llm_instance.call = AsyncMock(side_effect=["Macro", "Stock"])
        mock_llm_cls.return_value = llm_instance

        result = await market_context_node(_base_state())
        ms = result["market_summary"]

        assert 0.0 < ms["confidence"] <= 1.0
        # macro: 0.50 + 0.15 + 0.10 + 0.10 = 0.85
        # stock: 0.50 + 0.05 + 0.10 = 0.65
        # average ≈ 0.75
        assert ms["confidence"] == pytest.approx(0.75, abs=0.05)


# ---------------------------------------------------------------------------
# Logging verification tests
# ---------------------------------------------------------------------------

_PATCH_LOGGER = "services.app.agents.market_context.node.logger"


class TestMarketContextNodeLogging:
    """Verify structured logging events are emitted (Task 4 / AC #4)."""

    @patch(_PATCH_LOGGER)
    @patch(_PATCH_CONFIG)
    @patch(_PATCH_LOAD_PROMPT)
    @patch(_PATCH_LLM_CLIENT)
    @patch(_PATCH_SEARCH)
    async def test_success_emits_expected_log_events(
        self, mock_search, mock_llm_cls, mock_load, mock_config, mock_logger
    ):
        _setup_config_mock(mock_config)
        _setup_prompt_mock(mock_load)

        mock_search.side_effect = [
            [_make_article(category="macro")],
            [_make_article(category="stock")],
        ]
        llm_instance = AsyncMock()
        llm_instance.call = AsyncMock(side_effect=["Macro", "Stock"])
        mock_llm_cls.return_value = llm_instance

        await market_context_node(_base_state())

        logged_events = [call.args[0] for call in mock_logger.info.call_args_list]
        assert "agent_started" in logged_events
        assert logged_events.count("article_retrieval_started") == 2
        assert logged_events.count("article_retrieval_completed") == 2
        assert logged_events.count("llm_call_started") == 2
        assert logged_events.count("llm_call_completed") == 2
        assert "agent_completed" in logged_events

    @patch(_PATCH_LOGGER)
    @patch(_PATCH_CONFIG)
    @patch(_PATCH_LOAD_PROMPT)
    @patch(_PATCH_LLM_CLIENT)
    @patch(_PATCH_SEARCH)
    async def test_truncation_emits_warning(
        self, mock_search, mock_llm_cls, mock_load, mock_config, mock_logger
    ):
        _setup_config_mock(mock_config)
        _setup_prompt_mock(mock_load)

        long_article = _make_article(raw_content="x" * (MAX_INPUT_CHARS + 5000), category="macro")
        mock_search.side_effect = [[long_article], [_make_article(category="stock")]]
        llm_instance = AsyncMock()
        llm_instance.call = AsyncMock(side_effect=["Macro", "Stock"])
        mock_llm_cls.return_value = llm_instance

        await market_context_node(_base_state())

        warning_events = [call.args[0] for call in mock_logger.warning.call_args_list]
        assert "articles_truncated" in warning_events

    @patch(_PATCH_LOGGER)
    @patch(_PATCH_CONFIG)
    @patch(_PATCH_LOAD_PROMPT)
    @patch(_PATCH_LLM_CLIENT)
    @patch(_PATCH_SEARCH)
    async def test_llm_failure_emits_error(
        self, mock_search, mock_llm_cls, mock_load, mock_config, mock_logger
    ):
        from shared.llm.client import LLMCallError

        _setup_config_mock(mock_config)
        _setup_prompt_mock(mock_load)

        mock_search.side_effect = [
            [_make_article(category="macro")],
            [_make_article(category="stock")],
        ]
        llm_instance = AsyncMock()
        llm_instance.call = AsyncMock(
            side_effect=[LLMCallError("fail"), "Stock"]
        )
        mock_llm_cls.return_value = llm_instance

        await market_context_node(_base_state())

        error_events = [call.args[0] for call in mock_logger.error.call_args_list]
        assert "market_context_llm_failed" in error_events
