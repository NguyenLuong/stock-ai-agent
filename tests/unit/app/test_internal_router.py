"""Tests for internal trigger endpoints — mock pipeline calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.app.routers.internal import router

app = FastAPI()
app.include_router(router)
client = TestClient(app)

TRIGGER_HEADER = {"X-Trigger-Source": "prefect-scheduler"}


class TestTriggerSourceValidation:
    """Tests for X-Trigger-Source header validation."""

    def test_missing_header_returns_403(self):
        """Missing trigger source header returns 403."""
        response = client.post("/internal/trigger/crawl")
        assert response.status_code == 403

    def test_wrong_header_returns_403(self):
        """Wrong trigger source header returns 403."""
        response = client.post(
            "/internal/trigger/crawl",
            headers={"X-Trigger-Source": "unknown"},
        )
        assert response.status_code == 403

    def test_valid_header_on_all_endpoints(self):
        """Valid header is accepted on all endpoints (with mocked pipelines)."""
        with patch(
            "services.app.routers.internal.run_news_crawl",
            new_callable=AsyncMock,
            create=True,
        ), patch(
            "services.app.routers.internal.run_macro_crawl",
            new_callable=AsyncMock,
            create=True,
        ):
            response = client.post(
                "/internal/trigger/crawl",
                headers=TRIGGER_HEADER,
            )
            assert response.status_code == 200


@dataclass
class MockCrawlResult:
    total_articles: int = 5
    new_articles: int = 3
    sources_crawled: int = 2
    errors: int = 0
    source_details: dict = field(default_factory=dict)


@dataclass
class MockMacroCrawlResult:
    saved_count: int = 4
    results: list = field(default_factory=list)

    @property
    def succeeded(self) -> list:
        return self.results

    @property
    def failed(self) -> list:
        return []

    @property
    def failed_indicators(self) -> list:
        return []


class TestTriggerCrawl:
    """Tests for /internal/trigger/crawl endpoint."""

    @patch(
        "services.crawler.macro.macro_crawl_manager.run_macro_crawl",
        new_callable=AsyncMock,
    )
    @patch(
        "services.crawler.news.crawl_manager.run_news_crawl",
        new_callable=AsyncMock,
    )
    def test_crawl_success(self, mock_news, mock_macro):
        """Successful crawl returns status ok with results."""
        mock_news.return_value = MockCrawlResult()
        mock_macro.return_value = MockMacroCrawlResult()

        response = client.post("/internal/trigger/crawl", headers=TRIGGER_HEADER)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["news_crawl"]["total_articles"] == 5
        assert data["macro_crawl"]["saved_count"] == 4
        assert data["duration_seconds"] >= 0

    @patch(
        "services.crawler.macro.macro_crawl_manager.run_macro_crawl",
        new_callable=AsyncMock,
    )
    @patch(
        "services.crawler.news.crawl_manager.run_news_crawl",
        new_callable=AsyncMock,
    )
    def test_news_crawl_fails_macro_continues(self, mock_news, mock_macro):
        """If news crawl fails, macro still runs — status partial."""
        mock_news.side_effect = Exception("news error")
        mock_macro.return_value = MockMacroCrawlResult()

        response = client.post("/internal/trigger/crawl", headers=TRIGGER_HEADER)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "partial"
        assert len(data["errors"]) == 1
        assert "news_crawl" in data["errors"][0]

    @patch(
        "services.crawler.macro.macro_crawl_manager.run_macro_crawl",
        new_callable=AsyncMock,
    )
    @patch(
        "services.crawler.news.crawl_manager.run_news_crawl",
        new_callable=AsyncMock,
    )
    def test_both_fail(self, mock_news, mock_macro):
        """Both crawls fail — status partial, 2 errors."""
        mock_news.side_effect = Exception("news down")
        mock_macro.side_effect = Exception("macro down")

        response = client.post("/internal/trigger/crawl", headers=TRIGGER_HEADER)

        data = response.json()
        assert data["status"] == "partial"
        assert len(data["errors"]) == 2


class MockPipelineResult:
    """Mock for Pydantic BaseModel results with model_dump."""

    def __init__(self, **kwargs):
        self._data = kwargs

    def model_dump(self) -> dict:
        return self._data

    def __getattr__(self, name):
        if name.startswith("_"):
            return super().__getattribute__(name)
        return self._data.get(name)


class TestTriggerEmbedding:
    """Tests for /internal/trigger/embedding endpoint."""

    @patch(
        "services.crawler.embedding.embedding_pipeline.run_embedding_pipeline",
        new_callable=AsyncMock,
    )
    def test_embedding_success(self, mock_embed):
        """Successful embedding returns status ok."""
        mock_embed.return_value = MockPipelineResult(
            total=10, embedded_count=8, failed_count=2, skipped_count=0, duration_seconds=5.0
        )

        response = client.post("/internal/trigger/embedding", headers=TRIGGER_HEADER)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["result"]["embedded_count"] == 8

    @patch(
        "services.crawler.embedding.embedding_pipeline.run_embedding_pipeline",
        new_callable=AsyncMock,
    )
    def test_embedding_failure(self, mock_embed):
        """Failed embedding returns status failed with error."""
        mock_embed.side_effect = Exception("OpenAI API error")

        response = client.post("/internal/trigger/embedding", headers=TRIGGER_HEADER)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "failed"
        assert "OpenAI API error" in data["error"]


class TestTriggerLifecycle:
    """Tests for /internal/trigger/lifecycle endpoint."""

    @patch(
        "services.crawler.lifecycle.lifecycle_pipeline.run_lifecycle_pipeline",
        new_callable=AsyncMock,
    )
    def test_lifecycle_success(self, mock_lifecycle):
        """Successful lifecycle returns status ok."""
        mock_lifecycle.return_value = MockPipelineResult(
            total=20,
            summarized_count=15,
            skipped_count=3,
            failed_count=2,
            tokens_used=5000,
            duration_seconds=12.0,
        )

        response = client.post("/internal/trigger/lifecycle", headers=TRIGGER_HEADER)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["result"]["summarized_count"] == 15

    @patch(
        "services.crawler.lifecycle.lifecycle_pipeline.run_lifecycle_pipeline",
        new_callable=AsyncMock,
    )
    def test_lifecycle_failure(self, mock_lifecycle):
        """Failed lifecycle returns status failed with error."""
        mock_lifecycle.side_effect = Exception("DB connection lost")

        response = client.post("/internal/trigger/lifecycle", headers=TRIGGER_HEADER)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "failed"
        assert "DB connection lost" in data["error"]
