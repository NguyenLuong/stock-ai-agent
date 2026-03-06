"""Tests for shared.llm.embedder — text-embedding-3-small wrapper."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from shared.llm.client import reset_llm_client
from shared.llm.embedder import embed_single, embed_texts


@pytest.fixture(autouse=True)
def _reset_client():
    reset_llm_client()
    yield
    reset_llm_client()


def _make_embedding_response(texts: list[str]) -> MagicMock:
    """Create mock embedding response with 1536-dim vectors."""
    data = []
    for i, _ in enumerate(texts):
        item = MagicMock()
        item.embedding = [0.1] * 1536
        data.append(item)
    response = MagicMock()
    response.data = data
    return response


class TestEmbedTexts:
    async def test_batching_250_texts_makes_3_api_calls(self, monkeypatch):
        mock_client = AsyncMock()
        call_count = 0

        async def mock_create(**kwargs):
            nonlocal call_count
            call_count += 1
            return _make_embedding_response(kwargs["input"])

        mock_client.embeddings.create = mock_create
        monkeypatch.setattr("shared.llm.embedder._get_client", lambda: mock_client)

        texts = [f"text_{i}" for i in range(250)]
        result = await embed_texts(texts, batch_size=100)

        assert call_count == 3
        assert len(result) == 250

    async def test_return_shape_correct_dimension(self, monkeypatch):
        mock_client = AsyncMock()

        async def mock_create(**kwargs):
            return _make_embedding_response(kwargs["input"])

        mock_client.embeddings.create = mock_create
        monkeypatch.setattr("shared.llm.embedder._get_client", lambda: mock_client)

        result = await embed_texts(["hello", "world"])
        assert len(result) == 2
        assert len(result[0]) == 1536
        assert len(result[1]) == 1536
        assert all(isinstance(v, float) for v in result[0])

    async def test_embed_single_returns_single_vector(self, monkeypatch):
        mock_client = AsyncMock()

        async def mock_create(**kwargs):
            return _make_embedding_response(kwargs["input"])

        mock_client.embeddings.create = mock_create
        monkeypatch.setattr("shared.llm.embedder._get_client", lambda: mock_client)

        result = await embed_single("hello world")
        assert isinstance(result, list)
        assert len(result) == 1536
