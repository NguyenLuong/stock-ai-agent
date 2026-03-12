"""Pydantic models for the embedding pipeline."""

from __future__ import annotations

from pydantic import BaseModel


class EmbeddingPipelineResult(BaseModel):
    """Result summary of an embedding pipeline run."""

    total: int
    embedded_count: int
    failed_count: int
    skipped_count: int
    duration_seconds: float
