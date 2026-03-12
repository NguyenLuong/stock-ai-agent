"""Article embedding pipeline for RAG infrastructure."""

from services.crawler.embedding.embedding_pipeline import run_embedding_pipeline
from services.crawler.embedding.models import EmbeddingPipelineResult

__all__ = ["run_embedding_pipeline", "EmbeddingPipelineResult"]
