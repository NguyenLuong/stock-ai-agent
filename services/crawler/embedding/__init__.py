"""Article embedding pipeline for RAG infrastructure."""

from embedding.embedding_pipeline import run_embedding_pipeline
from embedding.models import EmbeddingPipelineResult

__all__ = ["run_embedding_pipeline", "EmbeddingPipelineResult"]
