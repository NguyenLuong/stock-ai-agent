"""Article embedding pipeline for RAG infrastructure."""

from .embedding_pipeline import run_embedding_pipeline
from .models import EmbeddingPipelineResult

__all__ = ["run_embedding_pipeline", "EmbeddingPipelineResult"]
