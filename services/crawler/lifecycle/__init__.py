"""Data lifecycle pipeline for summarizing old articles."""

from services.crawler.lifecycle.lifecycle_pipeline import run_lifecycle_pipeline
from services.crawler.lifecycle.models import LifecyclePipelineResult

__all__ = ["run_lifecycle_pipeline", "LifecyclePipelineResult"]
