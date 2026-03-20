"""Data lifecycle pipeline for summarizing old articles."""

from lifecycle.lifecycle_pipeline import run_lifecycle_pipeline
from lifecycle.models import LifecyclePipelineResult

__all__ = ["run_lifecycle_pipeline", "LifecyclePipelineResult"]
