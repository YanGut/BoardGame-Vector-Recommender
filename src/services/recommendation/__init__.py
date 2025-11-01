"""
Recommendation service submodules.

This package contains cohesive building blocks used by the
`RecommendationService` orchestrator while keeping the external API stable.
"""

from .mapper import document_to_game_dict, document_to_hybrid_game_dict
from .pipeline_factory import build_query_pipeline
from .ranker import hybrid_rank
from .retrieval import run_text_retrieval

__all__ = [
    "document_to_game_dict",
    "document_to_hybrid_game_dict",
    "build_query_pipeline",
    "hybrid_rank",
    "run_text_retrieval",
]
