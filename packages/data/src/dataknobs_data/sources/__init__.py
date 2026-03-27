"""Grounded source abstraction for structured retrieval.

Provides the types and ABC for queryable data sources that participate
in a grounded retrieval pipeline.  Sources declare schemas, receive
structured intent, and translate it deterministically to native queries.
"""

from .base import GroundedSource, RetrievalIntent, SourceResult, SourceSchema
from .database import DatabaseSource
from .processing import (
    CrossSourceNormalizer,
    EmbedFn,
    EmbeddingClusterer,
    QueryClusterScorer,
    QueryRelevanceRanker,
    RelativeRelevanceFilter,
    ResultPipeline,
    ResultProcessor,
    StrategyChain,
    StrategyUnavailable,
    TermOverlapClusterer,
    TfidfClusterer,
    build_pipeline,
    inject_embed_fn,
)

__all__ = [
    "CrossSourceNormalizer",
    "DatabaseSource",
    "EmbedFn",
    "EmbeddingClusterer",
    "GroundedSource",
    "QueryClusterScorer",
    "QueryRelevanceRanker",
    "RelativeRelevanceFilter",
    "ResultPipeline",
    "ResultProcessor",
    "RetrievalIntent",
    "SourceResult",
    "SourceSchema",
    "StrategyChain",
    "StrategyUnavailable",
    "TermOverlapClusterer",
    "TfidfClusterer",
    "build_pipeline",
    "inject_embed_fn",
]
