"""Grounded source abstraction for structured retrieval.

Provides the types and ABC for queryable data sources that participate
in a grounded retrieval pipeline.  Sources declare schemas, receive
structured intent, and translate it deterministically to native queries.
"""

from .base import (
    GroundedSource,
    RetrievalIntent,
    SourceResult,
    SourceSchema,
    StrategyUnavailable,
)
from .cluster_index import (
    ClusterTopicConfig,
    ClusterTopicIndex,
    DEFAULT_LABEL_MIN_WORD_LENGTH,
    DEFAULT_LABEL_TOP_TERMS,
)
from .cluster_index import VectorQueryFn as ClusterVectorQueryFn
from .database import DatabaseSource
from .topic_index import (
    DEFAULT_HEADING_EXCLUDE_PATTERNS,
    DEFAULT_HEADING_STOPWORDS,
    DEFAULT_MIN_WORD_LENGTH,
    HeadingMatchConfig,
    TopicIndex,
    TopicNode,
    build_heading_tree,
    expand_region,
    extract_query_words,
    find_heading_regions,
)
from .processing import (
    CrossSourceNormalizer,
    EmbeddingClusterer,
    agglomerative_cluster,
    cosine_similarity,
    QueryClusterScorer,
    QueryRelevanceRanker,
    RelativeRelevanceFilter,
    ResultPipeline,
    ResultProcessor,
    StrategyChain,
    TermOverlapClusterer,
    TfidfClusterer,
    build_pipeline,
    inject_embedder,
)

__all__ = [
    "ClusterTopicConfig",
    "ClusterTopicIndex",
    "ClusterVectorQueryFn",
    "CrossSourceNormalizer",
    "DEFAULT_HEADING_EXCLUDE_PATTERNS",
    "DEFAULT_HEADING_STOPWORDS",
    "DEFAULT_LABEL_MIN_WORD_LENGTH",
    "DEFAULT_LABEL_TOP_TERMS",
    "DEFAULT_MIN_WORD_LENGTH",
    "agglomerative_cluster",
    "cosine_similarity",
    "DatabaseSource",
    "EmbeddingClusterer",
    "GroundedSource",
    "HeadingMatchConfig",
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
    "TopicIndex",
    "TopicNode",
    "build_heading_tree",
    "build_pipeline",
    "expand_region",
    "extract_query_words",
    "find_heading_regions",
    "inject_embedder",
]
