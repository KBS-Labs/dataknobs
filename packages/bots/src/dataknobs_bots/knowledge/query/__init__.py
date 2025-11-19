"""Query transformation utilities for RAG knowledge bases.

This module provides query preprocessing to improve retrieval quality
by transforming user input into optimized search queries.
"""

from dataknobs_bots.knowledge.query.expander import (
    ContextualExpander,
    Message,
    is_ambiguous_query,
)
from dataknobs_bots.knowledge.query.transformer import (
    QueryTransformer,
    TransformerConfig,
    create_transformer,
)

__all__ = [
    "QueryTransformer",
    "TransformerConfig",
    "create_transformer",
    "ContextualExpander",
    "Message",
    "is_ambiguous_query",
]
