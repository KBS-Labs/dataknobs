"""Grounded source implementations for DynaBot.

Re-exports core types from ``dataknobs-data`` and ``dataknobs-llm``,
plus bot-specific source implementations.
"""

# Core types (from dataknobs-data)
from dataknobs_data.sources import (
    GroundedSource,
    RetrievalIntent,
    SourceResult,
    SourceSchema,
)

# Intent composition (from dataknobs-llm)
from dataknobs_llm.sources import compose_intent_schema, parse_intent

# Bot-specific sources
from .factory import create_source_from_config
from .heading_tree import HeadingTreeConfig, HeadingTreeIndex
from .vector import VectorKnowledgeSource

__all__ = [
    # Core types
    "GroundedSource",
    "RetrievalIntent",
    "SourceResult",
    "SourceSchema",
    # Intent composition
    "compose_intent_schema",
    "parse_intent",
    # Bot-specific
    "HeadingTreeConfig",
    "HeadingTreeIndex",
    "VectorKnowledgeSource",
    "create_source_from_config",
]
