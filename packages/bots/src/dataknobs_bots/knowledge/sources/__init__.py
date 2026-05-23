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
from .factory import (
    create_source_from_config,
    get_source_backend_factory,
    is_source_backend_registered,
    list_source_backends,
    register_source_backend,
    source_backends,
)
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
    "register_source_backend",
    "get_source_backend_factory",
    "is_source_backend_registered",
    "list_source_backends",
    "source_backends",
]
