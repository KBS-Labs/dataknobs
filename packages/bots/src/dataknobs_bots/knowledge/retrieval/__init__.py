"""Retrieval utilities for RAG knowledge bases.

This module provides post-retrieval processing for improving RAG quality,
including chunk merging, context formatting, and result optimization.
"""

from dataknobs_bots.knowledge.retrieval.formatter import (
    ContextFormatter,
    FormatterConfig,
)
from dataknobs_bots.knowledge.retrieval.merger import (
    ChunkMerger,
    MergedChunk,
    MergerConfig,
)

__all__ = [
    "ChunkMerger",
    "MergedChunk",
    "MergerConfig",
    "ContextFormatter",
    "FormatterConfig",
]
