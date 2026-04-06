"""Pluggable document chunking for RAG applications.

This module provides a ``Chunker`` abstraction, a ``ChunkTransform``
post-processing primitive, and ``PluginRegistry``-backed registries so
that consuming code (``RAGKnowledgeBase``, ``DirectoryProcessor``,
standalone pipelines) can select the chunker and transforms via
configuration.

Quick start::

    from dataknobs_xization.chunking import create_chunker

    # Default — wraps existing MarkdownChunker with content-level interface
    chunker = create_chunker({"max_chunk_size": 800})

    # With transforms — pipeline:
    chunker = create_chunker({
        "chunker": "markdown_tree",
        "max_chunk_size": 800,
        "transforms": [
            {"merge_small": {"min_size": 200}},
        ],
    })

    # Custom implementation via dotted import path
    chunker = create_chunker({"chunker": "my_project.chunkers.RFCChunker"})

Registering a custom chunker at import time::

    from dataknobs_xization.chunking import Chunker, register_chunker

    class PlaintextChunker(Chunker):
        ...

    register_chunker("plaintext", PlaintextChunker)
"""

from dataknobs_xization.chunking.base import ChunkTransform, Chunker, DocumentInfo
from dataknobs_xization.chunking.composite import CompositeChunker
from dataknobs_xization.chunking.markdown import MarkdownTreeChunker
from dataknobs_xization.chunking.registry import (
    chunker_registry,
    create_chunker,
    register_chunker,
    register_transform,
    transform_registry,
)
from dataknobs_xization.chunking.transforms import (
    MergeSmallChunks,
    QualityFilterTransform,
    SplitLargeChunks,
)
from dataknobs_xization.chunking.utils import split_text

__all__ = [
    "ChunkTransform",
    "Chunker",
    "CompositeChunker",
    "DocumentInfo",
    "MarkdownTreeChunker",
    "MergeSmallChunks",
    "QualityFilterTransform",
    "SplitLargeChunks",
    "chunker_registry",
    "create_chunker",
    "register_chunker",
    "register_transform",
    "split_text",
    "transform_registry",
]
