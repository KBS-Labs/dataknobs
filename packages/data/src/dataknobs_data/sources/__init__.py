"""Grounded source abstraction for structured retrieval.

Provides the types and ABC for queryable data sources that participate
in a grounded retrieval pipeline.  Sources declare schemas, receive
structured intent, and translate it deterministically to native queries.
"""

from .base import GroundedSource, RetrievalIntent, SourceResult, SourceSchema
from .database import DatabaseSource

__all__ = [
    "GroundedSource",
    "DatabaseSource",
    "RetrievalIntent",
    "SourceResult",
    "SourceSchema",
]
