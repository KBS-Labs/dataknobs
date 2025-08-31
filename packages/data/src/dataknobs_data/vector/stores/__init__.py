"""Specialized vector store implementations."""

from .base import VectorStore
from .factory import VectorStoreFactory

__all__ = [
    "VectorStore",
    "VectorStoreFactory",
]

# Import specialized stores when available
try:
    from .faiss import FaissVectorStore
    __all__ += ["FaissVectorStore"]
except ImportError:
    pass

try:
    from .chroma import ChromaVectorStore
    __all__ += ["ChromaVectorStore"]
except ImportError:
    pass
