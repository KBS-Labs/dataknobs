"""Specialized vector store implementations."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Type

from dataknobs_common.registry import PluginRegistry
from dataknobs_common.structured_config import StructuredConfig, config_registries

from .base import VectorStore

logger = logging.getLogger(__name__)


def _register_vector_backends(
    registry: PluginRegistry[Type[VectorStore]],
) -> None:
    """Auto-register all available built-in vector backends."""
    # Memory backend (always available)
    try:
        from .memory import MemoryVectorStore

        registry.register(
            "memory",
            MemoryVectorStore,
            metadata={
                "description": "In-memory vector storage for testing",
                "persistent": False,
                "requires_install": False,
                "config_options": {
                    "dimensions": "Vector dimensions (required)",
                    "metric": "Distance metric: cosine, euclidean, dot_product",
                },
            },
        )
    except ImportError:
        pass

    # Faiss backend
    try:
        from .faiss import FaissVectorStore

        registry.register(
            "faiss",
            FaissVectorStore,
            metadata={
                "description": "Facebook AI Similarity Search - efficient vector search",
                "persistent": True,
                "requires_install": "pip install faiss-cpu",
                "config_options": {
                    "dimensions": "Vector dimensions (required)",
                    "metric": "Distance metric: cosine, euclidean, dot_product",
                    "index_type": "Index type: flat, ivfflat, hnsw, auto",
                    "persist_path": "Path to save/load index",
                    "nlist": "Number of clusters for IVF index",
                    "m": "Number of connections for HNSW",
                },
            },
        )
    except ImportError:
        pass

    # Chroma backend
    try:
        from .chroma import ChromaVectorStore

        registry.register(
            "chroma",
            ChromaVectorStore,
            metadata={
                "description": "ChromaDB - AI-native vector database",
                "persistent": True,
                "requires_install": "pip install chromadb",
                "config_options": {
                    "collection_name": "Name of the collection",
                    "persist_path": "Path for persistent storage",
                    "embedding_function": "Embedding function name or object",
                    "metric": "Distance metric: cosine, euclidean, dot_product",
                },
            },
        )
        registry.register("chromadb", ChromaVectorStore)  # Alias
    except ImportError:
        pass

    # pgvector backend
    try:
        from .pgvector import PgVectorStore

        registry.register(
            "pgvector",
            PgVectorStore,
            metadata={
                "description": "PostgreSQL with pgvector extension - production vector database",
                "persistent": True,
                "requires_install": "pip install asyncpg",
                "config_options": {
                    "connection_string": "PostgreSQL connection URL (or use DATABASE_URL env)",
                    "dimensions": "Vector dimensions (required)",
                    "metric": "Distance metric: cosine, euclidean, inner_product",
                    "schema": "Database schema (default: public)",
                    "table_name": "Table name (default: knowledge_embeddings)",
                    "domain_id": "Domain ID for multi-tenant isolation (optional)",
                    "pool_min_size": "Min connection pool size (default: 2)",
                    "pool_max_size": "Max connection pool size (default: 10)",
                    "columns": "Column name mappings dict (optional)",
                    "auto_create_table": "Create table if missing (default: True)",
                    "id_type": "ID column type: uuid or text (default: text)",
                },
            },
        )
        registry.register("postgresql", PgVectorStore)  # Alias
    except ImportError:
        pass


# Create singleton instance BEFORE importing factory to avoid circular import
vector_backends: PluginRegistry[Type[VectorStore]] = PluginRegistry(
    "vector_backends",
    canonicalize_keys=True,
    on_first_access=_register_vector_backends,
)

# Keep VectorBackendRegistry as alias for backward compat.
# Use the unparameterized class so isinstance() checks still work at runtime.
VectorBackendRegistry = PluginRegistry

# Now import factory (which will import vector_backends from this module)
from .factory import VectorStoreFactory  # noqa: E402


def _resolve_vector_store_config_cls(
    raw: Mapping[str, Any],
) -> type[StructuredConfig] | None:
    """Resolve a ``vector_store`` section's dict to its config class.

    The resolver registered for the ``"vector_store"`` binding in
    :data:`~dataknobs_common.structured_config.config_registries`, used by
    :meth:`StructuredConfig.validate
    <dataknobs_common.structured_config.StructuredConfig.validate>` to
    validate a raw ``vector_store`` config without constructing the store.

    Delegates to ``vector_backends`` — the same registry the construction
    path uses — by reading ``CONFIG_CLS`` off the registered store class
    for the ``"backend"`` discriminator (defaulting to ``"memory"``, the
    factory's own default). Holding no independent backend→config-class
    table is the no-drift guarantee. Returns ``None`` for an unknown
    backend, which ``validate`` surfaces as a ``ConfigurationError``.
    """
    backend = raw.get("backend", "memory")
    store_cls = vector_backends.get_factory(backend)
    config_cls = getattr(store_cls, "CONFIG_CLS", None)
    if isinstance(config_cls, type) and issubclass(config_cls, StructuredConfig):
        return config_cls
    return None


# Eager registration (not on_first_access): importing this package is what
# makes the ``vector_store`` binding resolvable, and any parent config that
# holds a vector-store section already depends on this package. ``override``
# keeps re-import idempotent.
config_registries.register(
    "vector_store", _resolve_vector_store_config_cls, allow_overwrite=True
)


__all__ = [
    "VectorStore",
    "VectorStoreFactory",
    "VectorBackendRegistry",
    "vector_backends",
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

try:
    from .pgvector import PgVectorStore

    __all__ += ["PgVectorStore"]
except ImportError:
    pass
