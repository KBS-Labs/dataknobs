"""Knowledge base implementations for DynaBot."""

from typing import Any

# Re-export hybrid search types
from dataknobs_data.vector.hybrid import (
    FusionStrategy,
    HybridSearchConfig,
    HybridSearchResult,
)

# Re-export ingestion types for convenience
from dataknobs_xization.ingestion import (
    DirectoryProcessor,
    FilePatternConfig,
    KnowledgeBaseConfig,
    ProcessedDocument,
)

from .base import KnowledgeBase
from .config import RAGKnowledgeBaseConfig
from .events import (
    INGEST_DOMAIN_END,
    INGEST_DOMAIN_START,
    INGEST_METADATA_WRITE,
    INGEST_SNAPSHOT_WRITE,
    KnowledgeTriggerPayload,
    TenantFilteredCallback,
)
from .ingestion import (
    IngestionResult,
    IngestSwapMode,
    KnowledgeIngestionManager,
)
from .orchestration import IngestionManagerResolver, IngestOrchestrator
from .query import (
    ContextualExpander,
    Message,
    QueryTransformer,
    TransformerConfig,
    create_transformer,
    is_ambiguous_query,
)
from .rag import RAGKnowledgeBase
from .registry import (
    create_knowledge_base_from_config,
    get_knowledge_base_backend_factory,
    is_knowledge_base_backend_registered,
    knowledge_base_backends,
    list_knowledge_base_backends,
    register_knowledge_base_backend,
)
from .registry_mixin import AutoIngestionMixin
from .retrieval import (
    ChunkMerger,
    ContextFormatter,
    FormatterConfig,
    MergedChunk,
    MergerConfig,
)
from .service import (
    EnsureIngestionResult,
    KnowledgeIngestionService,
    ensure_knowledge_base_ingested,
    get_ingestion_service,
)
from .storage import (
    BackendKeyDiscriminator,
    ChangeSet,
    FileKnowledgeBackend,
    IngestionStatus,
    InMemoryKnowledgeBackend,
    InvalidVersionError,
    KnowledgeBaseInfo,
    KnowledgeFile,
    KnowledgeKeyKind,
    KnowledgeResourceBackend,
    KnowledgeResourceBackendMixin,
    create_knowledge_backend,
)

__all__ = [
    # Main knowledge base
    "KnowledgeBase",
    "RAGKnowledgeBase",
    "RAGKnowledgeBaseConfig",
    "create_knowledge_base_from_config",
    "register_knowledge_base_backend",
    "get_knowledge_base_backend_factory",
    "is_knowledge_base_backend_registered",
    "list_knowledge_base_backends",
    "knowledge_base_backends",
    # Retrieval utilities
    "ChunkMerger",
    "MergedChunk",
    "MergerConfig",
    "ContextFormatter",
    "FormatterConfig",
    # Query utilities
    "QueryTransformer",
    "TransformerConfig",
    "create_transformer",
    "ContextualExpander",
    "Message",
    "is_ambiguous_query",
    # Storage backends
    "KnowledgeResourceBackend",
    "KnowledgeResourceBackendMixin",
    "KnowledgeKeyKind",
    "KnowledgeFile",
    "KnowledgeBaseInfo",
    "IngestionStatus",
    "ChangeSet",
    "InvalidVersionError",
    "create_knowledge_backend",
    "InMemoryKnowledgeBackend",
    "FileKnowledgeBackend",
    "S3KnowledgeBackend",
    # Discriminator adapters
    "BackendKeyDiscriminator",
    # Ingestion manager (file-backend to vector-store)
    "KnowledgeIngestionManager",
    "IngestionResult",
    "IngestSwapMode",
    # Event-driven orchestration
    "IngestOrchestrator",
    "IngestionManagerResolver",
    # Knowledge-layer event topics + payload + filter adapter
    "INGEST_DOMAIN_START",
    "INGEST_DOMAIN_END",
    "INGEST_METADATA_WRITE",
    "INGEST_SNAPSHOT_WRITE",
    "KnowledgeTriggerPayload",
    "TenantFilteredCallback",
    # High-level ingestion service
    "KnowledgeIngestionService",
    "EnsureIngestionResult",
    "get_ingestion_service",
    "ensure_knowledge_base_ingested",
    "AutoIngestionMixin",
    # Ingestion types (from xization)
    "DirectoryProcessor",
    "FilePatternConfig",
    "KnowledgeBaseConfig",
    "ProcessedDocument",
    # Hybrid search types
    "FusionStrategy",
    "HybridSearchConfig",
    "HybridSearchResult",
]


def __getattr__(name: str) -> Any:
    """Lazy module-level attribute access for S3KnowledgeBackend.

    Avoids requiring boto3 at import time — only loaded when S3KnowledgeBackend
    is actually accessed.
    """
    if name == "S3KnowledgeBackend":
        from .storage import S3KnowledgeBackend

        return S3KnowledgeBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
