"""Typed configuration dataclass for the knowledge-base backends.

The RAG knowledge base documents a small set of configuration keys.
This :class:`~dataknobs_common.structured_config.StructuredConfig`
subclass captures those keys with defaults so the backend factory can
project a raw config dict onto a typed, validated, round-trippable
object before constructing the concrete knowledge base.

The ``vector_store``, ``embedding``, ``chunking``, ``merger``, and
``formatter`` sections stay raw mappings — each is consumed by a
factory owned elsewhere (``dataknobs-data`` for the vector store,
``dataknobs-llm`` for the embedder, ``dataknobs-xization`` for the
chunker, and the retrieval module for merger/formatter) rather than
being baked into this static field graph.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from dataknobs_common.structured_config import StructuredConfig


@dataclass(frozen=True)
class RAGKnowledgeBaseConfig(StructuredConfig):
    """Configuration for :class:`~dataknobs_bots.knowledge.rag.RAGKnowledgeBase`.

    Attributes:
        vector_store: Vector-store config forwarded to
            ``VectorStoreFactory``. Required in practice; kept defaulted
            (empty dict) so the config is default-constructible for the
            pre-built ``from_components`` path, with the missing-key
            failure surfacing from the factory as it does today.
        embedding: Nested embedding-provider config (preferred). Kept a
            raw mapping — embedder config is owned by ``dataknobs-llm``.
        embedding_provider: Legacy flat embedding-provider key.
        embedding_model: Legacy flat embedding-model key.
        dimensions: Embedder dimension, forwarded to the embedding
            provider as a legacy flat passthrough.
        api_base: Custom embedder endpoint, forwarded to the embedding
            provider as a legacy flat passthrough.
        api_key: Embedder credential, forwarded to the embedding provider
            as a legacy flat passthrough. Redacted from ``repr``.
        chunking: Chunking config forwarded to ``create_chunker``.
        merger: Optional chunk-merger config (raw mapping projected onto
            ``MergerConfig``).
        formatter: Optional context-formatter config (raw mapping
            projected onto ``FormatterConfig``).
        documents_path: Optional directory to ingest on async warmup.
        document_pattern: Glob pattern used when ``documents_path`` is set.
    """

    # Adopt polymorphic-section validation for the nested ``vector_store``
    # and ``embedding`` sections: a
    # ``RAGKnowledgeBaseConfig.from_dict(raw).validate()`` dry-run-builds each
    # bound section's config to surface field errors / an unknown discriminator
    # at config-parse time (without constructing the store or embedder). Both
    # bindings are strings — ``dataknobs-data`` registers the ``"vector_store"``
    # resolver and ``dataknobs-llm`` registers the ``"embedding"`` resolver
    # (an embedder config is an ``LLMConfig``, keyed by ``provider``) — so this
    # adds no import of any data/llm config type.
    #
    # Only the *nested* ``embedding`` dict is validated. The legacy flat
    # passthroughs (``embedding_provider`` / ``embedding_model`` / ``dimensions``
    # / ``api_base`` / ``api_key``) are intentionally left unvalidated: they are
    # legacy and slated for removal, and lack a ``provider`` discriminator to
    # resolve on. A config using only flat keys has an empty nested ``embedding``
    # → ``validate()`` skips it (empty-section rule), so no false positive.
    _polymorphic_fields: ClassVar[Mapping[str, str]] = {
        "vector_store": "vector_store",
        "embedding": "embedding",
    }

    vector_store: dict[str, Any] = field(default_factory=dict)
    embedding: dict[str, Any] | None = None
    embedding_provider: str | None = None
    embedding_model: str | None = None
    dimensions: int | None = None
    api_base: str | None = None
    api_key: str | None = None
    chunking: dict[str, Any] = field(default_factory=dict)
    merger: dict[str, Any] | None = None
    formatter: dict[str, Any] | None = None
    documents_path: str | None = None
    document_pattern: str = "**/*.md"
    # When set, every write auto-stamps ``tenant_id`` into chunk metadata
    # (auto-derived wins on collision so a caller cannot silently re-tag
    # chunks for another tenant) and every read AND-composes
    # ``{"tenant_id": tenant_id}`` into the vector-store search filter
    # (explicit-filter-wins, so admin tooling can legitimately read
    # across tenants by passing the explicit key). ``None`` (default) is
    # the single-tenant byte-identical posture.
    tenant_id: str | None = None

    # Redacted from ``repr`` by the StructuredConfig base. A secret nested
    # inside a raw mapping section (``embedding``'s ``api_key``,
    # ``vector_store``'s ``connection_string``) is also masked: the base's
    # repr descends into raw ``Mapping``/``list`` fields and masks interior
    # keys in its default sensitive-key set unioned with this set.
    _SENSITIVE_FIELDS: ClassVar[frozenset[str]] = frozenset({"api_key"})
