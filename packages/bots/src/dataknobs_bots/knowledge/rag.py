"""RAG (Retrieval-Augmented Generation) knowledge base implementation."""

import logging
import types
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self, TypeVar

from dataknobs_bots.knowledge.base import KnowledgeBase
from dataknobs_bots.knowledge.config import RAGKnowledgeBaseConfig
from dataknobs_bots.knowledge.retrieval import (
    ChunkMerger,
    ContextFormatter,
    FormatterConfig,
    MergerConfig,
)
from dataknobs_common.capabilities import (
    Capability,
    CapabilityLike,
    CapabilityMixin,
)
from dataknobs_common.lifecycle import close_if_owned
from dataknobs_common.metadata import enforce_immutable_keys
from dataknobs_common.structured_config import StructuredConfigConsumer
from dataknobs_xization import (
    ContentTransformer,
    create_chunker,
)
from dataknobs_xization.chunking.base import Chunker, DocumentInfo
from dataknobs_xization.ingestion import (
    BackendDocumentSource,
    DirectoryProcessor,
    KnowledgeBaseConfig,
)

if TYPE_CHECKING:
    from dataknobs_bots.knowledge.storage.backend import KnowledgeResourceBackend
    from dataknobs_bots.knowledge.storage.models import KnowledgeFile
    from dataknobs_common.ratelimit import RateLimiter

logger = logging.getLogger(__name__)

# Raw search-result row type for the shared stale-gate helper
# (a vector ``(id, score, meta)`` tuple or a ``HybridSearchResult``).
_R = TypeVar("_R")


class RAGKnowledgeBase(
    StructuredConfigConsumer[RAGKnowledgeBaseConfig],
    CapabilityMixin,
    KnowledgeBase,
):
    """RAG knowledge base using dataknobs-xization for chunking and vector search.

    This implementation:
    - Parses markdown documents using dataknobs-xization
    - Chunks documents intelligently based on structure
    - Stores chunks with embeddings in vector store
    - Provides semantic search for relevant context

    Construct from config (``await RAGKnowledgeBase.from_config({...})`` —
    builds the vector store + embedder and optionally ingests
    ``documents_path``) or from pre-built collaborators
    (``RAGKnowledgeBase.from_components(vector_store=…,
    embedding_provider=…)``). The chunker, merger, and formatter are
    built synchronously from config (``chunking`` / ``merger`` /
    ``formatter``) or accepted as pre-built ``chunker`` /
    ``merger_config`` / ``formatter_config`` collaborators.

    Attributes:
        vector_store: Vector store backend from dataknobs_data
        embedding_provider: LLM provider for generating embeddings
        chunking_config: Configuration for document chunking
    """

    CONFIG_CLS: ClassVar[type[RAGKnowledgeBaseConfig]] = RAGKnowledgeBaseConfig

    # Capability advertisement (per the chunk-layer Tenancy capability).
    # The class HAS the chunk-layer tenant-scoping code path
    # (``tenant_id`` folds into ``_CHUNK_ID_PREFIX_KEYS``, the bound
    # ``tenant_id`` stamps onto chunk metadata, and ``_resolve_read_filter``
    # AND-composes the bound tenant onto every read). Advertisement is
    # **structural**, not activation-state — an unbound instance still
    # advertises the capability because the class has the code path;
    # whether a specific instance is currently chunk-scoping is the
    # natural binding check (``kb._tenant_id is not None``).
    # ``Capability.TENANT_SCOPED_STATE`` is deliberately not declared
    # here: backend-state writes are still per-domain, not per-tenant,
    # at the :class:`KnowledgeResourceBackend` layer.
    SUPPORTED_CAPABILITIES: ClassVar[frozenset[CapabilityLike]] = frozenset(
        {Capability.TENANT_SCOPED_CHUNKS}
    )

    # Ordered metadata keys folded into the chunk-id prefix when present
    # (the present-and-truthy values join into the prefix in declared
    # order; ``source_stem`` is always last). Order is significant —
    # most-scoping first — so the prefix reads
    # ``<tenant>\x1f<domain>\x1f<gen>\x1f<stem>`` for a fully-tagged
    # multi-tenant store. Subclasses rebind to add fold positions (e.g.
    # ``("tenant_id", "domain_id", "region", "_generation")``) without
    # forking the derivation. Single-tenant consumers see no change:
    # ``tenant_id`` absent in metadata yields the historical
    # ``[domain_id?, generation?, source_stem]`` shape.
    _CHUNK_ID_PREFIX_KEYS: ClassVar[tuple[str, ...]] = (
        "tenant_id",
        "domain_id",
        "_generation",
    )

    # Identity tags the KB owns: auto-derived from the bound ``tenant_id``
    # / per-call ``domain_id`` / TOMBSTONE-generation token rather than
    # accepted from consumer-supplied ``extra_metadata``. A caller
    # passing these keys in ``extra_metadata`` is shadowed by the
    # auto-derived value so identity cannot be silently re-tagged for
    # another tenant. Non-identity keys (``region``, ``cohort``, any
    # custom tag) are preserved unchanged through the merge. Documented
    # in the multi-tenant USER_GUIDE under "Reserved vs. Consumer-
    # Extensible Metadata Keys".
    _RESERVED_METADATA_KEYS: ClassVar[frozenset[str]] = frozenset(
        {"tenant_id", "domain_id", "_generation"}
    )

    def _setup(self) -> None:
        """Build the chunker, merger, and formatter (synchronous).

        These are config-derived but require no async work, so they are
        built here for both construction paths. Injected pre-built
        collaborators (``chunker`` / ``merger_config`` /
        ``formatter_config``, available on ``self.components``) take
        precedence over the config dicts. The vector store and embedding
        provider are bound by :meth:`_ainit` (config-driven) or
        :meth:`_adopt_components` (pre-built injection).
        """
        self.chunking_config = self.config.chunking or {
            "max_chunk_size": 500,
            "combine_under_heading": True,
        }

        # Resolve chunker: injected instance > config-driven > default
        injected_chunker = self.components.get("chunker")
        self._chunker: Chunker = injected_chunker or create_chunker(
            self.chunking_config
        )

        # Resolve merger: injected MergerConfig > config dict > default
        merger_config = self.components.get("merger_config")
        if merger_config is None and self.config.merger is not None:
            merger_config = MergerConfig(**self.config.merger)
        self.merger = ChunkMerger(merger_config) if merger_config else ChunkMerger()

        # Resolve formatter: injected FormatterConfig > config dict > default
        formatter_config = self.components.get("formatter_config")
        if formatter_config is None and self.config.formatter is not None:
            formatter_config = FormatterConfig(**self.config.formatter)
        self.formatter = (
            ContextFormatter(formatter_config)
            if formatter_config
            else ContextFormatter()
        )

        self.vector_store: Any = None
        self.embedding_provider: Any = None
        # Ownership of the cascade-closed collaborators. Bound by
        # :meth:`_ainit` (config-driven build owns what it creates → True)
        # or :meth:`_adopt_components` (pre-built injection → caller-owned,
        # left False) — mirroring ``VectorMemory``. ``close()`` only tears
        # down collaborators this instance owns, so an injected shared
        # store/provider survives one holder's close.
        self._owns_vector_store = False
        self._owns_embedding_provider = False

        # Bound-tenant binding: when set, every write auto-stamps
        # ``tenant_id`` into chunk metadata and every read AND-composes
        # the bound tenant into the vector-store search filter. None
        # (default) preserves the single-tenant byte-identical posture.
        self._tenant_id: str | None = self.config.tenant_id

    @classmethod
    async def from_config(  # type: ignore[override]
        cls, config: Any, **components: Any
    ) -> "RAGKnowledgeBase":
        """Create RAG knowledge base from configuration (async warmup).

        Builds the vector store and embedding provider, awaits the
        store's ``initialize()``, and ingests ``documents_path`` when set.
        Accepts a config dict or a typed :class:`RAGKnowledgeBaseConfig`.
        The config keys are:

        - ``vector_store``: Vector store configuration
        - ``embedding``: Nested embedding config dict (preferred), e.g.
          ``{"provider": "ollama", "model": "nomic-embed-text"}``
        - ``embedding_provider`` / ``embedding_model``: Legacy flat keys
        - ``chunking``: Optional chunking configuration
        - ``merger`` / ``formatter``: Optional retrieval configuration
        - ``documents_path``: Optional path to load documents from
        - ``document_pattern``: Optional glob pattern for documents

        Example:
            ```python
            config = {
                "vector_store": {
                    "backend": "faiss",
                    "dimensions": 768,
                    "collection": "docs"
                },
                "embedding": {
                    "provider": "ollama",
                    "model": "nomic-embed-text",
                },
                "chunking": {
                    "max_chunk_size": 500
                },
                "documents_path": "./docs"
            }
            kb = await RAGKnowledgeBase.from_config(config)
            ```
        """
        return await cls.from_config_async(config, **components)

    async def _ainit(self, **_: Any) -> None:
        """Build the vector store + embedder and optionally ingest docs."""
        if self._prebuilt:
            return
        from dataknobs_data.vector.stores import VectorStoreFactory

        from ..providers import build_embedding_config, create_embedding_provider

        factory = VectorStoreFactory()
        self.vector_store = factory.create(**self.config.vector_store)
        await self.vector_store.initialize()
        self._owns_vector_store = True

        self.embedding_provider = await create_embedding_provider(
            build_embedding_config(
                embedding=self.config.embedding,
                embedding_provider=self.config.embedding_provider,
                embedding_model=self.config.embedding_model,
                dimensions=self.config.dimensions,
                api_base=self.config.api_base,
                api_key=self.config.api_key,
            )
        )
        self._owns_embedding_provider = True

        if self.config.documents_path is not None:
            await self.load_documents_from_directory(
                self.config.documents_path, self.config.document_pattern
            )

    def _adopt_components(
        self,
        *,
        vector_store: Any = None,
        embedding_provider: Any = None,
        **_: Any,
    ) -> None:
        """Adopt caller-owned store + embedder for ``from_components``."""
        if vector_store is None or embedding_provider is None:
            raise TypeError(
                "RAGKnowledgeBase.from_components requires vector_store and "
                "embedding_provider"
            )
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self._owns_vector_store = False
        self._owns_embedding_provider = False

    async def load_markdown_document(
        self,
        filepath: str | Path,
        metadata: dict[str, Any] | None = None,
        *,
        extra_metadata: Mapping[str, Any] | None = None,
        tenant_id: str | None = None,
    ) -> int:
        """Load and chunk a markdown document from a file.

        Reads the file and delegates to :meth:`load_markdown_text` for
        parsing, chunking, embedding, and storage. ``extra_metadata`` /
        ``tenant_id`` are forwarded uniformly — see
        :meth:`load_markdown_text` for the precedence rules.

        Args:
            filepath: Path to markdown file
            metadata: Optional per-document metadata
            extra_metadata: Optional cross-document identity carrier
                (see :meth:`load_markdown_text`).
            tenant_id: Optional convenience kwarg folded into
                ``extra_metadata`` (see :meth:`load_markdown_text`).

        Returns:
            Number of chunks created

        Example:
            ```python
            num_chunks = await kb.load_markdown_document(
                "docs/api.md",
                metadata={"category": "api", "version": "1.0"},
                tenant_id="acme",
            )
            ```
        """
        filepath = Path(filepath)
        with open(filepath, encoding="utf-8") as f:
            markdown_text = f.read()

        return await self.load_markdown_text(
            markdown_text,
            source=str(filepath),
            metadata=metadata,
            extra_metadata=extra_metadata,
            tenant_id=tenant_id,
        )

    async def load_documents_from_directory(
        self,
        directory: str | Path,
        pattern: str = "**/*.md",
        *,
        extra_metadata: Mapping[str, Any] | None = None,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        """Load all markdown documents from a directory.

        Args:
            directory: Directory path containing documents
            pattern: Glob pattern for files to load (default: **/*.md)
            extra_metadata: Optional cross-document identity carrier
                (see :meth:`load_markdown_text`).
            tenant_id: Optional convenience kwarg folded into
                ``extra_metadata`` (see :meth:`load_markdown_text`).

        Returns:
            Dictionary with loading statistics:
                - total_files: Number of files processed
                - total_chunks: Total chunks created
                - errors: List of errors encountered

        Example:
            ```python
            results = await kb.load_documents_from_directory(
                "docs/",
                pattern="**/*.md",
                tenant_id="acme",
            )
            print(f"Loaded {results['total_chunks']} chunks from {results['total_files']} files")
            ```
        """
        directory = Path(directory)
        results = {"total_files": 0, "total_chunks": 0, "errors": []}

        for filepath in directory.glob(pattern):
            if not filepath.is_file():
                continue

            try:
                num_chunks = await self.load_markdown_document(
                    filepath,
                    metadata={"filename": filepath.name},
                    extra_metadata=extra_metadata,
                    tenant_id=tenant_id,
                )
                results["total_files"] += 1
                results["total_chunks"] += num_chunks
            except Exception as e:
                results["errors"].append({"file": str(filepath), "error": str(e)})

        return results

    async def load_json_document(
        self,
        filepath: str | Path,
        metadata: dict[str, Any] | None = None,
        schema: str | None = None,
        transformer: ContentTransformer | None = None,
        title: str | None = None,
        *,
        extra_metadata: Mapping[str, Any] | None = None,
        tenant_id: str | None = None,
    ) -> int:
        """Load and chunk a JSON document by converting it to markdown.

        This method converts JSON data to markdown format using ContentTransformer,
        then processes it like any other markdown document. ``extra_metadata`` /
        ``tenant_id`` are forwarded uniformly — see
        :meth:`load_markdown_text` for the precedence rules.

        Args:
            filepath: Path to JSON file
            metadata: Optional metadata to attach to all chunks
            schema: Optional schema name (requires transformer with registered schema)
            transformer: Optional ContentTransformer instance with custom configuration
            title: Optional document title for the markdown
            extra_metadata: Optional cross-document identity carrier
                (see :meth:`load_markdown_text`).
            tenant_id: Optional convenience kwarg folded into
                ``extra_metadata`` (see :meth:`load_markdown_text`).

        Returns:
            Number of chunks created

        Example:
            ```python
            # Generic conversion
            num_chunks = await kb.load_json_document(
                "data/patterns.json",
                metadata={"content_type": "patterns"}
            )

            # With custom schema
            transformer = ContentTransformer()
            transformer.register_schema("pattern", {
                "title_field": "name",
                "sections": [
                    {"field": "description", "heading": "Description"},
                    {"field": "example", "heading": "Example", "format": "code"}
                ]
            })
            num_chunks = await kb.load_json_document(
                "data/patterns.json",
                transformer=transformer,
                schema="pattern"
            )
            ```
        """
        import json

        filepath = Path(filepath)

        # Read JSON
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        # Convert to markdown
        if transformer is None:
            transformer = ContentTransformer()

        markdown_text = transformer.transform_json(
            data,
            schema=schema,
            title=title or filepath.stem.replace("_", " ").title(),
        )

        return await self.load_markdown_text(
            markdown_text,
            source=str(filepath),
            metadata=metadata,
            extra_metadata=extra_metadata,
            tenant_id=tenant_id,
        )

    async def load_yaml_document(
        self,
        filepath: str | Path,
        metadata: dict[str, Any] | None = None,
        schema: str | None = None,
        transformer: ContentTransformer | None = None,
        title: str | None = None,
        *,
        extra_metadata: Mapping[str, Any] | None = None,
        tenant_id: str | None = None,
    ) -> int:
        """Load and chunk a YAML document by converting it to markdown.

        Args:
            filepath: Path to YAML file
            metadata: Optional metadata to attach to all chunks
            schema: Optional schema name (requires transformer with registered schema)
            transformer: Optional ContentTransformer instance with custom configuration
            title: Optional document title for the markdown
            extra_metadata: Optional cross-document identity carrier
                (see :meth:`load_markdown_text`).
            tenant_id: Optional convenience kwarg folded into
                ``extra_metadata`` (see :meth:`load_markdown_text`).

        Returns:
            Number of chunks created

        Example:
            ```python
            num_chunks = await kb.load_yaml_document(
                "data/config.yaml",
                metadata={"content_type": "configuration"}
            )
            ```
        """
        filepath = Path(filepath)

        # Convert to markdown
        if transformer is None:
            transformer = ContentTransformer()

        markdown_text = transformer.transform_yaml(
            filepath,
            schema=schema,
            title=title or filepath.stem.replace("_", " ").title(),
        )

        return await self.load_markdown_text(
            markdown_text,
            source=str(filepath),
            metadata=metadata,
            extra_metadata=extra_metadata,
            tenant_id=tenant_id,
        )

    async def load_csv_document(
        self,
        filepath: str | Path,
        metadata: dict[str, Any] | None = None,
        title: str | None = None,
        title_field: str | None = None,
        transformer: ContentTransformer | None = None,
        *,
        extra_metadata: Mapping[str, Any] | None = None,
        tenant_id: str | None = None,
    ) -> int:
        """Load and chunk a CSV document by converting it to markdown.

        Each row becomes a section with the first column (or title_field) as heading.

        Args:
            filepath: Path to CSV file
            metadata: Optional metadata to attach to all chunks
            title: Optional document title for the markdown
            title_field: Column to use as section title (default: first column)
            transformer: Optional ContentTransformer instance with custom configuration
            extra_metadata: Optional cross-document identity carrier
                (see :meth:`load_markdown_text`).
            tenant_id: Optional convenience kwarg folded into
                ``extra_metadata`` (see :meth:`load_markdown_text`).

        Returns:
            Number of chunks created

        Example:
            ```python
            num_chunks = await kb.load_csv_document(
                "data/faq.csv",
                title="Frequently Asked Questions",
                title_field="question"
            )
            ```
        """
        filepath = Path(filepath)

        # Convert to markdown
        if transformer is None:
            transformer = ContentTransformer()

        markdown_text = transformer.transform_csv(
            filepath,
            title=title or filepath.stem.replace("_", " ").title(),
            title_field=title_field,
        )

        return await self.load_markdown_text(
            markdown_text,
            source=str(filepath),
            metadata=metadata,
            extra_metadata=extra_metadata,
            tenant_id=tenant_id,
        )

    async def load_from_directory(
        self,
        directory: str | Path,
        config: KnowledgeBaseConfig | None = None,
        progress_callback: Callable[[str, int], None] | None = None,
        *,
        extra_metadata: Mapping[str, Any] | None = None,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        """Load documents from a directory using KnowledgeBaseConfig.

        This method uses the xization DirectoryProcessor to process documents
        with configurable patterns, chunking, and metadata. It supports markdown,
        JSON, and JSONL files with streaming for large files.

        Args:
            directory: Directory path containing documents
            config: Optional KnowledgeBaseConfig. If not provided, attempts to load
                   from knowledge_base.json/yaml in the directory, or uses defaults.
            progress_callback: Optional callback function(file_path, num_chunks) for progress
            extra_metadata: Optional cross-document identity carrier
                (see :meth:`load_markdown_text`). Composed through
                :meth:`_compose_extra_metadata` so the bound tenant
                wins over any caller-supplied ``tenant_id`` key.
            tenant_id: Optional convenience kwarg folded into
                ``extra_metadata`` as ``{"tenant_id": tenant_id}``.

        Returns:
            Dictionary with loading statistics:
                - total_files: Number of files processed
                - total_chunks: Total chunks created
                - files_by_type: Count of files by type (markdown, json, jsonl)
                - errors: List of errors encountered
                - documents: List of processed document info

        Example:
            ```python
            # With auto-loaded config from directory
            results = await kb.load_from_directory("./docs")

            # With explicit config
            config = KnowledgeBaseConfig(
                name="product-docs",
                default_chunking={"max_chunk_size": 800},
                patterns=[
                    FilePatternConfig(pattern="api/**/*.json", text_fields=["title", "description"]),
                    FilePatternConfig(pattern="**/*.md"),
                ],
                exclude_patterns=["**/drafts/**"],
            )
            results = await kb.load_from_directory("./docs", config=config)
            print(f"Loaded {results['total_chunks']} chunks from {results['total_files']} files")
            ```
        """
        directory = Path(directory)

        if config is None:
            config = KnowledgeBaseConfig.load(directory)

        effective_extra: dict[str, Any] = dict(extra_metadata or {})
        if tenant_id is not None:
            effective_extra["tenant_id"] = tenant_id
        composed = self._compose_extra_metadata(effective_extra)

        processor = DirectoryProcessor(config, directory, chunker=self._chunker)
        return await self._ingest_from_processor_async(
            processor,
            progress_callback,
            extra_metadata=composed,
        )

    async def ingest_from_backend(
        self,
        backend: "KnowledgeResourceBackend",
        domain_id: str,
        config: KnowledgeBaseConfig | None = None,
        progress_callback: Callable[[str, int], None] | None = None,
        extra_metadata: Mapping[str, Any] | None = None,
        *,
        tenant_id: str | None = None,
        file_filter: Callable[["KnowledgeFile"], bool] | None = None,
        rate_limiter: "RateLimiter | None" = None,
    ) -> dict[str, Any]:
        """Ingest documents from a :class:`KnowledgeResourceBackend`.

        Equivalent to :meth:`load_from_directory` but drives the same
        :class:`DirectoryProcessor` pipeline against any backend (local
        :class:`FileKnowledgeBackend`, :class:`InMemoryKnowledgeBackend`,
        :class:`S3KnowledgeBackend`). All ``KnowledgeBaseConfig``
        richness — pattern-based chunking, exclude patterns,
        per-pattern metadata, streaming JSON — applies.

        When ``config`` is ``None``, looks for a config document in
        the backend's ``domain_id`` namespace. The domain root is
        checked first (``knowledge_base.(yaml|yml|json)``, matching the
        local-corpus convention so a directory can be promoted to a
        backend without relocation), then ``_metadata/knowledge_base.*``
        as a fallback. Falls back to defaults when neither is found.

        Args:
            backend: Storage backend; must already be
                :meth:`initialize`-d by the caller.
            domain_id: Domain/KB identifier within the backend.
            config: Optional :class:`KnowledgeBaseConfig`; see above
                for resolution when omitted.
            progress_callback: Optional ``callback(file_path: str,
                num_chunks: int) -> None`` invoked after each
                successfully-ingested document.
            extra_metadata: Optional cross-document identity carrier
                merged into every chunk's metadata. Composed through
                :meth:`_compose_extra_metadata` so a bound ``tenant_id``
                on the KB wins over any caller-supplied ``tenant_id``
                key — identity is sacred at the write boundary. Used by
                :class:`KnowledgeIngestionManager` to thread
                ``domain_id`` (and ``tenant_id`` when bound on the
                manager) onto each chunk so multi-tenant consumers can
                filter on either at query time.
            tenant_id: Optional keyword-only convenience kwarg folded
                into ``extra_metadata`` as ``{"tenant_id": tenant_id}``
                (wins over an ``extra_metadata["tenant_id"]`` entry
                from the same call — the kwarg is more specific). Same
                bound-tenant precedence rule applies.
            file_filter: Optional keyword-only predicate evaluated
                against each :class:`KnowledgeFile` *after* the
                pattern match. Files for which it returns ``False``
                are skipped at enumeration. ``None`` (default)
                enumerates every matching file (unchanged behavior).
                Used by
                :meth:`KnowledgeIngestionManager.ingest_changes` to
                re-embed only the changed files through the same
                pattern/chunking pipeline.
            rate_limiter: Optional keyword-only
                :class:`~dataknobs_common.ratelimit.RateLimiter`. When
                set, each per-chunk embed call is preceded by
                ``await rate_limiter.acquire("embed")`` so a
                rate-limited embedding provider cannot fail the ingest
                under burst. ``None`` (default) is unchanged behaviour
                (no pacing). Threaded by
                :class:`KnowledgeIngestionManager` from its own
                ``rate_limiter``.

        Returns:
            Statistics dict matching :meth:`load_from_directory`:
            ``{"total_files", "total_chunks", "files_by_type",
            "errors", "documents"}``.
        """
        if config is None:
            loaded = await self._load_kb_config_from_backend(backend, domain_id)
            config = loaded if loaded is not None else KnowledgeBaseConfig(
                name=domain_id
            )

        effective_extra: dict[str, Any] = dict(extra_metadata or {})
        if tenant_id is not None:
            effective_extra["tenant_id"] = tenant_id
        composed = self._compose_extra_metadata(effective_extra)

        source = BackendDocumentSource(
            backend, domain_id, file_filter=file_filter
        )
        processor = DirectoryProcessor(config, source, chunker=self._chunker)
        return await self._ingest_from_processor_async(
            processor,
            progress_callback,
            extra_metadata=composed,
            rate_limiter=rate_limiter,
        )

    async def _load_kb_config_from_backend(
        self,
        backend: "KnowledgeResourceBackend",
        domain_id: str,
    ) -> KnowledgeBaseConfig | None:
        """Read a KB config from the backend's domain namespace.

        Checks the domain root first
        (``knowledge_base.(yaml|yml|json)``), then falls back to the
        ``_metadata/`` subdirectory. The domain-root location mirrors
        the local-corpus convention used by
        :meth:`KnowledgeBaseConfig.load`, so a user promoting a local
        corpus to a backend doesn't have to relocate the file.
        ``_metadata/`` remains supported for consumers that prefer to
        keep metadata visually separated from content in the backend
        namespace. Returns ``None`` when no config document is present.
        When a file IS present but fails to parse or does not decode to
        a dict, raises :class:`IngestionConfigError` so the failure is
        loud — symmetric with the local-directory path.
        """
        from dataknobs_common.config_loading import (
            ConfigLoadError,
            ConfigShapeError,
            ConfigYAMLNotInstalledError,
            parse_yaml_or_json,
        )
        from dataknobs_xization.ingestion.config import IngestionConfigError

        for filename in (
            "knowledge_base.yaml",
            "knowledge_base.yml",
            "knowledge_base.json",
            "_metadata/knowledge_base.yaml",
            "_metadata/knowledge_base.yml",
            "_metadata/knowledge_base.json",
        ):
            data = await backend.get_file(domain_id, filename)
            if data is None:
                continue
            fmt: Literal["yaml", "json"] = (
                "json" if filename.endswith(".json") else "yaml"
            )
            try:
                raw = parse_yaml_or_json(
                    data,
                    format=fmt,
                    source_name=f"{domain_id}/{filename}",
                )
            except ConfigYAMLNotInstalledError as e:
                # Preserve the historical user-facing message verbatim.
                raise IngestionConfigError(
                    f"YAML config {filename} found in backend but "
                    "PyYAML is not installed; install 'pyyaml' or "
                    "use a .json config"
                ) from e
            except ConfigShapeError as e:
                raise IngestionConfigError(
                    f"knowledge_base config {filename} for domain "
                    f"{domain_id} did not decode to a dict"
                ) from e
            except ConfigLoadError as e:
                raise IngestionConfigError(
                    f"Failed to parse knowledge_base config {filename} "
                    f"for domain {domain_id}: {e}"
                ) from e
            return KnowledgeBaseConfig.from_dict(raw, default_name=domain_id)
        return None

    async def _ingest_from_processor_async(
        self,
        processor: DirectoryProcessor,
        progress_callback: Callable[[str, int], None] | None = None,
        extra_metadata: dict[str, Any] | None = None,
        *,
        rate_limiter: "RateLimiter | None" = None,
    ) -> dict[str, Any]:
        """Drive a :class:`DirectoryProcessor` and embed each chunk.

        Shared implementation for :meth:`load_from_directory` and
        :meth:`ingest_from_backend`. Iterates the processor's async
        output and delegates per-file embed+store to
        :meth:`_embed_and_store_chunks`. Each file is wrapped in its
        own try/except so that a transient embed/add_vectors failure
        on one document captures an error entry and lets the batch
        continue — matching the pre-unification
        ``KnowledgeIngestionManager._ingest_file`` semantics.

        ``rate_limiter`` (keyword-only, default ``None``) is forwarded
        unchanged to :meth:`_embed_and_store_chunks`; ``None`` keeps
        the embed path un-paced.
        """
        results: dict[str, Any] = {
            "total_files": 0,
            "total_chunks": 0,
            "files_skipped": 0,
            "files_by_type": {"markdown": 0, "json": 0, "jsonl": 0},
            "errors": [],
            "documents": [],
        }

        async for doc in processor.process_async():
            doc_info: dict[str, Any] = {
                "source": doc.source_file,
                "type": doc.document_type,
                "chunks": 0,
                "errors": list(doc.errors),
            }

            if doc.has_errors:
                results["errors"].extend(
                    {"file": doc.source_file, "error": err}
                    for err in doc.errors
                )
                results["documents"].append(doc_info)
                continue

            try:
                merged_metadata: dict[str, Any] = {}
                if doc.metadata:
                    merged_metadata.update(doc.metadata)
                if extra_metadata:
                    merged_metadata.update(extra_metadata)

                stored = await self._embed_and_store_chunks(
                    chunks=doc.chunks,
                    source_file=doc.source_file,
                    document_type=doc.document_type,
                    source_path=doc.source_path,
                    metadata=merged_metadata or None,
                    rate_limiter=rate_limiter,
                )
            except Exception as e:
                logger.exception(
                    "Failed to embed/store chunks for %s", doc.source_file
                )
                results["errors"].append(
                    {"file": doc.source_file, "error": str(e)}
                )
                doc_info["errors"].append(str(e))
                results["documents"].append(doc_info)
                continue

            doc_info["chunks"] = stored
            results["total_files"] += 1
            results["total_chunks"] += stored
            results["files_by_type"][doc.document_type] += 1
            results["documents"].append(doc_info)

            if progress_callback:
                progress_callback(doc.source_file, stored)

        # DirectoryProcessor tallies config-file, excluded, and
        # unsupported-extension skips during iteration — read it here
        # so callers (e.g. KnowledgeIngestionManager) can report
        # files_skipped in their own result types.
        results["files_skipped"] = processor.files_skipped
        return results

    @classmethod
    def _derive_chunk_prefix(
        cls,
        source_stem: str,
        metadata: dict[str, Any] | None,
    ) -> tuple[str, str]:
        r"""Return ``(chunk_prefix, chunk_separator)`` for a chunk's id.

        Walks :attr:`_CHUNK_ID_PREFIX_KEYS` against ``metadata`` and
        folds every present, truthy value into the prefix in declared
        order; ``source_stem`` is always last. When at least one fold
        key is present the record-separator (``\x1f``) is used so
        snake_case-tag collisions are impossible (``my`` + ``team_doc``
        vs ``my_team`` + ``doc`` would both produce ``my_team_doc`` under
        ``_``); when none are present the historical ``_`` separator is
        preserved so existing populated stores keep matching on
        re-ingest. Switching to ``\x1f`` unconditionally would silently
        double up every old chunk on the next ingest (``stem_index`` and
        ``stem\x1findex`` are different keys, so UPSERT inserts instead
        of overwriting).

        Single-tenant single-domain consumers (``metadata`` empty or
        carrying none of the declared keys) see the historical
        ``(source_stem, "_")`` shape. Multi-tenant shared stores that
        thread ``tenant_id`` get the deepest fold position by default
        so tenant identity participates in the primary key — fixing the
        chunk-id UPSERT collision class at the derivation layer.
        """
        md = metadata or {}
        parts: list[str] = []
        for key in cls._CHUNK_ID_PREFIX_KEYS:
            value = md.get(key)
            if value:
                parts.append(str(value))
        if not parts:
            # Legacy single-segment path — byte-identical to pre-fold.
            return source_stem, "_"
        parts.append(source_stem)
        return "\x1f".join(parts), "\x1f"

    async def _embed_and_store_chunks(
        self,
        chunks: list[dict[str, Any]],
        source_file: str,
        document_type: str,
        source_path: str | None = None,
        metadata: dict[str, Any] | None = None,
        *,
        rate_limiter: "RateLimiter | None" = None,
    ) -> int:
        """Embed chunk dicts and add them to the vector store.

        Shared by :meth:`load_markdown_text` and
        :meth:`_ingest_from_processor_async`. Chunks with empty
        ``embedding_text`` and ``text`` are skipped.

        Args:
            chunks: Chunk dicts (``text``, ``embedding_text``,
                ``chunk_index``, ``metadata``).
            source_file: Display path written into each chunk's
                ``"source"`` metadata field.
            document_type: Written into each chunk's
                ``"document_type"`` metadata field (``"markdown"``,
                ``"json"``, ``"jsonl"``).
            source_path: Optional source-relative path written into
                each chunk's ``"source_path"`` metadata field.
            metadata: Optional metadata merged into every chunk's
                metadata; caller-provided entries win over per-chunk
                fields. Used for pattern-level and ``domain_id``
                injection.
            rate_limiter: Optional keyword-only
                :class:`~dataknobs_common.ratelimit.RateLimiter`. When
                set, each per-chunk embed is preceded by
                ``await rate_limiter.acquire("embed")``. ``None``
                (default) leaves the embed path un-paced (unchanged).

        Returns:
            Number of chunks actually embedded and stored.
        """
        import numpy as np

        vectors: list[Any] = []
        ids: list[str] = []
        metadatas: list[dict[str, Any]] = []

        # ``KnowledgeBaseConfig.get_metadata`` (xization) and many
        # direct callers add ``source`` / ``filename`` to caller
        # metadata as redundant copies of the explicit ``source_file``
        # argument. They legitimately differ from the precise
        # display-URI ``source_file`` (e.g. relative path vs full S3
        # URI) but are redundant — not caller-as-attacker overrides.
        # Strip them here, ONCE, at the shared layer so every entry
        # point (including direct ``load_markdown_text`` callers)
        # benefits without triggering a spurious immutable-key
        # warning per chunk.
        if metadata:
            metadata = {
                k: v
                for k, v in metadata.items()
                if k not in ("source", "filename")
            }
            if not metadata:
                metadata = None

        source_stem = Path(source_file).stem if source_file else "doc"
        chunk_prefix, chunk_separator = self._derive_chunk_prefix(
            source_stem, metadata
        )

        # Detect caller-attempted system-field overrides ONCE per
        # call rather than per chunk. The same ``metadata`` dict
        # would otherwise emit N identical warnings for an N-chunk
        # document. We pass ``caller=metadata`` to the helper on the
        # first chunk only (warning emission), then ``caller=None``
        # on subsequent chunks (silent enforcement).
        warn_caller: dict[str, Any] | None = metadata

        for i, chunk in enumerate(chunks):
            text_for_embedding = chunk.get("embedding_text") or chunk.get(
                "text", ""
            )
            if not text_for_embedding:
                continue

            # Ingest-path rate-limit seam: pace the
            # per-chunk embed against an injected limiter so a
            # rate-limited embedding provider cannot fail the whole
            # ingest under burst. ``None`` (default) is a no-op — the
            # local Ollama embedder needs no pacing.
            if rate_limiter is not None:
                await rate_limiter.acquire("embed")

            embedding = await self.embedding_provider.embed(text_for_embedding)
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding, dtype=np.float32)

            chunk_index = chunk.get("chunk_index", i)
            chunk_id = f"{chunk_prefix}{chunk_separator}{chunk_index}"

            # System-controlled fields — caller-supplied ``metadata``
            # cannot override these. Tracked separately so the helper
            # can compare caller against system source.
            system_fields: dict[str, Any] = {
                "text": chunk.get("text", ""),
                "source": source_file,
                "chunk_index": chunk_index,
                "document_type": document_type,
            }
            if source_path is not None:
                system_fields["source_path"] = source_path

            chunk_metadata: dict[str, Any] = dict(system_fields)
            inner = chunk.get("metadata") or {}
            if inner:
                chunk_metadata.update(inner)
            if metadata:
                chunk_metadata.update(metadata)

            # Restore system-field values if caller-supplied metadata
            # tried to overwrite them (e.g. ``metadata={"text":
            # "tampered"}``). On the first chunk pass ``caller=metadata``
            # so the helper emits a warning; subsequent chunks pass
            # ``caller=None`` to enforce silently.
            enforce_immutable_keys(
                target=chunk_metadata,
                caller=warn_caller,
                source=system_fields,
                keys=system_fields.keys(),
                logger=logger,
                context="RAGKnowledgeBase._embed_and_store_chunks",
            )
            warn_caller = None

            vectors.append(embedding)
            ids.append(chunk_id)
            metadatas.append(chunk_metadata)

        if vectors:
            await self.vector_store.add_vectors(
                vectors=vectors, ids=ids, metadata=metadatas
            )

        return len(vectors)

    async def load_markdown_text(
        self,
        markdown_text: str,
        source: str,
        metadata: dict[str, Any] | None = None,
        *,
        extra_metadata: Mapping[str, Any] | None = None,
        tenant_id: str | None = None,
    ) -> int:
        """Load markdown content from a string.

        Parses, chunks, embeds, and stores the markdown text. Shared
        implementation used by :meth:`load_markdown_document`,
        :meth:`load_json_document`, :meth:`load_yaml_document`, and
        :meth:`load_csv_document`. Delegates embed+store to
        :meth:`_embed_and_store_chunks` so ``load_from_directory`` and
        ``ingest_from_backend`` share the same final stage.

        Args:
            markdown_text: Markdown content to load
            source: Source identifier for metadata
            metadata: Optional per-document metadata merged into every
                chunk (cross-document identity carriers belong in
                ``extra_metadata`` instead).
            extra_metadata: Optional keyword-only cross-document
                identity carrier (e.g. ``{"tenant_id": "acme"}``).
                Merged OVER ``metadata`` so identity tags cannot be
                shadowed by per-document fields. Composed through
                :meth:`_compose_extra_metadata` so a bound tenant on
                the KB wins over any caller-supplied ``tenant_id`` key
                — identity is sacred at the write boundary.
            tenant_id: Optional keyword-only convenience kwarg folded
                into ``extra_metadata`` as ``{"tenant_id": tenant_id}``
                (wins over an ``extra_metadata["tenant_id"]`` entry
                from the same call — the kwarg is more specific). Same
                bound-tenant precedence rule applies.

        Returns:
            Number of chunks created
        """
        effective_extra: dict[str, Any] = dict(extra_metadata or {})
        if tenant_id is not None:
            effective_extra["tenant_id"] = tenant_id
        composed = self._compose_extra_metadata(effective_extra)
        merged_metadata: dict[str, Any] = dict(metadata or {})
        if composed:
            merged_metadata.update(composed)

        doc_info = DocumentInfo(source=source, metadata=merged_metadata)
        chunks = self._chunker.chunk(markdown_text, doc_info)

        chunk_dicts: list[dict[str, Any]] = []
        for i, chunk in enumerate(chunks):
            chunk_dicts.append(
                {
                    "text": chunk.text,
                    "embedding_text": chunk.metadata.embedding_text or chunk.text,
                    "chunk_index": i,
                    "metadata": {
                        "heading_path": chunk.metadata.heading_display
                        or chunk.metadata.get_heading_path(),
                        "headings": chunk.metadata.headings,
                        "heading_levels": chunk.metadata.heading_levels,
                        "line_number": chunk.metadata.line_number,
                        "char_start": chunk.metadata.char_start,
                        "char_end": chunk.metadata.char_end,
                        "chunk_size": chunk.metadata.chunk_size,
                        "content_length": chunk.metadata.content_length,
                    },
                }
            )

        await self._embed_and_store_chunks(
            chunks=chunk_dicts,
            source_file=source,
            document_type="markdown",
            metadata=merged_metadata or None,
        )
        return len(chunks)

    # Over-fetch factor applied when stale chunks must be hidden: the
    # vector-store filter dict is equality/containment only and cannot
    # express ``_stale IS NULL OR _stale = false``, so the not-stale
    # gate is a post-filter. Tombstoned chunks exist only transiently
    # (during a TOMBSTONE swap), so a 4x over-fetch comfortably absorbs
    # them while keeping the post-filter store-agnostic.
    _STALE_OVERFETCH = 4

    def _compose_extra_metadata(
        self,
        extra_metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Return the effective ``extra_metadata`` seen by the chunk pipeline.

        Identity tags the KB owns at this layer (the bound
        ``tenant_id``) win over caller-supplied keys on collision — a
        caller cannot silently re-tag chunks for another tenant by
        passing ``extra_metadata={"tenant_id": ...}``. Non-identity
        keys are preserved as-is so callers can still attach
        per-document ``region``, ``cohort``, or any custom tag through
        the same channel.

        ``domain_id`` is NOT stamped here: it lives one layer up at
        :meth:`KnowledgeIngestionManager._compose_extra_metadata`,
        which subclasses this method's contract and layers the
        per-call ``domain_id`` over the same identity-wins rule before
        forwarding into :meth:`ingest_from_backend`. Direct
        :class:`RAGKnowledgeBase` consumers without a manager attach
        ``domain_id`` themselves (or don't, for single-domain stores).

        Returns a fresh dict so the caller's mapping is never mutated.
        """
        composed: dict[str, Any] = dict(extra_metadata or {})
        if self._tenant_id is not None:
            composed["tenant_id"] = self._tenant_id
        return composed

    def _resolve_read_filter(
        self, filter_metadata: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """The single place a read filter is formed for the store.

        Both :meth:`query` and :meth:`hybrid_query` (every
        ``vector_store.search`` / ``hybrid_search`` call) route their
        ``filter_metadata`` through here, so any future read-filter
        shaping has exactly one home. The store filter is returned
        unchanged when no bound tenant is configured: stale
        (tombstoned) exclusion is *not* expressible as an equality
        filter (it needs ``_stale IS NULL OR = false``) and is applied
        as a post-filter via :meth:`_is_stale`.

        When the KB has a bound ``tenant_id`` the bound value is
        AND-composed into the supplied filter with **explicit-filter-
        wins on collision** — the inverse of the write-side precedence
        (auto-derived wins) — because the two sides have different
        threat models. On write, identity is sacred and a caller cannot
        silently re-tag chunks for another tenant. On read, admin
        tooling legitimately needs to read across tenants by passing
        the explicit key; the asymmetry is therefore deliberate.
        """
        if self._tenant_id is None:
            return filter_metadata
        effective: dict[str, Any] = {"tenant_id": self._tenant_id}
        if filter_metadata:
            effective.update(filter_metadata)  # explicit-filter-wins
        return effective

    @staticmethod
    def _is_stale(metadata: dict[str, Any] | None) -> bool:
        """True only for an explicitly tombstoned chunk.

        Missing / ``None`` / ``False`` ``_stale`` is visible — this is
        the ``_stale IS NULL OR _stale = false`` read semantics that an
        equality filter cannot express.

        The guard is a ``None``-guard, not a truthiness gate: callers
        legitimately pass ``dict | None`` (a metadata-less row), and
        ``None.get`` would raise. ``metadata is not None`` keeps an
        empty ``{}`` flowing into ``.get`` (correctly not stale) rather
        than short-circuiting on falsy-but-present metadata.
        """
        return metadata is not None and metadata.get("_stale") is True

    async def _fetch_drop_stale_truncate(
        self,
        *,
        search: Callable[[int], Awaitable[list[_R]]],
        k: int,
        include_stale: bool,
        extract_meta: Callable[[_R], dict[str, Any] | None],
    ) -> list[_R]:
        """Fetch, drop tombstoned rows, truncate to ``k``.

        The vector-store filter is equality/containment only and
        cannot express ``_stale IS NULL OR _stale = false``, so the
        not-stale gate is a post-filter. When stale rows must be
        hidden this over-fetches ``k * _STALE_OVERFETCH`` so the
        post-filter still yields a full ``k`` mid-swap; with
        ``include_stale`` it is a thin pass-through (fetch exactly
        ``k``, no gate).

        Shared by the plain vector path **and** the native-hybrid
        path so the zero-downtime read guarantee — and specifically
        the over-fetch that stops it under-returning mid-swap — is
        identical on both. ``search`` is invoked with the effective
        fetch size; ``extract_meta`` adapts a raw row to its metadata
        dict (a ``(id, score, meta)`` tuple, or a record-bearing
        ``HybridSearchResult``).
        """
        if include_stale:
            return await search(k)
        raw = await search(k * self._STALE_OVERFETCH)
        visible = [r for r in raw if not self._is_stale(extract_meta(r))]
        return visible[:k]

    async def _vector_search(
        self,
        query_vector: Any,
        *,
        k: int,
        filter_metadata: dict[str, Any] | None,
        include_stale: bool,
    ) -> list[tuple[str, float, dict[str, Any] | None]]:
        """Shared vector-search chokepoint feeding query + hybrid.

        Hides tombstoned chunks unless ``include_stale`` via the
        shared :meth:`_fetch_drop_stale_truncate` (over-fetch, drop
        ``_stale``-true rows, return at most ``k``).
        """
        store_filter = self._resolve_read_filter(filter_metadata)
        return await self._fetch_drop_stale_truncate(
            search=lambda eff_k: self.vector_store.search(
                query_vector=query_vector,
                k=eff_k,
                filter=store_filter,
                include_metadata=True,
            ),
            k=k,
            include_stale=include_stale,
            extract_meta=lambda r: r[2],
        )

    async def query(
        self,
        query: str,
        k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
        min_similarity: float = 0.0,
        merge_adjacent: bool = False,
        max_chunk_size: int | None = None,
        *,
        include_stale: bool = False,
    ) -> list[dict[str, Any]]:
        """Query knowledge base for relevant chunks.

        Args:
            query: Query text to search for
            k: Number of results to return
            filter_metadata: Optional metadata filters
            min_similarity: Minimum similarity score (0-1)
            merge_adjacent: Whether to merge adjacent chunks with same heading
            max_chunk_size: Maximum size for merged chunks (uses merger config default if not specified)
            include_stale: When ``False`` (default), chunks tombstoned
                by an in-progress :attr:`IngestSwapMode.TOMBSTONE`
                re-ingest are hidden, so a concurrent reader never sees
                the superseded generation. ``True`` returns them
                (introspection / debugging).

        Returns:
            List of result dictionaries with:
                - text: Chunk text
                - source: Source file
                - heading_path: Heading hierarchy
                - similarity: Similarity score
                - metadata: Full chunk metadata

        Example:
            ```python
            results = await kb.query(
                "How do I configure the database?",
                k=3,
                merge_adjacent=True
            )
            for result in results:
                print(f"[{result['similarity']:.2f}] {result['heading_path']}")
                print(result['text'])
            ```
        """
        import numpy as np

        # Generate query embedding
        query_embedding = await self.embedding_provider.embed(query)

        # Convert to numpy if needed
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding, dtype=np.float32)

        # Search vector store (stale chunks hidden via the shared
        # read chokepoint unless include_stale).
        search_results = await self._vector_search(
            query_embedding,
            k=k,
            filter_metadata=filter_metadata,
            include_stale=include_stale,
        )

        # Format results
        results = []
        for _chunk_id, similarity, chunk_metadata in search_results:
            if chunk_metadata and similarity >= min_similarity:
                results.append(
                    {
                        "text": chunk_metadata.get("text", ""),
                        "source": chunk_metadata.get("source", ""),
                        "heading_path": chunk_metadata.get("heading_path", ""),
                        "similarity": similarity,
                        "metadata": chunk_metadata,
                    }
                )

        # Apply chunk merging if requested
        if merge_adjacent and results:
            # Update merger config if max_chunk_size specified
            if max_chunk_size is not None:
                merger = ChunkMerger(MergerConfig(max_merged_size=max_chunk_size))
            else:
                merger = self.merger

            merged_chunks = merger.merge(results)
            results = merger.to_result_list(merged_chunks)

        return results

    async def hybrid_query(
        self,
        query: str,
        k: int = 5,
        text_weight: float = 0.5,
        vector_weight: float = 0.5,
        fusion_strategy: str = "rrf",
        text_fields: list[str] | None = None,
        filter_metadata: dict[str, Any] | None = None,
        min_similarity: float = 0.0,
        merge_adjacent: bool = False,
        max_chunk_size: int | None = None,
        *,
        include_stale: bool = False,
    ) -> list[dict[str, Any]]:
        """Query knowledge base using hybrid search (text + vector).

        Combines keyword matching with semantic vector search for improved
        retrieval quality. Uses Reciprocal Rank Fusion (RRF) or weighted
        score fusion to combine results.

        Args:
            query: Query text to search for
            k: Number of results to return
            text_weight: Weight for text search (0.0 to 1.0)
            vector_weight: Weight for vector search (0.0 to 1.0)
            fusion_strategy: Fusion method - "rrf" (default), "weighted_sum", or "native"
            text_fields: Fields to search for text matching (default: ["text"])
            filter_metadata: Optional metadata filters
            min_similarity: Minimum combined score (0-1)
            merge_adjacent: Whether to merge adjacent chunks with same heading
            max_chunk_size: Maximum size for merged chunks
            include_stale: When ``False`` (default), chunks tombstoned
                by an in-progress :attr:`IngestSwapMode.TOMBSTONE`
                re-ingest are hidden on both the native and
                client-side fusion paths. ``True`` returns them.

        Returns:
            List of result dictionaries with:
                - text: Chunk text
                - source: Source file
                - heading_path: Heading hierarchy
                - similarity: Combined similarity score
                - text_score: Score from text search (if available)
                - vector_score: Score from vector search (if available)
                - metadata: Full chunk metadata

        Example:
            ```python
            # Default RRF fusion
            results = await kb.hybrid_query(
                "How do I configure the database?",
                k=5,
            )

            # Weighted toward vector search
            results = await kb.hybrid_query(
                "database configuration",
                k=5,
                text_weight=0.3,
                vector_weight=0.7,
            )

            # Weighted sum fusion
            results = await kb.hybrid_query(
                "configure database",
                k=5,
                fusion_strategy="weighted_sum",
            )

            for result in results:
                print(f"[{result['similarity']:.2f}] {result['heading_path']}")
                print(f"  text_score={result.get('text_score', 'N/A')}")
                print(f"  vector_score={result.get('vector_score', 'N/A')}")
                print(result['text'])
            ```
        """
        import numpy as np

        from dataknobs_data.vector.hybrid import (
            FusionStrategy,
            HybridSearchConfig,
            reciprocal_rank_fusion,
            weighted_score_fusion,
        )

        # Generate query embedding
        query_embedding = await self.embedding_provider.embed(query)

        # Convert to numpy if needed
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding, dtype=np.float32)

        # Check if vector store supports hybrid search natively
        has_hybrid = hasattr(self.vector_store, "hybrid_search")

        # Default text fields for knowledge base chunks
        search_text_fields = text_fields or ["text"]

        # Map string to FusionStrategy enum
        strategy_map = {
            "rrf": FusionStrategy.RRF,
            "weighted_sum": FusionStrategy.WEIGHTED_SUM,
            "native": FusionStrategy.NATIVE,
        }
        strategy = strategy_map.get(fusion_strategy.lower(), FusionStrategy.RRF)

        if has_hybrid and strategy == FusionStrategy.NATIVE:
            # Use vector store's native hybrid search
            config = HybridSearchConfig(
                text_weight=text_weight,
                vector_weight=vector_weight,
                fusion_strategy=strategy,
                text_fields=search_text_fields,
            )
            store_filter = self._resolve_read_filter(filter_metadata)

            def _hr_meta(hr: Any) -> dict[str, Any] | None:
                if hasattr(hr.record, "data"):
                    return hr.record.data or {}
                if hasattr(hr.record, "metadata"):
                    return hr.record.metadata or {}
                return {}

            # Over-fetch before the stale gate (shared with the vector
            # path) so a tombstoned top-k chunk never makes native
            # fusion under-return mid-swap.
            hybrid_results = await self._fetch_drop_stale_truncate(
                search=lambda eff_k: self.vector_store.hybrid_search(
                    query_text=query,
                    query_vector=query_embedding,
                    text_fields=search_text_fields,
                    k=eff_k,
                    config=config,
                    filter=store_filter,
                ),
                k=k,
                include_stale=include_stale,
                extract_meta=_hr_meta,
            )

            # Convert HybridSearchResult to our result format
            results = []
            for hr in hybrid_results:
                if hr.combined_score >= min_similarity:
                    record_metadata = _hr_meta(hr) or {}

                    results.append({
                        "text": record_metadata.get("text", ""),
                        "source": record_metadata.get("source", ""),
                        "heading_path": record_metadata.get("heading_path", ""),
                        "similarity": hr.combined_score,
                        "text_score": hr.text_score,
                        "vector_score": hr.vector_score,
                        "metadata": record_metadata,
                    })
        else:
            # Client-side hybrid search implementation
            # Step 1: Vector search (stale chunks hidden via the
            # shared read chokepoint unless include_stale).
            vector_results = await self._vector_search(
                query_embedding,
                k=k * 2,  # Get more for fusion
                filter_metadata=filter_metadata,
                include_stale=include_stale,
            )

            # Step 2: Text search (simple keyword matching on stored chunks)
            # For vector stores without text search, we search in retrieved chunks
            # and also do a broader metadata-based text match if supported

            # Build vector result map
            vector_scores: list[tuple[str, float]] = []
            chunks_by_id: dict[str, dict[str, Any]] = {}

            for chunk_id, similarity, chunk_metadata in vector_results:
                if chunk_metadata:
                    vector_scores.append((chunk_id, similarity))
                    chunks_by_id[chunk_id] = chunk_metadata

            # Simple text matching on chunk content
            query_lower = query.lower()
            query_terms = query_lower.split()
            text_scores: list[tuple[str, float]] = []

            for chunk_id, chunk_metadata in chunks_by_id.items():
                text_content = ""
                for field in search_text_fields:
                    value = chunk_metadata.get(field, "")
                    if value:
                        text_content += " " + str(value)

                text_content_lower = text_content.lower()

                # Calculate text match score
                if query_lower in text_content_lower:
                    # Exact phrase match
                    score = 1.0
                else:
                    # Term overlap score
                    matched_terms = sum(1 for term in query_terms if term in text_content_lower)
                    score = matched_terms / len(query_terms) if query_terms else 0.0

                if score > 0:
                    text_scores.append((chunk_id, score))

            # Sort text scores descending
            text_scores.sort(key=lambda x: x[1], reverse=True)

            # Step 3: Fuse results
            if strategy == FusionStrategy.WEIGHTED_SUM:
                total = text_weight + vector_weight
                if total > 0:
                    norm_text = text_weight / total
                    norm_vector = vector_weight / total
                else:
                    norm_text = norm_vector = 0.5

                fused = weighted_score_fusion(
                    text_results=text_scores,
                    vector_results=vector_scores,
                    text_weight=norm_text,
                    vector_weight=norm_vector,
                    normalize_scores=True,
                )
            else:
                # Default to RRF
                fused = reciprocal_rank_fusion(
                    text_results=text_scores,
                    vector_results=vector_scores,
                    k=60,
                    text_weight=text_weight,
                    vector_weight=vector_weight,
                )

            # Build result list
            text_score_map = dict(text_scores)
            vector_score_map = dict(vector_scores)

            results = []
            for chunk_id, combined_score in fused[:k]:
                if combined_score < min_similarity:
                    continue

                chunk_metadata = chunks_by_id.get(chunk_id)
                if not chunk_metadata:
                    continue

                results.append({
                    "text": chunk_metadata.get("text", ""),
                    "source": chunk_metadata.get("source", ""),
                    "heading_path": chunk_metadata.get("heading_path", ""),
                    "similarity": combined_score,
                    "text_score": text_score_map.get(chunk_id),
                    "vector_score": vector_score_map.get(chunk_id),
                    "metadata": chunk_metadata,
                })

        # Apply chunk merging if requested
        if merge_adjacent and results:
            if max_chunk_size is not None:
                merger = ChunkMerger(MergerConfig(max_merged_size=max_chunk_size))
            else:
                merger = self.merger

            merged_chunks = merger.merge(results)
            results = merger.to_result_list(merged_chunks)

        return results

    def format_context(
        self,
        results: list[dict[str, Any]],
        wrap_in_tags: bool = True,
        *,
        envelope: Any = None,
    ) -> str:
        """Format search results for LLM context.

        Convenience method to format results using the configured formatter.

        Args:
            results: Search results from query()
            wrap_in_tags: Whether to wrap the rendered context. When
                ``False``, the body is returned unwrapped. When
                ``True``: if ``envelope`` is provided, it renders the
                wrapper section; otherwise the legacy
                ``<knowledge_base>...</knowledge_base>`` shape is
                preserved for direct callers that have not migrated.
            envelope: Optional
                :class:`~dataknobs_bots.prompts.PromptEnvelope` used to
                render the wrapper when ``wrap_in_tags`` is ``True``.

        Returns:
            Formatted context string
        """
        context = self.formatter.format(results)
        if wrap_in_tags:
            context = self.formatter.wrap_for_prompt(context, envelope=envelope)
        return context

    async def count(
        self,
        filter: dict[str, Any] | None = None,
        *,
        include_stale: bool = False,
    ) -> int:
        """Get the number of chunks in the knowledge base.

        Excludes tombstoned (``_stale``) chunks by default, matching
        what :meth:`query` / :meth:`hybrid_query` actually return: a
        mid-``TOMBSTONE``-swap ``count`` would otherwise report
        old+new (roughly double) while reads only see the new
        generation. The store filter cannot express
        ``_stale IS NULL OR _stale = false``, so the visible count is
        ``count(filter) - count(filter ∧ _stale=True)`` (two cheap
        store counts, store-agnostic). ``include_stale=True`` restores
        the prior single delegated count (every stored chunk).

        Args:
            filter: Optional metadata filter to count only matching
                chunks
            include_stale: When ``True``, also count tombstoned chunks
                (the raw stored total). Default ``False`` counts only
                read-visible chunks.

        Returns:
            Number of chunks stored (optionally filtered)

        Example:
            ```python
            visible = await kb.count(filter={"domain_id": "my-domain"})
            raw = await kb.count(
                filter={"domain_id": "my-domain"}, include_stale=True
            )
            ```
        """
        total = await self.vector_store.count(filter)
        if include_stale:
            return total
        stale_filter = {**(filter or {}), "_stale": True}
        stale = await self.vector_store.count(stale_filter)
        return total - stale

    async def clear(self, filter: dict[str, Any] | None = None) -> None:
        """Clear documents from the knowledge base.

        Warning: When ``filter`` is ``None``, this removes all stored
        chunks and embeddings.  Pass a metadata filter to scope the
        clear (e.g. ``filter={"domain_id": "docs"}``) so that only
        matching chunks are removed.

        Args:
            filter: Optional metadata filter.  When ``None`` (default),
                all chunks are removed.  When provided, only chunks
                whose metadata matches the filter are removed; the
                filter shape is the same as for :meth:`query`.

        Raises:
            NotImplementedError: If the backing vector store does
                not support ``clear()``.
        """
        if hasattr(self.vector_store, "clear"):
            await self.vector_store.clear(filter=filter)
        else:
            raise NotImplementedError(
                "Vector store does not support clearing. "
                "Consider creating a new knowledge base with a fresh "
                "collection."
            )

    async def update_metadata_where(
        self,
        filter: dict[str, Any] | None,
        set_: dict[str, Any],
    ) -> int:
        """Bulk-merge ``set_`` into the metadata of matching chunks.

        Delegates to the backing vector store's
        :meth:`~dataknobs_data.vector.stores.base.VectorStore.update_metadata_where`.
        This is the destination-side primitive the
        :attr:`~dataknobs_bots.knowledge.IngestSwapMode.TOMBSTONE`
        swap uses to mark (or, on rollback, un-mark) a generation
        ``_stale`` without enumerating ids. Every in-tree store
        implements it; an out-of-tree store that does not raises
        ``NotImplementedError`` (the ABC contract) rather than
        silently mis-swapping.

        Args:
            filter: Metadata filter selecting chunks to update
                (same shape as :meth:`query` / :meth:`clear`).
            set_: Key/value pairs merged into each matched chunk's
                metadata.

        Returns:
            Number of chunks whose metadata was updated.
        """
        return await self.vector_store.update_metadata_where(filter, set_)

    async def save(self) -> None:
        """Save the knowledge base to persistent storage.

        This persists the vector store index and metadata to disk.
        Only applicable for vector stores that support persistence (e.g., FAISS).

        Example:
            ```python
            await kb.load_markdown_document("docs/api.md")
            await kb.save()  # Persist to disk
            ```
        """
        if hasattr(self.vector_store, "save"):
            await self.vector_store.save()

    def providers(self) -> dict[str, Any]:
        """Return the embedding provider, keyed by role."""
        from dataknobs_bots.bot.base import PROVIDER_ROLE_KB_EMBEDDING

        if self.embedding_provider is not None:
            return {PROVIDER_ROLE_KB_EMBEDDING: self.embedding_provider}
        return {}

    def set_provider(self, role: str, provider: Any) -> bool:
        """Replace the embedding provider if the role matches."""
        from dataknobs_bots.bot.base import PROVIDER_ROLE_KB_EMBEDDING

        if role == PROVIDER_ROLE_KB_EMBEDDING:
            self.embedding_provider = provider
            return True
        return False

    async def close(self) -> None:
        """Close the knowledge base and release resources.

        This method:
        - Saves the vector store to disk (if persistence is configured)
        - Closes the vector store connection
        - Closes the embedding provider (releases HTTP sessions)

        Only collaborators this instance *owns* (built from config) are
        closed. A vector store or embedding provider supplied externally
        via :meth:`from_components` is caller-owned and left open, so a
        consumer sharing one store/provider across several knowledge bases
        can close each base independently without tearing down a resource
        the others still depend on.

        Should be called when done using the knowledge base to prevent
        resource leaks (e.g., unclosed aiohttp sessions).

        Example:
            ```python
            kb = await RAGKnowledgeBase.from_config(config)
            try:
                await kb.load_markdown_document("docs/api.md")
                results = await kb.query("How do I configure?")
            finally:
                await kb.close()
            ```
        """
        # Close vector store (will save if persist_path is set) — only
        # when owned; an injected store is left open for the caller.
        await close_if_owned(self.vector_store, self._owns_vector_store)

        # Close embedding provider (releases HTTP client sessions) — only
        # when owned; an injected provider is left open for the caller.
        await close_if_owned(
            self.embedding_provider, self._owns_embedding_provider
        )

    async def __aenter__(self) -> Self:
        """Async context manager entry.

        Returns:
            Self for use in async with statement

        Example:
            ```python
            async with await RAGKnowledgeBase.from_config(config) as kb:
                await kb.load_markdown_document("docs/api.md")
                results = await kb.query("How do I configure?")
            # Automatically saved and closed
            ```
        """
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Async context manager exit - ensures cleanup.

        Args:
            exc_type: Exception type if an exception occurred
            exc_val: Exception value if an exception occurred
            exc_tb: Exception traceback if an exception occurred
        """
        await self.close()
