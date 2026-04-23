"""RAG (Retrieval-Augmented Generation) knowledge base implementation."""

import logging
import types
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

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
from dataknobs_bots.knowledge.base import KnowledgeBase
from dataknobs_bots.knowledge.retrieval import (
    ChunkMerger,
    ContextFormatter,
    FormatterConfig,
    MergerConfig,
)

if TYPE_CHECKING:
    from dataknobs_bots.knowledge.storage.backend import KnowledgeResourceBackend

logger = logging.getLogger(__name__)


class RAGKnowledgeBase(KnowledgeBase):
    """RAG knowledge base using dataknobs-xization for chunking and vector search.

    This implementation:
    - Parses markdown documents using dataknobs-xization
    - Chunks documents intelligently based on structure
    - Stores chunks with embeddings in vector store
    - Provides semantic search for relevant context

    Attributes:
        vector_store: Vector store backend from dataknobs_data
        embedding_provider: LLM provider for generating embeddings
        chunking_config: Configuration for document chunking
    """

    def __init__(
        self,
        vector_store: Any,
        embedding_provider: Any,
        chunking_config: dict[str, Any] | None = None,
        merger_config: MergerConfig | None = None,
        formatter_config: FormatterConfig | None = None,
        chunker: Chunker | None = None,
    ):
        """Initialize RAG knowledge base.

        Args:
            vector_store: Vector store backend instance
            embedding_provider: LLM provider with embed() method
            chunking_config: Configuration for chunking.  The ``chunker``
                key selects the chunker implementation (default:
                ``"markdown_tree"``).  Remaining keys are forwarded to the
                chunker's ``from_config()`` method.  Legacy keys
                (``max_chunk_size``, ``combine_under_heading``, etc.) are
                preserved for backward compatibility.
            merger_config: Configuration for chunk merging (optional)
            formatter_config: Configuration for context formatting (optional)
            chunker: Pre-built chunker instance.  When provided, takes
                precedence over ``chunking_config`` for chunker selection.
        """
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.chunking_config = chunking_config or {
            "max_chunk_size": 500,
            "combine_under_heading": True,
        }

        # Resolve chunker: explicit instance > config-driven > default
        self._chunker = chunker or create_chunker(self.chunking_config)

        # Initialize merger and formatter
        self.merger = ChunkMerger(merger_config) if merger_config else ChunkMerger()
        self.formatter = ContextFormatter(formatter_config) if formatter_config else ContextFormatter()

    @classmethod
    async def from_config(cls, config: dict[str, Any]) -> "RAGKnowledgeBase":
        """Create RAG knowledge base from configuration.

        Args:
            config: Configuration dictionary with:
                - vector_store: Vector store configuration
                - embedding: Nested embedding config dict (preferred), e.g.
                  ``{"provider": "ollama", "model": "nomic-embed-text"}``
                - embedding_provider / embedding_model: Legacy flat keys
                - chunking: Optional chunking configuration
                - documents_path: Optional path to load documents from
                - document_pattern: Optional glob pattern for documents

        Returns:
            Configured RAGKnowledgeBase instance

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
        from dataknobs_data.vector.stores import VectorStoreFactory

        from ..providers import create_embedding_provider

        # Create vector store
        vs_config = config["vector_store"]
        factory = VectorStoreFactory()
        vector_store = factory.create(**vs_config)
        await vector_store.initialize()

        # Create embedding provider
        embedding_provider = await create_embedding_provider(config)

        # Create merger config if specified
        merger_config = None
        if "merger" in config:
            merger_config = MergerConfig(**config["merger"])

        # Create formatter config if specified
        formatter_config = None
        if "formatter" in config:
            formatter_config = FormatterConfig(**config["formatter"])

        # Create instance
        kb = cls(
            vector_store=vector_store,
            embedding_provider=embedding_provider,
            chunking_config=config.get("chunking", {}),
            merger_config=merger_config,
            formatter_config=formatter_config,
        )

        # Load documents if path provided
        if "documents_path" in config:
            await kb.load_documents_from_directory(
                config["documents_path"], config.get("document_pattern", "**/*.md")
            )

        return kb

    async def load_markdown_document(
        self, filepath: str | Path, metadata: dict[str, Any] | None = None
    ) -> int:
        """Load and chunk a markdown document from a file.

        Reads the file and delegates to :meth:`load_markdown_text` for
        parsing, chunking, embedding, and storage.

        Args:
            filepath: Path to markdown file
            metadata: Optional metadata to attach to all chunks

        Returns:
            Number of chunks created

        Example:
            ```python
            num_chunks = await kb.load_markdown_document(
                "docs/api.md",
                metadata={"category": "api", "version": "1.0"}
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
        )

    async def load_documents_from_directory(
        self, directory: str | Path, pattern: str = "**/*.md"
    ) -> dict[str, Any]:
        """Load all markdown documents from a directory.

        Args:
            directory: Directory path containing documents
            pattern: Glob pattern for files to load (default: **/*.md)

        Returns:
            Dictionary with loading statistics:
                - total_files: Number of files processed
                - total_chunks: Total chunks created
                - errors: List of errors encountered

        Example:
            ```python
            results = await kb.load_documents_from_directory(
                "docs/",
                pattern="**/*.md"
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
                    filepath, metadata={"filename": filepath.name}
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
    ) -> int:
        """Load and chunk a JSON document by converting it to markdown.

        This method converts JSON data to markdown format using ContentTransformer,
        then processes it like any other markdown document.

        Args:
            filepath: Path to JSON file
            metadata: Optional metadata to attach to all chunks
            schema: Optional schema name (requires transformer with registered schema)
            transformer: Optional ContentTransformer instance with custom configuration
            title: Optional document title for the markdown

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
        )

    async def load_yaml_document(
        self,
        filepath: str | Path,
        metadata: dict[str, Any] | None = None,
        schema: str | None = None,
        transformer: ContentTransformer | None = None,
        title: str | None = None,
    ) -> int:
        """Load and chunk a YAML document by converting it to markdown.

        Args:
            filepath: Path to YAML file
            metadata: Optional metadata to attach to all chunks
            schema: Optional schema name (requires transformer with registered schema)
            transformer: Optional ContentTransformer instance with custom configuration
            title: Optional document title for the markdown

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
        )

    async def load_csv_document(
        self,
        filepath: str | Path,
        metadata: dict[str, Any] | None = None,
        title: str | None = None,
        title_field: str | None = None,
        transformer: ContentTransformer | None = None,
    ) -> int:
        """Load and chunk a CSV document by converting it to markdown.

        Each row becomes a section with the first column (or title_field) as heading.

        Args:
            filepath: Path to CSV file
            metadata: Optional metadata to attach to all chunks
            title: Optional document title for the markdown
            title_field: Column to use as section title (default: first column)
            transformer: Optional ContentTransformer instance with custom configuration

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
        )

    async def load_from_directory(
        self,
        directory: str | Path,
        config: KnowledgeBaseConfig | None = None,
        progress_callback: Callable[[str, int], None] | None = None,
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

        processor = DirectoryProcessor(config, directory, chunker=self._chunker)
        return await self._ingest_from_processor_async(
            processor, progress_callback
        )

    async def ingest_from_backend(
        self,
        backend: "KnowledgeResourceBackend",
        domain_id: str,
        config: KnowledgeBaseConfig | None = None,
        progress_callback: Callable[[str, int], None] | None = None,
        extra_metadata: dict[str, Any] | None = None,
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
            extra_metadata: Optional metadata dict merged into every
                chunk's metadata (caller-provided entries win over
                pattern-config / per-chunk entries). Used by
                :class:`KnowledgeIngestionManager` to thread the
                ``domain_id`` onto each chunk so that multi-tenant
                consumers can filter on it at query time.

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

        source = BackendDocumentSource(backend, domain_id)
        processor = DirectoryProcessor(config, source, chunker=self._chunker)
        return await self._ingest_from_processor_async(
            processor, progress_callback, extra_metadata=extra_metadata
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
        import json as json_lib

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
            try:
                if filename.endswith(".json"):
                    raw = json_lib.loads(data.decode("utf-8"))
                else:
                    try:
                        import yaml
                    except ImportError as e:
                        raise IngestionConfigError(
                            f"YAML config {filename} found in backend but "
                            "PyYAML is not installed; install 'pyyaml' or "
                            "use a .json config"
                        ) from e
                    raw = yaml.safe_load(data.decode("utf-8"))
            except IngestionConfigError:
                raise
            except Exception as e:
                raise IngestionConfigError(
                    f"Failed to parse knowledge_base config {filename} "
                    f"for domain {domain_id}: {e}"
                ) from e
            if not isinstance(raw, dict):
                raise IngestionConfigError(
                    f"knowledge_base config {filename} for domain "
                    f"{domain_id} did not decode to a dict"
                )
            return KnowledgeBaseConfig.from_dict(raw, default_name=domain_id)
        return None

    async def _ingest_from_processor_async(
        self,
        processor: DirectoryProcessor,
        progress_callback: Callable[[str, int], None] | None = None,
        extra_metadata: dict[str, Any] | None = None,
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

    async def _embed_and_store_chunks(
        self,
        chunks: list[dict[str, Any]],
        source_file: str,
        document_type: str,
        source_path: str | None = None,
        metadata: dict[str, Any] | None = None,
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

        Returns:
            Number of chunks actually embedded and stored.
        """
        import numpy as np

        vectors: list[Any] = []
        ids: list[str] = []
        metadatas: list[dict[str, Any]] = []

        source_stem = Path(source_file).stem if source_file else "doc"

        for i, chunk in enumerate(chunks):
            text_for_embedding = chunk.get("embedding_text") or chunk.get(
                "text", ""
            )
            if not text_for_embedding:
                continue

            embedding = await self.embedding_provider.embed(text_for_embedding)
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding, dtype=np.float32)

            chunk_index = chunk.get("chunk_index", i)
            chunk_id = f"{source_stem}_{chunk_index}"

            chunk_metadata: dict[str, Any] = {
                "text": chunk.get("text", ""),
                "source": source_file,
                "chunk_index": chunk_index,
                "document_type": document_type,
            }
            if source_path is not None:
                chunk_metadata["source_path"] = source_path
            inner = chunk.get("metadata") or {}
            if inner:
                chunk_metadata.update(inner)
            if metadata:
                chunk_metadata.update(metadata)

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
            metadata: Optional metadata merged into every chunk
                (caller-provided entries win)

        Returns:
            Number of chunks created
        """
        doc_info = DocumentInfo(source=source, metadata=metadata or {})
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
            metadata=metadata,
        )
        return len(chunks)

    async def query(
        self,
        query: str,
        k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
        min_similarity: float = 0.0,
        merge_adjacent: bool = False,
        max_chunk_size: int | None = None,
    ) -> list[dict[str, Any]]:
        """Query knowledge base for relevant chunks.

        Args:
            query: Query text to search for
            k: Number of results to return
            filter_metadata: Optional metadata filters
            min_similarity: Minimum similarity score (0-1)
            merge_adjacent: Whether to merge adjacent chunks with same heading
            max_chunk_size: Maximum size for merged chunks (uses merger config default if not specified)

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

        # Search vector store
        search_results = await self.vector_store.search(
            query_vector=query_embedding,
            k=k,
            filter=filter_metadata,
            include_metadata=True,
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
        from dataknobs_data.vector.hybrid import (
            FusionStrategy,
            HybridSearchConfig,
            reciprocal_rank_fusion,
            weighted_score_fusion,
        )
        import numpy as np

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
            hybrid_results = await self.vector_store.hybrid_search(
                query_text=query,
                query_vector=query_embedding,
                text_fields=search_text_fields,
                k=k,
                config=config,
                filter=filter_metadata,
            )

            # Convert HybridSearchResult to our result format
            results = []
            for hr in hybrid_results:
                if hr.combined_score >= min_similarity:
                    # Extract metadata from record
                    record_metadata = {}
                    if hasattr(hr.record, "data"):
                        record_metadata = hr.record.data or {}
                    elif hasattr(hr.record, "metadata"):
                        record_metadata = hr.record.metadata or {}

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
            # Step 1: Vector search
            vector_results = await self.vector_store.search(
                query_vector=query_embedding,
                k=k * 2,  # Get more for fusion
                filter=filter_metadata,
                include_metadata=True,
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
    ) -> str:
        """Format search results for LLM context.

        Convenience method to format results using the configured formatter.

        Args:
            results: Search results from query()
            wrap_in_tags: Whether to wrap in <knowledge_base> tags

        Returns:
            Formatted context string
        """
        context = self.formatter.format(results)
        if wrap_in_tags:
            context = self.formatter.wrap_for_prompt(context)
        return context

    async def count(self, filter: dict[str, Any] | None = None) -> int:
        """Get the number of chunks in the knowledge base.

        Delegates to the underlying vector store's count method.

        Args:
            filter: Optional metadata filter to count only matching chunks

        Returns:
            Number of chunks stored (optionally filtered)

        Example:
            ```python
            total = await kb.count()
            domain_count = await kb.count(filter={"domain_id": "my-domain"})
            ```
        """
        return await self.vector_store.count(filter)

    async def clear(self) -> None:
        """Clear all documents from the knowledge base.

        Warning: This removes all stored chunks and embeddings.
        """
        if hasattr(self.vector_store, "clear"):
            await self.vector_store.clear()
        else:
            raise NotImplementedError(
                "Vector store does not support clearing. "
                "Consider creating a new knowledge base with a fresh collection."
            )

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
        # Close vector store (will save if persist_path is set)
        if hasattr(self.vector_store, "close"):
            await self.vector_store.close()

        # Close embedding provider (releases HTTP client sessions)
        if hasattr(self.embedding_provider, "close"):
            await self.embedding_provider.close()

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
