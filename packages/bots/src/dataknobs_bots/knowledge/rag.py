"""RAG (Retrieval-Augmented Generation) knowledge base implementation."""

import types
from pathlib import Path
from typing import Any

from dataknobs_xization import (
    ChunkQualityConfig,
    ContentTransformer,
    HeadingInclusion,
    chunk_markdown_tree,
    parse_markdown,
)
from dataknobs_xization.ingestion import (
    DirectoryProcessor,
    KnowledgeBaseConfig,
)
from dataknobs_bots.knowledge.retrieval import (
    ChunkMerger,
    ContextFormatter,
    FormatterConfig,
    MergerConfig,
)


class RAGKnowledgeBase:
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
    ):
        """Initialize RAG knowledge base.

        Args:
            vector_store: Vector store backend instance
            embedding_provider: LLM provider with embed() method
            chunking_config: Configuration for chunking:
                - max_chunk_size: Maximum chunk size in characters
                - chunk_overlap: Overlap between chunks
                - combine_under_heading: Combine text under same heading
                - quality_filter: ChunkQualityConfig for filtering
                - generate_embeddings: Whether to generate enriched embedding text
            merger_config: Configuration for chunk merging (optional)
            formatter_config: Configuration for context formatting (optional)
        """
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.chunking_config = chunking_config or {
            "max_chunk_size": 500,
            "chunk_overlap": 50,
            "combine_under_heading": True,
        }

        # Initialize merger and formatter
        self.merger = ChunkMerger(merger_config) if merger_config else ChunkMerger()
        self.formatter = ContextFormatter(formatter_config) if formatter_config else ContextFormatter()

    @classmethod
    async def from_config(cls, config: dict[str, Any]) -> "RAGKnowledgeBase":
        """Create RAG knowledge base from configuration.

        Args:
            config: Configuration dictionary with:
                - vector_store: Vector store configuration
                - embedding_provider: LLM provider name
                - embedding_model: Model for embeddings
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
                    "dimensions": 1536,
                    "collection": "docs"
                },
                "embedding_provider": "openai",
                "embedding_model": "text-embedding-3-small",
                "chunking": {
                    "max_chunk_size": 500,
                    "chunk_overlap": 50
                },
                "documents_path": "./docs"
            }
            kb = await RAGKnowledgeBase.from_config(config)
            ```
        """
        from dataknobs_data.vector.stores import VectorStoreFactory
        from dataknobs_llm.llm import LLMProviderFactory

        # Create vector store
        vs_config = config["vector_store"]
        factory = VectorStoreFactory()
        vector_store = factory.create(**vs_config)
        await vector_store.initialize()

        # Create embedding provider
        llm_factory = LLMProviderFactory(is_async=True)
        embedding_provider = llm_factory.create(
            {
                "provider": config.get("embedding_provider", "openai"),
                "model": config.get("embedding_model", "text-embedding-ada-002"),
            }
        )
        await embedding_provider.initialize()

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
        """Load and chunk a markdown document.

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
        import numpy as np

        # Read document
        filepath = Path(filepath)
        with open(filepath, encoding="utf-8") as f:
            markdown_text = f.read()

        # Parse markdown
        tree = parse_markdown(markdown_text)

        # Build quality filter config if specified
        quality_filter = None
        if "quality_filter" in self.chunking_config:
            qf_config = self.chunking_config["quality_filter"]
            if isinstance(qf_config, ChunkQualityConfig):
                quality_filter = qf_config
            elif isinstance(qf_config, dict):
                quality_filter = ChunkQualityConfig(**qf_config)

        # Chunk the document with enhanced options
        chunks = chunk_markdown_tree(
            tree,
            max_chunk_size=self.chunking_config.get("max_chunk_size", 500),
            chunk_overlap=self.chunking_config.get("chunk_overlap", 50),
            heading_inclusion=HeadingInclusion.IN_METADATA,  # Keep headings in metadata only
            combine_under_heading=self.chunking_config.get("combine_under_heading", True),
            quality_filter=quality_filter,
            generate_embeddings=self.chunking_config.get("generate_embeddings", True),
        )

        # Process and store chunks
        vectors = []
        ids = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            # Use embedding_text if available, otherwise use chunk text
            text_for_embedding = chunk.metadata.embedding_text or chunk.text

            # Generate embedding
            embedding = await self.embedding_provider.embed(text_for_embedding)

            # Convert to numpy if needed
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding, dtype=np.float32)

            # Prepare metadata with new fields
            chunk_id = f"{filepath.stem}_{i}"
            chunk_metadata = {
                "text": chunk.text,
                "source": str(filepath),
                "chunk_index": i,
                "heading_path": chunk.metadata.heading_display or chunk.metadata.get_heading_path(),
                "headings": chunk.metadata.headings,
                "heading_levels": chunk.metadata.heading_levels,
                "line_number": chunk.metadata.line_number,
                "chunk_size": chunk.metadata.chunk_size,
                "content_length": chunk.metadata.content_length,
            }

            # Merge with user metadata
            if metadata:
                chunk_metadata.update(metadata)

            vectors.append(embedding)
            ids.append(chunk_id)
            metadatas.append(chunk_metadata)

        # Batch insert into vector store
        if vectors:
            await self.vector_store.add_vectors(
                vectors=vectors, ids=ids, metadata=metadatas
            )

        return len(chunks)

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

        return await self._load_markdown_text(
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

        return await self._load_markdown_text(
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

        return await self._load_markdown_text(
            markdown_text,
            source=str(filepath),
            metadata=metadata,
        )

    async def load_from_directory(
        self,
        directory: str | Path,
        config: KnowledgeBaseConfig | None = None,
        progress_callback: Any | None = None,
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
        import numpy as np

        directory = Path(directory)

        # Load or use provided config
        if config is None:
            config = KnowledgeBaseConfig.load(directory)

        # Create processor
        processor = DirectoryProcessor(config, directory)

        # Track results
        results: dict[str, Any] = {
            "total_files": 0,
            "total_chunks": 0,
            "files_by_type": {"markdown": 0, "json": 0, "jsonl": 0},
            "errors": [],
            "documents": [],
        }

        # Process each document
        for doc in processor.process():
            doc_info: dict[str, Any] = {
                "source": doc.source_file,
                "type": doc.document_type,
                "chunks": 0,
                "errors": doc.errors,
            }

            if doc.has_errors:
                results["errors"].extend([
                    {"file": doc.source_file, "error": err}
                    for err in doc.errors
                ])
                results["documents"].append(doc_info)
                continue

            # Process chunks for this document
            vectors = []
            ids = []
            metadatas = []

            source_stem = Path(doc.source_file).stem

            for chunk in doc.chunks:
                # Get text for embedding
                text_for_embedding = chunk.get("embedding_text") or chunk.get("text", "")

                if not text_for_embedding:
                    continue

                # Generate embedding
                embedding = await self.embedding_provider.embed(text_for_embedding)

                # Convert to numpy if needed
                if not isinstance(embedding, np.ndarray):
                    embedding = np.array(embedding, dtype=np.float32)

                # Build chunk ID
                chunk_index = chunk.get("chunk_index", len(vectors))
                chunk_id = f"{source_stem}_{chunk_index}"

                # Build metadata
                chunk_metadata = {
                    "text": chunk.get("text", ""),
                    "source": doc.source_file,
                    "chunk_index": chunk_index,
                    "document_type": doc.document_type,
                }

                # Add chunk-specific metadata
                if "metadata" in chunk:
                    chunk_metadata.update(chunk["metadata"])

                # Add document-level metadata
                if doc.metadata:
                    for key, value in doc.metadata.items():
                        if key not in chunk_metadata:
                            chunk_metadata[key] = value

                vectors.append(embedding)
                ids.append(chunk_id)
                metadatas.append(chunk_metadata)

            # Batch insert into vector store
            if vectors:
                await self.vector_store.add_vectors(
                    vectors=vectors, ids=ids, metadata=metadatas
                )

            doc_info["chunks"] = len(vectors)
            results["total_files"] += 1
            results["total_chunks"] += len(vectors)
            results["files_by_type"][doc.document_type] += 1
            results["documents"].append(doc_info)

            # Call progress callback if provided
            if progress_callback:
                progress_callback(doc.source_file, len(vectors))

        return results

    async def _load_markdown_text(
        self,
        markdown_text: str,
        source: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Internal method to load markdown text directly.

        Used by load_json_document, load_yaml_document, and load_csv_document.

        Args:
            markdown_text: Markdown content to load
            source: Source identifier for metadata
            metadata: Optional metadata to attach to all chunks

        Returns:
            Number of chunks created
        """
        import numpy as np

        # Parse markdown
        tree = parse_markdown(markdown_text)

        # Build quality filter config if specified
        quality_filter = None
        if "quality_filter" in self.chunking_config:
            qf_config = self.chunking_config["quality_filter"]
            if isinstance(qf_config, ChunkQualityConfig):
                quality_filter = qf_config
            elif isinstance(qf_config, dict):
                quality_filter = ChunkQualityConfig(**qf_config)

        # Chunk the document with enhanced options
        chunks = chunk_markdown_tree(
            tree,
            max_chunk_size=self.chunking_config.get("max_chunk_size", 500),
            chunk_overlap=self.chunking_config.get("chunk_overlap", 50),
            heading_inclusion=HeadingInclusion.IN_METADATA,
            combine_under_heading=self.chunking_config.get("combine_under_heading", True),
            quality_filter=quality_filter,
            generate_embeddings=self.chunking_config.get("generate_embeddings", True),
        )

        # Process and store chunks
        vectors = []
        ids = []
        metadatas = []

        # Generate a base ID from source
        source_stem = Path(source).stem if source else "doc"

        for i, chunk in enumerate(chunks):
            # Use embedding_text if available, otherwise use chunk text
            text_for_embedding = chunk.metadata.embedding_text or chunk.text

            # Generate embedding
            embedding = await self.embedding_provider.embed(text_for_embedding)

            # Convert to numpy if needed
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding, dtype=np.float32)

            # Prepare metadata with new fields
            chunk_id = f"{source_stem}_{i}"
            chunk_metadata = {
                "text": chunk.text,
                "source": source,
                "chunk_index": i,
                "heading_path": chunk.metadata.heading_display or chunk.metadata.get_heading_path(),
                "headings": chunk.metadata.headings,
                "heading_levels": chunk.metadata.heading_levels,
                "line_number": chunk.metadata.line_number,
                "chunk_size": chunk.metadata.chunk_size,
                "content_length": chunk.metadata.content_length,
            }

            # Merge with user metadata
            if metadata:
                chunk_metadata.update(metadata)

            vectors.append(embedding)
            ids.append(chunk_id)
            metadatas.append(chunk_metadata)

        # Batch insert into vector store
        if vectors:
            await self.vector_store.add_vectors(
                vectors=vectors, ids=ids, metadata=metadatas
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

    async def __aenter__(self) -> "RAGKnowledgeBase":
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
