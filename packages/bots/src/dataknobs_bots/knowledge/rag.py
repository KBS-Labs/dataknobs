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
