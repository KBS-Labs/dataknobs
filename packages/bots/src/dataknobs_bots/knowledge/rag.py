"""RAG (Retrieval-Augmented Generation) knowledge base implementation."""

from pathlib import Path
from typing import Any

from dataknobs_xization import chunk_markdown_tree, parse_markdown


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
    ):
        """Initialize RAG knowledge base.

        Args:
            vector_store: Vector store backend instance
            embedding_provider: LLM provider with embed() method
            chunking_config: Configuration for chunking:
                - max_chunk_size: Maximum chunk size in characters
                - chunk_overlap: Overlap between chunks
                - combine_under_heading: Combine text under same heading
        """
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.chunking_config = chunking_config or {
            "max_chunk_size": 500,
            "chunk_overlap": 50,
            "combine_under_heading": True,
        }

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

        # Create instance
        kb = cls(
            vector_store=vector_store,
            embedding_provider=embedding_provider,
            chunking_config=config.get("chunking", {}),
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

        # Chunk the document
        chunks = chunk_markdown_tree(
            tree,
            max_chunk_size=self.chunking_config.get("max_chunk_size", 500),
            chunk_overlap=self.chunking_config.get("chunk_overlap", 50),
            combine_under_heading=self.chunking_config.get("combine_under_heading", True),
        )

        # Process and store chunks
        vectors = []
        ids = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            # Generate embedding
            embedding = await self.embedding_provider.embed(chunk.text)

            # Convert to numpy if needed
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding, dtype=np.float32)

            # Prepare metadata
            chunk_id = f"{filepath.stem}_{i}"
            chunk_metadata = {
                "text": chunk.text,
                "source": str(filepath),
                "chunk_index": i,
                "heading_path": chunk.metadata.get_heading_path(),
                "headings": chunk.metadata.headings,
                "line_number": chunk.metadata.line_number,
                "chunk_size": chunk.metadata.chunk_size,
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
            print(f"Loaded {results['total_chunks']} chunks from {results['total_files']} files")  # validate: ignore-print
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

    async def query(
        self,
        query: str,
        k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
        min_similarity: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Query knowledge base for relevant chunks.

        Args:
            query: Query text to search for
            k: Number of results to return
            filter_metadata: Optional metadata filters
            min_similarity: Minimum similarity score (0-1)

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
                k=3
            )
            for result in results:
                print(f"[{result['similarity']:.2f}] {result['heading_path']}")  # validate: ignore-print
                print(result['text'])  # validate: ignore-print
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

        return results

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
