"""Tests for knowledge base implementations."""

import os
import tempfile
from pathlib import Path

import pytest

from dataknobs_bots.knowledge import RAGKnowledgeBase, create_knowledge_base_from_config
from dataknobs_bots.tools import KnowledgeSearchTool
from dataknobs_data.vector.stores import VectorStoreFactory
from dataknobs_llm.llm import LLMProviderFactory


class TestRAGKnowledgeBase:
    """Tests for RAGKnowledgeBase."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test basic initialization."""
        # Create vector store
        factory = VectorStoreFactory()
        vector_store = factory.create(backend="memory", dimensions=384)
        await vector_store.initialize()

        # Create embedding provider
        llm_factory = LLMProviderFactory(is_async=True)
        embedding_provider = llm_factory.create({"provider": "echo", "model": "test"})
        await embedding_provider.initialize()

        # Create knowledge base
        kb = RAGKnowledgeBase(
            vector_store=vector_store, embedding_provider=embedding_provider
        )

        assert kb.vector_store is not None
        assert kb.embedding_provider is not None
        assert kb.chunking_config["max_chunk_size"] == 500

    @pytest.mark.asyncio
    async def test_from_config(self):
        """Test creating from configuration."""
        config = {
            "vector_store": {"backend": "memory", "dimensions": 384},
            "embedding_provider": "echo",
            "embedding_model": "test",
            "chunking": {"max_chunk_size": 300, "chunk_overlap": 30},
        }

        kb = await RAGKnowledgeBase.from_config(config)
        assert kb.chunking_config["max_chunk_size"] == 300
        assert kb.chunking_config["chunk_overlap"] == 30

    @pytest.mark.asyncio
    async def test_load_markdown_document(self):
        """Test loading a markdown document."""
        config = {
            "vector_store": {"backend": "memory", "dimensions": 384},
            "embedding_provider": "echo",
            "embedding_model": "test",
        }

        kb = await RAGKnowledgeBase.from_config(config)

        # Load test document
        test_doc = Path(__file__).parent / "test_docs" / "quickstart.md"
        num_chunks = await kb.load_markdown_document(test_doc)

        assert num_chunks > 0

    @pytest.mark.asyncio
    async def test_load_documents_from_directory(self):
        """Test loading multiple documents from a directory."""
        config = {
            "vector_store": {"backend": "memory", "dimensions": 384},
            "embedding_provider": "echo",
            "embedding_model": "test",
        }

        kb = await RAGKnowledgeBase.from_config(config)

        # Load all test documents
        test_docs_dir = Path(__file__).parent / "test_docs"
        results = await kb.load_documents_from_directory(test_docs_dir)

        assert results["total_files"] >= 2  # quickstart.md and configuration.md
        assert results["total_chunks"] > 0
        assert len(results["errors"]) == 0

    @pytest.mark.asyncio
    async def test_query(self):
        """Test querying the knowledge base."""
        config = {
            "vector_store": {"backend": "memory", "dimensions": 384},
            "embedding_provider": "echo",
            "embedding_model": "test",
        }

        kb = await RAGKnowledgeBase.from_config(config)

        # Load test document
        test_doc = Path(__file__).parent / "test_docs" / "quickstart.md"
        await kb.load_markdown_document(test_doc)

        # Query
        results = await kb.query("How do I install DynaBot?", k=3)

        assert len(results) > 0
        assert all("text" in r for r in results)
        assert all("source" in r for r in results)
        assert all("similarity" in r for r in results)

    @pytest.mark.asyncio
    async def test_query_with_filter(self):
        """Test querying with metadata filter."""
        config = {
            "vector_store": {"backend": "memory", "dimensions": 384},
            "embedding_provider": "echo",
            "embedding_model": "test",
        }

        kb = await RAGKnowledgeBase.from_config(config)

        # Load documents with metadata
        test_docs_dir = Path(__file__).parent / "test_docs"
        await kb.load_documents_from_directory(test_docs_dir)

        # Query (filter functionality depends on vector store implementation)
        results = await kb.query("configuration", k=5)

        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_query_min_similarity(self):
        """Test query with minimum similarity threshold."""
        config = {
            "vector_store": {"backend": "memory", "dimensions": 384},
            "embedding_provider": "echo",
            "embedding_model": "test",
        }

        kb = await RAGKnowledgeBase.from_config(config)

        # Load test document
        test_doc = Path(__file__).parent / "test_docs" / "quickstart.md"
        await kb.load_markdown_document(test_doc)

        # Query with high threshold (may return fewer results)
        results = await kb.query("installation", k=10, min_similarity=0.9)

        # Should only return highly similar chunks
        assert isinstance(results, list)
        assert all(r["similarity"] >= 0.9 for r in results)

    @pytest.mark.asyncio
    async def test_save_with_persistence(self):
        """Test saving knowledge base to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_path = os.path.join(tmpdir, "test_index")

            config = {
                "vector_store": {
                    "backend": "memory",
                    "dimensions": 384,
                    "persist_path": persist_path,
                },
                "embedding_provider": "echo",
                "embedding_model": "test",
            }

            kb = await RAGKnowledgeBase.from_config(config)

            # Load test document
            test_doc = Path(__file__).parent / "test_docs" / "quickstart.md"
            await kb.load_markdown_document(test_doc)

            # Save to disk
            await kb.save()

            # Verify file was created
            assert os.path.exists(persist_path), "Vector store file not created"

    @pytest.mark.asyncio
    async def test_close_saves_and_releases_resources(self):
        """Test that close() saves to disk and releases resources."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_path = os.path.join(tmpdir, "test_index")

            config = {
                "vector_store": {
                    "backend": "memory",
                    "dimensions": 384,
                    "persist_path": persist_path,
                },
                "embedding_provider": "echo",
                "embedding_model": "test",
            }

            kb = await RAGKnowledgeBase.from_config(config)

            # Load test document
            test_doc = Path(__file__).parent / "test_docs" / "quickstart.md"
            num_chunks = await kb.load_markdown_document(test_doc)
            assert num_chunks > 0

            # Close (should save automatically)
            await kb.close()

            # Verify file was created
            assert os.path.exists(persist_path), "Vector store not saved on close"

    @pytest.mark.asyncio
    async def test_close_without_persist_path(self):
        """Test that close() works even without persistence configured."""
        config = {
            "vector_store": {"backend": "memory", "dimensions": 384},
            "embedding_provider": "echo",
            "embedding_model": "test",
        }

        kb = await RAGKnowledgeBase.from_config(config)

        # Load test document
        test_doc = Path(__file__).parent / "test_docs" / "quickstart.md"
        await kb.load_markdown_document(test_doc)

        # Close should not raise even without persist_path
        await kb.close()

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test async context manager saves and closes automatically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_path = os.path.join(tmpdir, "test_index")

            config = {
                "vector_store": {
                    "backend": "memory",
                    "dimensions": 384,
                    "persist_path": persist_path,
                },
                "embedding_provider": "echo",
                "embedding_model": "test",
            }

            # Use as async context manager
            async with await RAGKnowledgeBase.from_config(config) as kb:
                # Load test document
                test_doc = Path(__file__).parent / "test_docs" / "quickstart.md"
                num_chunks = await kb.load_markdown_document(test_doc)
                assert num_chunks > 0

                # Query to verify it works
                results = await kb.query("installation", k=3)
                assert len(results) > 0

            # After exiting context, file should be saved
            assert os.path.exists(persist_path), "Vector store not saved after context exit"

    @pytest.mark.asyncio
    async def test_persistence_round_trip(self):
        """Test that saved data can be loaded back correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_path = os.path.join(tmpdir, "test_index")

            config = {
                "vector_store": {
                    "backend": "memory",
                    "dimensions": 384,
                    "persist_path": persist_path,
                },
                "embedding_provider": "echo",
                "embedding_model": "test",
            }

            # First: create and save
            kb1 = await RAGKnowledgeBase.from_config(config)
            test_doc = Path(__file__).parent / "test_docs" / "quickstart.md"
            num_chunks = await kb1.load_markdown_document(test_doc)
            await kb1.close()

            # Second: load from disk and query
            kb2 = await RAGKnowledgeBase.from_config(config)

            # Should be able to query the loaded data
            results = await kb2.query("installation", k=3)
            assert len(results) > 0, "No results after loading from disk"

            # Verify chunk count matches
            count = await kb2.vector_store.count()
            assert count == num_chunks, f"Expected {num_chunks} chunks, got {count}"

            await kb2.close()


class TestKnowledgeFactory:
    """Tests for knowledge base factory function."""

    @pytest.mark.asyncio
    async def test_create_rag_kb(self):
        """Test creating RAG knowledge base from config."""
        config = {
            "type": "rag",
            "vector_store": {"backend": "memory", "dimensions": 384},
            "embedding_provider": "echo",
            "embedding_model": "test",
        }

        kb = await create_knowledge_base_from_config(config)
        assert isinstance(kb, RAGKnowledgeBase)

    @pytest.mark.asyncio
    async def test_default_type(self):
        """Test that default type is RAG."""
        config = {
            "vector_store": {"backend": "memory", "dimensions": 384},
            "embedding_provider": "echo",
            "embedding_model": "test",
        }

        kb = await create_knowledge_base_from_config(config)
        assert isinstance(kb, RAGKnowledgeBase)

    @pytest.mark.asyncio
    async def test_invalid_type(self):
        """Test error handling for invalid type."""
        config = {
            "type": "invalid",
            "vector_store": {"backend": "memory", "dimensions": 384},
        }

        with pytest.raises(ValueError, match="Unknown knowledge base type"):
            await create_knowledge_base_from_config(config)


class TestKnowledgeSearchTool:
    """Tests for KnowledgeSearchTool."""

    @pytest.mark.asyncio
    async def test_tool_initialization(self):
        """Test tool initialization."""
        config = {
            "vector_store": {"backend": "memory", "dimensions": 384},
            "embedding_provider": "echo",
            "embedding_model": "test",
        }

        kb = await RAGKnowledgeBase.from_config(config)
        tool = KnowledgeSearchTool(knowledge_base=kb)

        assert tool.name == "knowledge_search"
        assert tool.knowledge_base is kb

    @pytest.mark.asyncio
    async def test_tool_schema(self):
        """Test tool schema definition."""
        config = {
            "vector_store": {"backend": "memory", "dimensions": 384},
            "embedding_provider": "echo",
            "embedding_model": "test",
        }

        kb = await RAGKnowledgeBase.from_config(config)
        tool = KnowledgeSearchTool(knowledge_base=kb)

        schema = tool.schema
        assert schema["type"] == "object"
        assert "query" in schema["properties"]
        assert "max_results" in schema["properties"]
        assert "query" in schema["required"]

    @pytest.mark.asyncio
    async def test_tool_execute(self):
        """Test tool execution."""
        config = {
            "vector_store": {"backend": "memory", "dimensions": 384},
            "embedding_provider": "echo",
            "embedding_model": "test",
        }

        kb = await RAGKnowledgeBase.from_config(config)

        # Load test document
        test_doc = Path(__file__).parent / "test_docs" / "quickstart.md"
        await kb.load_markdown_document(test_doc)

        tool = KnowledgeSearchTool(knowledge_base=kb)

        # Execute tool
        result = await tool.execute(query="How to install?", max_results=2)

        assert "query" in result
        assert "results" in result
        assert "num_results" in result
        assert result["query"] == "How to install?"
        assert isinstance(result["results"], list)

    @pytest.mark.asyncio
    async def test_tool_max_results_clamping(self):
        """Test that max_results is clamped to valid range."""
        config = {
            "vector_store": {"backend": "memory", "dimensions": 384},
            "embedding_provider": "echo",
            "embedding_model": "test",
        }

        kb = await RAGKnowledgeBase.from_config(config)
        tool = KnowledgeSearchTool(knowledge_base=kb)

        # Test with value too high
        result = await tool.execute(query="test", max_results=100)
        assert result["num_results"] <= 10

        # Test with value too low
        result = await tool.execute(query="test", max_results=0)
        # Should be clamped to at least 1
        assert isinstance(result, dict)
