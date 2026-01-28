"""Tests for KnowledgeIngestionService.

Uses real DataKnobs implementations for testing:
- MemoryVectorStore: In-memory vector store with full API (via factory)
- EchoProvider: Deterministic embeddings for reproducible tests

This approach catches integration issues early and validates real behavior.
"""

import tempfile
from pathlib import Path

import pytest

from dataknobs_bots.knowledge import (
    EnsureIngestionResult,
    IngestionResult,
    KnowledgeIngestionService,
    RAGKnowledgeBase,
)
from dataknobs_data.vector.stores import VectorStoreFactory
from dataknobs_llm.llm import LLMProviderFactory


# ============================================================================
# Fixtures - Real implementations
# ============================================================================


@pytest.fixture
async def memory_vector_store():
    """Create a real in-memory vector store for testing."""
    factory = VectorStoreFactory()
    store = factory.create(backend="memory", dimensions=384)
    await store.initialize()
    yield store
    await store.close()


@pytest.fixture
async def echo_provider():
    """Create a real echo provider for deterministic embeddings."""
    llm_factory = LLMProviderFactory(is_async=True)
    provider = llm_factory.create({"provider": "echo", "model": "test"})
    await provider.initialize()
    yield provider
    await provider.close()


@pytest.fixture
async def real_knowledge_base(memory_vector_store, echo_provider):
    """Create a real RAGKnowledgeBase with in-memory components."""
    kb = RAGKnowledgeBase(
        vector_store=memory_vector_store,
        embedding_provider=echo_provider,
        chunking_config={"max_chunk_size": 500, "chunk_overlap": 50},
    )
    yield kb
    # Note: Don't close here since we yield the vector store/provider separately
    # and they handle their own cleanup


# ============================================================================
# Tests
# ============================================================================


class TestKnowledgeIngestionService:
    """Tests for the KnowledgeIngestionService."""

    @pytest.mark.asyncio
    async def test_check_needs_ingestion_empty_store(
        self,
        real_knowledge_base: RAGKnowledgeBase,
    ) -> None:
        """Should detect when vector store is empty."""
        service = KnowledgeIngestionService()

        # Real knowledge base starts empty
        needs_ingestion = await service.check_needs_ingestion(real_knowledge_base)
        assert needs_ingestion is True

        # Verify via actual vector store count
        count = await real_knowledge_base.count()
        assert count == 0

    @pytest.mark.asyncio
    async def test_check_needs_ingestion_populated_store(
        self,
        real_knowledge_base: RAGKnowledgeBase,
    ) -> None:
        """Should detect when vector store is already populated."""
        service = KnowledgeIngestionService()

        # Pre-populate with real documents
        with tempfile.TemporaryDirectory() as tmp_dir:
            doc_path = Path(tmp_dir) / "test.md"
            doc_path.write_text("# Test Document\n\nSome content here.")
            await real_knowledge_base.load_markdown_document(str(doc_path))

        # Should now detect as populated
        needs_ingestion = await service.check_needs_ingestion(real_knowledge_base)
        assert needs_ingestion is False

        # Verify actual count
        count = await real_knowledge_base.count()
        assert count >= 1

    @pytest.mark.asyncio
    async def test_check_needs_ingestion_force_reingest(
        self,
        real_knowledge_base: RAGKnowledgeBase,
    ) -> None:
        """Should always return True when force_reingest is set."""
        service = KnowledgeIngestionService(force_reingest=True)

        # Pre-populate
        with tempfile.TemporaryDirectory() as tmp_dir:
            doc_path = Path(tmp_dir) / "test.md"
            doc_path.write_text("# Test\n\nContent.")
            await real_knowledge_base.load_markdown_document(str(doc_path))

        # Should still need ingestion due to force flag
        needs_ingestion = await service.check_needs_ingestion(real_knowledge_base)
        assert needs_ingestion is True

    @pytest.mark.asyncio
    async def test_ensure_ingested_skips_disabled(
        self,
        real_knowledge_base: RAGKnowledgeBase,
    ) -> None:
        """Should skip if knowledge base is disabled."""
        service = KnowledgeIngestionService()

        config = {"enabled": False}

        result = await service.ensure_ingested(real_knowledge_base, config)

        assert result.skipped is True
        assert result.reason == "knowledge_base_disabled"

    @pytest.mark.asyncio
    async def test_ensure_ingested_skips_if_populated(
        self,
        real_knowledge_base: RAGKnowledgeBase,
    ) -> None:
        """Should skip if already populated."""
        service = KnowledgeIngestionService()

        # Pre-populate
        with tempfile.TemporaryDirectory() as tmp_dir:
            doc_path = Path(tmp_dir) / "existing.md"
            doc_path.write_text("# Existing\n\nAlready here.")
            await real_knowledge_base.load_markdown_document(str(doc_path))

            config = {
                "enabled": True,
                "documents_path": tmp_dir,
            }

            result = await service.ensure_ingested(real_knowledge_base, config)

            assert result.skipped is True
            assert result.reason == "already_populated"

    @pytest.mark.asyncio
    async def test_ensure_ingested_runs_if_empty(
        self,
        real_knowledge_base: RAGKnowledgeBase,
    ) -> None:
        """Should run ingestion if empty."""
        service = KnowledgeIngestionService()

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create test documents
            Path(tmp_dir, "doc1.md").write_text("# Document 1\n\nFirst document.")
            Path(tmp_dir, "doc2.md").write_text("# Document 2\n\nSecond document.")

            config = {
                "enabled": True,
                "documents_path": tmp_dir,
                "document_pattern": "**/*.md",
            }

            result = await service.ensure_ingested(real_knowledge_base, config)

            assert result.skipped is False
            assert result.total_files >= 1
            assert result.total_chunks >= 1

            # Verify actual vector store was populated
            count = await real_knowledge_base.count()
            assert count > 0

    @pytest.mark.asyncio
    async def test_handles_missing_documents_path(
        self,
        real_knowledge_base: RAGKnowledgeBase,
    ) -> None:
        """Should handle missing documents_path gracefully."""
        service = KnowledgeIngestionService()

        config = {"enabled": True}  # No documents_path

        result = await service.ensure_ingested(real_knowledge_base, config)

        assert result.error is not None
        assert "documents_path" in result.error

    @pytest.mark.asyncio
    async def test_handles_nonexistent_path(
        self,
        real_knowledge_base: RAGKnowledgeBase,
    ) -> None:
        """Should handle nonexistent documents_path gracefully."""
        service = KnowledgeIngestionService()

        config = {
            "enabled": True,
            "documents_path": "/nonexistent/path/that/does/not/exist",
        }

        result = await service.ensure_ingested(real_knowledge_base, config)

        assert result.error is not None
        assert "does not exist" in result.error


class TestEnsureIngestionResult:
    """Tests for EnsureIngestionResult dataclass."""

    def test_success_property_skipped(self) -> None:
        """Skipped is considered success."""
        result = EnsureIngestionResult(skipped=True, reason="already_populated")
        assert result.success is True

    def test_success_property_completed(self) -> None:
        """Completed without errors is success."""
        result = EnsureIngestionResult(total_files=2, total_chunks=10)
        assert result.success is True

    def test_success_property_with_error(self) -> None:
        """With error is not success."""
        result = EnsureIngestionResult(error="Something failed")
        assert result.success is False

    def test_success_property_with_file_errors(self) -> None:
        """With individual file errors is not success."""
        result = EnsureIngestionResult(
            errors=[{"file": "bad.md", "error": "Parse error"}]
        )
        assert result.success is False

    def test_to_dict(self) -> None:
        """Should convert to dictionary correctly."""
        result = EnsureIngestionResult(
            skipped=True,
            reason="already_populated",
        )
        d = result.to_dict()
        assert d["skipped"] is True
        assert d["reason"] == "already_populated"
        assert d["success"] is True
        assert "total_files" in d
        assert "total_chunks" in d

    @pytest.mark.asyncio
    async def test_from_ingestion_result(self) -> None:
        """Should convert IngestionResult to EnsureIngestionResult."""
        from datetime import datetime, timezone

        # Create an IngestionResult (as from KnowledgeIngestionManager)
        ingestion_result = IngestionResult(
            domain_id="test-domain",
            files_processed=5,
            chunks_created=25,
            files_skipped=1,
            errors=[{"file": "bad.md", "error": "Parse error"}],
        )
        ingestion_result.completed_at = datetime.now(timezone.utc)

        # Convert to EnsureIngestionResult
        ensure_result = EnsureIngestionResult.from_ingestion_result(ingestion_result)

        assert ensure_result.total_files == 5
        assert ensure_result.total_chunks == 25
        assert len(ensure_result.errors) == 1
        assert ensure_result.success is False  # Has errors


class TestRAGKnowledgeBaseCount:
    """Tests for the new count() method on RAGKnowledgeBase."""

    @pytest.mark.asyncio
    async def test_count_empty(self) -> None:
        """Should return 0 for empty knowledge base."""
        config = {
            "vector_store": {"backend": "memory", "dimensions": 384},
            "embedding_provider": "echo",
            "embedding_model": "test",
        }

        kb = await RAGKnowledgeBase.from_config(config)
        count = await kb.count()
        assert count == 0

    @pytest.mark.asyncio
    async def test_count_after_loading(self) -> None:
        """Should return correct count after loading documents."""
        config = {
            "vector_store": {"backend": "memory", "dimensions": 384},
            "embedding_provider": "echo",
            "embedding_model": "test",
        }

        kb = await RAGKnowledgeBase.from_config(config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            doc_path = Path(tmp_dir) / "test.md"
            doc_path.write_text("# Test\n\nSome content here.")
            num_chunks = await kb.load_markdown_document(str(doc_path))

            count = await kb.count()
            assert count == num_chunks
            assert count > 0
