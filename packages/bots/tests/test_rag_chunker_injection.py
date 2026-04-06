"""Tests for custom chunker injection in RAGKnowledgeBase."""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_bots.knowledge import RAGKnowledgeBase
from dataknobs_data.vector.stores import VectorStoreFactory
from dataknobs_llm.llm import LLMProviderFactory
from dataknobs_xization.chunking import Chunker, DocumentInfo, MarkdownTreeChunker
from dataknobs_xization.markdown.md_chunker import Chunk, ChunkMetadata


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

SAMPLE_MD = """\
# Title

Introduction paragraph.

## Section A

Content of section A.

## Section B

Content of section B.
"""


class StubChunker(Chunker):
    """Chunker that returns a single chunk with a marker."""

    def __init__(self, marker: str = "stub"):
        self.marker = marker

    def chunk(
        self,
        content: str,
        document_info: DocumentInfo | None = None,
    ) -> list[Chunk]:
        return [
            Chunk(
                text=f"[{self.marker}] {content[:50]}",
                metadata=ChunkMetadata(
                    chunk_index=0,
                    chunk_size=len(content),
                    content_length=len(content),
                    embedding_text=f"[{self.marker}] {content[:50]}",
                ),
            ),
        ]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> StubChunker:
        return cls(marker=config.get("marker", "stub"))


async def _make_kb(
    chunking_config: dict[str, Any] | None = None,
    chunker: Chunker | None = None,
) -> RAGKnowledgeBase:
    """Create a RAGKnowledgeBase with memory vector store + echo provider."""
    factory = VectorStoreFactory()
    vector_store = factory.create(backend="memory", dimensions=384)
    await vector_store.initialize()

    llm_factory = LLMProviderFactory(is_async=True)
    provider = llm_factory.create({"provider": "echo", "model": "test"})
    await provider.initialize()

    return RAGKnowledgeBase(
        vector_store=vector_store,
        embedding_provider=provider,
        chunking_config=chunking_config,
        chunker=chunker,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRAGChunkerInjection:
    """Test that RAGKnowledgeBase uses the chunker abstraction."""

    @pytest.mark.asyncio
    async def test_default_chunker_is_markdown_tree(self):
        """Without explicit chunker or 'chunker' config key, the default
        MarkdownTreeChunker is used (backward compat)."""
        kb = await _make_kb()
        assert isinstance(kb._chunker, MarkdownTreeChunker)

    @pytest.mark.asyncio
    async def test_default_produces_chunks(self):
        """Default chunker produces the same kind of chunks as before."""
        kb = await _make_kb()
        count = await kb.load_markdown_text(SAMPLE_MD, source="test.md")
        assert count > 0
        total = await kb.count()
        assert total == count

    @pytest.mark.asyncio
    async def test_chunker_from_config_key(self):
        """Config with 'chunker' key selects a registered chunker."""
        from dataknobs_xization.chunking import register_chunker, chunker_registry

        register_chunker("stub_test", StubChunker, override=True)
        try:
            kb = await _make_kb(chunking_config={"chunker": "stub_test", "marker": "via-config"})
            assert isinstance(kb._chunker, StubChunker)

            count = await kb.load_markdown_text(SAMPLE_MD, source="test.md")
            assert count == 1  # StubChunker always returns 1 chunk
        finally:
            chunker_registry.unregister("stub_test")

    @pytest.mark.asyncio
    async def test_explicit_chunker_instance(self):
        """Pre-built chunker instance takes precedence over config."""
        stub = StubChunker(marker="explicit")
        kb = await _make_kb(chunker=stub)
        assert kb._chunker is stub

        count = await kb.load_markdown_text(SAMPLE_MD, source="test.md")
        assert count == 1

    @pytest.mark.asyncio
    async def test_explicit_chunker_overrides_config(self):
        """When both chunker= and chunking_config= are provided, the
        explicit instance wins."""
        stub = StubChunker(marker="winner")
        kb = await _make_kb(
            chunking_config={"max_chunk_size": 100},
            chunker=stub,
        )
        assert kb._chunker is stub

    @pytest.mark.asyncio
    async def test_from_config_passes_chunking_to_registry(self):
        """RAGKnowledgeBase.from_config() passes the chunking dict
        through to create_chunker()."""
        config = {
            "vector_store": {"backend": "memory", "dimensions": 384},
            "embedding_provider": "echo",
            "embedding_model": "test",
            "chunking": {"max_chunk_size": 300},
        }
        kb = await RAGKnowledgeBase.from_config(config)
        assert isinstance(kb._chunker, MarkdownTreeChunker)

    @pytest.mark.asyncio
    async def test_custom_chunker_receives_document_info(self):
        """The chunker receives DocumentInfo with source and metadata."""
        received: list[DocumentInfo | None] = []

        class SpyChunker(Chunker):
            def chunk(self, content: str, document_info: DocumentInfo | None = None) -> list[Chunk]:
                received.append(document_info)
                return [
                    Chunk(
                        text=content[:20],
                        metadata=ChunkMetadata(
                            chunk_index=0,
                            chunk_size=20,
                            content_length=20,
                            embedding_text=content[:20],
                        ),
                    ),
                ]

        kb = await _make_kb(chunker=SpyChunker())
        await kb.load_markdown_text(SAMPLE_MD, source="spy_test.md", metadata={"key": "val"})

        assert len(received) == 1
        info = received[0]
        assert info is not None
        assert info.source == "spy_test.md"
        assert info.metadata == {"key": "val"}
