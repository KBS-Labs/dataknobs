"""Tests for RAG caching functionality in prompt builders."""

import pytest

from dataknobs_llm.prompts.builders import AsyncPromptBuilder, PromptBuilder
from dataknobs_llm.prompts.implementations import ConfigPromptLibrary
from dataknobs_llm.prompts.adapters import (
    InMemoryAdapter,
    InMemoryAsyncAdapter,
)


class TestRAGMetadataCapture:
    """Test RAG metadata capture in prompt builders."""

    def test_async_builder_captures_metadata(self):
        """Test AsyncPromptBuilder captures RAG metadata when requested."""
        config = {
            "system": {
                "test": {
                    "template": "Context: {{RAG_CONTENT}}",
                    "rag_configs": [
                        {
                            "adapter_name": "docs",
                            "query": "test query",
                            "k": 3,
                            "placeholder": "RAG_CONTENT"
                        }
                    ]
                }
            }
        }

        library = ConfigPromptLibrary(config)

        # InMemory adapter that returns test results
        adapter = InMemoryAsyncAdapter(
            search_results=[
                {"content": "Result 1", "score": 0.9},
                {"content": "Result 2", "score": 0.8},
            ],
            name="docs"
        )

        builder = AsyncPromptBuilder(
            library=library,
            adapters={"docs": adapter}
        )

        # Render with metadata capture
        import asyncio
        result = asyncio.run(builder.render_system_prompt(
            "test",
            return_rag_metadata=True
        ))

        # Verify metadata was captured
        assert result.rag_metadata is not None
        assert "RAG_CONTENT" in result.rag_metadata

        rag_data = result.rag_metadata["RAG_CONTENT"]
        assert rag_data["adapter_name"] == "docs"
        assert rag_data["query"] == "test query"
        assert rag_data["k"] == 3
        assert "query_hash" in rag_data
        assert "timestamp" in rag_data
        assert "results" in rag_data
        assert "formatted_content" in rag_data
        assert len(rag_data["results"]) == 2

    def test_sync_builder_captures_metadata(self):
        """Test PromptBuilder captures RAG metadata when requested."""
        config = {
            "system": {
                "test": {
                    "template": "Context: {{RAG_CONTENT}}",
                    "rag_configs": [
                        {
                            "adapter_name": "docs",
                            "query": "test query",
                            "k": 3,
                            "placeholder": "RAG_CONTENT"
                        }
                    ]
                }
            }
        }

        library = ConfigPromptLibrary(config)

        # InMemory adapter that returns test results
        adapter = InMemoryAdapter(
            search_results=[
                {"content": "Result 1", "score": 0.9},
                {"content": "Result 2", "score": 0.8},
            ],
            name="docs"
        )

        builder = PromptBuilder(
            library=library,
            adapters={"docs": adapter}
        )

        # Render with metadata capture
        result = builder.render_system_prompt(
            "test",
            return_rag_metadata=True
        )

        # Verify metadata was captured
        assert result.rag_metadata is not None
        assert "RAG_CONTENT" in result.rag_metadata

        rag_data = result.rag_metadata["RAG_CONTENT"]
        assert rag_data["adapter_name"] == "docs"
        assert rag_data["query"] == "test query"
        assert rag_data["k"] == 3
        assert "query_hash" in rag_data
        assert "timestamp" in rag_data
        assert "results" in rag_data
        assert "formatted_content" in rag_data

    def test_metadata_not_captured_by_default(self):
        """Test that RAG metadata is not captured by default."""
        config = {
            "system": {
                "test": {
                    "template": "Context: {{RAG_CONTENT}}",
                    "rag_configs": [
                        {
                            "adapter_name": "docs",
                            "query": "test query",
                            "placeholder": "RAG_CONTENT"
                        }
                    ]
                }
            }
        }

        library = ConfigPromptLibrary(config)
        adapter = InMemoryAdapter(
            search_results=[{"content": "Result 1"}],
            name="docs"
        )

        builder = PromptBuilder(
            library=library,
            adapters={"docs": adapter}
        )

        # Render without requesting metadata
        result = builder.render_system_prompt("test")

        # Verify metadata was not captured
        assert result.rag_metadata is None

    def test_multiple_rag_configs_captured(self):
        """Test metadata capture with multiple RAG configs."""
        config = {
            "system": {
                "test": {
                    "template": "Docs: {{DOCS}}\nExamples: {{EXAMPLES}}",
                    "rag_configs": [
                        {
                            "adapter_name": "docs",
                            "query": "documentation",
                            "placeholder": "DOCS"
                        },
                        {
                            "adapter_name": "examples",
                            "query": "code examples",
                            "placeholder": "EXAMPLES"
                        }
                    ]
                }
            }
        }

        library = ConfigPromptLibrary(config)

        docs_adapter = InMemoryAdapter(
            search_results=[{"content": "Doc 1"}],
            name="docs"
        )

        examples_adapter = InMemoryAdapter(
            search_results=[{"content": "Example 1"}],
            name="examples"
        )

        builder = PromptBuilder(
            library=library,
            adapters={"docs": docs_adapter, "examples": examples_adapter}
        )

        result = builder.render_system_prompt("test", return_rag_metadata=True)

        # Verify both RAG configs captured
        assert result.rag_metadata is not None
        assert "DOCS" in result.rag_metadata
        assert "EXAMPLES" in result.rag_metadata
        assert result.rag_metadata["DOCS"]["query"] == "documentation"
        assert result.rag_metadata["EXAMPLES"]["query"] == "code examples"


class TestRAGCacheReuse:
    """Test RAG cache reuse functionality."""

    def test_cached_rag_reused_in_sync_builder(self):
        """Test that cached RAG is reused in PromptBuilder."""
        config = {
            "system": {
                "test": {
                    "template": "Context: {{RAG_CONTENT}}",
                    "rag_configs": [
                        {
                            "adapter_name": "docs",
                            "query": "test query",
                            "placeholder": "RAG_CONTENT"
                        }
                    ]
                }
            }
        }

        library = ConfigPromptLibrary(config)
        adapter = InMemoryAdapter(
            search_results=[{"content": "Original result"}],
            name="docs"
        )

        builder = PromptBuilder(
            library=library,
            adapters={"docs": adapter}
        )

        # First render - captures metadata
        result1 = builder.render_system_prompt("test", return_rag_metadata=True)
        original_content = result1.content

        # Verify search was called
        assert adapter.search_count == 1

        # Second render - reuses cached RAG
        result2 = builder.render_system_prompt(
            "test",
            cached_rag=result1.rag_metadata
        )

        # Verify search was NOT called again
        assert adapter.search_count == 1

        # Verify content is identical
        assert result2.content == original_content

        # Verify metadata indicates cache was used
        assert result2.metadata["used_cached_rag"] is True

    def test_cached_rag_reused_in_async_builder(self):
        """Test that cached RAG is reused in AsyncPromptBuilder."""
        config = {
            "system": {
                "test": {
                    "template": "Context: {{RAG_CONTENT}}",
                    "rag_configs": [
                        {
                            "adapter_name": "docs",
                            "query": "test query",
                            "placeholder": "RAG_CONTENT"
                        }
                    ]
                }
            }
        }

        library = ConfigPromptLibrary(config)
        adapter = InMemoryAsyncAdapter(
            search_results=[{"content": "Original result"}],
            name="docs"
        )

        builder = AsyncPromptBuilder(
            library=library,
            adapters={"docs": adapter}
        )

        import asyncio

        async def test_caching():
            # First render - captures metadata
            result1 = await builder.render_system_prompt("test", return_rag_metadata=True)
            original_content = result1.content

            # Verify search was called
            assert adapter.search_count == 1

            # Second render - reuses cached RAG
            result2 = await builder.render_system_prompt(
                "test",
                cached_rag=result1.rag_metadata
            )

            # Verify search was NOT called again
            assert adapter.search_count == 1

            # Verify content is identical
            assert result2.content == original_content

            # Verify metadata indicates cache was used
            assert result2.metadata["used_cached_rag"] is True

        asyncio.run(test_caching())

    def test_cached_rag_passthrough_metadata(self):
        """Test that cached RAG metadata is passed through when reused."""
        config = {
            "system": {
                "test": {
                    "template": "{{RAG_CONTENT}}",
                    "rag_configs": [
                        {
                            "adapter_name": "docs",
                            "query": "test",
                            "placeholder": "RAG_CONTENT"
                        }
                    ]
                }
            }
        }

        library = ConfigPromptLibrary(config)
        adapter = InMemoryAdapter(
            search_results=[{"content": "Result"}],
            name="docs"
        )

        builder = PromptBuilder(library=library, adapters={"docs": adapter})

        # First render with metadata capture
        result1 = builder.render_system_prompt("test", return_rag_metadata=True)

        # Second render with cache and metadata request
        result2 = builder.render_system_prompt(
            "test",
            cached_rag=result1.rag_metadata,
            return_rag_metadata=True
        )

        # Verify metadata was passed through
        assert result2.rag_metadata is not None
        assert result2.rag_metadata == result1.rag_metadata


class TestRAGCacheMatchingLogic:
    """Test RAG cache matching and hashing logic."""

    def test_query_hash_computation(self):
        """Test that query hashes are computed correctly."""
        config = {
            "system": {
                "test": {
                    "template": "{{RAG_CONTENT}}",
                    "rag_configs": [
                        {
                            "adapter_name": "docs",
                            "query": "python decorators",
                            "placeholder": "RAG_CONTENT"
                        }
                    ]
                }
            }
        }

        library = ConfigPromptLibrary(config)
        adapter = InMemoryAdapter(
            search_results=[{"content": "Result"}],
            name="docs"
        )

        builder = PromptBuilder(library=library, adapters={"docs": adapter})

        # Render and capture metadata
        result = builder.render_system_prompt("test", return_rag_metadata=True)

        # Verify query hash exists and is SHA256 (64 hex chars)
        query_hash = result.rag_metadata["RAG_CONTENT"]["query_hash"]
        assert isinstance(query_hash, str)
        assert len(query_hash) == 64
        assert all(c in "0123456789abcdef" for c in query_hash)

    def test_identical_queries_produce_same_hash(self):
        """Test that identical queries produce the same hash."""
        config = {
            "system": {
                "test": {
                    "template": "{{RAG_CONTENT}}",
                    "rag_configs": [
                        {
                            "adapter_name": "docs",
                            "query": "python decorators",
                            "placeholder": "RAG_CONTENT"
                        }
                    ]
                }
            }
        }

        library = ConfigPromptLibrary(config)
        adapter = InMemoryAdapter(
            search_results=[{"content": "Result"}],
            name="docs"
        )

        builder = PromptBuilder(library=library, adapters={"docs": adapter})

        # Render twice
        result1 = builder.render_system_prompt("test", return_rag_metadata=True)
        adapter.reset()
        result2 = builder.render_system_prompt("test", return_rag_metadata=True)

        # Verify hashes are identical
        hash1 = result1.rag_metadata["RAG_CONTENT"]["query_hash"]
        hash2 = result2.rag_metadata["RAG_CONTENT"]["query_hash"]
        assert hash1 == hash2

    def test_different_queries_produce_different_hashes(self):
        """Test that different queries produce different hashes."""
        library = ConfigPromptLibrary({
            "system": {
                "test1": {
                    "template": "{{RAG_CONTENT}}",
                    "rag_configs": [{
                        "adapter_name": "docs",
                        "query": "python decorators",
                        "placeholder": "RAG_CONTENT"
                    }]
                },
                "test2": {
                    "template": "{{RAG_CONTENT}}",
                    "rag_configs": [{
                        "adapter_name": "docs",
                        "query": "javascript promises",
                        "placeholder": "RAG_CONTENT"
                    }]
                }
            }
        })

        adapter = InMemoryAdapter(
            search_results=[{"content": "Result"}],
            name="docs"
        )

        builder = PromptBuilder(library=library, adapters={"docs": adapter})

        result1 = builder.render_system_prompt("test1", return_rag_metadata=True)
        result2 = builder.render_system_prompt("test2", return_rag_metadata=True)

        hash1 = result1.rag_metadata["RAG_CONTENT"]["query_hash"]
        hash2 = result2.rag_metadata["RAG_CONTENT"]["query_hash"]

        # Hashes should be different
        assert hash1 != hash2

    def test_extract_formatted_content_from_cache(self):
        """Test that formatted content can be extracted from cache."""
        config = {
            "system": {
                "test": {
                    "template": "Context: {{RAG_CONTENT}}",
                    "rag_configs": [
                        {
                            "adapter_name": "docs",
                            "query": "test",
                            "placeholder": "RAG_CONTENT",
                            "header": "# Documentation\n",
                            "item_template": "- {{content}}\n"
                        }
                    ]
                }
            }
        }

        library = ConfigPromptLibrary(config)
        adapter = InMemoryAdapter(
            search_results=[
                {"content": "Item 1"},
                {"content": "Item 2"}
            ],
            name="docs"
        )

        builder = PromptBuilder(library=library, adapters={"docs": adapter})

        # Capture metadata
        result1 = builder.render_system_prompt("test", return_rag_metadata=True)

        # Extract formatted content using the helper method
        extracted = builder._extract_formatted_content_from_cache(result1.rag_metadata)

        # Verify extracted content matches what was in the cache
        assert "RAG_CONTENT" in extracted
        expected_content = result1.rag_metadata["RAG_CONTENT"]["formatted_content"]
        assert extracted["RAG_CONTENT"] == expected_content


class TestRAGCachingWithParameters:
    """Test RAG caching with parameterized queries."""

    def test_parameterized_query_hashing(self):
        """Test that parameterized queries are hashed after substitution."""
        config = {
            "system": {
                "test": {
                    "template": "{{RAG_CONTENT}}",
                    "rag_configs": [
                        {
                            "adapter_name": "docs",
                            "query": "{{language}} documentation",
                            "placeholder": "RAG_CONTENT"
                        }
                    ]
                }
            }
        }

        library = ConfigPromptLibrary(config)
        adapter = InMemoryAdapter(
            search_results=[{"content": "Result"}],
            name="docs"
        )

        builder = PromptBuilder(library=library, adapters={"docs": adapter})

        # Render with python
        result1 = builder.render_system_prompt(
            "test",
            params={"language": "python"},
            return_rag_metadata=True
        )

        # Render with javascript
        result2 = builder.render_system_prompt(
            "test",
            params={"language": "javascript"},
            return_rag_metadata=True
        )

        # Verify queries are different after substitution
        query1 = result1.rag_metadata["RAG_CONTENT"]["query"]
        query2 = result2.rag_metadata["RAG_CONTENT"]["query"]
        assert query1 == "python documentation"
        assert query2 == "javascript documentation"

        # Verify hashes are different
        hash1 = result1.rag_metadata["RAG_CONTENT"]["query_hash"]
        hash2 = result2.rag_metadata["RAG_CONTENT"]["query_hash"]
        assert hash1 != hash2
