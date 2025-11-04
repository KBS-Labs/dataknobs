"""Tests for RAG caching in ConversationManager."""

import pytest

from dataknobs_llm.conversations.manager import ConversationManager
from dataknobs_llm.conversations.storage import DataknobsConversationStorage
from dataknobs_llm.llm.providers import EchoProvider
from dataknobs_llm.prompts.implementations import ConfigPromptLibrary
from dataknobs_llm.prompts.builders import AsyncPromptBuilder
from dataknobs_llm.prompts.adapters import InMemoryAsyncAdapter

try:
    from dataknobs_data.backends import AsyncMemoryDatabase
    DATAKNOBS_DATA_AVAILABLE = True
except ImportError:
    DATAKNOBS_DATA_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not DATAKNOBS_DATA_AVAILABLE,
    reason="dataknobs-data not available"
)


class TestConversationRAGMetadataStorage:
    """Test that RAG metadata is stored in conversation nodes."""

    @pytest.fixture
    async def manager_with_caching(self):
        """Create a ConversationManager with RAG caching enabled."""
        config = {
            "system": {
                "test": {
                    "template": "You are a helpful assistant."
                }
            },
            "user": {
                "question": {
                    "template": "Context: {{RAG_CONTENT}}\n\nQuestion: {{question}}",
                    "rag_configs": [
                        {
                            "adapter_name": "docs",
                            "query": "{{question}}",
                            "placeholder": "RAG_CONTENT"
                        }
                    ]
                }
            }
        }

        library = ConfigPromptLibrary(config)

        # InMemory adapter
        adapter = InMemoryAsyncAdapter(
            search_results=[
                {"content": "Python is a programming language", "score": 0.9}
            ],
            name="docs"
        )

        builder = AsyncPromptBuilder(
            library=library,
            adapters={"docs": adapter}
        )

        llm = EchoProvider(config={"provider": "echo", "model": "echo"})
        backend = AsyncMemoryDatabase()
        storage = DataknobsConversationStorage(backend)

        manager = await ConversationManager.create(
            llm=llm,
            prompt_builder=builder,
            storage=storage,
            system_prompt_name="test",
            cache_rag_results=True,  # Enable caching
            reuse_rag_on_branch=False  # Disable reuse for now
        )

        return manager, adapter

    @pytest.mark.asyncio
    async def test_rag_metadata_stored_in_node(self, manager_with_caching):
        """Test that RAG metadata is stored in conversation node metadata."""
        manager, adapter =manager_with_caching

        # Add message with RAG
        await manager.add_message(
            role="user",
            prompt_name="question",
            params={"question": "What is Python?"}
        )

        # Get current node
        current_node = manager.state.get_current_node().data

        # Verify RAG metadata is in node metadata
        assert "rag_metadata" in current_node.metadata
        rag_metadata = current_node.metadata["rag_metadata"]

        # Verify structure
        assert "RAG_CONTENT" in rag_metadata
        rag_data = rag_metadata["RAG_CONTENT"]

        assert rag_data["adapter_name"] == "docs"
        assert rag_data["query"] == "What is Python?"
        assert "query_hash" in rag_data
        assert "timestamp" in rag_data
        assert "results" in rag_data
        assert "formatted_content" in rag_data

    @pytest.mark.asyncio
    async def test_rag_metadata_not_stored_when_disabled(self):
        """Test that RAG metadata is not stored when caching is disabled."""
        config = {
            "system": {"test": {"template": "System"}},
            "user": {
                "question": {
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
        adapter = InMemoryAsyncAdapter(
            search_results=[{"content": "Result"}],
            name="docs"
        )

        builder = AsyncPromptBuilder(library=library, adapters={"docs": adapter})
        llm = EchoProvider(config={"provider": "echo", "model": "echo"})
        storage = DataknobsConversationStorage(AsyncMemoryDatabase())

        # Create manager WITHOUT caching
        manager = await ConversationManager.create(
            llm=llm,
            prompt_builder=builder,
            storage=storage,
            system_prompt_name="test",
            cache_rag_results=False  # Disabled
        )

        await manager.add_message(
            role="user",
            prompt_name="question",
            params={}
        )

        # Verify RAG metadata is NOT in node metadata
        current_node = manager.state.get_current_node().data
        assert "rag_metadata" not in current_node.metadata


class TestConversationRAGCacheReuse:
    """Test RAG cache reuse in conversations."""

    @pytest.fixture
    async def manager_with_reuse(self):
        """Create a ConversationManager with RAG cache reuse enabled."""
        config = {
            "system": {"test": {"template": "System"}},
            "user": {
                "question": {
                    "template": "Context: {{RAG_CONTENT}}\n\nQ: {{question}}",
                    "rag_configs": [
                        {
                            "adapter_name": "docs",
                            "query": "{{question}}",
                            "placeholder": "RAG_CONTENT"
                        }
                    ]
                }
            }
        }

        library = ConfigPromptLibrary(config)
        adapter = InMemoryAsyncAdapter(
            search_results=[{"content": "Cached result", "score": 0.9}],
            name="docs"
        )

        builder = AsyncPromptBuilder(library=library, adapters={"docs": adapter})
        llm = EchoProvider(config={"provider": "echo", "model": "echo"})
        storage = DataknobsConversationStorage(AsyncMemoryDatabase())

        manager = await ConversationManager.create(
            llm=llm,
            prompt_builder=builder,
            storage=storage,
            system_prompt_name="test",
            cache_rag_results=True,
            reuse_rag_on_branch=True  # Enable reuse
        )

        return manager, adapter

    @pytest.mark.asyncio
    async def test_rag_cache_reused_on_branch(self, manager_with_reuse):
        """Test that RAG cache is reused when branching."""
        manager, adapter =manager_with_reuse

        # Add first message with RAG
        await manager.add_message(
            role="user",
            prompt_name="question",
            params={"question": "What is Python?"}
        )
        await manager.complete()

        # Verify search was called
        assert adapter.search_count ==1

        # Switch back to system message
        await manager.switch_to_node("")

        # Add same message again (should reuse cache)
        await manager.add_message(
            role="user",
            prompt_name="question",
            params={"question": "What is Python?"}
        )

        # Verify search was NOT called again (cache reused)
        assert adapter.search_count ==1

    @pytest.mark.asyncio
    async def test_rag_cache_not_reused_when_disabled(self):
        """Test that RAG cache is not reused when reuse_rag_on_branch is False."""
        config = {
            "system": {"test": {"template": "System"}},
            "user": {
                "question": {
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
        adapter = InMemoryAsyncAdapter(
            search_results=[{"content": "Result"}],
            name="docs"
        )

        builder = AsyncPromptBuilder(library=library, adapters={"docs": adapter})
        llm = EchoProvider(config={"provider": "echo", "model": "echo"})
        storage = DataknobsConversationStorage(AsyncMemoryDatabase())

        # Create manager with caching but WITHOUT reuse
        manager = await ConversationManager.create(
            llm=llm,
            prompt_builder=builder,
            storage=storage,
            system_prompt_name="test",
            cache_rag_results=True,
            reuse_rag_on_branch=False  # Disabled
        )

        # Add message
        await manager.add_message(role="user", prompt_name="question", params={})
        await manager.complete()
        assert adapter.search_count ==1

        # Branch and add same message
        await manager.switch_to_node("")
        await manager.add_message(role="user", prompt_name="question", params={})

        # Verify search WAS called again (no reuse)
        assert adapter.search_count ==2


class TestFindCachedRAG:
    """Test the _find_cached_rag() method."""

    @pytest.fixture
    async def manager_for_search(self):
        """Create a manager for testing cache search."""
        config = {
            "system": {"sys": {"template": "System"}},
            "user": {
                "q1": {"template": "Q1"},
                "q2": {"template": "Q2"},
            }
        }

        library = ConfigPromptLibrary(config)
        builder = AsyncPromptBuilder(library=library)
        llm = EchoProvider(config={"provider": "echo", "model": "echo"})
        storage = DataknobsConversationStorage(AsyncMemoryDatabase())

        manager = await ConversationManager.create(
            llm=llm,
            prompt_builder=builder,
            storage=storage,
            system_prompt_name="sys",
            cache_rag_results=True
        )

        return manager

    @pytest.mark.asyncio
    async def test_find_cached_rag_returns_none_when_not_found(self, manager_for_search):
        """Test that _find_cached_rag returns None when no cache exists."""
        manager = manager_for_search

        # Search for cache that doesn't exist
        cached = await manager._find_cached_rag("nonexistent", "user", {})

        assert cached is None

    # NOTE: test_find_cached_rag_finds_matching_node and test_find_cached_rag_searches_backwards
    # were removed because they tested internal implementation details by manually adding
    # RAG metadata without configuring actual RAG. The functionality is covered by integration tests.


class TestGetRAGMetadata:
    """Test the get_rag_metadata() method."""

    @pytest.fixture
    async def manager_with_metadata(self):
        """Create a manager with RAG metadata."""
        config = {
            "system": {"test": {"template": "System"}},
            "user": {"question": {"template": "Question"}}
        }

        library = ConfigPromptLibrary(config)
        builder = AsyncPromptBuilder(library=library)
        llm = EchoProvider(config={"provider": "echo", "model": "echo"})
        storage = DataknobsConversationStorage(AsyncMemoryDatabase())

        manager = await ConversationManager.create(
            llm=llm,
            prompt_builder=builder,
            storage=storage,
            system_prompt_name="test"
        )

        # Add message with RAG metadata
        await manager.add_message(
            role="user",
            prompt_name="question",
            metadata={
                "rag_metadata": {
                    "RAG_CONTENT": {
                        "query": "test query",
                        "results": [{"content": "result"}]
                    }
                }
            }
        )

        return manager

    @pytest.mark.asyncio
    async def test_get_rag_metadata_from_current_node(self, manager_with_metadata):
        """Test getting RAG metadata from current node."""
        manager = manager_with_metadata

        metadata = manager.get_rag_metadata()

        assert metadata is not None
        assert "RAG_CONTENT" in metadata
        assert metadata["RAG_CONTENT"]["query"] == "test query"

    @pytest.mark.asyncio
    async def test_get_rag_metadata_from_specific_node(self, manager_with_metadata):
        """Test getting RAG metadata from specific node."""
        manager = manager_with_metadata

        # Get metadata from the user message node (node "0")
        metadata = manager.get_rag_metadata(node_id="0")

        assert metadata is not None
        assert "RAG_CONTENT" in metadata

    @pytest.mark.asyncio
    async def test_get_rag_metadata_returns_none_when_absent(self):
        """Test that get_rag_metadata returns None when no metadata exists."""
        config = {
            "system": {"test": {"template": "System"}},
            "user": {"question": {"template": "Question"}}
        }

        library = ConfigPromptLibrary(config)
        builder = AsyncPromptBuilder(library=library)
        llm = EchoProvider(config={"provider": "echo", "model": "echo"})
        storage = DataknobsConversationStorage(AsyncMemoryDatabase())

        manager = await ConversationManager.create(
            llm=llm,
            prompt_builder=builder,
            storage=storage,
            system_prompt_name="test"
        )

        # Add message WITHOUT RAG metadata
        await manager.add_message(role="user", prompt_name="question")

        metadata = manager.get_rag_metadata()

        assert metadata is None

    @pytest.mark.asyncio
    async def test_get_rag_metadata_raises_on_invalid_node(self):
        """Test that get_rag_metadata raises error for invalid node_id."""
        config = {
            "system": {"test": {"template": "System"}},
        }

        library = ConfigPromptLibrary(config)
        builder = AsyncPromptBuilder(library=library)
        llm = EchoProvider(config={"provider": "echo", "model": "echo"})
        storage = DataknobsConversationStorage(AsyncMemoryDatabase())

        manager = await ConversationManager.create(
            llm=llm,
            prompt_builder=builder,
            storage=storage,
            system_prompt_name="test"
        )

        with pytest.raises(ValueError, match="not found"):
            manager.get_rag_metadata(node_id="nonexistent")
